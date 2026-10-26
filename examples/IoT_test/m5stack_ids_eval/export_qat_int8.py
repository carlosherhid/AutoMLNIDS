"""
Export a QAT-converted checkpoint (qat_pytorch_model_exp*.pth) to a C header with the
integer parameters of every layer (uint8 activations, int8 weights, int32 biases and a
float requantization multiplier), and verify, before anything is flashed, that the integer
arithmetic the firmware performs reproduces PyTorch's quantized model.

The firmware (arduino_m5_ids_int8/arduino_m5_ids_int8.ino) computes, for every layer l:

    acc[o] = bias_i32[o] + sum_i (q_in[i] - zp_in) * w[o][i]            (int32)
    q_out  = clamp(lrintf((float) acc * M) + zp_out, 0, 255)            M = s_in * s_w / s_out
    q_out  = max(q_out, zp_out)  after the hidden layers (ReLU on a quantized tensor)

and quantizes the input as q = clamp(lrintf(x / s_0) + zp_0, 0, 255). `emulate()` below
performs exactly these operations with numpy (float32 for the same steps as the firmware) and
is compared with PyTorch's output; the header is then parsed back and a scalar re-implementation
of the firmware loop is run on a few rows to check the exported layout (row-major weights).

Usage:
    python export_qat_int8.py --qat_model <qat_checkpoint> \
        --output_header arduino_m5_ids_int8/exp6_qat_int8_model.h --model_name exp6_qat_int8 \
        --check_csv <test_device_scaled.csv>
The check CSV has 61 features and the label (last column), without header, as sent to the device.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.quantization import DeQuantStub, QuantStub, convert, prepare_qat


class Base(nn.Module):
    def __init__(self, main_block):
        super().__init__()
        self.main_block = main_block

    def forward(self, x):
        return self.main_block(x)


class QATWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.quant, self.model, self.dequant = QuantStub(), model, DeQuantStub()

    def forward(self, x):
        return self.dequant(self.model(self.quant(x)))


def load_quantized_model(state_dict):
    """Rebuild the converted model from the checkpoint alone (layer shapes come from the packed weights)."""
    indices = sorted(int(m.group(1)) for k in state_dict if (m := re.match(r'model\.main_block\.(\d+)\._packed_params\._packed_params$', k)))
    if not indices:
        raise ValueError('No quantized Linear layers found: is this a QAT-converted checkpoint?')
    modules = [nn.Identity() for _ in range(indices[-1] + 1)]
    for n, idx in enumerate(indices):
        weight, _ = state_dict[f'model.main_block.{idx}._packed_params._packed_params']
        modules[idx] = nn.Linear(weight.shape[1], weight.shape[0])
        if n < len(indices) - 1:
            modules[idx + 1] = nn.ReLU()  # the activation follows every hidden Linear (tanh is not supported by the firmware)
    model = QATWrapper(Base(nn.Sequential(*modules)))
    model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
    prepare_qat(model, inplace=True)
    model = convert(model.eval())
    model.load_state_dict(state_dict)
    return model.eval(), indices


def extract_parameters(model, indices):
    """Integer parameters of the converted model."""
    s_in, z_in = float(model.quant.scale), int(model.quant.zero_point)
    params = {'input_scale': np.float32(s_in), 'input_zp': z_in, 'layers': []}
    for n, idx in enumerate(indices):
        layer = model.model.main_block[idx]
        qweight = layer.weight()
        assert qweight.dtype == torch.qint8 and qweight.q_zero_point() == 0, 'expected symmetric int8 weights'
        w = qweight.int_repr().numpy().astype(np.int8)
        s_w, s_out, z_out = float(qweight.q_scale()), float(layer.scale), int(layer.zero_point)
        bias = layer.bias().detach().numpy().astype(np.float64)
        params['layers'].append({
            'weight': w,
            'bias_i32': np.rint(bias / (s_in * s_w)).astype(np.int32),
            'requant': np.float32(s_in * s_w / s_out),
            'in_zp': z_in, 'out_zp': z_out, 'relu': n < len(indices) - 1,
            'out_scale': s_out, 'in_scale': s_in, 'w_scale': s_w,
        })
        s_in, z_in = s_out, z_out
    return params


def emulate(params, x):
    """Integer forward pass identical to the firmware; returns the output codes (uint8) of the last layer."""
    x = x.astype(np.float32)
    q = np.clip(np.rint(x / params['input_scale']) + params['input_zp'], 0, 255).astype(np.int64)
    for layer in params['layers']:
        acc = (q - layer['in_zp']).astype(np.float64) @ layer['weight'].astype(np.float64).T  # exact: |acc| << 2**53
        acc = acc + layer['bias_i32']
        v = acc.astype(np.float32) * layer['requant']  # float32 like the firmware
        q = np.clip(np.rint(v) + layer['out_zp'], 0, 255).astype(np.int64)
        if layer['relu']:
            q = np.maximum(q, layer['out_zp'])
    return q


def format_array(values):
    return ', '.join(str(int(v)) for v in values.reshape(-1))


def write_header(params, path, name, source):
    layers = params['layers']
    sizes = [layers[0]['weight'].shape[1]] + [l['weight'].shape[0] for l in layers]
    f32 = lambda v: f'{float(v):.9g}f'  # 9 significant digits round-trip a float32
    lines = ['#pragma once', '#include <stdint.h>', f'// Auto-generated by export_qat_int8.py from {source}', '',
             f'static const int {name}_num_layers = {len(layers)};',
             f'static const int {name}_layer_sizes[{len(sizes)}] = {{{", ".join(map(str, sizes))}}};',
             f'static const float {name}_input_scale = {f32(params["input_scale"])};',
             f'static const int {name}_input_zero_point = {params["input_zp"]};',
             f'static const float {name}_requant[{len(layers)}] = {{{", ".join(f32(l["requant"]) for l in layers)}}};',
             f'static const int {name}_in_zero_point[{len(layers)}] = {{{", ".join(str(l["in_zp"]) for l in layers)}}};',
             f'static const int {name}_out_zero_point[{len(layers)}] = {{{", ".join(str(l["out_zp"]) for l in layers)}}};', '']
    for i, l in enumerate(layers):
        out_f, in_f = l['weight'].shape
        lines.append(f'// layer {i}: {in_f} -> {out_f}, row-major [out][in]')
        lines.append(f'static const int8_t {name}_w_{i}[{out_f * in_f}] = {{{format_array(l["weight"])}}};')
        lines.append(f'static const int32_t {name}_b_{i}[{out_f}] = {{{format_array(l["bias_i32"])}}};')
        lines.append('')
    lines.append(f'static const int8_t* const {name}_weights[{len(layers)}] = {{{", ".join(f"{name}_w_{i}" for i in range(len(layers)))}}};')
    lines.append(f'static const int32_t* const {name}_biases[{len(layers)}] = {{{", ".join(f"{name}_b_{i}" for i in range(len(layers)))}}};')
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return sizes


def parse_header(path, name):
    """Read the header back (as the C compiler would see it) into the same parameter structure."""
    text = Path(path).read_text(encoding='utf-8')

    def array(pattern):
        return np.array([float(v.strip().rstrip('f')) for v in re.search(pattern, text).group(1).split(',')])

    n = int(re.search(rf'{name}_num_layers = (\d+);', text).group(1))
    params = {'input_scale': np.float32(re.search(rf'{name}_input_scale = ([^;]+)f;', text).group(1)),
              'input_zp': int(re.search(rf'{name}_input_zero_point = (\d+);', text).group(1)), 'layers': []}
    requant, in_zp, out_zp = (array(rf'{name}_{key}\[\d+\] = \{{([^}}]*)\}}') for key in ('requant', 'in_zero_point', 'out_zero_point'))
    sizes = array(rf'{name}_layer_sizes\[\d+\] = \{{([^}}]*)\}}').astype(int)
    for i in range(n):
        w = array(rf'{name}_w_{i}\[\d+\] = \{{([^}}]*)\}}').astype(np.int8).reshape(sizes[i + 1], sizes[i])
        b = array(rf'{name}_b_{i}\[\d+\] = \{{([^}}]*)\}}').astype(np.int32)
        params['layers'].append({'weight': w, 'bias_i32': b, 'requant': np.float32(requant[i]), 'in_zp': int(in_zp[i]),
                                 'out_zp': int(out_zp[i]), 'relu': i < n - 1})
    return params


def firmware_scalar(params, x):
    """Line-by-line transcription of the C loop (python ints / float32), for a single sample."""
    q = [int(min(max(int(np.rint(np.float32(v) / params['input_scale'])) + params['input_zp'], 0), 255)) for v in x]
    for layer in params['layers']:
        w, out = layer['weight'], []
        for o in range(w.shape[0]):
            acc = int(layer['bias_i32'][o])
            for i in range(w.shape[1]):
                acc += (q[i] - layer['in_zp']) * int(w[o, i])
            v = int(np.rint(np.float32(acc) * layer['requant'])) + layer['out_zp']
            v = min(max(v, 0), 255)
            if layer['relu'] and v < layer['out_zp']:
                v = layer['out_zp']
            out.append(v)
        q = out
    return q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qat_model', required=True)
    parser.add_argument('--output_header', required=True)
    parser.add_argument('--model_name', default='exp6_qat_int8')
    parser.add_argument('--check_csv', default=None, help='device CSV (61 features + label, no header) used to check the export')
    parser.add_argument('--engine', default='onednn', help='PyTorch engine for the reference predictions (an engine that does not saturate)')
    args = parser.parse_args()

    if args.engine in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = args.engine
    print('reference engine:', torch.backends.quantized.engine)
    state = torch.load(args.qat_model, map_location='cpu')
    model, indices = load_quantized_model(state)
    params = extract_parameters(model, indices)
    sizes = write_header(params, args.output_header, args.model_name, Path(args.qat_model).name)
    print('layer sizes:', sizes, '| int8 weights:', sum(l['weight'].size for l in params['layers']), 'bytes | int32 biases:',
          sum(l['bias_i32'].size for l in params['layers']) * 4, 'bytes')

    # the header, as the compiler will read it, must describe the same network
    parsed = parse_header(args.output_header, args.model_name)
    assert all((a['weight'] == b['weight']).all() and (a['bias_i32'] == b['bias_i32']).all() and a['requant'] == b['requant']
               for a, b in zip(params['layers'], parsed['layers'])) and parsed['input_scale'] == params['input_scale']
    print('header parsed back: identical parameters')

    if args.check_csv:
        data = pd.read_csv(args.check_csv, header=None)
        x, y = data.iloc[:, :-1].values.astype(np.float32), data.iloc[:, -1].values.astype(int)
        with torch.no_grad():
            logits = model(torch.tensor(x)).numpy()
        torch_pred = logits.argmax(1)
        codes = emulate(params, x)
        emu_pred = codes.argmax(1)
        last = params['layers'][-1]
        deq = (codes - last['out_zp']) * last['out_scale']
        print(f'{len(x)} rows | accuracy: PyTorch quantized {np.mean(torch_pred == y):.4f} | integer emulation {np.mean(emu_pred == y):.4f}')
        print(f'prediction agreement emulation vs PyTorch: {np.mean(emu_pred == torch_pred):.4f} '
              f'| max |logit difference|: {np.abs(deq - logits).max():.4f} (one output step = {last["out_scale"]:.4f})')
        sample = min(20, len(x))
        scalar = np.array([firmware_scalar(parsed, row) for row in x[:sample]])
        assert (scalar == codes[:sample]).all(), 'scalar firmware loop differs from the vectorized emulation'
        print(f'scalar transcription of the firmware loop == emulation on {sample} rows')


if __name__ == '__main__':
    main()
