"""Reference predictions for the device runs, on the raw Min-Max rows rescaled in float32 exactly like the board does:

  ref_fp32.csv  PyTorch FP32 model (normal_pytorch_model_exp6.pth)
  ref_int8.csv  integer emulation of the INT8 firmware (qat_pytorch_model_exp6.pth), the reference of the board

    python device_bench/reference_predictions.py        (run it from m5stack_ids_eval/)
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import export_qat_int8 as ex  # noqa: E402


def rescale_f32(x):
    """Equation 2 in float32, as in the firmware: x' = 1e6 * x if |x| < 1e-6."""
    x = x.astype(np.float32)
    return np.where(np.abs(x) < np.float32(1e-6), x * np.float32(1e6), x).astype(np.float32)


def load_fp32(state):
    idx = sorted(int(m.group(1)) for k in state if (m := re.match(r'main_block\.(\d+)\.weight$', k)))
    sizes = [state[f'main_block.{idx[0]}.weight'].shape[1]] + [state[f'main_block.{i}.weight'].shape[0] for i in idx]
    layers = []
    for n in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[n], sizes[n + 1]))
        if n < len(sizes) - 2:
            layers.append(nn.ReLU())
    model = nn.Sequential(*layers)
    remapped = {}
    for n, i in enumerate(idx):
        remapped[f'{2 * n}.weight'] = state[f'main_block.{i}.weight']
        remapped[f'{2 * n}.bias'] = state[f'main_block.{i}.bias']
    model.load_state_dict(remapped)
    return model.eval()


def metrics(pred, y, name):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    print(f'{name:28s} n={len(y)}  accuracy {(tp + tn) / len(y) * 100:.2f}%  recall {tp / (tp + fn) * 100:.2f}%  FPR {fp / (fp + tn) * 100:.2f}%')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=str(ROOT / 'data' / 'test_device.csv'))
    parser.add_argument('--fp32', default=str(ROOT / 'normal_pytorch_model_exp6.pth'))
    parser.add_argument('--qat', default=str(ROOT / 'qat_pytorch_model_exp6.pth'))
    parser.add_argument('--out_dir', default=str(HERE / 'ref'))
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data, header=None, nrows=args.limit)
    x = rescale_f32(df.iloc[:, :-1].values.astype(np.float32))
    y = df.iloc[:, -1].values.astype(int)

    fp32 = load_fp32(torch.load(args.fp32, map_location='cpu'))
    with torch.no_grad():
        p_fp32 = np.concatenate([fp32(torch.tensor(x[i:i + 8192])).argmax(1).numpy() for i in range(0, len(x), 8192)])

    if 'onednn' in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = 'onednn'
    model, indices = ex.load_quantized_model(torch.load(args.qat, map_location='cpu'))
    params = ex.extract_parameters(model, indices)
    p_int8 = np.concatenate([ex.emulate(params, x[i:i + 8192]).argmax(1) for i in range(0, len(x), 8192)])

    pd.DataFrame({'pred': p_fp32, 'true': y}).to_csv(out / 'ref_fp32.csv', index=False)
    pd.DataFrame({'pred': p_int8, 'true': y}).to_csv(out / 'ref_int8.csv', index=False)
    metrics(p_fp32, y, 'FP32 (PyTorch)')
    metrics(p_int8, y, 'INT8 (integer emulation)')
    print('FP32 vs INT8 predictions differ on', int((p_fp32 != p_int8).sum()), 'rows | files in', out)


if __name__ == '__main__':
    main()
