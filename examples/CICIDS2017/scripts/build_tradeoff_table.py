"""
Consolidated accuracy / F1 / size / latency / sparsity trade-off table across
experiments, reporting latency percentiles rather than only the mean.

Parses compressed_models/execution_log_exp*.txt (accuracy, size) and, if the
matching checkpoint + test.csv are present, also measures actual CPU inference
latency (per-sample, at each of --latency_batch_sizes) and per-layer weight
sparsity. Latency is measured at multiple batch sizes because at batch_size=1
QAT's per-call quantize/dequantize overhead can dominate for models this small,
making QAT *slower* than FP32 despite being smaller - --latency_batch_sizes lets
that overhead amortize over a bigger int8 GEMM to see whether/where it crosses over.

Usage:
    python scripts/build_tradeoff_table.py --predictor_path <predictor_dir> --data_dir <data_dir> --logs_dir <logs_dir>
"""
import argparse
import glob
import os
import platform
import re
import sys
import time

import pandas as pd
import torch
from autogluon.tabular import TabularPredictor

sys.path.insert(0, os.path.dirname(__file__))
from utils.log_parsing import parse_log_file  # noqa: E402
from utils.model import network_layers_sparsity, get_prunable  # noqa: E402
from autogluon_to_pytorch_qat import AutoReplicatedNN, QATWrapper, get_model_architecture  # noqa: E402

# Internal --experiment_number -> paper Experiment (see docs/EXPERIMENTS_LOG.md)
PAPER_EXPERIMENTS = {1: 'Exp. 1', 2: 'Exp. 2', 3: 'Exp. 3', 6: 'Exp. 4', 7: 'Exp. 5'}


def cpu_model():
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def measure_latency(model, X_sample, batch_size=1, n_repeats=200):
    """Per-sample latency in ms: runs `batch_size`-sized batches through the
    model and divides each call's wall-clock time by batch_size, so results
    at different batch sizes are directly comparable to the batch_size=1
    (single-sample, the realistic edge-inference case) measurement."""
    model.eval()
    n = len(X_sample)
    with torch.no_grad():
        warm_batch = X_sample[[i % n for i in range(batch_size)]]
        for _ in range(10):  # warm-up
            model(warm_batch)
        latencies_ms = []
        for i in range(n_repeats):
            idx = [(i * batch_size + j) % n for j in range(batch_size)]
            batch = X_sample[idx]
            start = time.perf_counter()
            model(batch)
            latencies_ms.append((time.perf_counter() - start) * 1000 / batch_size)
    s = pd.Series(latencies_ms)
    return {
        'latency_mean_ms': s.mean(), 'latency_p50_ms': s.median(),
        'latency_p95_ms': s.quantile(0.95), 'latency_p99_ms': s.quantile(0.99),
        'latency_std_ms': s.std(),
    }


def try_load_checkpoint(path, kind, architecture, input_feature_size):
    if not os.path.exists(path):
        return None
    base = AutoReplicatedNN(architecture, input_feature_size)
    if kind == 'normal':
        base.load_state_dict(torch.load(path, map_location='cpu'))
        base.eval()
        return base
    wrapper = QATWrapper(base)
    wrapper.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
    torch.quantization.prepare_qat(wrapper, inplace=True)  # requires model.training == True
    wrapper = torch.quantization.convert(wrapper.eval())
    wrapper.load_state_dict(torch.load(path, map_location='cpu'))
    wrapper.eval()
    return wrapper


def main():
    parser = argparse.ArgumentParser(description='Build the consolidated accuracy/size/latency/sparsity table')
    parser.add_argument('--predictor_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--logs_dir', type=str, required=True, help='Directory with execution_log_exp*.txt and *_expN.pth')
    parser.add_argument('--output_csv', type=str, default=None)
    parser.add_argument('--skip_latency', action='store_true', help='Skip the (slower) on-CPU latency measurement')
    parser.add_argument('--latency_batch_sizes', type=str, default='1,8,32,64,128',
                         help='Comma-separated batch sizes to measure per-sample latency at')
    parser.add_argument('--engine', type=str, default=None,
                        help='torch.backends.quantized.engine for the quantized models (default: PyTorch default, x86)')
    parser.add_argument('--threads', type=int, default=None,
                        help='torch.set_num_threads for the latency measurements (default: PyTorch default)')
    parser.add_argument('--only_paper_experiments', action='store_true',
                        help='Only process the experiments reported in the paper (internal numbers 1, 2, 3, 6, 7)')
    args = parser.parse_args()
    batch_sizes = [int(b) for b in args.latency_batch_sizes.split(',')]

    if args.engine:
        if args.engine not in torch.backends.quantized.supported_engines:
            raise SystemExit(f'engine {args.engine!r} not supported; supported: {torch.backends.quantized.supported_engines}')
        torch.backends.quantized.engine = args.engine
    if args.threads:
        torch.set_num_threads(args.threads)
    environment = {'engine': torch.backends.quantized.engine, 'threads': torch.get_num_threads(),
                   'node': platform.node(), 'cpu_model': cpu_model(), 'torch_version': torch.__version__}
    print('measurement environment:', environment)

    log_files = sorted(
        glob.glob(os.path.join(args.logs_dir, 'execution_log_exp*.txt')),
        key=lambda p: int(re.search(r'exp(\d+)', p).group(1)))
    if args.only_paper_experiments:
        log_files = [p for p in log_files if int(re.search(r'exp(\d+)', p).group(1)) in PAPER_EXPERIMENTS]

    predictor = None
    architecture = None
    X_sample = None
    if not args.skip_latency:
        # require_version_match=False / require_py_version_match=False:
        # predictor may have been trained on an older AutoGluon/Python than
        # what's installed - see autogluon_to_pytorch_qat.py.
        predictor = TabularPredictor.load(
            args.predictor_path, require_version_match=False, require_py_version_match=False)
        test_data = pd.read_csv(os.path.join(args.data_dir, 'test.csv'), index_col='ID')
        X_sample = torch.tensor(test_data.drop(columns=['Label']).values, dtype=torch.float32)
        architecture, _, input_feature_size = get_model_architecture(predictor, X_sample.shape[1])

    rows = []
    for log_file in log_files:
        exp_number = int(re.search(r'exp(\d+)', log_file).group(1))
        parsed = parse_log_file(log_file)
        parsed['experiment_number'] = exp_number
        parsed['paper_experiment'] = PAPER_EXPERIMENTS.get(exp_number, '')
        parsed.update(environment)

        if not args.skip_latency:
            normal_path = os.path.join(args.logs_dir, f'normal_pytorch_model_exp{exp_number}.pth')
            qat_path = os.path.join(args.logs_dir, f'qat_pytorch_model_exp{exp_number}.pth')

            normal_model = try_load_checkpoint(normal_path, 'normal', architecture, input_feature_size)
            if normal_model is not None:
                for bs in batch_sizes:
                    suffix = '' if bs == 1 else f'_b{bs}'
                    for k, v in measure_latency(normal_model, X_sample, batch_size=bs).items():
                        parsed[f'normal_{k.replace("latency_", f"latency{suffix}_", 1)}'] = v
                sparsity, _ = network_layers_sparsity(get_prunable(normal_model))
                parsed['normal_weight_sparsity'] = sparsity[0]

            qat_model = try_load_checkpoint(qat_path, 'qat', architecture, input_feature_size)
            if qat_model is not None:
                for bs in batch_sizes:
                    suffix = '' if bs == 1 else f'_b{bs}'
                    for k, v in measure_latency(qat_model, X_sample, batch_size=bs).items():
                        parsed[f'qat_{k.replace("latency_", f"latency{suffix}_", 1)}'] = v

        rows.append(parsed)

    df = pd.DataFrame(rows)
    out_path = args.output_csv or os.path.join(args.logs_dir, 'tradeoff_table.csv')
    df.to_csv(out_path, index=False)
    print(f'Saved trade-off table to {out_path}')

    cols = ['experiment_number', 'paper_experiment', 'autogluon_accuracy',
            'normal_test_accuracy', 'qat_test_accuracy', 'normal_size_kb', 'qat_size_kb']
    print(df[[c for c in cols if c in df.columns]].to_string(index=False))
    print('\nStarred rows (paper_experiment set) are the ones reported in the paper (Table 1).')


if __name__ == '__main__':
    main()
