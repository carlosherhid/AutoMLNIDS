"""
Cross-platform table for the paper from the prediction CSVs written by run_serial_eval.py (device) and
benchmark_fp32.py (Mac): accuracy, precision, recall, F1 and latency statistics of

  * Mac, PyTorch FP32                     preds_mac_exp6_fp32_<n>.csv
  * M5Stack Basic, FP32 export            preds_m5_exp6_<n>.csv
  * M5Stack Basic, INT8 (QAT) export      preds_m5_exp6_int8_<n>.csv

and, if --qat_model is given, the agreement of the device's INT8 predictions with the bit-exact integer
emulation of the firmware (export_qat_int8.py) and with the FP32 predictions on the same samples.

Usage:
    python build_m5_table.py --dir ../../../paper_results/m5stack \
        --qat_model qat_pytorch_model_exp6.pth --device_csv_dir data
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

PLATFORMS = [('Mac (PyTorch, FP32)', 'preds_mac_exp6_fp32_{n}.csv', 'mac'),
             ('M5Stack Basic (FP32 export)', 'preds_m5_exp6_{n}.csv', 'm5'),
             ('M5Stack Basic (INT8, QAT export)', 'preds_m5_exp6_int8_{n}.csv', 'm5')]


def load(path, unit_us):
    df = pd.read_csv(path)
    df = df[(df['pred'] >= 0) & (df['true'] >= 0) & (df['latency_us'] >= 0)].reset_index(drop=True)
    df['latency_ms'] = df['latency_us'] / 1000.0
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True)
    parser.add_argument('--sizes', type=int, nargs='+', default=[100, 1000])
    parser.add_argument('--qat_model', default=None, help='QAT checkpoint for the emulation agreement check')
    parser.add_argument('--device_csv_dir', default='data', help='folder with test_device_scaled_<n>.csv')
    args = parser.parse_args()

    emulate_pred = {}
    if args.qat_model:
        import torch
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import export_qat_int8 as ex
        torch.backends.quantized.engine = 'onednn' if 'onednn' in torch.backends.quantized.supported_engines else torch.backends.quantized.engine
        model, idx = ex.load_quantized_model(torch.load(args.qat_model, map_location='cpu'))
        params = ex.extract_parameters(model, idx)
        for n in args.sizes:
            data = pd.read_csv(os.path.join(args.device_csv_dir, f'test_device_scaled_{n}.csv'), header=None)
            emulate_pred[n] = ex.emulate(params, data.iloc[:, :-1].values.astype(np.float32)).argmax(1)

    print('% Platform & Samples & Accuracy & Precision & Recall & F1-score & Mean latency (ms)')
    latency_rows, preds = [], {}
    for n in args.sizes:
        for label, pattern, _ in PLATFORMS:
            path = os.path.join(args.dir, pattern.format(n=n))
            if not os.path.exists(path):
                print(f'% missing: {path}')
                continue
            df = load(path, 'us')
            preds[(label, n)] = df
            y, p = df['true'].values, df['pred'].values
            print(f'{label} & {n} & {accuracy_score(y, p):.4f} & {precision_score(y, p, zero_division=0):.4f} & '
                  f'{recall_score(y, p, zero_division=0):.4f} & {f1_score(y, p, zero_division=0):.4f} & {df.latency_ms.mean():.4f} \\\\')
            lat = df['latency_ms']
            latency_rows.append((label, n, lat.mean(), lat.median(), lat.quantile(0.95), lat.quantile(0.99), lat.min(), lat.max(), int((p != y).sum())))

    print('\n% Latency statistics (ms): label, n, mean, median, p95, p99, min, max, errors')
    for row in latency_rows:
        print(' | '.join([str(row[0]), str(row[1])] + [f'{v:.4f}' for v in row[2:8]] + [str(row[8])]))

    print('\n% Agreement')
    for n in args.sizes:
        int8 = preds.get(('M5Stack Basic (INT8, QAT export)', n))
        fp32 = preds.get(('M5Stack Basic (FP32 export)', n))
        if int8 is None:
            continue
        if fp32 is not None:
            print(f'n={n}: INT8 vs FP32 on the board differ on {int((int8.pred.values != fp32.pred.values).sum())} of {len(int8)} samples')
        if n in emulate_pred:
            diff = int((int8.pred.values != emulate_pred[n][:len(int8)]).sum())
            print(f'n={n}: INT8 on the board vs bit-exact emulation differ on {diff} of {len(int8)} samples')
    fp32 = preds.get(('M5Stack Basic (FP32 export)', args.sizes[-1]))
    int8 = preds.get(('M5Stack Basic (INT8, QAT export)', args.sizes[-1]))
    if fp32 is not None and int8 is not None:
        print(f'mean latency ratio FP32/INT8 on the board (n={args.sizes[-1]}): {fp32.latency_ms.mean() / int8.latency_ms.mean():.2f}x')


if __name__ == '__main__':
    main()
