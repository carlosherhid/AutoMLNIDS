"""
How much of each feature's resolution survives 8-bit input quantization, for each input
preprocessing compared in the paper (Min-Max only, near-zero rescaling, standardization,
log Min-Max, quantile).

The QAT model quantizes its whole input tensor with a single scale (QuantStub with a
min/max observer), so a feature whose values occupy a tiny part of the tensor-wide range
is mapped to a handful of the 256 integer codes. This script reproduces that mapping on the
training split (one scale from the global min/max over all features, 256 levels) and reports,
over the features that are not constant, how many distinct codes a feature uses and which
share of its values falls on its most frequent code. It is a diagnostic of the input
representation, not a measurement of the trained model's observers.

Usage:
    python scripts/input_resolution_diagnostic.py --data_dir ./balanced_binary \
        --output_csv ./paper_results/CICIDS2017/input_resolution_diagnostic.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from utils.preprocessing import apply_preprocessing  # noqa: E402

TAU, LAMBDA = 1e-6, 1e6  # near-zero threshold and factor of the proposed rescaling


def rescale_near_zero(train):
    features = [c for c in train.columns if c not in ('ID', 'Label')]
    out = train.copy()
    values = out[features].values
    out[features] = np.where(np.abs(values) < TAU, values * LAMBDA, values)
    return out


def quantize_codes(matrix, levels=256):
    """Affine per-tensor quantization with min/max range, as a QuantStub with a min/max observer."""
    low, high = float(matrix.min()), float(matrix.max())
    scale = (high - low) / (levels - 1)
    codes = np.clip(np.rint((matrix - low) / scale), 0, levels - 1).astype(np.int64)
    return codes, low, high, scale


def main():
    parser = argparse.ArgumentParser(description='Input resolution under 8-bit quantization, per preprocessing')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_csv', type=str, default=None)
    parser.add_argument('--max_rows', type=int, default=None, help='Subsample the training split (default: all rows)')
    args = parser.parse_args()

    train = pd.read_csv(os.path.join(args.data_dir, 'train.csv'))
    if args.max_rows:
        train = train.sample(args.max_rows, random_state=0)
    features = [c for c in train.columns if c not in ('ID', 'Label')]
    raw = train[features].values
    varying = [j for j in range(len(features)) if raw[:, j].min() != raw[:, j].max()]
    print(f'{len(train)} training rows, {len(features)} features ({len(varying)} not constant)')

    variants = {
        'Min-Max only': train,
        'Near-zero rescaling': rescale_near_zero(train),
        'Standardization': apply_preprocessing(train, train.iloc[:1], train.iloc[:1], 'standard')[0],
        'Log Min-Max': apply_preprocessing(train, train.iloc[:1], train.iloc[:1], 'logminmax')[0],
        'Quantile': apply_preprocessing(train, train.iloc[:1], train.iloc[:1], 'quantile')[0],
    }

    rows = []
    for name, data in variants.items():
        matrix = data[features].values.astype(np.float64)
        codes, low, high, scale = quantize_codes(matrix)
        distinct, mode_share = [], []
        for j in varying:
            counts = np.bincount(codes[:, j], minlength=256)
            distinct.append(int((counts > 0).sum()))
            mode_share.append(counts.max() / len(codes))
        distinct, mode_share = np.array(distinct), np.array(mode_share)
        rows.append({
            'preprocessing': name, 'tensor_min': low, 'tensor_max': high, 'step': scale,
            'mean_distinct_codes': distinct.mean(), 'median_distinct_codes': float(np.median(distinct)),
            'features_with_le_8_codes': int((distinct <= 8).sum()),
            'median_modal_code_share': float(np.median(mode_share)),
            'features_with_modal_share_gt_90pct': int((mode_share > 0.9).sum()),
            'n_features': len(varying),
        })
        print(f'{name:20s} range=[{low:.3g}, {high:.3g}] step={scale:.3g}  mean distinct codes={distinct.mean():.1f}  '
              f'features with <=8 codes={(distinct <= 8).sum()}  median modal share={np.median(mode_share):.3f}  '
              f'features with >90% on one code={(mode_share > 0.9).sum()}')

    if args.output_csv:
        pd.DataFrame(rows).to_csv(args.output_csv, index=False)
        print(f'Saved {args.output_csv}')


if __name__ == '__main__':
    main()
