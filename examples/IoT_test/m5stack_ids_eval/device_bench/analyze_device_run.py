"""Metrics, latency distribution, stability and agreement with the reference of one device run.

    python analyze_device_run.py --preds runs/int8_full.csv --ref ref/ref_int8.csv [--name INT8]

Accepts the CSV of run_device_eval.py (idx,pred,true,latency_us,rescale_us) and the older 3-column files
(pred,true,latency_us). Writes <preds>_analysis.json next to the input.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds', required=True)
    parser.add_argument('--ref', default=None, help='reference predictions (pred,true) for the same rows, e.g. ref/ref_int8.csv')
    parser.add_argument('--name', default=None)
    args = parser.parse_args()

    d = pd.read_csv(args.preds)
    if 'idx' not in d.columns:
        d.insert(0, 'idx', np.arange(len(d)))
    if 'rescale_us' not in d.columns:
        d['rescale_us'] = np.nan
    name = args.name or Path(args.preds).stem
    answered = d[d['pred'] >= 0].reset_index(drop=True)
    res = {'name': name, 'rows': int(len(d)), 'answered': int(len(answered)), 'missing': int(len(d) - len(answered))}

    y, p = answered['true'].values.astype(int), answered['pred'].values.astype(int)
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    tn = int(((p == 0) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    n = len(y)
    acc, rec, fpr = (tp + tn) / n, tp / max(tp + fn, 1), fp / max(fp + tn, 1)
    prec = tp / max(tp + fp, 1)
    lo, hi = wilson(tp + tn, n)
    res.update({'accuracy_pct': acc * 100, 'accuracy_ci95_pct': [lo * 100, hi * 100], 'recall_pct': rec * 100, 'fpr_pct': fpr * 100,
                'precision_pct': prec * 100, 'f1_pct': 2 * prec * rec / max(prec + rec, 1e-12) * 100, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn})

    lat = answered['latency_us'].values.astype(float)
    res['latency_us'] = {'mean': lat.mean(), 'std': lat.std(ddof=1) if n > 1 else 0.0, 'min': lat.min(), 'p50': np.percentile(lat, 50),
                         'p95': np.percentile(lat, 95), 'p99': np.percentile(lat, 99), 'max': lat.max(),
                         'cv_pct': lat.std(ddof=1) / lat.mean() * 100 if n > 1 else 0.0}
    res['compute_throughput_per_s'] = 1e6 / lat.mean()
    if answered['rescale_us'].notna().any():
        res['rescale_us_mean'] = float(answered['rescale_us'].mean())
        res['rescale_share_of_latency_pct'] = float(answered['rescale_us'].mean() / lat.mean() * 100)

    block = max(len(answered) // 10, 1)
    blocks = [answered['latency_us'].iloc[i:i + block] for i in range(0, len(answered), block)]
    res['latency_mean_first_tenth_us'] = float(blocks[0].mean())
    res['latency_mean_last_tenth_us'] = float(blocks[-1].mean())
    res['latency_drift_pct'] = float((blocks[-1].mean() / blocks[0].mean() - 1) * 100)

    summ = Path(args.preds).with_name(Path(args.preds).stem + '_summary.json')
    if summ.exists():
        s = json.loads(summ.read_text(encoding='utf-8'))
        res['wall_seconds'] = s['wall_seconds']
        res['end_to_end_throughput_per_s'] = s['rows_answered'] / s['wall_seconds']

    if args.ref:
        ref = pd.read_csv(args.ref)
        ref = ref.iloc[answered['idx'].values].reset_index(drop=True)
        diff = np.flatnonzero(ref['pred'].values != p)
        res['reference_differences'] = int(len(diff))
        res['reference_first_differences_idx'] = [int(answered['idx'].iloc[i]) for i in diff[:10]]
        res['reference_accuracy_pct'] = float((ref['pred'].values == ref['true'].values).mean() * 100)

    print(f"== {name} ==")
    print(f"rows {res['rows']} | answered {res['answered']} | missing {res['missing']}")
    print(f"accuracy {res['accuracy_pct']:.2f}% (95% CI {lo * 100:.2f}-{hi * 100:.2f}) | recall {res['recall_pct']:.2f}% | FPR {res['fpr_pct']:.2f}% | "
          f"precision {res['precision_pct']:.2f}% | F1 {res['f1_pct']:.2f}%")
    L = res['latency_us']
    print(f"latency us: mean {L['mean']:.0f} | std {L['std']:.1f} | p50 {L['p50']:.0f} | p95 {L['p95']:.0f} | p99 {L['p99']:.0f} | "
          f"max {L['max']:.0f} | CV {L['cv_pct']:.2f}%")
    print(f"compute throughput {res['compute_throughput_per_s']:.1f} samples/s", end='')
    if 'end_to_end_throughput_per_s' in res:
        print(f" | end-to-end (with serial) {res['end_to_end_throughput_per_s']:.1f} samples/s over {res['wall_seconds'] / 60:.1f} min", end='')
    print()
    print(f"stability: mean latency first tenth {res['latency_mean_first_tenth_us']:.0f} us, last tenth {res['latency_mean_last_tenth_us']:.0f} us "
          f"({res['latency_drift_pct']:+.2f}%)")
    if 'rescale_us_mean' in res:
        print(f"rescaling on the board: {res['rescale_us_mean']:.1f} us per sample ({res['rescale_share_of_latency_pct']:.2f}% of the latency)")
    if args.ref:
        print(f"agreement with the reference: {res['reference_differences']} different predictions "
              f"(first rows: {res['reference_first_differences_idx']}) | reference accuracy {res['reference_accuracy_pct']:.2f}%")
    Path(args.preds).with_name(Path(args.preds).stem + '_analysis.json').write_text(json.dumps(res, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
