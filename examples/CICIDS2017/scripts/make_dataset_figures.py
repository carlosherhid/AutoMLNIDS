"""
Three data-description figures of the paper, regenerated from the data files in the common style of the other figures:

  * original_vs_processed_binary_distribution.pdf - class balance of CICIDS2017 before/after the preprocessing
    (original: label column of the raw CSVs; processed: the four splits of balanced_binary);
  * fig5.pdf - for each of the 61 features, the share of values that are exactly zero, non-zero but below the
    rescaling threshold tau, and between tau and half the 8-bit step (these still round to zero), on the training split;
  * fig8.pdf - accuracy of the AutoGluon network as features are removed one by one (sequential backward elimination,
    finetune split), with the 7 features kept in Experiment 5 marked.

Usage:
    python scripts/make_dataset_figures.py --data_dir ./balanced_binary --raw_dir ./original --output_dir ./paper_results/figures
"""
import argparse
import glob
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

STYLE = os.path.join(os.path.dirname(__file__), 'utils', 'presentation.mplstyle')
BLUE, ORANGE, GREY, PURPLE = '#4C72B0', '#DD8452', '#B8B8B8', '#8172B3'
TAU = 1e-6
STEP = 1 / 255  # 8-bit step of an input tensor in [0, 1]


def class_counts_raw(raw_dir):
    benign = malicious = 0
    for path in sorted(glob.glob(os.path.join(raw_dir, '*.csv'))):
        labels = pd.read_csv(path, usecols=lambda c: c.strip() == 'Label', encoding='latin-1').iloc[:, 0].astype(str).str.strip()
        benign += int((labels.str.upper() == 'BENIGN').sum())
        malicious += int((labels.str.upper() != 'BENIGN').sum())
    return benign, malicious


def class_counts_processed(data_dir):
    counts = {}
    for split in ('train', 'validation', 'test', 'finetune'):
        labels = pd.read_csv(os.path.join(data_dir, f'{split}.csv'), usecols=['Label'])['Label']
        counts[split] = labels.value_counts().sort_index().tolist()
    return counts


def plot_distribution(raw, processed, path):
    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    groups = ['Original', 'Processed']
    benign, malicious = [raw[0], processed[0]], [raw[1], processed[1]]
    x, width = np.arange(2), 0.36
    ax.bar(x - width / 2, benign, width, color=BLUE, label='Benign')
    ax.bar(x + width / 2, malicious, width, color=ORANGE, label='Malicious')
    for i in range(2):
        total = benign[i] + malicious[i]
        for offset, value in ((-width / 2, benign[i]), (width / 2, malicious[i])):
            ax.text(i + offset, value + 0.02e6, f'{value / total * 100:.1f}%\n({value:,})'.replace(',', '{,}').replace('{,}', ','), ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel('Number of flows')
    ax.set_ylim(0, max(benign + malicious) * 1.22)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(6, 6))
    ax.set_title('CICIDS2017: binary class distribution')
    ax.legend(loc='upper right', frameon=False)
    fig.savefig(path)
    plt.close(fig)


def plot_near_zero(train, path):
    plt.style.use(STYLE)
    features = [c for c in train.columns if c not in ('ID', 'Label')]
    values = train[features].values
    zero = (values == 0).mean(0) * 100
    tiny = ((np.abs(values) < TAU) & (values != 0)).mean(0) * 100
    mid = ((np.abs(values) >= TAU) & (np.abs(values) < STEP / 2)).mean(0) * 100
    x = np.arange(1, len(features) + 1)
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.bar(x, zero, color=GREY, label='exactly zero')
    ax.bar(x, tiny, bottom=zero, color=ORANGE, label=r'non-zero, below $\tau = 10^{-6}$ (rescaled in Experiment 4)')
    ax.bar(x, mid, bottom=zero + tiny, color=BLUE, label=r'between $\tau$ and $s/2$ (still rounds to zero)')
    ax.set_xlabel('Feature index')
    ax.set_ylabel('Share of values (%)')
    ax.set_xlim(0, len(features) + 1)
    ax.set_ylim(0, 100)
    ticks = [1] + list(range(10, len(features) + 1, 10))  # 1, 10, ..., 60 (a tick at 61 would overlap the one at 60)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], fontsize=8)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=1, frameon=False)
    fig.savefig(path)
    plt.close(fig)
    return {'n_features': len(features), 'zero_pct': zero.mean(), 'tiny_pct': tiny.mean(), 'mid_pct': mid.mean(),
            'features_ge50_round_to_zero': int(((zero + tiny + mid) >= 50).sum()), 'features_with_tiny': int((tiny > 0).sum())}


def plot_ranking(leaderboard, n_kept, path):
    plt.style.use(STYLE)
    data = leaderboard.sort_values('ID', ascending=False)
    fig, ax = plt.subplots(figsize=(3.7, 3.2))  # drawn small: it is shown at about 45% of the text width
    ax.plot(data['ID'], data['accuracy'] * 100, 'o', ms=3.0, color=BLUE, label='AutoGluon network, features removed one by one')
    ax.axhline(50, color=GREY, ls='--', lw=1.4, label='constant prediction (50%)')
    kept = data[data['ID'] == n_kept].iloc[0]
    ax.plot([n_kept], [kept['accuracy'] * 100], 'o', ms=8, mfc='none', mec=ORANGE, mew=2, label=f'{n_kept} features kept in Experiment 5')
    ax.set_xlabel('Number of active features')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlim(max(data['ID']) + 2, -2)
    ax.set_ylim(40, 101)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=1, frameon=False, fontsize=7.5)
    fig.savefig(path)
    plt.close(fig)
    return float(kept['accuracy'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--raw_dir', default=None)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    train = pd.read_csv(os.path.join(args.data_dir, 'train.csv'))
    stats = plot_near_zero(train, os.path.join(args.output_dir, 'fig5.pdf'))
    print('near-zero statistics (training split):', {k: round(v, 2) if isinstance(v, float) else v for k, v in stats.items()})

    ranking = pd.read_csv(os.path.join(args.data_dir, 'automl_search', 'feature_ranking_custom_sbe', 'leaderboard.csv'))
    important = [line.strip() for line in open(os.path.join(args.data_dir, 'automl_search', 'feature_ranking_custom_sbe', 'important_features.txt'),
                                               encoding='utf-8') if line.strip() and line.strip() != 'ID']
    accuracy = plot_ranking(ranking, len(important), os.path.join(args.output_dir, 'fig8.pdf'))
    print(f'features kept in Experiment 5: {len(important)} -> {important}; AutoGluon accuracy with only those features: {accuracy:.4f}')

    processed = class_counts_processed(args.data_dir)
    totals = np.sum(list(processed.values()), axis=0)
    print('processed class counts per split (benign, malicious):', processed, '| total', totals.tolist())
    if args.raw_dir:
        raw = class_counts_raw(args.raw_dir)
        print('original class counts (benign, malicious):', raw, '| share benign %.2f%%' % (raw[0] / sum(raw) * 100))
        plot_distribution(raw, totals.tolist(), os.path.join(args.output_dir, 'original_vs_processed_binary_distribution.pdf'))


if __name__ == '__main__':
    main()
