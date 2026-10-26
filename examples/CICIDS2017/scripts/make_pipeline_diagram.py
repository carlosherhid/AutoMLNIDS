"""
Pipeline diagram of the paper (Figure "Pipeline Diagram of the process"), vector, same black-and-white
box style as the original raster figure but with the steps the paper actually describes:
data preparation -> AutoML (AutoGluon) -> PyTorch replica -> input scaling / feature selection ->
QAT (INT8) -> evaluation -> on-device deployment.

Usage:
    python scripts/make_pipeline_diagram.py --output ./paper_results/figures/diagram.pdf
"""
import argparse
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.family': 'DejaVu Sans'})

INK = '#1a1a1a'


def box(ax, cx, cy, w, h, text, size=8.2):
    ax.add_patch(FancyBboxPatch((cx - w / 2 + 0.07, cy - h / 2 - 0.07), w, h, boxstyle='round,pad=0,rounding_size=0.12', fc='#777777', ec='none', zorder=1))
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle='round,pad=0,rounding_size=0.12', fc='white', ec=INK, lw=1.6, zorder=2))
    ax.text(cx, cy, text, ha='center', va='center', fontsize=size, fontweight='bold', color=INK, linespacing=1.2, zorder=3)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=14, lw=1.6, color=INK, shrinkA=0, shrinkB=0, zorder=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 5.4)
    ax.set_aspect('equal')
    ax.axis('off')

    w, h = 2.8, 1.8
    xs = [1.5, 4.9, 8.3, 11.7]          # first row, left to right
    y1, y2 = 4.35, 1.05
    row1 = ['Data\npreparation', 'AutoML search\n(AutoGluon)', 'Replication\nin PyTorch', 'Input scaling\nand feature\nselection']
    row2 = ['QAT training\n(INT8)', 'Evaluation and\ncomparison', 'On-device\ndeployment']
    for x, text in zip(xs, row1):
        box(ax, x, y1, w, h, text)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + w / 2 + 0.05, y1), (b - w / 2 - 0.05, y1))

    xs2 = [11.7, 8.3, 4.9]               # second row, right to left
    for x, text in zip(xs2, row2):
        box(ax, x, y2, w, h, text)
    arrow(ax, (xs[-1], y1 - h / 2 - 0.05), (xs2[0], y2 + h / 2 + 0.05))
    for a, b in zip(xs2[:-1], xs2[1:]):
        arrow(ax, (a - w / 2 - 0.05, y2), (b + w / 2 + 0.05, y2))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fig.savefig(args.output, bbox_inches='tight', pad_inches=0.04)
    fig.savefig(os.path.splitext(args.output)[0] + '.png', dpi=170, bbox_inches='tight', pad_inches=0.04)
    print('saved', args.output)


if __name__ == '__main__':
    main()
