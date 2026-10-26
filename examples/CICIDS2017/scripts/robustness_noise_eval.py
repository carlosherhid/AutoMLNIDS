"""
Lightweight robustness check under input perturbation. This is *not* a full
adversarial-robustness study (no gradient-based attacks such as FGSM/PGD) - it
measures how accuracy degrades as increasing Gaussian noise is added to the (already
min-max-normalized, in [0,1]) input features, comparing the FP32 and QAT models.

Usage:
    python scripts/robustness_noise_eval.py --predictor_path <predictor_dir> \
        --data_dir <data_dir> --normal_model_path <normal_checkpoint> \
        --qat_model_path <qat_checkpoint> \
        --scale_close_to_zero   # exp6/exp7 were trained with near-zero scaling; exp1-3 were not
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from autogluon.tabular import TabularPredictor
from sklearn.preprocessing import LabelEncoder

from autogluon_to_pytorch_qat import AutoReplicatedNN, QATWrapper, get_model_architecture, set_all_seeds


def load_normal(path, architecture, input_feature_size):
    model = AutoReplicatedNN(architecture, input_feature_size)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model


def load_qat(path, architecture, input_feature_size):
    wrapper = QATWrapper(AutoReplicatedNN(architecture, input_feature_size))
    wrapper.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
    torch.quantization.prepare_qat(wrapper, inplace=True)  # requires model.training == True
    wrapper = torch.quantization.convert(wrapper.eval())
    wrapper.load_state_dict(torch.load(path, map_location='cpu'))
    wrapper.eval()
    return wrapper


def accuracy_under_noise(model, X, feature_std, y, noise_level, seed):
    """noise_level is a multiplier of each feature's own std (computed on X) - not
    an absolute value - so this works whether or not --scale_close_to_zero was used
    upstream (which pushes some near-zero features well outside the original [0, 1]
    min-max range, making a fixed absolute noise std/clamp meaningless)."""
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(X.shape, generator=generator) * (feature_std * noise_level)
    X_noisy = X + noise
    with torch.no_grad():
        y_pred = torch.softmax(model(X_noisy), dim=1).argmax(dim=1)
    return float((y_pred == y).float().mean())


def main():
    parser = argparse.ArgumentParser(description='Accuracy vs. Gaussian input noise, FP32 vs. QAT')
    parser.add_argument('--predictor_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--normal_model_path', type=str, required=True)
    parser.add_argument('--qat_model_path', type=str, required=True)
    parser.add_argument('--noise_levels', type=float, nargs='+',
                         default=[0.0, 0.1, 0.25, 0.5, 1.0, 2.0],
                         help='Multipliers of each feature\'s own std (computed on the test set)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--scale_close_to_zero', action='store_true',
                         help='Must match how the checkpoints were trained (e.g. paper Experiments 4/5)')
    parser.add_argument('--scale_factor', type=float, default=1e6)
    args = parser.parse_args()

    set_all_seeds(args.seed)
    args.output_dir = args.output_dir or os.path.dirname(args.qat_model_path)

    test_data = pd.read_csv(os.path.join(args.data_dir, 'test.csv'), index_col='ID')
    y = torch.tensor(LabelEncoder().fit_transform(test_data['Label']), dtype=torch.long)
    X_values = test_data.drop(columns=['Label']).values
    if args.scale_close_to_zero:
        # Same rule as load_data() in autogluon_to_pytorch_qat.py, applied in memory
        # (load_data() also writes *_scaled.csv files into data_dir, which would race
        # with any other job scaling the same dataset at the same time).
        X_values = np.where(np.abs(X_values) < 1e-6, X_values * args.scale_factor, X_values)
    X = torch.tensor(X_values, dtype=torch.float32)
    feature_std = X.std(dim=0, keepdim=True)

    # require_version_match=False / require_py_version_match=False: predictor
    # may have been trained on an older AutoGluon/Python than what's installed
    # - see autogluon_to_pytorch_qat.py.
    predictor = TabularPredictor.load(
        args.predictor_path, require_version_match=False, require_py_version_match=False)
    architecture, _, input_feature_size = get_model_architecture(predictor, X.shape[1])

    normal_model = load_normal(args.normal_model_path, architecture, input_feature_size)
    qat_model = load_qat(args.qat_model_path, architecture, input_feature_size)

    rows = []
    for noise_level in args.noise_levels:
        normal_acc = accuracy_under_noise(normal_model, X, feature_std, y, noise_level, args.seed)
        qat_acc = accuracy_under_noise(qat_model, X, feature_std, y, noise_level, args.seed)
        print(f'noise_level={noise_level:.3f}x  normal_acc={normal_acc * 100:.2f}%  qat_acc={qat_acc * 100:.2f}%')
        rows.append({'noise_level': noise_level, 'normal_accuracy': normal_acc, 'qat_accuracy': qat_acc})

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.output_dir, 'robustness_noise_eval.csv')
    df.to_csv(csv_path, index=False)
    print(f'Saved results to {csv_path}')

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df['noise_level'], df['normal_accuracy'] * 100, 'o-', label='Normal (FP32)')
    ax.plot(df['noise_level'], df['qat_accuracy'] * 100, 's-', label='QAT (INT8)')
    ax.set_xlabel("Gaussian noise (x feature std)")
    ax.set_ylabel('Test accuracy (%)')
    ax.set_title('Accuracy under input noise: FP32 vs. QAT')
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    fig_path = os.path.join(args.output_dir, 'robustness_noise_eval.svg')
    fig.savefig(fig_path)
    print(f'Saved plot to {fig_path}')


if __name__ == '__main__':
    main()
