"""
Functional equivalence check between the original AutoGluon model and its FP32
PyTorch replica. Aggregate test-set accuracy being similar is not proof
the two models behave the same on individual inputs - this compares their raw
per-sample outputs directly.

Reports: prediction agreement rate, Pearson correlation of the predicted
probability for the positive (malicious) class, and the max/mean absolute
probability difference.

Usage:
    python scripts/verify_functional_equivalence.py --predictor_path <predictor_dir> \
        --data_dir <data_dir> --normal_model_path <normal_checkpoint>
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from autogluon.tabular import TabularPredictor
from sklearn.preprocessing import LabelEncoder

from autogluon_to_pytorch_qat import AutoReplicatedNN, get_model_architecture

sys.path.insert(0, os.path.dirname(__file__))
from utils.common import patch_stale_predictor  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description='Verify AutoGluon <-> PyTorch replica functional equivalence')
    parser.add_argument('--predictor_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--normal_model_path', type=str, required=True,
                         help='Path to a *_normal_pytorch_model state_dict (FP32, not QAT)')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'validation', 'finetune'])
    parser.add_argument('--output_csv', type=str, default=None,
                         help='Optional path to save per-sample comparison rows')
    args = parser.parse_args()

    # require_version_match=False / require_py_version_match=False: predictor
    # may have been trained on an older AutoGluon/Python than what's installed
    # - see autogluon_to_pytorch_qat.py.
    predictor = TabularPredictor.load(
        args.predictor_path, require_version_match=False, require_py_version_match=False)
    patch_stale_predictor(predictor)

    data = pd.read_csv(os.path.join(args.data_dir, f'{args.split}.csv'), index_col='ID')
    y_true_raw = data['Label']
    X = data.drop(columns=['Label'])

    label_encoder = LabelEncoder()
    y_true = label_encoder.fit_transform(y_true_raw)
    positive_class_idx = int(np.where(label_encoder.classes_ == label_encoder.classes_.max())[0][0])

    # AutoGluon side
    ag_proba = predictor.predict_proba(X)
    ag_proba = ag_proba.reindex(sorted(ag_proba.columns), axis=1).values  # align class order
    ag_pred = ag_proba.argmax(axis=1)
    ag_positive_proba = ag_proba[:, positive_class_idx]

    # PyTorch replica side
    architecture, _, input_feature_size = get_model_architecture(predictor, X.shape[1])
    torch_model = AutoReplicatedNN(architecture, input_feature_size)
    torch_model.load_state_dict(torch.load(args.normal_model_path, map_location='cpu'))
    torch_model.eval()

    with torch.no_grad():
        logits = torch_model(torch.tensor(X.values, dtype=torch.float32))
        torch_proba = torch.softmax(logits, dim=1).numpy()
    torch_pred = torch_proba.argmax(axis=1)
    torch_positive_proba = torch_proba[:, positive_class_idx]

    agreement_rate = float((ag_pred == torch_pred).mean())
    abs_diff = np.abs(ag_positive_proba - torch_positive_proba)
    correlation = float(np.corrcoef(ag_positive_proba, torch_positive_proba)[0, 1])

    print(f'Samples compared: {len(X)}')
    print(f'Prediction agreement rate (AutoGluon vs. PyTorch FP32 replica): {agreement_rate * 100:.2f}%')
    print(f'Pearson correlation of P(malicious):                           {correlation:.4f}')
    print(f'Mean |P_AutoGluon - P_PyTorch|:                                {abs_diff.mean():.4f}')
    print(f'Max  |P_AutoGluon - P_PyTorch|:                                {abs_diff.max():.4f}')
    print(f'Accuracy AutoGluon:      {(ag_pred == y_true).mean() * 100:.2f}%')
    print(f'Accuracy PyTorch replica: {(torch_pred == y_true).mean() * 100:.2f}%')

    if args.output_csv:
        pd.DataFrame({
            'ID': X.index,
            'y_true': y_true,
            'autogluon_pred': ag_pred,
            'pytorch_pred': torch_pred,
            'autogluon_proba_positive': ag_positive_proba,
            'pytorch_proba_positive': torch_positive_proba,
            'abs_diff': abs_diff,
        }).to_csv(args.output_csv, index=False)
        print(f'Saved per-sample comparison to {args.output_csv}')


if __name__ == '__main__':
    main()
