"""
Classical ML baselines (Random Forest, Logistic Regression, and optionally XGBoost)
on the same train/test split used for the AutoML/QAT pipeline, for a fair
size/accuracy/latency comparison table.

Usage:
    python scripts/baselines_classical_ml.py --data_dir <data_dir> --output_dir <output_dir>
"""
import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def load_xy(data_dir, split, label_encoder=None):
    df = pd.read_csv(os.path.join(data_dir, f'{split}.csv'))
    y_raw = df['Label']
    X = df.drop(columns=['Label', 'ID'], errors='ignore')
    if label_encoder is None:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw)
    else:
        y = label_encoder.transform(y_raw)
    return X, y, label_encoder


def evaluate(name, model, X_test, y_test, fit_time, model_path):
    start = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start

    return {
        'model': name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'fit_time_s': fit_time,
        'mean_inference_time_ms_per_sample': (inference_time / len(X_test)) * 1000,
        'model_size_kb': os.path.getsize(model_path) / 1e3,
    }


def main():
    parser = argparse.ArgumentParser(description='Train/evaluate classical ML baselines for comparison with the QAT pipeline')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing train.csv/test.csv')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save fitted models and the results table')
    parser.add_argument('--rf_estimators', type=int, default=200)
    parser.add_argument('--rf_max_depth', type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    X_train, y_train, label_encoder = load_xy(args.data_dir, 'train')
    X_test, y_test, _ = load_xy(args.data_dir, 'test', label_encoder)

    results = []

    models = {
        'random_forest': RandomForestClassifier(
            n_estimators=args.rf_estimators, max_depth=args.rf_max_depth,
            n_jobs=-1, random_state=42),
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
    }
    if HAS_XGBOOST:
        models['xgboost'] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            eval_metric='logloss', n_jobs=-1, random_state=42)
    else:
        print('xgboost not installed - skipping that baseline (pip install xgboost to include it)')

    for name, model in models.items():
        print(f'\nTraining {name}...')
        start = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - start

        model_path = os.path.join(args.output_dir, f'baseline_{name}.joblib')
        joblib.dump(model, model_path)

        result = evaluate(name, model, X_test, y_test, fit_time, model_path)
        print(json.dumps(result, indent=2))
        results.append(result)

    results_df = pd.DataFrame(results)
    out_path = os.path.join(args.output_dir, 'baselines_classical_ml_results.csv')
    results_df.to_csv(out_path, index=False)
    print(f'\nSaved results table to {out_path}')
    print(results_df.to_string(index=False))


if __name__ == '__main__':
    main()
