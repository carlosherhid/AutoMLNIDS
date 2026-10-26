"""
Post-Training (static) Quantization baseline, to compare against Quantization-Aware
Training. Reuses the exact same architecture-replication and data-loading code
as scripts/autogluon_to_pytorch_qat.py - only the training/quantization procedure
differs: a model is trained in FP32 as usual, then quantized *after* training (no
fake-quantization during training), using a handful of calibration batches to
estimate activation ranges.

Usage:
    python scripts/post_training_quantization.py --predictor_path ./balanced_binary/automl_search \
        --data_dir ./balanced_binary --output_dir ./balanced_binary/compressed_models \
        --experiment_number 6 --epochs 100
    # Or, to skip retraining and quantize an already-trained FP32 checkpoint:
    python scripts/post_training_quantization.py --predictor_path ./balanced_binary/automl_search \
        --data_dir ./balanced_binary --output_dir ./balanced_binary/compressed_models \
        --experiment_number 6 --normal_model_path ./balanced_binary/compressed_models/normal_pytorch_model_exp6.pth
"""
import argparse
import os
import sys

import torch
import torch.quantization as tq
from autogluon.tabular import TabularPredictor

# Reuse everything from the main QAT script instead of duplicating it - safe to
# import since its own execution is guarded by `if __name__ == "__main__"`.
from autogluon_to_pytorch_qat import (
    AutoReplicatedNN, QATWrapper, create_dataloaders, evaluate_model,
    get_model_architecture, load_data, print_size_of_model, set_all_seeds,
    train_pytorch_model,
)


def calibrate(model, calibration_loader, device, n_batches=20):
    model.eval()
    with torch.no_grad():
        for i, (inputs, _) in enumerate(calibration_loader):
            if i >= n_batches:
                break
            model(inputs.to(device))


def main():
    parser = argparse.ArgumentParser(description='Post-training static quantization baseline (vs. QAT)')
    parser.add_argument('--predictor_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--experiment_number', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=100, help='Only used if --normal_model_path is not given')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--scale_close_to_zero', action='store_true')
    parser.add_argument('--scale_factor', type=float, default=1e6)
    parser.add_argument('--calibration_batches', type=int, default=20,
                         help='Number of training batches used to calibrate activation ranges')
    parser.add_argument('--normal_model_path', type=str, default=None,
                         help='Path to an already-trained FP32 state_dict (from autogluon_to_pytorch_qat.py '
                              '--normal) to reuse instead of retraining')
    args = parser.parse_args()

    set_all_seeds(args.seed)
    device = torch.device('cpu')  # PTQ (static) quantized models run on CPU in PyTorch

    train_data, test_data, val_data = load_data(
        args.data_dir, modify_test=False,
        scale_close_to_zero=args.scale_close_to_zero, scale_factor=args.scale_factor)
    train_loader, test_loader, val_loader = create_dataloaders(train_data, test_data, val_data, args.batch_size)

    # require_version_match=False / require_py_version_match=False: predictor
    # may have been trained on an older AutoGluon/Python than what's installed
    # - see autogluon_to_pytorch_qat.py.
    predictor = TabularPredictor.load(
        args.predictor_path, require_version_match=False, require_py_version_match=False)
    architecture, _, input_feature_size = get_model_architecture(predictor, len(train_data.columns) - 2)

    fp32_model = AutoReplicatedNN(architecture, input_feature_size)
    if args.normal_model_path:
        print(f'Loading pretrained FP32 model from {args.normal_model_path}')
        fp32_model.load_state_dict(torch.load(args.normal_model_path, map_location='cpu'))
        fp32_model.to(device)
    else:
        fp32_model.to(device)
        print('Training FP32 model from scratch (no --normal_model_path given)...')
        fp32_model = train_pytorch_model(fp32_model, train_loader, val_loader, args.epochs, device)

    fp32_accuracy = evaluate_model(fp32_model, test_loader, device)
    print(f'FP32 model accuracy: {fp32_accuracy * 100:.2f}%')
    print_size_of_model(fp32_model)

    # --- Post-training static quantization ---
    ptq_model = QATWrapper(fp32_model)
    ptq_model.eval()
    ptq_model.qconfig = tq.get_default_qconfig('qnnpack')
    tq.prepare(ptq_model, inplace=True)

    print(f'\nCalibrating on {args.calibration_batches} training batches...')
    calibrate(ptq_model, train_loader, device, args.calibration_batches)

    tq.convert(ptq_model, inplace=True)

    ptq_accuracy = evaluate_model(ptq_model, test_loader, device)
    print(f'\nPost-Training Quantized (INT8) model accuracy: {ptq_accuracy * 100:.2f}%')
    print_size_of_model(ptq_model)

    model_path = os.path.join(args.output_dir, f'ptq_pytorch_model_exp{args.experiment_number}.pth')
    torch.save(ptq_model.state_dict(), model_path)
    print(f'Saved PTQ model to {model_path}')

    print('\n--- Summary (compare against the matching QAT experiment in docs/EXPERIMENTS_LOG.md) ---')
    print(f'FP32 accuracy:            {fp32_accuracy * 100:.2f}%')
    print(f'Post-Training Quant (PTQ) accuracy: {ptq_accuracy * 100:.2f}%')


if __name__ == '__main__':
    main()
