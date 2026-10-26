import argparse
import copy
import os
import random
import sys
import tempfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import time  # Import the time module to measure execution time
from sklearn.preprocessing import LabelEncoder
from autogluon.tabular import TabularPredictor
from torch.utils.data import DataLoader, TensorDataset
from torch.quantization import QuantStub, DeQuantStub, prepare_qat, convert
from tqdm import tqdm
import torch.nn.utils.prune as prune

sys.path.insert(0, os.path.dirname(__file__))
from utils.common import patch_stale_predictor  # noqa: E402
from utils.preprocessing import apply_preprocessing  # noqa: E402


def set_all_seeds(seed: int):
    """Seed python/numpy/torch so a run can be repeated (or swept across seeds -
    see scripts/run_multiseed.py). Does not touch cuDNN determinism flags, which
    would slow down training; exact bit-for-bit reproducibility on GPU is not
    guaranteed, only statistical repeatability across CPU/GPU runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class Logger(object):
    def __init__(self, output_dir, experiment_number):
        self.terminal = sys.stdout
        self.log = open(os.path.join(output_dir, f"execution_log_exp{experiment_number}.txt"), "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def load_data(data_dir, modify_test=False, scale_close_to_zero=False, scale_factor=1e6):
    # Load original data
    train_data = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test_data = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    val_data = pd.read_csv(os.path.join(data_dir, 'validation.csv'))

    # Encode labels
    label_encoder = LabelEncoder()
    train_data['Label'] = label_encoder.fit_transform(train_data['Label'])
    test_data['Label'] = label_encoder.transform(test_data['Label'])
    val_data['Label'] = label_encoder.transform(val_data['Label'])

    modified_data_suffix = ""

    def modify_dataset(dataset, operation):
        # Perform operations like modifying or scaling
        modified_dataset = dataset.copy()
        original_types = dataset.dtypes
        modified_dataset.iloc[:, :-1] = modified_dataset.iloc[:, :-1].applymap(operation)
        # Ensure the original types are preserved
        for col, dtype in original_types.items():
            modified_dataset[col] = modified_dataset[col].astype(dtype)
        return modified_dataset

    if modify_test:
        train_data = modify_dataset(train_data, lambda x: 0 if abs(x) < 1e-6 else x)
        test_data = modify_dataset(test_data, lambda x: 0 if abs(x) < 1e-6 else x)
        val_data = modify_dataset(val_data, lambda x: 0 if abs(x) < 1e-6 else x)
        modified_data_suffix += "_modified"

    if scale_close_to_zero:
        train_data = modify_dataset(train_data, lambda x: x * scale_factor if abs(x) < 1e-6 else x)
        test_data = modify_dataset(test_data, lambda x: x * scale_factor if abs(x) < 1e-6 else x)
        val_data = modify_dataset(val_data, lambda x: x * scale_factor if abs(x) < 1e-6 else x)
        modified_data_suffix += "_scaled"

    # Save modified datasets to new CSV files
    if modified_data_suffix:
        train_data.to_csv(os.path.join(data_dir, f'train{modified_data_suffix}.csv'), index=False)
        test_data.to_csv(os.path.join(data_dir, f'test{modified_data_suffix}.csv'), index=False)
        val_data.to_csv(os.path.join(data_dir, f'validation{modified_data_suffix}.csv'), index=False)

    return train_data, test_data, val_data

def prune_features(train_data, test_data, val_data, important_features_file):
    # Read the important features from the file, assuming no header
    pruned_features = pd.read_csv(important_features_file, header=None).squeeze().tolist()
    important_features = ['ID'] + pruned_features + ['Label']

    # Set unimportant features to zero instead of removing them
    def set_unimportant_features_to_zero(dataset):
        unimportant_features = [col for col in dataset.columns if col not in important_features]
        dataset[unimportant_features] = 0
        return dataset

    train_data_pruned = set_unimportant_features_to_zero(train_data)
    test_data_pruned = set_unimportant_features_to_zero(test_data)
    val_data_pruned = set_unimportant_features_to_zero(val_data)

    return train_data_pruned, test_data_pruned, val_data_pruned

def create_dataloaders(train_data, test_data, val_data, batch_size):
    # Split features and labels
    X_train, y_train = train_data.drop(columns=['Label', 'ID']).values, train_data['Label'].values
    X_test, y_test = test_data.drop(columns=['Label', 'ID']).values, test_data['Label'].values
    X_val, y_val = val_data.drop(columns=['Label', 'ID']).values, val_data['Label'].values

    # Convert to torch tensors
    train_tensor = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_tensor = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    val_tensor = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

    # Create DataLoader
    train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_tensor, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_tensor, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, val_loader

def get_model_architecture(predictor, input_feature_size):
    best_model_name = predictor._trainer.model_best
    best_model = predictor._trainer.load_model(best_model_name)

    architecture = []
    for name, layer in best_model.model.named_children():
        if isinstance(layer, nn.Sequential):
            for sub_layer in layer:
                architecture.append((type(sub_layer), sub_layer))
        else:
            architecture.append((type(layer), layer))

    return architecture, best_model, input_feature_size

class AutoReplicatedNN(nn.Module):
    def __init__(self, architecture, input_feature_size):
        super(AutoReplicatedNN, self).__init__()
        self.input_feature_size = input_feature_size  # needed by restructure_pruned_model()
        layers = []
        current_input_size = input_feature_size
        for layer_type, layer_obj in architecture:
            if layer_type == nn.Linear:
                layers.append(nn.Linear(current_input_size, layer_obj.out_features))
                current_input_size = layer_obj.out_features
            elif layer_type == nn.ReLU:
                layers.append(nn.ReLU())
            elif layer_type == nn.Tanh:
                layers.append(nn.Tanh())
            elif layer_type == nn.Dropout:
                layers.append(nn.Dropout(p=layer_obj.p))
            elif layer_type == nn.BatchNorm1d:
                layers.append(nn.Identity())  # Replace BatchNorm1d with Identity
            elif layer_type == nn.Identity or layer_type == nn.Sequential:
                layers.append(layer_obj)
            elif layer_type != nn.Softmax:  # Exclude Softmax
                raise ValueError(f"Unhandled layer type: {layer_type}")
        self.main_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.main_block(x)

class QATWrapper(nn.Module):
    def __init__(self, model):
        super(QATWrapper, self).__init__()
        self.quant = QuantStub()
        self.model = model
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x

def train_pytorch_model(model, train_loader, val_loader, epochs, device, use_early_stopping=True, use_lr_scheduler=True):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5) if use_lr_scheduler else None

    # Introduce Early Stopping Mechanism
    best_val_accuracy = 0
    best_state_dict = copy.deepcopy(model.state_dict())
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for data in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', unit='batch'):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

        # Validate the model
        val_loss, val_accuracy = validate_pytorch_model(model, val_loader, device)
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        # Step the scheduler
        if scheduler:
            scheduler.step(val_loss)

        # Track the best checkpoint by val_accuracy regardless of whether early
        # stopping is enabled - without this, training instability late in a run
        # (common with QAT's noisier fake-quantized gradients, especially with
        # --disable_lr_scheduler's fixed LR) silently returns a collapsed
        # last-epoch model even though a much better checkpoint existed earlier
        # (e.g. exp2 seed=0 on CICIDS2017: peaked at val_accuracy=0.9588 on
        # epoch 14, then collapsed to 0.5000 by epoch 24 - the un-fixed code
        # returned the collapsed epoch-24 model since early stopping only halts
        # further epochs, it never rolled back to the best one).
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state_dict = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if use_early_stopping and patience_counter >= patience:
                print("Early stopping triggered")
                break

    model.load_state_dict(best_state_dict)
    print(f"Restored best checkpoint (val_accuracy={best_val_accuracy:.4f})")
    return model

def validate_pytorch_model(model, val_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_accuracy = correct / total
    return val_loss, val_accuracy

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    softmax = nn.Softmax(dim=1)  # Initialize Softmax

    with torch.no_grad():
        for data in tqdm(test_loader, desc='Testing', unit='batch'):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            outputs = softmax(outputs)  # Apply Softmax
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy


def evaluate_model_detailed(model, loader, device):
    """Confusion-matrix metrics on a loader; the positive class is label 1 (malicious).
    Recall = TPR = TP/(TP+FN); FPR = FP/(FP+TN). TPR and FPR do not depend on the class
    prevalence, so they transfer to the unbalanced traffic of a real deployment (precision
    does not: it can be projected from them for any prevalence)."""
    model.eval()
    tp = fp = tn = fn = 0
    with torch.no_grad():
        for inputs, labels in loader:
            predicted = model(inputs.to(device)).argmax(dim=1).cpu()
            labels = labels.cpu()
            tp += int(((predicted == 1) & (labels == 1)).sum())
            fp += int(((predicted == 1) & (labels == 0)).sum())
            tn += int(((predicted == 0) & (labels == 0)).sum())
            fn += int(((predicted == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {'accuracy': (tp + tn) / (tp + fp + tn + fn), 'precision': precision, 'recall': recall,
            'fpr': fpr, 'f1': f1, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}


def print_detailed_metrics(name, metrics):
    # Parsed by utils/log_parsing.py - keep the format in sync.
    print(f"Detailed metrics of the {name} model on the test dataset: precision={metrics['precision']:.5f} "
          f"recall={metrics['recall']:.5f} fpr={metrics['fpr']:.5f} f1={metrics['f1']:.5f}")


def print_size_of_model(model):
    # Unique directory but always the file name "temp.p": jobs running in parallel
    # from the same working directory used to overwrite/delete each other's temp.p,
    # and the file *name* changes the size of torch.save's zip archive by ~1 KB, so
    # it must stay "temp.p" for sizes to remain comparable with earlier results.
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, "temp.p")
    torch.save(model.state_dict(), tmp_path)
    print('Size (KB):', os.path.getsize(tmp_path)/1e3)
    os.remove(tmp_path)
    os.rmdir(tmp_dir)

def prune_model(model, amount=0.2):
    """
    Prunes the model by removing `amount` proportion of the smallest weights.
    """
    for layer in model.main_block:
        if isinstance(layer, nn.Linear):
            prune.l1_unstructured(layer, name='weight', amount=amount)
            prune.remove(layer, 'weight')
    return model

def restructure_pruned_model(original_model):
    """
    Rebuilds the model architecture by removing neurons or channels that have been pruned away.
    """
    pruned_layers = []

    for layer in original_model.main_block:
        if isinstance(layer, nn.Linear):
            # Check which neurons (input/output channels) have all weights set to zero
            weight_mask = layer.weight.abs().sum(dim=1) != 0
            bias_mask = layer.bias.abs() != 0

            # Create a new linear layer with the reduced number of neurons
            new_layer = nn.Linear(int(weight_mask.sum()), layer.out_features)
            new_layer.weight = torch.nn.Parameter(layer.weight[weight_mask].clone())
            new_layer.bias = torch.nn.Parameter(layer.bias[bias_mask].clone())

            pruned_layers.append(new_layer)
        else:
            pruned_layers.append(layer)

    # Build a new model with the pruned layers
    new_model = AutoReplicatedNN(pruned_layers, original_model.input_feature_size)
    return new_model

def save_model(model, output_dir, model_name, experiment_number):
    model_path = os.path.join(output_dir, f"{model_name}_exp{experiment_number}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def main():
    # Start the timer
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Replicate AutoGluon model in PyTorch and apply QAT')
    parser.add_argument('--predictor_path', type=str, required=True, help='Path to the AutoGluon predictor')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the dataset CSV files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output models')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train (default: 1000)')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training (default: 256)')
    parser.add_argument('--qat', action='store_true', help='Apply Quantization-Aware Training')
    parser.add_argument('--normal', action='store_true', help='Train a normal (non-quantized) model')
    parser.add_argument('--experiment_number', type=int, required=True, help='Experiment number to include in model name')
    parser.add_argument('--modify_test', action='store_true', help='Modify test dataset by setting values close to 0 to 0')
    parser.add_argument('--scale_close_to_zero', action='store_true', help='Scale values close to 0 by a large factor')
    parser.add_argument('--disable_early_stopping', action='store_true', help='Disable early stopping during training')
    parser.add_argument('--disable_lr_scheduler', action='store_true', help='Disable the learning rate scheduler during training')
    parser.add_argument('--prune_features', action='store_true', help='Prune features based on importance')
    parser.add_argument('--important_features_file', type=str, help='File containing important features to keep after pruning')
    parser.add_argument('--prune_model', action='store_true', help='Prune the model after training (works with --normal and/or --qat; combine with --qat to prune after QAT training as before, or with --normal alone for a pruning-only, non-quantized baseline)')
    parser.add_argument('--structured_pruning', action='store_true', help='Apply structured pruning with architectural changes')
    parser.add_argument('--prune_amount', type=float, default=0.2, help='Fraction of weights to prune when --prune_model is set (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for python/numpy/torch, for repeatability and multi-seed sweeps (default: 42)')
    parser.add_argument('--scale_factor', type=float, default=1e6, help='Multiplicative factor applied to near-zero feature values when --scale_close_to_zero is set (default: 1e6; see scripts/scale_factor_sweep.py)')
    parser.add_argument('--preprocess', type=str, default='none', choices=['none', 'standard', 'logminmax', 'quantile'],
                        help='Alternative input transformation to compare against --scale_close_to_zero (see apply_preprocessing); fitted on the training split only')

    args = parser.parse_args()

    set_all_seeds(args.seed)

    # Setup logger
    sys.stdout = Logger(args.output_dir, args.experiment_number)
    print(f'Using seed={args.seed}')
    if args.scale_close_to_zero:
        print(f'Using scale_factor={args.scale_factor}')
    if args.preprocess != 'none':
        print(f'Using preprocess={args.preprocess}')

    # Load the data
    train_data, test_data, val_data = load_data(
        args.data_dir,
        args.modify_test,
        args.scale_close_to_zero,
        args.scale_factor
    )
    train_data, test_data, val_data = apply_preprocessing(train_data, test_data, val_data, args.preprocess)

    # Apply feature pruning if the flag is set
    if args.prune_features:
        train_data, test_data, val_data = prune_features(
            train_data,
            test_data,
            val_data,
            args.important_features_file
        )
        # Save pruned datasets to new CSV files
        train_data.to_csv(os.path.join(args.data_dir, 'train_pruned.csv'), index=False)
        test_data.to_csv(os.path.join(args.data_dir, 'test_pruned.csv'), index=False)
        val_data.to_csv(os.path.join(args.data_dir, 'validation_pruned.csv'), index=False)

    train_loader, test_loader, val_loader = create_dataloaders(train_data, test_data, val_data, args.batch_size)

    # Load the AutoGluon predictor
    # require_version_match=False / require_py_version_match=False: predictors
    # trained on an older AutoGluon/Python (e.g. 1.1.1 on Python 3.10) still
    # load fine on the cluster's newer 1.6.x/Python 3.11 in practice.
    predictor = TabularPredictor.load(
        args.predictor_path, require_version_match=False, require_py_version_match=False)
    patch_stale_predictor(predictor)

    # Get the model architecture from the AutoGluon predictor
    architecture, best_model, input_feature_size = get_model_architecture(predictor, len(train_data.columns) - 2)  # Exclude ID and Label

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.normal:
        # Replicate and train normal PyTorch model
        normal_model = AutoReplicatedNN(architecture, input_feature_size).to(device)
        print("\nTraining Normal PyTorch Model:")
        normal_model = train_pytorch_model(normal_model, train_loader, val_loader, args.epochs, device, not args.disable_early_stopping, not args.disable_lr_scheduler)

        if args.prune_model and not args.qat:
            # Pruning-only baseline (no quantization at all): prune the trained FP32
            # model directly, to isolate pruning's effect from QAT's.
            print("\nModel size before pruning (FP32, no QAT):")
            print_size_of_model(normal_model)
            normal_model = prune_model(normal_model, amount=args.prune_amount)
            if args.structured_pruning:
                normal_model = restructure_pruned_model(normal_model)
            print("\nModel size after pruning (FP32, no QAT):")

        save_model(normal_model, args.output_dir, "normal_pytorch_model", args.experiment_number)
        normal_accuracy = evaluate_model(normal_model, test_loader, device)
        print(f'Accuracy of the normal model on the test dataset: {normal_accuracy * 100:.2f}%')
        print_size_of_model(normal_model)  # must directly follow the accuracy line (utils/log_parsing.py)
        print_detailed_metrics('normal', evaluate_model_detailed(normal_model, test_loader, device))

    if args.qat:
        # Apply QAT and train QAT PyTorch model
        qat_model = AutoReplicatedNN(architecture, input_feature_size).to(device)

        qat_model = QATWrapper(qat_model).to(device)
        qat_model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
        prepare_qat(qat_model, inplace=True)

        print("\nTraining QAT PyTorch Model:")
        qat_model = train_pytorch_model(qat_model, train_loader, val_loader, args.epochs, device, not args.disable_early_stopping, not args.disable_lr_scheduler)

        # Show the size of the model after QAT but before quantization or pruning
        print("\nModel size after QAT training but before pruning or quantization:")
        print_size_of_model(qat_model.model)

        if args.prune_model:
            # Prune the model after QAT training but before quantization
            pruned_model = prune_model(qat_model.model)
            print("\nModel size after pruning but before quantization:")
            print_size_of_model(pruned_model)

            # Rebuild model to reflect pruned structure
            if args.structured_pruning:
                pruned_model = restructure_pruned_model(pruned_model)
                print("\nRebuilt Model size after pruning:")
                print_size_of_model(pruned_model)

            # Re-evaluate after pruning
            qat_model.model = pruned_model  # Update the QAT wrapper's model
            pruned_accuracy = evaluate_model(qat_model, test_loader, device)
            print(f'Accuracy of the pruned QAT model on the test dataset: {pruned_accuracy * 100:.2f}%')

        qat_model.eval()
        qat_model = qat_model.cpu()  # Move to CPU for quantization
        qat_model = convert(qat_model)
        save_model(qat_model, args.output_dir, "qat_pytorch_model", args.experiment_number)

        qat_accuracy = evaluate_model(qat_model, test_loader, torch.device("cpu"))
        print(f'Accuracy of the QAT model on the test dataset: {qat_accuracy * 100:.2f}%')
        print_detailed_metrics('QAT', evaluate_model_detailed(qat_model, test_loader, torch.device("cpu")))
        print_size_of_model(qat_model)

        print("Weights after quantization:")
        # Print weights from the first Linear layer
        for layer in qat_model.model.main_block:
            if isinstance(layer, nn.Linear):
                print(torch.int_repr(layer.weight))
                break

    # Evaluate AutoGluon model on the modified test set if any modifications were applied.
    # This is a sanity-check comparison only - the PyTorch normal/QAT checkpoints
    # above are already trained and saved by this point. Wrapped in try/except
    # because predictor.evaluate() can hit AutoGluon-internal AttributeErrors when
    # a predictor pickled on an older AutoGluon (e.g. 1.1.1) is loaded on a newer
    # one (1.6.2) - AutoGluon's own load() warning calls exceptions like this
    # "very cryptic" but expected. Letting it propagate would abort the whole
    # run_multiseed.py sweep (subprocess.run(check=True)) after seed 0, discarding
    # every already-trained checkpoint for the remaining seeds.
    try:
        if args.modify_test or args.scale_close_to_zero or args.prune_features:
            print("Evaluating AutoGluon model on the modified test dataset.")
            ag_test_data = test_data.drop(columns=['Label'])
            ag_labels = test_data['Label']
            original_ag_model_accuracy = predictor.evaluate(pd.concat([ag_test_data, ag_labels], axis=1))['accuracy']
        else:
            original_ag_model_accuracy = predictor.evaluate(pd.read_csv(os.path.join(args.data_dir, 'test.csv')))['accuracy']
        print(f'Accuracy on the test set (Original AutoGluon Model): {original_ag_model_accuracy:.4f}')
    except Exception as e:
        print(f'WARNING: skipping original-AutoGluon-model comparison, predictor.evaluate() failed: {e!r}')

    # End the timer and print the total execution time
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()