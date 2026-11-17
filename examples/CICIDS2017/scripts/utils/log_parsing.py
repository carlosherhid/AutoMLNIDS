"""
Shared regex-based parser for compressed_models/execution_log_exp*.txt, produced by
Logger in autogluon_to_pytorch_qat.py. Shared by scripts/scale_factor_sweep.py,
scripts/run_multiseed.py and scripts/build_tradeoff_table.py so that they don't each
reimplement it.
"""
import re


def parse_log_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    result = {
        'normal_test_accuracy': None,
        'normal_size_kb': None,
        'qat_test_accuracy': None,
        'qat_size_kb': None,
        'autogluon_accuracy': None,
        'epochs_run': None,
        'total_execution_time_s': None,
    }
    for model_name in ('normal', 'qat'):
        for metric in ('precision', 'recall', 'fpr', 'f1'):
            result[f'{model_name}_{metric}'] = None

    # Written by autogluon_to_pytorch_qat.py's print_detailed_metrics(); absent in logs of older runs.
    for model_name, precision, recall, fpr, f1 in re.findall(
            r'Detailed metrics of the (normal|QAT) model on the test dataset: '
            r'precision=([0-9.]+) recall=([0-9.]+) fpr=([0-9.]+) f1=([0-9.]+)', content):
        prefix = model_name.lower()
        result[f'{prefix}_precision'] = float(precision)
        result[f'{prefix}_recall'] = float(recall)
        result[f'{prefix}_fpr'] = float(fpr)
        result[f'{prefix}_f1'] = float(f1)

    normal_match = re.search(
        r'Accuracy of the normal model on the test dataset: ([0-9.]+)%\nSize \(KB\): ([0-9.]+)', content)
    if normal_match:
        result['normal_test_accuracy'] = float(normal_match.group(1))
        result['normal_size_kb'] = float(normal_match.group(2))

    if 'Training QAT PyTorch Model:' in content:
        qat_section = content.split('Training QAT PyTorch Model:')[1]
        result['epochs_run'] = len(re.findall(r'Epoch \[\d+/\d+\]', qat_section))

        qat_accuracy = re.findall(r'Accuracy of the QAT model on the test dataset: ([0-9.]+)%', qat_section)
        if qat_accuracy:
            result['qat_test_accuracy'] = float(qat_accuracy[0])

        qat_sizes = re.findall(r'Size \(KB\): ([0-9.]+)', qat_section)
        if qat_sizes:
            result['qat_size_kb'] = float(qat_sizes[-1])  # last "Size (KB):" in the QAT section is post-quantization

    # Added later: the same QAT weights scored as the fake-quantized model and as INT8 under each engine (None in older logs)
    fake_accuracy = re.search(r'Accuracy of the QAT model before conversion \(fake quantization\) on the test dataset: ([0-9.]+)%', content)
    result['qat_fake_accuracy'] = float(fake_accuracy.group(1)) if fake_accuracy else None
    for engine in ('x86', 'onednn'):
        engine_accuracy = re.search(rf'Accuracy of the QAT model \(INT8, {engine} engine\) on the test dataset: ([0-9.]+)%', content)
        result[f'qat_{engine}_accuracy'] = float(engine_accuracy.group(1)) if engine_accuracy else None
    default_engine = re.search(r'INT8 engine of the accuracy above: (\w+)', content)
    if default_engine and result.get('qat_test_accuracy') is not None:
        result[f'qat_{default_engine.group(1)}_accuracy'] = result['qat_test_accuracy']

    ag_accuracy = re.search(r'Accuracy on the test set \(Original AutoGluon Model\): ([0-9.]+)', content)
    if ag_accuracy:
        result['autogluon_accuracy'] = float(ag_accuracy.group(1)) * 100

    exec_time = re.search(r'Total execution time: ([0-9.]+) seconds', content)
    if exec_time:
        result['total_execution_time_s'] = float(exec_time.group(1))

    return result
