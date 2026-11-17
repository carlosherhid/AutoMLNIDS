"""
The experiments of the paper, defined once. `run_multiseed.py --experiment N` and the analysis scripts read their flags and
labels from here, so that every dataset is run with exactly the same configurations.

Experiments 1-5 are the five experiments of Table 1 of the paper. Experiments 6-8 replace the near-zero rescaling of
Experiment 4 by one of the alternative input representations compared with it (standardization, logarithmic Min-Max and
quantile transform); together with Experiments 3 and 4 they make up the input-preprocessing comparison.

A run of experiment N with seed S is stored as `*_exp<100000 + N * 100 + S>.pth` / `execution_log_exp<...>.txt`.
"""

# number -> (label, description, flags of autogluon_to_pytorch_qat.py). Experiment 5 also needs --important_features_file.
EXPERIMENTS = {
    1: ('Exp. 1', 'baseline: no early stopping, no LR scheduling',
        '--normal --qat --disable_early_stopping --disable_lr_scheduler'),
    2: ('Exp. 2', 'early stopping enabled',
        '--normal --qat --disable_lr_scheduler'),
    3: ('Exp. 3', 'LR scheduling enabled',
        '--normal --qat --disable_early_stopping'),
    4: ('Exp. 4', 'near-zero rescaling (x1e6)',
        '--normal --qat --disable_early_stopping --scale_close_to_zero'),
    5: ('Exp. 5', 'Exp. 4 + feature selection',
        '--normal --qat --disable_early_stopping --scale_close_to_zero --prune_features'),
    6: ('Standardization', 'input representation: standardization instead of the rescaling',
        '--normal --qat --disable_early_stopping --preprocess standard'),
    7: ('Log Min-Max', 'input representation: logarithmic Min-Max instead of the rescaling',
        '--normal --qat --disable_early_stopping --preprocess logminmax'),
    8: ('Quantile', 'input representation: quantile transform instead of the rescaling',
        '--normal --qat --disable_early_stopping --preprocess quantile'),
    # Not part of the default sequence: the same Exp. 3 / Exp. 4 trained with the x86 QAT configuration (reduce_range), the one that matches the
    # x86 INT8 kernels; used to separate real QAT degradation from INT8-kernel saturation of the qnnpack configuration.
    9: ('Exp. 3, x86 qconfig', 'Exp. 3 trained with the x86 QAT configuration',
        '--normal --qat --disable_early_stopping --qconfig x86'),
    10: ('Exp. 4, x86 qconfig', 'Exp. 4 trained with the x86 QAT configuration',
         '--normal --qat --disable_early_stopping --scale_close_to_zero --qconfig x86'),
    # Not part of the default sequence: the FP32 network keeps the BatchNorm layers of the AutoGluon network, they are folded into the next Linear
    # after the FP32 training, and the QAT starts from the folded weights (AutoGluon places the BatchNorm before each Linear, so Linear+BN fusion
    # does not apply).
    11: ('Exp. 3, BatchNorm folded', 'Exp. 3 with the BatchNorm kept in training and folded into the next Linear',
         '--normal --qat --disable_early_stopping --keep_batchnorm'),
    12: ('Exp. 4, BatchNorm folded', 'Exp. 4 with the BatchNorm kept in training and folded into the next Linear',
         '--normal --qat --disable_early_stopping --scale_close_to_zero --keep_batchnorm'),
}

# Experiment 5 (feature selection) is still produced for reproducibility, but it is not reported in the paper.
PAPER_EXPERIMENTS = (1, 2, 3, 4, 5)
SELECTS_FEATURES = (5,)
RESCALES = (4, 5, 10, 12)


def experiment_number(base, seed):
    """Value of --experiment_number of the run of experiment `base` with `seed`."""
    return 100000 + base * 100 + seed


def split_experiment_number(number):
    """Inverse of experiment_number(): (base, seed); a plain number (single run) gives (number, None)."""
    if number >= 100000:
        return (number - 100000) // 100, (number - 100000) % 100
    return number, None


def experiment_flags(base, important_features_file=None):
    """Flags of autogluon_to_pytorch_qat.py that define experiment `base`."""
    flags = EXPERIMENTS[base][2]
    if base in SELECTS_FEATURES:
        if not important_features_file:
            raise ValueError(f'experiment {base} needs an important-features file (see select_features.py)')
        flags += f' --important_features_file {important_features_file}'
    return flags
