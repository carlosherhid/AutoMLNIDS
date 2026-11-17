"""
Runs, for one dataset, the analyses of the paper that need the trained models, with the arguments that tie them to the
multi-seed runs (seed i is evaluated on split_i). Run it after Experiments 1-5 have finished:

  ptq          static post-training quantization of the FP32 model of every seed (post_training_quantization.py), for the experiments of
               --ptq_experiments (default: Experiment 1). Each one is quantized with the input transformation it was trained with
               (rescaling of Experiment 4, standardization / log / quantile of Experiments 6-8), so PTQ and QAT are compared on the same data
  equivalence  prediction agreement between the AutoGluon model and the FP32 replica of --equivalence_experiment (default Experiment 3,
               which has the same Min-Max input as the AutoGluon model), seed 0, on split_0
  noise        accuracy of the Experiment 4 FP32 and QAT models, seed 0, under Gaussian input noise (robustness_noise_eval.py)
  resolution   share of each feature's resolution that survives 8-bit input quantization, on split_0
  tradeoff     size and latency of the model of Experiments 1-5, seed 0, one thread (build_tradeoff_table.py)
  engines      re-evaluation of the checkpoints of Experiments 1-5 with other quantized engines (evaluate_checkpoint_metrics.py)
  sweep        Experiment 4 for a range of rescaling factors, seed 0, on split_0 (scale_factor_sweep.py)

Everything is written under <data_root>/analyses/, one log per stage.

Usage:
    python scripts/run_analyses.py --data_root <data>/balanced_binary [--stages ptq noise ...] [--seeds 0 1 2 3 4]
    # runs made with the BatchNorm kept and folded (submit_experiments.sh --keep-batchnorm):
    python scripts/run_analyses.py --data_root <data>/balanced_binary --models_dir compressed_models_bnfold --out_dir analyses_bnfold --keep_batchnorm
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.experiments import EXPERIMENTS, SELECTS_FEATURES  # noqa: E402
from utils.quantization import DEFAULT_ENGINE  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = ('ptq', 'equivalence', 'noise', 'resolution', 'tradeoff', 'engines', 'sweep')
EXPERIMENT_1, EXPERIMENT_3, EXPERIMENT_4 = 1, 3, 4     # numbers of utils/experiments.py
# experiments whose FP32 replica can be compared with AutoGluon: no feature selection and no alternative representation (verify_functional_equivalence.py
# only knows the rescaling)
EQUIVALENCE_EXPERIMENTS = [n for n in EXPERIMENTS if n not in SELECTS_FEATURES and '--preprocess' not in EXPERIMENTS[n][2].split()]


def input_flags(experiment):
    """The flags of post_training_quantization.py that give an experiment's FP32 network the input it was trained on (the rescaling or the
    alternative representation of utils/experiments.py; the other flags of the experiment only concern the training)."""
    tokens = EXPERIMENTS[experiment][2].split()
    flags = ['--scale_close_to_zero'] if '--scale_close_to_zero' in tokens else []
    if '--preprocess' in tokens:
        flags += ['--preprocess', tokens[tokens.index('--preprocess') + 1]]
    return flags


def run(command, log_path):
    """Runs a command, echoing its output to the console and to log_path; returns its exit code."""
    print('+', ' '.join(command), flush=True)
    with open(log_path, 'w', encoding='utf-8') as log:
        log.write('+ ' + ' '.join(command) + '\n')
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace')
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        return process.wait()


def main():
    parser = argparse.ArgumentParser(description='Analyses of the paper for one dataset')
    parser.add_argument('--data_root', type=str, required=True, help='<data>/balanced_binary: split_<seed>/ and compressed_models/')
    parser.add_argument('--models_dir', type=str, default='compressed_models',
                        help='folder of <data_root> with the checkpoints and logs of the experiments (compressed_models_bnfold for --keep_batchnorm runs)')
    parser.add_argument('--out_dir', type=str, default='analyses', help='folder of <data_root> where the analyses are written')
    parser.add_argument('--keep_batchnorm', action='store_true',
                        help='the experiments were run with the BatchNorm kept and folded: the rescaling-factor sweep is run the same way')
    parser.add_argument('--stages', nargs='+', choices=STAGES, default=list(STAGES))
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--ptq_experiments', type=int, nargs='+', default=[EXPERIMENT_1],
                        choices=[n for n in EXPERIMENTS if n not in SELECTS_FEATURES],
                        help='experiments whose FP32 network the ptq stage quantizes (Experiment 5 selects features: not supported). Experiment 1 keeps its '
                             'log name ptq_seed<S>.txt; the others are written as ptq_exp<N>_seed<S>.txt')
    parser.add_argument('--equivalence_experiment', type=int, default=EXPERIMENT_3, choices=EQUIVALENCE_EXPERIMENTS,
                        help='experiment whose FP32 replica the equivalence stage compares with the AutoGluon model; the replica is fed the input it '
                             'was trained with. The default, Experiment 3, uses the raw Min-Max input of the AutoGluon model (no rescaling to undo)')
    parser.add_argument('--paper_seed', type=int, default=0, help='seed of the single model shown in the paper (split_0 by default)')
    parser.add_argument('--engines', nargs='+', default=['x86', 'onednn', 'qnnpack'])
    parser.add_argument('--latency_engine', type=str, default=DEFAULT_ENGINE, help='quantized engine of the size/latency table (default: the engine of every other INT8 number)')
    parser.add_argument('--latency_threads', type=int, default=1)
    parser.add_argument('--sweep_epochs', type=int, default=100)
    parser.add_argument('--sweep_factors', type=float, nargs='+', default=None)
    args = parser.parse_args()

    root = os.path.abspath(args.data_root)
    predictor = os.path.join(root, 'split_0', 'automl_search')
    logs = os.path.join(root, args.models_dir)
    out = os.path.join(root, args.out_dir)
    features_file = os.path.join(predictor, 'feature_ranking_custom_sbe', 'important_features.txt')
    split = lambda seed: os.path.join(root, f'split_{seed}')           # noqa: E731
    checkpoint = lambda kind, experiment, seed: os.path.join(            # noqa: E731
        logs, f'{kind}_pytorch_model_exp{100000 + experiment * 100 + seed}.pth')
    python = sys.executable
    script = lambda name: os.path.join(HERE, name)                       # noqa: E731
    os.makedirs(out, exist_ok=True)

    failures = []

    def stage(name, command, log_name=None):
        if run(command, os.path.join(out, (log_name or name) + '.txt')) != 0:
            failures.append(log_name or name)

    paper = args.paper_seed
    for name in args.stages:
        print(f'\n===== {name} =====', flush=True)
        if name == 'ptq':
            os.makedirs(os.path.join(out, 'ptq'), exist_ok=True)
            for experiment in args.ptq_experiments:
                for seed in args.seeds:
                    stage(name, [python, script('post_training_quantization.py'), '--predictor_path', predictor,
                                 '--data_dir', split(seed), '--output_dir', os.path.join(out, 'ptq'),
                                 '--experiment_number', str(200000 + experiment * 100 + seed), '--seed', str(seed),
                                 '--normal_model_path', checkpoint('normal', experiment, seed)] + input_flags(experiment),
                          f'ptq_seed{seed}' if experiment == EXPERIMENT_1 else f'ptq_exp{experiment}_seed{seed}')
        elif name == 'equivalence':
            experiment = args.equivalence_experiment
            stage(name, [python, script('verify_functional_equivalence.py'), '--predictor_path', predictor,
                         '--data_dir', split(paper), '--normal_model_path', checkpoint('normal', experiment, paper),
                         '--output_csv', os.path.join(out, f'functional_equivalence_exp{experiment}.csv')] +
                  [flag for flag in input_flags(experiment) if flag == '--scale_close_to_zero'], f'equivalence_exp{experiment}')
        elif name == 'noise':
            os.makedirs(os.path.join(out, 'noise'), exist_ok=True)
            stage(name, [python, script('robustness_noise_eval.py'), '--predictor_path', predictor,
                         '--data_dir', split(paper), '--normal_model_path', checkpoint('normal', EXPERIMENT_4, paper),
                         '--qat_model_path', checkpoint('qat', EXPERIMENT_4, paper), '--scale_close_to_zero',
                         '--seed', str(paper), '--output_dir', os.path.join(out, 'noise')])
        elif name == 'resolution':
            stage(name, [python, script('input_resolution_diagnostic.py'), '--data_dir', split(paper),
                         '--output_csv', os.path.join(out, 'input_resolution.csv')])
        elif name == 'tradeoff':
            stage(name, [python, script('build_tradeoff_table.py'), '--predictor_path', predictor, '--data_dir', split(paper),
                         '--logs_dir', logs, '--only_paper_experiments', '--paper_seed', str(paper),
                         '--engine', args.latency_engine, '--threads', str(args.latency_threads),
                         '--output_csv', os.path.join(out, f'tradeoff_table_{args.latency_engine}_{args.latency_threads}thread.csv')])
        elif name == 'engines':
            for engine in args.engines:
                stage(name, [python, script('evaluate_checkpoint_metrics.py'), '--predictor_path', predictor,
                             '--data_dir_template', os.path.join(root, 'split_{seed}'), '--logs_dir', logs,
                             '--important_features_file', features_file, '--engine', engine,
                             '--seeds'] + [str(s) for s in args.seeds] +
                      ['--output_csv', os.path.join(out, f'detailed_metrics_{engine}.csv')], f'engines_{engine}')
        elif name == 'sweep':
            os.makedirs(os.path.join(out, 'scale_sweep'), exist_ok=True)
            command = [python, script('scale_factor_sweep.py'), '--predictor_path', predictor, '--data_dir', split(paper),
                       '--output_dir', os.path.join(out, 'scale_sweep'), '--epochs', str(args.sweep_epochs), '--seed', str(paper)]
            if args.sweep_factors:
                command += ['--factors'] + [str(f) for f in args.sweep_factors]
            if args.keep_batchnorm:      # scale_factor_sweep.py forwards --extra_args to autogluon_to_pytorch_qat.py (default: --disable_early_stopping)
                command += ['--extra_args', '--disable_early_stopping --keep_batchnorm']
            stage(name, command)

    print('\n===== done =====')
    if failures:
        print('FAILED stages:', ', '.join(failures))
        sys.exit(1)


if __name__ == '__main__':
    main()
