"""
Tests of the parts of the pipeline that run on a CPU without AutoGluon, a GPU or the real datasets: the data splits, the tables, the exporters and
their guard, and the consistency of the experiment definitions. They use small synthetic data and take about a minute.

    python -m unittest discover -s tests -v
"""
import ast
import itertools
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'examples' / 'CICIDS2017' / 'scripts'
DEVICE = ROOT / 'examples' / 'IoT_test' / 'm5stack_ids_eval'
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(DEVICE))


def run(command, **kwargs):
    return subprocess.run([str(c) for c in command], capture_output=True, text=True, **kwargs)


def synthetic_unsw_files(directory, n_files=2, seed=0):
    """UNSW-NB15-like CSV files (label 0 = benign, `id` and `attack_cat` columns, heavy-tailed features, some exact duplicates)."""
    rng = np.random.default_rng(seed)
    os.makedirs(directory, exist_ok=True)
    for part in range(n_files):
        n_benign, n_attack = 1500, 600
        rows = []
        for label, n in ((0, n_benign), (1, n_attack)):
            shift = 0.0 if label == 0 else 1.2
            frame = pd.DataFrame({f'f{i}': rng.lognormal(mean=shift * (i % 3), sigma=1.5, size=n) for i in range(8)})
            frame['tiny'] = rng.choice([1e-7, 3e-7, 1.0, 5.0], size=n)           # values below the near-zero threshold
            frame['attack_cat'] = 'Normal' if label == 0 else rng.choice(['DoS', 'Fuzzers'], size=n)
            frame['label'] = label
            rows.append(frame)
        data = pd.concat(rows, ignore_index=True)
        data = pd.concat([data, data.sample(60, random_state=seed)], ignore_index=True)       # exact duplicates
        data.insert(0, 'id', range(len(data)))
        data.sample(frac=1.0, random_state=seed + part).to_csv(os.path.join(directory, f'UNSW_NB15_part{part + 1}.csv'), index=False)


def feature_rows(frame):
    features = [c for c in frame.columns if c not in ('ID', 'Label')]
    return set(map(tuple, frame[features].round(12).itertuples(index=False, name=None)))


class DatasetCreatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.directory = os.path.join(cls.tmp, 'UNSW_NB15', 'original')
        synthetic_unsw_files(cls.directory)
        result = run([sys.executable, SCRIPTS / 'dataset_creator.py', '--directory', cls.directory, '--seeds', '0,1', '--cpus', '1', '--keep-attack-type'])
        assert result.returncode == 0, result.stderr[-2000:]
        cls.base = os.path.join(cls.tmp, 'UNSW_NB15', 'balanced_binary')
        cls.splits = {s: {part: pd.read_csv(os.path.join(cls.base, f'split_{s}', f'{part}.csv'))
                          for part in ('train', 'validation', 'test', 'finetune')} for s in (0, 1)}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_splits_do_not_share_rows(self):
        for seed, parts in self.splits.items():
            rows = {name: feature_rows(frame) for name, frame in parts.items()}
            for a, b in itertools.combinations(rows, 2):
                self.assertFalse(rows[a] & rows[b], f'split_{seed}: {a} and {b} share feature vectors')

    def test_no_duplicates_inside_a_split(self):
        for parts in self.splits.values():
            for name, frame in parts.items():
                features = [c for c in frame.columns if c not in ('ID', 'Label')]
                self.assertFalse(frame[features].duplicated().any(), name)

    def test_min_max_is_fitted_on_the_training_part(self):
        for seed, parts in self.splits.items():
            features = [c for c in parts['train'].columns if c not in ('ID', 'Label')]
            train = parts['train'][features]
            self.assertTrue(np.allclose(train.min().values, 0) and np.allclose(train.max().values, 1))
            for name in ('validation', 'test', 'finetune'):
                other = parts[name][features]
                self.assertTrue((other.values >= 0).all() and (other.values <= 1).all(), f'{name} of split_{seed} is not clipped to [0, 1]')

    def test_row_id_carries_no_information_about_the_class(self):
        train = self.splits[0]['train']
        correlation = pd.Series(train['ID']).corr(train['Label'], method='spearman')
        self.assertLess(abs(correlation), 0.1)

    def test_classes_are_balanced_and_split_info_is_written(self):
        info = json.load(open(os.path.join(self.base, 'split_0', 'split_info.json')))
        for part in ('train', 'validation', 'test', 'finetune'):
            counts = info['splits'][part]
            self.assertAlmostEqual(counts['attack_rows'] / counts['rows'], 0.5, delta=0.02)
        self.assertEqual(info['files_in_reading_order'], ['UNSW_NB15_part1.csv', 'UNSW_NB15_part2.csv'])
        self.assertTrue((self.splits[0]['test'].columns == self.splits[0]['train'].columns).all())

    def test_every_seed_has_its_own_split(self):
        self.assertFalse(self.splits[0]['test'].equals(self.splits[1]['test']))

    def test_same_input_gives_the_same_splits_and_file_order_matters(self):
        def create(name, extra=()):
            directory = os.path.join(self.tmp, name, 'UNSW_NB15', 'original')
            shutil.copytree(self.directory, directory)
            result = run([sys.executable, SCRIPTS / 'dataset_creator.py', '--directory', directory, '--seeds', '0', '--cpus', '1', *extra])
            self.assertEqual(result.returncode, 0, result.stderr[-1500:])
            return pd.read_csv(os.path.join(self.tmp, name, 'UNSW_NB15', 'balanced_binary', 'split_0', 'test.csv')), result
        again, _ = create('again')
        self.assertTrue(again.equals(self.splits[0]['test']), 'the same files and seed must give identical splits')
        reordered, _ = create('reordered', ['--file-order', 'UNSW_NB15_part2.csv,UNSW_NB15_part1.csv'])
        self.assertFalse(reordered.equals(self.splits[0]['test']), 'another file order is another row order, hence other splits')
        explicit, _ = create('explicit', ['--file-order', 'UNSW_NB15_part1.csv,UNSW_NB15_part2.csv'])
        self.assertTrue(explicit.equals(self.splits[0]['test']), 'the default order is the alphabetical one')

    def test_wrong_file_order_is_rejected(self):
        directory = os.path.join(self.tmp, 'wrong', 'UNSW_NB15', 'original')
        shutil.copytree(self.directory, directory)
        result = run([sys.executable, SCRIPTS / 'dataset_creator.py', '--directory', directory, '--seeds', '0', '--cpus', '1', '--file-order', 'a.csv,b.csv'])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('--file-order must name every CSV file', result.stderr)


class ResultTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.root = os.path.join(cls.tmp, 'DS', 'balanced_binary')
        rng = np.random.default_rng(1)
        for seed in range(5):
            split = os.path.join(cls.root, f'split_{seed}')
            os.makedirs(split)
            json.dump({'splits': {'test': {'rows': 1000, 'attack_rows': 300}}, 'features': ['a']}, open(os.path.join(split, 'split_info.json'), 'w'))
            base = os.path.join(cls.root, 'baselines', f'split_{seed}')
            os.makedirs(base)
            # accuracy and recall of a test set with 30 % of malicious flows and a false-positive rate of exactly 5 %
            recall, fpr, share = 0.9, 0.05, 0.3
            accuracy = share * recall + (1 - share) * (1 - fpr)
            pd.DataFrame([{'model': 'random_forest', 'accuracy': accuracy, 'precision': 0.8, 'recall': recall, 'f1': 0.85}]).to_csv(
                os.path.join(base, 'baselines_classical_ml_results.csv'), index=False)
        models = os.path.join(cls.root, 'compressed_models')
        os.makedirs(models)
        # experiment -> shift of the INT8 accuracy; the seed-to-seed noise is shared, and every run has a little noise of its own (a paired t-test
        # needs differences with some variance), so the paired differences are the shifts up to about 0.01
        noise = rng.normal(0, 0.2, 5)
        for experiment, shift in {3: 0.0, 4: -1.0, 6: -2.0, 7: 0.5, 8: 0.25}.items():
            rows = []
            for seed in range(5):
                rows.append({'seed': seed, 'normal_test_accuracy': 90 + noise[seed],
                             'qat_test_accuracy': 89 + shift + noise[seed] + rng.normal(0, 0.01),
                             'normal_recall': 0.9, 'normal_fpr': 0.1, 'normal_f1': 0.9, 'qat_recall': 0.88, 'qat_fpr': 0.1 - shift / 100, 'qat_f1': 0.89})
            pd.DataFrame(rows).to_csv(os.path.join(models, f'multiseed_exp{experiment}_results.csv'), index=False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_fpr_derivation_is_exact_for_any_prevalence(self):
        from utils.results import derive_fpr
        for share, recall, fpr in itertools.product((0.1, 0.3, 0.5, 0.9), (0.6, 0.95), (0.01, 0.2)):
            accuracy = share * recall + (1 - share) * (1 - fpr)
            self.assertAlmostEqual(derive_fpr(accuracy, recall, share), fpr, places=12)

    def test_dataset_argument(self):
        from utils.results import parse_dataset
        self.assertEqual(parse_dataset('A=/x/y'), ('A', '/x/y', 'compressed_models'))
        self.assertEqual(parse_dataset('A=/x/y,compressed_models_bnfold'), ('A', '/x/y', 'compressed_models_bnfold'))
        with self.assertRaises(ValueError):
            parse_dataset('no-equals-sign')

    def test_exact_wilcoxon_and_holm(self):
        from scipy import stats
        sys.argv = ['x']
        import build_preprocessing_table as bpt
        self.assertAlmostEqual(bpt.exact_wilcoxon_p(np.array([1., 2, 3, 4, 5])), 0.0625)
        rng = np.random.default_rng(3)
        for _ in range(50):
            d = rng.normal(size=int(rng.integers(3, 9)))
            self.assertAlmostEqual(bpt.exact_wilcoxon_p(d), stats.wilcoxon(d).pvalue, places=10)
        adjusted = bpt.holm([0.01, 0.04, 0.03, float('nan')])
        self.assertTrue(np.allclose(adjusted[:3], [0.03, 0.06, 0.06]) and np.isnan(adjusted[3]))

    def test_metrics_table_derives_the_baseline_fpr_with_the_real_share(self):
        out = os.path.join(self.tmp, 'metrics_out')
        result = run([sys.executable, SCRIPTS / 'build_metrics_table.py', '--dataset', f'DS={self.root}', '--output_dir', out])
        self.assertEqual(result.returncode, 0, result.stderr[-1500:])
        baselines = pd.read_csv(os.path.join(out, 'baselines_table.csv'))
        fpr = baselines[(baselines.metric == 'fpr') & (baselines.model == 'random_forest')].iloc[0]
        self.assertAlmostEqual(fpr['mean'], 0.05, places=9)
        self.assertEqual(fpr['n_splits'], 5)
        table = pd.read_csv(os.path.join(out, 'metrics_table.csv'))
        self.assertEqual(set(table.experiment), {3, 4, 6, 7, 8})            # experiments without a CSV are skipped with a warning
        self.assertIn('WARNING', result.stdout)

    def test_preprocessing_table_pairs_the_seeds(self):
        out = os.path.join(self.tmp, 'prep_out')
        result = run([sys.executable, SCRIPTS / 'build_preprocessing_table.py', '--dataset', f'DS={self.root}', '--output_dir', out])
        self.assertEqual(result.returncode, 0, result.stderr[-1500:])
        tests = pd.read_csv(os.path.join(out, 'preprocessing_tests.csv'))
        row = tests[(tests.alternative == 6) & (tests.metric == 'qat_acc')].iloc[0]       # proposed (4) - standardization (6) = -1.0 - (-2.0) = +1.0
        self.assertAlmostEqual(row['mean_diff'], 1.0, delta=0.05)
        self.assertEqual(row['ahead'], 5)
        self.assertAlmostEqual(row['p_wilcoxon'], 0.0625)
        self.assertEqual(len(tests), 12)                                    # 4 alternatives x 3 metrics, Holm-adjusted together
        self.assertTrue(tests['p_holm'].notna().sum() >= 1)


class ExporterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import torch.nn as nn
        from torch.quantization import convert, prepare_qat
        import export_qat_int8 as ex
        cls.tmp = tempfile.mkdtemp()
        torch.manual_seed(0)
        layers = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Dropout(0.0), nn.Linear(8, 8), nn.ReLU(), nn.Dropout(0.0), nn.Linear(8, 2))
        net = ex.QATWrapper(ex.Base(layers))
        net.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
        prepare_qat(net, inplace=True)
        net.train()
        for _ in range(30):
            net(torch.rand(64, 6))
        cls.checkpoint = os.path.join(cls.tmp, 'toy_qat.pth')
        torch.save(convert(net.eval()).state_dict(), cls.checkpoint)
        rng = np.random.default_rng(0)
        data = pd.DataFrame(rng.random((200, 6)))
        data['label'] = rng.integers(0, 2, 200)
        cls.check_csv = os.path.join(cls.tmp, 'check.csv')
        data.to_csv(cls.check_csv, header=False, index=False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def export(self, *extra):
        return run([sys.executable, DEVICE / 'export_qat_int8.py', '--qat_model', self.checkpoint, '--output_header', os.path.join(self.tmp, 'toy.h'),
                    '--model_name', 'toy', *extra])

    def summary(self, layer_types, first_inputs=6):
        layers = [{'type': 'Linear', 'inputs': first_inputs, 'outputs': 8}, {'type': layer_types}, {'type': 'Linear', 'inputs': 8, 'outputs': 8},
                  {'type': layer_types}, {'type': 'Linear', 'inputs': 8, 'outputs': 2}, {'type': 'Softmax'}]
        path = os.path.join(self.tmp, f'summary_{layer_types}_{first_inputs}.json')
        json.dump({'layers': layers}, open(path, 'w'))
        return path

    def test_stops_without_knowing_the_activation(self):
        result = self.export()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('does not record its activation function', result.stderr)

    def test_assume_relu_exports_and_the_checks_pass(self):
        result = self.export('--assume_relu', '--check_csv', self.check_csv)
        self.assertEqual(result.returncode, 0, result.stderr[-1500:])
        self.assertIn('header parsed back: identical parameters', result.stdout)
        self.assertIn('scalar transcription of the firmware loop == emulation', result.stdout)
        self.assertIn('layer sizes: [6, 8, 8, 2]', result.stdout)

    def test_relu_summary_passes_even_if_the_first_layer_has_more_inputs(self):
        for first_inputs in (6, 7):                       # the AutoGluon network can take more inputs than the data has columns
            result = self.export('--architecture_json', self.summary('ReLU', first_inputs))
            self.assertEqual(result.returncode, 0, result.stderr[-1500:])

    def test_tanh_summary_stops_the_export(self):
        result = self.export('--architecture_json', self.summary('Tanh'))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Tanh', result.stderr)

    def test_summary_of_another_network_stops_the_export(self):
        path = os.path.join(self.tmp, 'other.json')
        json.dump({'layers': [{'type': 'Linear', 'inputs': 6, 'outputs': 16}, {'type': 'ReLU'}, {'type': 'Linear', 'inputs': 16, 'outputs': 2}]}, open(path, 'w'))
        result = self.export('--architecture_json', path)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('wrong summary for this checkpoint', result.stderr)

    def test_integer_emulation_follows_the_pytorch_model(self):
        import torch
        import export_qat_int8 as ex
        torch.backends.quantized.engine = 'onednn' if 'onednn' in torch.backends.quantized.supported_engines else 'qnnpack'
        model, indices = ex.load_quantized_model(torch.load(self.checkpoint, map_location='cpu'))
        params = ex.extract_parameters(model, indices)
        x = np.random.default_rng(5).random((500, 6)).astype(np.float32)
        with torch.no_grad():
            torch_pred = model(torch.tensor(x)).numpy().argmax(1)
        emulated = ex.emulate(params, x).argmax(1)
        self.assertGreater(np.mean(torch_pred == emulated), 0.98)


class DefinitionTests(unittest.TestCase):
    def test_experiment_numbers_round_trip(self):
        from utils.experiments import EXPERIMENTS, experiment_number, split_experiment_number
        for base in EXPERIMENTS:
            for seed in range(5):
                self.assertEqual(split_experiment_number(experiment_number(base, seed)), (base, seed))

    def test_analysis_flags_follow_the_experiment_definitions(self):
        import run_analyses as ra
        self.assertEqual(ra.input_flags(1), [])
        self.assertEqual(ra.input_flags(3), [])
        self.assertEqual(ra.input_flags(4), ['--scale_close_to_zero'])
        self.assertEqual(ra.input_flags(6), ['--preprocess', 'standard'])
        self.assertEqual(ra.input_flags(8), ['--preprocess', 'quantile'])
        self.assertEqual(sorted(ra.EQUIVALENCE_EXPERIMENTS), [1, 2, 3, 4, 9, 10, 11, 12])      # no feature selection, no alternative representation

    def test_every_python_file_parses_and_every_shell_script_is_valid(self):
        for path in ROOT.rglob('*.py'):
            if '.git' not in path.parts:
                ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        if shutil.which('bash'):
            for path in list((ROOT / 'cluster').glob('*.sh')) + list((ROOT / 'cluster').glob('*.sbatch')):
                self.assertEqual(run(['bash', '-n', path]).returncode, 0, path.name)


if __name__ == '__main__':
    unittest.main()
