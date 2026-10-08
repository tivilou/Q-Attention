"""Source configuration/scalar state checks, not neural or AMP certification."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT/'baselines/retacred/luke_training_recipe.py'
SPEC = importlib.util.spec_from_file_location('recipe_binding_fixture', PATH)
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)
CACHE = Path(os.environ.get('LUKE_RECIPE_SOURCE_CACHE',
    ROOT/'.external/retacred/luke-recipe-binding-20261008/source'))


class TrainingRecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec, cls.data = r.checked_sources(CACHE)

    def test_locked_dependency_versions(self):
        config = r.binding(self.data)
        self.assertEqual(config['versions'], {'allennlp': '2.9.2', 'torch': '1.10.2', 'transformers': '4.13.0'})
        self.assertEqual(self.spec['framework']['archive_sha256'],
            '004d6a4ad64891779d90f9c407b49141a5214aef0c6cc0794dba776f9927e687')

    def test_configuration_and_environment(self):
        config = r.binding(self.data)
        self.assertEqual(config['trainer']['num_epochs'], 5)
        self.assertEqual(config['trainer']['num_gradient_accumulation_steps'], 8)
        self.assertEqual(config['trainer']['validation_metric'], '+micro_fscore')
        self.assertEqual(config['trainer']['patience'], 3)
        self.assertEqual(config['data_loader'], {'batch_size': 4, 'shuffle': True})
        self.assertEqual(config['seeds'], {'random_seed': 0, 'numpy_seed': 0, 'pytorch_seed': 0})
        self.assertEqual(config['environment'], ['DATASET', 'TRAIN_DATA_PATH', 'TRANSFORMERS_MODEL_NAME', 'VALIDATION_DATA_PATH'])

    def test_optimizer_and_clipping_defaults_bound(self):
        config = r.binding(self.data)
        self.assertEqual(config['optimizer_backend'], 'torch.optim.AdamW')
        self.assertEqual(config['optimizer_defaults'], {'betas': (0.9, 0.999), 'eps': 1e-8, 'amsgrad': False})
        self.assertEqual(config['trainer_defaults'], {'grad_norm': False, 'grad_clipping': None,
            'use_amp': False, 'grad_scaling': True})
        groups = config['trainer']['optimizer']['parameter_groups']
        self.assertEqual(groups, [[['bias', 'LayerNorm.weight'], {'weight_decay': 0}]])

    def test_source_update_count_and_partial_group(self):
        self.assertEqual(r.source_update_count(self.data, 9, 8), 2)
        self.assertEqual(r.source_update_count(self.data, 8, 8), 1)
        self.assertEqual(r.source_update_count(self.data, 1, 8), 1)
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                r.source_update_count(self.data, value, 8)

    def test_source_schedule_initial_zero_then_first_update_lr(self):
        schedule, _ = r.load_scalar_classes(self.data)
        optimizer = SimpleNamespace(param_groups=[{'lr': 1e-5}])
        scheduler = schedule(optimizer, 5, 10, 0.06)
        self.assertEqual(scheduler.warmup_steps, 3)
        self.assertEqual(optimizer.param_groups[0]['lr'], 0)
        scheduler.step_batch(1)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 1e-5/3)
        scheduler.step_batch(3)
        self.assertEqual(optimizer.param_groups[0]['lr'], 1e-5)
        scheduler.step_batch(50)
        self.assertEqual(optimizer.param_groups[0]['lr'], 0)

    def test_source_warmup_floor_and_no_warmup(self):
        schedule, _ = r.load_scalar_classes(self.data)
        optimizer = SimpleNamespace(param_groups=[{'lr': 1e-5}])
        scheduler = schedule(optimizer, 5, 3, 0.06)
        self.assertEqual(scheduler.warmup_steps, 0)
        self.assertEqual(optimizer.param_groups[0]['lr'], 1e-5)
        scheduler.step_batch(1)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 14/15*1e-5)

    def test_tracker_ties_do_not_replace_best(self):
        _, tracker_class = r.load_scalar_classes(self.data)
        tracker = tracker_class('+micro_fscore', 3)
        for score in (0.4, 0.4, 0.4):
            tracker.add_metrics({'micro_fscore': score})
        self.assertEqual(tracker.best_epoch, 0)
        self.assertFalse(tracker.should_stop_early())
        tracker.add_metrics({'micro_fscore': 0.4})
        self.assertTrue(tracker.should_stop_early())

    def test_scalar_scheduler_state_roundtrip(self):
        schedule, _ = r.load_scalar_classes(self.data)
        optimizer = SimpleNamespace(param_groups=[{'lr': 1e-5}])
        scheduler = schedule(optimizer, 5, 10, 0.06)
        scheduler.step_batch(7)
        state = copy.deepcopy(scheduler.state_dict())
        expected = scheduler.get_values()
        scheduler.step_batch(15)
        scheduler.load_state_dict(state)
        self.assertEqual(scheduler.get_values(), expected)

    def test_full_gate_retains_neural_pending(self):
        gate = r.run_gate(CACHE)
        self.assertEqual(gate['update_count_checks'], 260)
        self.assertEqual(gate['learning_rate_checks'], 583)
        self.assertFalse(gate['formal_comparison_admitted'])
        self.assertFalse(gate['actual_AllenNLP_runtime'])
        self.assertFalse(gate['actual_Jsonnet_execution'])
        self.assertEqual(gate['model_jobs'], 0)
        self.assertEqual(gate['partition_counterexample']['source_equal_batch_mean'], 2.)
        self.assertEqual(gate['partition_counterexample']['example_weighted_mean'], 1.8)
        json.dumps(gate, allow_nan=False)

    def test_import_closure_rejected_if_nested(self):
        data = dict(self.data)
        name = 'examples/relation_classification/configs/lib/base.libsonnet'
        data[name] += '\nlocal hidden = import "not-bound.jsonnet";\n'
        with self.assertRaisesRegex(ValueError, 'nested'):
            r.binding(data)

    def test_corrupt_cache_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/next(iter(self.spec['files'].values()))['file']
            target.write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'size/SHA'):
                r.checked_sources(directory)
            self.assertEqual(target.read_bytes(), b'tampered')

    def test_cli_offline_no_backend(self):
        script = '''
import importlib.util,sys
def guard(event,args):
 if event=='import' and args[0].split('.')[0] in ('torch','numpy','allennlp','tensorflow','jax'): raise AssertionError('backend import')
 if event.startswith('socket.'): raise AssertionError('network')
sys.addaudithook(guard)
spec=importlib.util.spec_from_file_location('audited',sys.argv[1])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
raise SystemExit(module.main(['--source-cache',sys.argv[2]]))
'''
        done = subprocess.run([sys.executable, '-c', script, str(PATH), str(CACHE)],
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), text=True, capture_output=True, timeout=30)
        self.assertEqual(done.returncode, 2, done.stderr)
        self.assertEqual(json.loads(done.stdout)['status'], 'source_recipe_bound_neural_runtime_pending')


if __name__ == '__main__':
    unittest.main()
