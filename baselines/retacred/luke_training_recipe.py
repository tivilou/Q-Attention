"""Hash-bound LUKE recipe/scalar scheduler audit; no neural certification."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
LOCK = Path(__file__).with_name('luke_recipe.lock.json')


def checked_sources(cache, lock_path=LOCK):
    spec = json.loads(Path(lock_path).read_text())
    if (spec['schema'] != 'luke-training-recipe.v1'
            or spec['revision'] != '251654548a15b9a8ddc708a53b715bdf3a112102'
            or spec['framework']['version'] != '2.9.2'):
        raise ValueError('recipe source identity invalid')
    cache = Path(cache).absolute()
    data = {}
    for name, row in spec['files'].items():
        if Path(row['file']).name != row['file'] or row['file'] in ('.', '..'):
            raise ValueError('unsafe source cache filename')
        path = cache/row['file']
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError('symlinked source rejected')
        raw = path.read_bytes()
        if len(raw) != row['size_bytes'] or hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise ValueError('recipe source size/SHA mismatch: '+name)
        data[name] = raw.decode('utf-8')
    return spec, data


def literal_object(text, name):
    """Read one literal object in locked Jsonnet; never evaluate Jsonnet/env."""
    marker = '"'+name+'":'
    start = text.index('{', text.index(marker)+len(marker))
    depth, quoted, escaped = 0, False, False
    for index in range(start, len(text)):
        char = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return json.loads(re.sub(r',\s*([}\]])', r'\1', text[start:index+1]))
    raise ValueError('unterminated source literal object')


def source_class(text, name):
    return next(node for node in ast.parse(text).body if isinstance(node, ast.ClassDef) and node.name == name)


def method(node, name):
    return next(child for child in node.body if isinstance(child, ast.FunctionDef) and child.name == name)


def default_values(node, names):
    pairs = zip(node.args.args[-len(node.args.defaults):], node.args.defaults)
    return {argument.arg: ast.literal_eval(value) for argument, value in pairs if argument.arg in names}


class RegistryCarrier:
    """No package registry state; class bodies only."""


def load_scalar_classes(data):
    namespace = {'Registrable': RegistryCarrier, 'ConfigurationError': ValueError}
    sources = (
        ('framework/training/scheduler.py', 'Scheduler'),
        ('framework/training/learning_rate_schedulers/learning_rate_scheduler.py', 'LearningRateScheduler'),
        ('framework/training/learning_rate_schedulers/polynomial_decay.py', 'PolynomialDecay'),
        ('examples/utils/learning_rate_schedulers/linear_with_warmup.py', 'CustomLinearWithWarmup'),
        ('framework/training/metric_tracker.py', 'MetricTracker'))
    for path, name in sources:
        node = copy.deepcopy(source_class(data[path], name))
        node.decorator_list = []
        tree = ast.Module(body=[ast.ImportFrom(module='__future__',
                          names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
        ast.fix_missing_locations(tree)
        exec(compile(tree, path, 'exec'), namespace)
    return namespace['CustomLinearWithWarmup'], namespace['MetricTracker']


def binding(data):
    base = data['examples/relation_classification/configs/lib/base.libsonnet']
    trainer = literal_object(base, 'trainer')
    loader = literal_object(base, 'data_loader')
    entry = data['examples/relation_classification/configs/transformers_luke_with_entity_aware_attention.jsonnet']
    imports = re.findall(r'import "([^"]+)"', entry)
    if imports != ['lib/base.libsonnet', 'lib/transformers_model_luke_with_entity_aware_attention.jsonnet']:
        raise ValueError('entry import closure differs')
    model_path = 'examples/relation_classification/configs/'+imports[1]
    model = data[model_path]
    if re.search(r'\bimport\s', base+'\n'+model):
        raise ValueError('unbound nested Jsonnet import')
    versions = dict(re.findall(r'\[\[package\]\]\s+name = "([^"]+)"\s+version = "([^"]+)"', data['poetry.lock']))
    seeds = {key: int(re.search(r'"'+key+r'":\s*(\d+)', base).group(1))
             for key in ('random_seed', 'numpy_seed', 'pytorch_seed')}
    optimizer = source_class(data['framework/training/optimizers.py'], 'AdamWOptimizer')
    opt_defaults = default_values(method(optimizer, '__init__'), {'betas', 'eps', 'amsgrad'})
    if not any(ast.unparse(parent) == 'torch.optim.AdamW' for parent in optimizer.bases):
        raise ValueError('adamw backend differs')
    source_trainer = source_class(data['framework/training/gradient_descent_trainer.py'], 'GradientDescentTrainer')
    names = {'grad_norm', 'grad_clipping', 'use_amp', 'grad_scaling'}
    defaults = default_values(method(source_trainer, '__init__'), names)
    construction_defaults = default_values(method(source_trainer, 'from_partial_objects'), names)
    if construction_defaults != defaults:
        raise ValueError('trainer constructor/default boundary differs')
    return {'trainer': trainer, 'data_loader': loader, 'seeds': seeds,
            'versions': {key: versions[key] for key in ('allennlp', 'torch', 'transformers')},
            'optimizer_backend': 'torch.optim.AdamW', 'optimizer_defaults': opt_defaults,
            'trainer_defaults': defaults, 'imports': imports,
            'environment': sorted(set(re.findall(r'std.extVar\("([^"]+)"\)', base+'\n'+model)))}


def source_update_count(data, physical_batches, accumulation):
    if type(physical_batches) is not int or physical_batches < 1 or type(accumulation) is not int or accumulation < 1:
        raise ValueError('positive integer batch counts required')
    cls = source_class(data['framework/training/gradient_descent_trainer.py'], 'GradientDescentTrainer')
    function = method(cls, 'from_partial_objects')
    statement = next(node for node in ast.walk(function) if isinstance(node, ast.Assign)
                     and ast.unparse(node) == 'batches_per_epoch = math.ceil(batches_per_epoch / num_gradient_accumulation_steps)')
    namespace = {'math': math, 'batches_per_epoch': physical_batches, 'num_gradient_accumulation_steps': accumulation}
    exec(compile(ast.Module(body=[statement], type_ignores=[]), 'locked_trainer_update_count', 'exec'), namespace)
    return namespace['batches_per_epoch']


def run_gate(cache, lock_path=LOCK):
    spec, data = checked_sources(cache, lock_path)
    config = binding(data)
    schedule, tracker_class = load_scalar_classes(data)
    count_checks = 0
    for batches in range(1, 66):
        for accumulation in (1, 2, 4, 8):
            if source_update_count(data, batches, accumulation) != (batches+accumulation-1)//accumulation:
                raise AssertionError('source update count differs')
            count_checks += 1
    lr_checks = 0
    for steps_per_epoch in (1, 3, 10, 101):
        total = config['trainer']['num_epochs']*steps_per_epoch
        warmup = int(total*config['trainer']['learning_rate_scheduler']['warmup_ratio'])
        optimizer = SimpleNamespace(param_groups=[{'lr': 1e-5}])
        scheduler = schedule(optimizer, 5, steps_per_epoch, 0.06)
        for step in range(total+2):
            scheduler.step_batch(step)
            expected = (0. if step >= total else step/warmup if warmup and step < warmup
                        else (total-step)/(total-warmup))*1e-5
            if not math.isclose(optimizer.param_groups[0]['lr'], expected, rel_tol=0, abs_tol=1e-16):
                raise AssertionError('locked scheduler differs from count oracle')
            lr_checks += 1
        state = copy.deepcopy(scheduler.state_dict())
        scheduler.steps = -100
        scheduler.load_state_dict(state)
        if scheduler.state_dict() != state:
            raise AssertionError('scalar scheduler state roundtrip failed')
    trainer = method(source_class(data['framework/training/gradient_descent_trainer.py'], 'GradientDescentTrainer'), '_train_epoch')
    statements = [ast.unparse(node) for node in ast.walk(trainer)]
    if 'loss = loss / len(batch_group)' not in statements:
        raise ValueError('source accumulation loss normalization differs')
    calls = [node for node in ast.walk(trainer) if isinstance(node, ast.Call)]
    schedule_line = next(node.lineno for node in calls if ast.unparse(node.func) == 'self._learning_rate_scheduler.step_batch')
    optimizer_line = next(node.lineno for node in calls if ast.unparse(node.func) == 'self.optimizer.step')
    if schedule_line >= optimizer_line:
        raise AssertionError('source trainer step ordering differs')
    tracker = tracker_class('+micro_fscore', patience=3)
    for score in (0.2, 0.3, 0.3, 0.3, 0.29):
        tracker.add_metrics({'micro_fscore': score})
    if tracker.best_epoch != 1 or not tracker.should_stop_early():
        raise AssertionError('strict-best tie/patience rule differs')
    state = tracker.state_dict()
    tracker.clear()
    tracker.load_state_dict(state)
    if tracker.state_dict() != state:
        raise AssertionError('scalar metric tracker state roundtrip failed')
    return {'schema': 'luke-training-recipe-audit.v1', 'status': 'source_recipe_bound_neural_runtime_pending',
            'revision': spec['revision'], 'source_files_verified': len(data), 'configuration': config,
            'config_evidence': 'locked literal blocks and imports; not Jsonnet execution',
            'scheduler_evidence': 'locked source classes with scalar optimizer/registry carriers',
            'update_count_checks': count_checks, 'learning_rate_checks': lr_checks,
            'warmup_rounding': 'floor_int(epochs*ceil(physical_batches/accumulation)*ratio)',
            'step_order': 'scheduler.step_batch(group_index+1) BEFORE optimizer.step',
            'accumulation': 'average physical-batch means by actual len(batch_group), including tail',
            'partition_counterexample': {'physical_sizes': [4, 4, 2], 'physical_mean_losses': [1., 2., 3.],
                'source_equal_batch_mean': 2., 'example_weighted_mean': 1.8},
            'tracker': {'best_epoch': 1, 'strict_improvement': True, 'ties_increment_patience': True,
                        'stop_after': 3, 'scalar_state_roundtrip': True},
            'framework_defaults_bound': True, 'source_config_import_closure_bound': True,
            'actual_AllenNLP_runtime': False, 'actual_Jsonnet_execution': False,
            'formal_comparison_admitted': False, 'model_jobs': 0, 'backend_imported': False,
            'network_requests': 0, 'pending': ['actual_framework_tokenizer_indexer_collation_equivalence',
                'Torch1.10.2_vs_installed_optimizer_autograd_dropout_equivalence',
                'CUDA_AMP_scaler_overflow_and_scheduler_resume_equivalence',
                'full_weight_numerical_runtime_and_method_specific_CoRE_CFIE_recipe']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-cache', type=Path, default=ROOT/'.external/retacred/luke-recipe-binding-20261008/source')
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run_gate(args.source_cache), indent=2, allow_nan=False))
        return 2
    except Exception as error:
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__, 'reason': str(error), 'model_jobs': 0}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
