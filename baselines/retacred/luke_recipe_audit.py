"""Offline LUKE evaluator algebra audit; no AllenNLP/model execution certificate."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
from itertools import product
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

REVISION = '251654548a15b9a8ddc708a53b715bdf3a112102'
METRIC_FILE = 'examples--relation_classification--metrics--multiway_f1.py'
METRIC_SHA256 = '9992976c88a3b9bca8b549486320fcc44cfe8950ea14acc30bbc10171585fc72'
ROOT = Path(__file__).resolve().parents[2]


class ScalarIds(list):
    """Declared list carrier for equality only; not a Torch tensor."""
    def __eq__(self, other):
        return [left == right for left, right in zip(self, other)]


class MetricCarrier:
    @staticmethod
    def detach_tensors(*values):
        return values


def mean(values):
    return sum(values)/len(values) if values else float('nan')


def checked_source(cache):
    path = Path(cache).absolute()/METRIC_FILE
    if any(part.is_symlink() for part in (path,*path.parents)):
        raise ValueError('symlinked source cache rejected')
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != METRIC_SHA256:
        raise ValueError('pinned LUKE scorer SHA mismatch')
    return path,data


def load_scorer(cache):
    path,data = checked_source(cache)
    tree = ast.parse(data)
    names = ('DEFAULT_IGNORED_LABEL','normalize_kbp37_label','MultiwayF1')
    chosen = [node for node in tree.body if
        isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names
        or isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id in names
                                              for target in node.targets)]
    if len(chosen) != 3:
        raise ValueError('unexpected source scorer structure')
    namespace = {'Counter':Counter, 'Dict':Dict, 'List':List, 'Metric':MetricCarrier,
                 'torch':SimpleNamespace(LongTensor=ScalarIds), 'np':SimpleNamespace(mean=mean)}
    exec(compile(ast.Module(body=chosen,type_ignores=[]),str(path),'exec'),namespace)
    return namespace['MultiwayF1'],namespace['normalize_kbp37_label']


def score(cache, gold, prediction, labels):
    if (not gold or len(gold) != len(prediction) or len(set(labels)) != len(labels)
            or 'no_relation' not in labels or any(type(x) is not int or not 0 <= x < len(labels)
                                                 for x in gold+prediction)):
        raise ValueError('invalid scalar classification fixture')
    scorer,_ = load_scorer(cache)
    value = scorer()
    value(ScalarIds(prediction),ScalarIds(gold),[labels[i] for i in prediction],[labels[i] for i in gold])
    return value.get_metric(reset=False)


def common_oracle(gold, prediction, labels):
    """Independent count oracle for common-protocol metrics, no model inputs."""
    correct = Counter(labels[g] for g,p in zip(gold,prediction) if g == p)
    support = Counter(labels[g] for g in gold)
    guesses = Counter(labels[p] for p in prediction)
    numerator = sum(value for label,value in correct.items() if label != 'no_relation')
    positive_gold = sum(value for label,value in support.items() if label != 'no_relation')
    positive_guess = sum(value for label,value in guesses.items() if label != 'no_relation')
    micro = 2*numerator/(positive_gold+positive_guess) if numerator else 0.
    f1 = {name:2*correct[name]/(support[name]+guesses[name]) if correct[name] else 0. for name in labels}
    observed = (support.keys() | guesses.keys()) - {'no_relation'}
    return {'micro_f1':micro, 'macro_all_labels':sum(f1.values())/len(labels),
            'macro_observed_non_null':mean([f1[name] for name in observed])}


def run_gate(cache):
    checked_source(cache)
    scorer,normalize = load_scorer(cache)
    labels = ['no_relation','per:employee_of','per:schools_attended']
    checked = 0
    for length in (1,2,3):
        for pairs in product(product(range(3),repeat=2),repeat=length):
            gold,prediction = [pair[0] for pair in pairs],[pair[1] for pair in pairs]
            actual = score(cache,gold,prediction,labels)
            expected = common_oracle(gold,prediction,labels)
            if not math.isclose(actual['micro_fscore'],expected['micro_f1'],rel_tol=0,abs_tol=1e-12):
                raise AssertionError('source/common non-null micro-F1 differs')
            macro = expected['macro_observed_non_null']
            if not (math.isnan(macro) and math.isnan(actual['macro_fscore'])):
                if not math.isclose(actual['macro_fscore'],macro,rel_tol=0,abs_tol=1e-12):
                    raise AssertionError('source observed non-null macro-F1 differs from count oracle')
            checked += 1
    metric = scorer()
    metric(ScalarIds([1,0]),ScalarIds([1,2]),[labels[1],labels[0]],[labels[1],labels[2]])
    old = metric.get_metric(reset=True)
    if old['micro_fscore'] != 2/3 or metric.get_metric(reset=False)['micro_fscore'] != 0:
        raise AssertionError('source accumulation/reset behavior differs')
    if any(normalize(label) != label for label in labels):
        raise AssertionError('canonical relation fixture unexpectedly normalized')
    if normalize('per:employee_of(e2,e1)') != 'per:employee_of':
        raise AssertionError('source default KBP direction normalizer changed')
    labels40 = labels+['synthetic-unused-'+str(i) for i in range(37)]
    gold,prediction = [0,1,2],[0,1,0]
    example = {'gold':gold,'prediction':prediction,'labels':labels40,
               'label_scope':'three_named_relations_plus37_explicit_unused_synthetic_labels',
               'source':score(cache,gold,prediction,labels40),
               'common':common_oracle(gold,prediction,labels40)}
    all_null = score(cache,[0],[0],labels)
    if not math.isnan(all_null['macro_fscore']):
        raise AssertionError('source all-null macro was silently redefined')
    return {'schema':'luke-recipe-audit.v1','status':'metric_algebra_verified_recipe_pending',
            'source_revision':REVISION,'metric_sha256':METRIC_SHA256,
            'scalar_fixture_count':checked,'metric_scope':'source_AST_with_scalar_carriers_not_AllenNLP_runtime',
            'micro_convention':'all_examples_non_null_precision_recall',
            'source_macro_convention':'observed_gold_or_predicted_non_null_classes_only',
            'common_macro_convention':'all40_labels_including_no_relation_zero_division0',
            'all_null_source_macro':'undefined_NaN_not_coerced_to_zero',
            'macro_counterexample':example,'source_metric_reset_verified':True,
            'default_normalizer':'kbp37_removes_parenthesized_direction_suffixes',
            'normalization_adaptation':'must_preserve_canonical_relation_strings; no direction-merging rule introduced',
            'training_recipe_status':'separate_luke_training_recipe_audit_required',
            'pending':['run_separate_recipe_audit_not_certified_by_this_metric_gate',
                       'AllenNLP_tokenizer_indexer_loader_runtime_equivalence',
                       'paper_CoRE_CFIE_resample_focal_training_binding','full_weight_CUDA_numerical_fidelity'],
            'formal_comparison_admitted':False,'model_jobs':0,'backend_imported':False,'network_requests':0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-cache',type=Path,default=ROOT/'.external/retacred/luke-foundation-20261008')
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run_gate(args.source_cache),indent=2,allow_nan=False))
        return 2  # Source recipe/runtime admission remains explicitly open.
    except Exception as error:
        print(json.dumps({'status':'failed','error_type':type(error).__name__,'reason':str(error),'model_jobs':0}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
