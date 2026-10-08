"""Scalar evaluator contracts, explicitly not neural or AllenNLP certification."""
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT/'baselines/retacred/luke_recipe_audit.py'
SPEC = importlib.util.spec_from_file_location('luke_recipe_audit_fixture',PATH)
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)
CACHE = Path(os.environ.get('LUKE_AUDIT_SOURCE_CACHE',ROOT/'.external/retacred/luke-foundation-20261008'))


class RecipeTests(unittest.TestCase):
    def test_exact_source_checksum(self):
        path,data = r.checked_source(CACHE)
        self.assertEqual(len(data),3752)
        self.assertEqual(path.name,r.METRIC_FILE)

    def test_exhaustive_count_and_scope(self):
        gate = r.run_gate(CACHE)
        self.assertEqual(gate['scalar_fixture_count'],819)
        self.assertFalse(gate['formal_comparison_admitted'])
        self.assertFalse(gate['backend_imported'])
        self.assertEqual(gate['model_jobs'],0)
        self.assertTrue(gate['source_metric_reset_verified'])
        json.dumps(gate,allow_nan=False)

    def test_source_and_common_macro_are_not_same_metric(self):
        gate = r.run_gate(CACHE)
        values = gate['macro_counterexample']
        self.assertAlmostEqual(values['source']['micro_fscore'],2/3)
        self.assertEqual(values['source']['macro_fscore'],0.5)
        self.assertAlmostEqual(values['common']['macro_all_labels'],1/24)

    def test_all_null_source_macro_is_undefined(self):
        actual = r.score(CACHE,[0],[0],['no_relation','relation'])
        self.assertTrue(math.isnan(actual['macro_fscore']))
        self.assertEqual(actual['micro_fscore'],0)

    def test_false_positive_on_null_relation_counts(self):
        actual = r.score(CACHE,[0,1],[1,1],['no_relation','relation'])
        self.assertAlmostEqual(actual['micro_fscore'],2/3)

    def test_reset_and_default_normalization(self):
        scorer,normalizer = r.load_scorer(CACHE)
        self.assertEqual(normalizer('relation(e1,e2)'),'relation')
        metric = scorer()
        metric(r.ScalarIds([1]),r.ScalarIds([1]),['relation'],['relation'])
        self.assertEqual(metric.get_metric(reset=True)['micro_fscore'],1)
        self.assertEqual(metric.get_metric(reset=False)['micro_fscore'],0)

    def test_tampered_source_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            (cache/r.METRIC_FILE).write_bytes(b'raise AssertionError("must not execute")\n')
            with self.assertRaisesRegex(ValueError,'SHA mismatch'):
                r.load_scorer(cache)

    def test_invalid_fixtures_rejected(self):
        for gold,prediction,labels in (([],[],['no_relation']),([0],[0,1],['no_relation','A']),
                                      ([2],[0],['no_relation','A']),([0],[0],['A']),
                                      ([True],[0],['no_relation','A']),([0],[0],['no_relation','no_relation'])):
            with self.subTest(gold=gold,prediction=prediction,labels=labels):
                with self.assertRaises(ValueError):
                    r.score(CACHE,gold,prediction,labels)

    def test_cli_read_only_no_backend_or_network(self):
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
        done = subprocess.run([sys.executable,'-c',script,str(PATH),str(CACHE)],
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),text=True,capture_output=True,timeout=25)
        self.assertEqual(done.returncode,2,done.stderr)
        self.assertEqual(json.loads(done.stdout)['scalar_fixture_count'],819)


if __name__ == '__main__':
    unittest.main()
