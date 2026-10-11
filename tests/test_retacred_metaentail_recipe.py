"""Explicit recipe and paper-scope tests; zero neural/final experiments."""
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import unittest

from baselines.retacred import metaentail_recipe as r


def proposed_recipe():
    return {'provenance': 'declared_common_protocol', 'provenance_note': 'synthetic validator fixture only',
            'epochs': 3, 'pair_batch_size': 4, 'accumulation_steps': 2, 'learning_rate': 2e-5,
            'optimizer_backend': 'torch_adamw', 'betas': [0.9, 0.999], 'epsilon': 1e-8,
            'weight_decay': 0.01, 'scheduler': 'linear', 'scheduler_total_steps': 'expanded_pair_budget',
            'warmup_steps': 0, 'max_pair_tokens': 512,
            'overflow_policy': 'reject_with_length_diagnostic_no_truncation_or_drop',
            'max_grad_norm': None, 'seed': 13,
            'checkpoint_selection': 'dev_all_example_non_null_micro_f1', 'amp': False}


class MetaEntailRecipeTests(unittest.TestCase):
    def test_actual_pdf_and_receipt_scope(self):
        gate = r.audit_paper(r.ROOT/'.external/retacred/metaentail-paper-recovery-20261008A',
            r.ROOT/'.external/retacred/metaentail-recipe-recovery-20261008G/collection.json')
        self.assertEqual(gate['source_observation']['http_status'], 404)
        self.assertEqual(gate['paper_table6']['caption_scope'], 'BioLinkBERTlarge')
        self.assertEqual(gate['paper_table6']['scope_status'], 'general_domain_transfer_ambiguous_not_certified')
        self.assertEqual(gate['paper_table6']['max_sequence_length'], 1024)
        self.assertEqual(gate['checkpoint_pair_capacity'], 512)
        self.assertTrue(gate['length_conflict'])
        self.assertFalse(gate['formal_comparison_admitted'])

    def test_native_table_reader_rejects_unbound_caption(self):
        with self.assertRaisesRegex(ValueError, 'scope'):
            r.parse_table6('Table 6: Hyperparameters used to fine-tune RoBERTa large')

    def test_explicit_fixture_keeps_pair_tail_and_closed_admission(self):
        gate = r.validate_declared_recipe(proposed_recipe(), [40, 1])
        budget = gate['budget']
        self.assertEqual(budget['pairs_per_epoch'], 41)
        self.assertEqual(budget['mini_batches_per_epoch'], 11)
        self.assertEqual(budget['optimizer_updates_per_epoch'], 6)
        self.assertEqual(budget['total_optimizer_updates'], 18)
        self.assertEqual(budget['last_update_pairs'], 1)
        self.assertEqual(budget['loss_denominator'], 'actual_pairs_in_optimizer_window')
        self.assertEqual(gate['fidelity_status'], 'paper_interpretive')
        self.assertFalse(gate['paper_general_domain_recipe_recovered'])
        self.assertFalse(gate['formal_comparison_admitted'])

    def test_every_choice_required_no_defaults(self):
        for field in r.REQUIRED:
            recipe = proposed_recipe()
            del recipe[field]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'explicitly'):
                r.validate_declared_recipe(recipe, [1])

    def test_no_author_recipe_or_unknown_fields(self):
        recipe = proposed_recipe()
        recipe['provenance'] = 'author_recipe'
        with self.assertRaisesRegex(ValueError, 'provenance'):
            r.validate_declared_recipe(recipe, [1])
        recipe = proposed_recipe()
        recipe['hidden_default'] = 1024
        with self.assertRaisesRegex(ValueError, 'unknown'):
            r.validate_declared_recipe(recipe, [1])

    def test_table6_length_and_fixed_steps_rejected(self):
        for field, value in (('max_pair_tokens', 1024), ('scheduler_total_steps', 1000)):
            recipe = proposed_recipe()
            recipe[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                r.validate_declared_recipe(recipe, [40, 1])

    def test_nonfinite_lr_and_implicit_bool_rejected(self):
        for field, value in (('learning_rate', math.nan), ('learning_rate', math.inf),
                             ('epsilon', 0), ('weight_decay', -1), ('learning_rate', True)):
            recipe = proposed_recipe()
            recipe[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                r.validate_declared_recipe(recipe, [1])

    def test_betas_clipping_and_rng_choices(self):
        for field, value in (('betas', [0.9, 1]), ('betas', [True, 0.999]), ('betas', [0.9]),
                             ('max_grad_norm', 0), ('seed', True), ('amp', 'auto')):
            recipe = proposed_recipe()
            recipe[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                r.validate_declared_recipe(recipe, [1])

    def test_warmup_over_budget_rejected(self):
        recipe = proposed_recipe()
        recipe['warmup_steps'] = 3
        with self.assertRaisesRegex(ValueError, 'warmup'):
            r.validate_declared_recipe(recipe, [1])

    def test_implicit_overflow_and_test_selection_rejected(self):
        for field, value in (('overflow_policy', 'drop'), ('checkpoint_selection', 'test_micro_f1'),
                             ('scheduler', 'cosine'), ('optimizer_backend', 'default_adamw')):
            recipe = proposed_recipe()
            recipe[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                r.validate_declared_recipe(recipe, [1])

    def test_count_budget_uses_existing_validator(self):
        for counts in ([0], [41], [], [True]):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                r.validate_declared_recipe(proposed_recipe(), counts)

    def test_fingerprint_inputs_unchanged(self):
        recipe = proposed_recipe()
        original = copy.deepcopy(recipe)
        a = r.validate_declared_recipe(recipe, [1, 2])
        b = r.validate_declared_recipe(recipe, [1, 2])
        self.assertEqual(recipe, original)
        self.assertEqual(a['recipe_sha256'], b['recipe_sha256'])

    def test_actual_namespace_cli_offline_guard(self):
        script = '''import runpy,sys
def guard(event,args):
 if event=='import' and args[0].split('.')[0] in ('torch','numpy','allennlp','tensorflow','jax'): raise AssertionError('backend import')
 if event.startswith('socket.'): raise AssertionError('network')
sys.addaudithook(guard)
runpy.run_module('baselines.retacred.metaentail_recipe',run_name='__main__')
'''
        done = subprocess.run([sys.executable, '-c', script], cwd=r.ROOT,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), text=True, capture_output=True, timeout=30)
        self.assertEqual(done.returncode, 2, done.stderr)
        self.assertEqual(json.loads(done.stdout)['status'], 'general_domain_author_recipe_unresolved')


if __name__ == '__main__':
    unittest.main()
