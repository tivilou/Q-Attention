"""MetaEntail paper-scope and explicit common-recipe guard, never a trainer."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

from .metaentail_checkpoint import checkpoint_spec, expanded_pair_budget

ROOT = Path(__file__).resolve().parents[2]
LOCK = Path(__file__).with_name('metaentail_recipe_evidence.lock.json')
PAPER_LOCK = Path(__file__).with_name('metaentail_paper.lock.json')
REQUIRED = ('provenance', 'provenance_note', 'epochs', 'pair_batch_size', 'accumulation_steps',
            'learning_rate', 'optimizer_backend', 'betas', 'epsilon', 'weight_decay',
            'scheduler', 'scheduler_total_steps', 'warmup_steps', 'max_pair_tokens',
            'overflow_policy', 'max_grad_norm', 'seed', 'checkpoint_selection', 'amp')


def parse_table6(text):
    text = re.sub(r'\s+', ' ', text)
    caption = re.search(r'Table 6:\s*Hyperparameters used to fine-tune BioLinkBERT\s*large', text)
    if not caption:
        raise ValueError('Table6 model scope not found')
    start = text.rfind('Parameter', 0, caption.start())
    if start < 0:
        raise ValueError('Table6 header not found')
    section = text[start:caption.start()]
    expected = {'Epochs': '3', 'Batch size': '32', 'Grad. accumulation steps': '1',
                'Max seq. length': '1,024', 'Learning rate': '2e-5',
                'Seeds': '{41, 42, 43, 44, 45}', 'Optimizer': 'AdamW',
                'LR Scheduler warm-up steps': '0', 'LR Scheduler training steps': '1,000'}
    for name, value in expected.items():
        if re.search(re.escape(name)+r'\s+'+re.escape(value)+r'(?:\s|$)', section) is None:
            raise ValueError('Table6 value not bound: '+name)
    if not re.search(r'A\.2\.4\s+Hyperparameters for', text):
        raise ValueError('generic appendix heading missing')
    return {'caption_scope': 'BioLinkBERTlarge', 'appendix_context': 'A.2.4 generic MetaEntail heading',
            'scope_status': 'general_domain_transfer_ambiguous_not_certified',
            'epochs': 3, 'batch_size': 32, 'accumulation_steps': 1, 'max_sequence_length': 1024,
            'learning_rate': 2e-5, 'seeds': [41, 42, 43, 44, 45], 'optimizer': 'AdamW',
            'warmup_steps': 0, 'scheduler_training_steps': 1000,
            'batch_unit_defined_in_table': False, 'scheduler_kind_defined_in_table': False,
            'optimizer_betas_epsilon_defined_in_table': False,
            'checkpoint_selection_defined_in_table': False}


def audit_paper(paper_cache, source_receipt, evidence_lock=LOCK, paper_lock=PAPER_LOCK):
    evidence = json.loads(Path(evidence_lock).read_text())
    paper = json.loads(Path(paper_lock).read_text())['paper']
    if (evidence['schema'] != 'metaentail-recipe-evidence.v1'
            or evidence['provenance_mode'] != 'paper'
            or evidence['fidelity_status'] != 'not_verifiable'
            or evidence['formal_comparison_admitted'] is not False
            or evidence['general_domain_recipe'] is not None
            or evidence['paper_sha256'] != paper['sha256']):
        raise ValueError('recipe evidence/admission contract differs')
    path = Path(paper_cache)/'paper.pdf'
    receipt_path = Path(source_receipt)
    if any(p.is_symlink() for target in (path, receipt_path) for p in (target, *target.parents)):
        raise ValueError('symlinked evidence rejected')
    raw = path.read_bytes()
    if len(raw) != paper['size_bytes'] or hashlib.sha256(raw).hexdigest() != paper['sha256']:
        raise ValueError('official PDF size/SHA differs')
    receipt_bytes = receipt_path.read_bytes()
    if hashlib.sha256(receipt_bytes).hexdigest() != evidence['author_repository_check']['receipt_sha256']:
        raise ValueError('source receipt SHA differs')
    receipt = json.loads(receipt_bytes)
    checks = [row for row in receipt['requests'] if row['url'] == evidence['author_repository_check']['url']]
    if len(checks) != 1 or checks[0]['status'] != 404:
        raise ValueError('locked public repository observation differs')
    done = subprocess.run(['pdftotext', '-layout', str(path), '-'], text=True, capture_output=True, timeout=30)
    if done.returncode:
        raise ValueError('native PDF text extraction failed')
    table = parse_table6(done.stdout)
    capacity = checkpoint_spec()['input']['max_pair_tokens_including_specials']
    return {'status': 'general_domain_author_recipe_unresolved', 'provenance_mode': 'paper',
            'fidelity_status': 'not_verifiable', 'paper_sha256': paper['sha256'],
            'source_observation': {'http_status': 404, 'url': checks[0]['url'], 'at': checks[0]['at'],
                'interpretation': 'public endpoint unavailable at observed time; deletion/private cause not inferred'},
            'paper_table6': table, 'checkpoint_pair_capacity': capacity,
            'length_conflict': table['max_sequence_length'] > capacity,
            'missing_general_domain_fields': list(REQUIRED),
            'formal_comparison_admitted': False, 'model_jobs': 0, 'backend_imported': False,
            'network_requests': 0, 'paper_score_reproduced': False}


def finite_number(value, name, *, positive=False):
    if type(value) not in (float, int) or not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError('invalid finite number: '+name)


def validate_declared_recipe(recipe, candidate_counts):
    """Validate an explicit hypothetical protocol; never mark it author-faithful."""
    if not isinstance(recipe, dict) or set(recipe) != set(REQUIRED):
        raise ValueError('all recipe choices required explicitly; unknown fields rejected')
    if recipe['provenance'] != 'declared_common_protocol' or not isinstance(recipe['provenance_note'], str) or not recipe['provenance_note'].strip():
        raise ValueError('explicit declared-common provenance required; no author-recipe claim')
    budget = expanded_pair_budget(candidate_counts, epochs=recipe['epochs'],
        pair_batch_size=recipe['pair_batch_size'], accumulation_steps=recipe['accumulation_steps'])
    for name in ('learning_rate', 'epsilon'):
        finite_number(recipe[name], name, positive=True)
    finite_number(recipe['weight_decay'], 'weight_decay')
    betas = recipe['betas']
    if not isinstance(betas, list) or len(betas) != 2:
        raise ValueError('two explicit AdamW betas required')
    for value in betas:
        finite_number(value, 'beta')
        if value >= 1:
            raise ValueError('AdamW beta must be below1')
    if recipe['optimizer_backend'] != 'torch_adamw' or recipe['scheduler'] != 'linear':
        raise ValueError('only explicitly proposed torch_adamw/linear path supported')
    if recipe['scheduler_total_steps'] != 'expanded_pair_budget':
        raise ValueError('fixed scheduler training steps not silently imported from Table6')
    warmup = recipe['warmup_steps']
    if type(warmup) is not int or not 0 <= warmup < budget['total_optimizer_updates']:
        raise ValueError('explicit warmup must fit actual expanded update budget')
    limit = recipe['max_pair_tokens']
    capacity = checkpoint_spec()['input']['max_pair_tokens_including_specials']
    if type(limit) is not int or not 4 < limit <= capacity:
        raise ValueError('explicit pair length exceeds locked checkpoint capacity')
    if recipe['overflow_policy'] != 'reject_with_length_diagnostic_no_truncation_or_drop':
        raise ValueError('overflow adaptation must be explicit; no silent drop or truncate')
    if recipe['max_grad_norm'] is not None:
        finite_number(recipe['max_grad_norm'], 'max_grad_norm', positive=True)
    if type(recipe['seed']) is not int or recipe['seed'] < 0 or type(recipe['amp']) is not bool:
        raise ValueError('explicit nonnegative seed and boolean AMP required')
    if recipe['checkpoint_selection'] != 'dev_all_example_non_null_micro_f1':
        raise ValueError('explicit dev-only common checkpoint selection required')
    frozen = json.dumps(recipe, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return {'status': 'declared_common_recipe_neural_and_paper_equivalence_pending',
        'fidelity_status': 'paper_interpretive', 'budget': budget,
        'recipe_sha256': hashlib.sha256(frozen.encode()).hexdigest(),
        'paper_general_domain_recipe_recovered': False, 'formal_comparison_admitted': False,
        'model_jobs': 0, 'actual_optimizer_or_scheduler_executed': False,
        'example_length_and_tensor_capture_coverage_verified': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper-cache', type=Path, default=ROOT/'.external/retacred/metaentail-paper-recovery-20261008A')
    parser.add_argument('--source-receipt', type=Path, default=ROOT/'.external/retacred/metaentail-recipe-recovery-20261008G/collection.json')
    parser.add_argument('--recipe-file', type=Path)
    parser.add_argument('--candidate-counts', type=Path)
    args = parser.parse_args(argv)
    try:
        result = audit_paper(args.paper_cache, args.source_receipt)
        if (args.recipe_file is None) != (args.candidate_counts is None):
            raise ValueError('recipe-file and explicit candidate-counts must be supplied together')
        if args.recipe_file is not None:
            result['declared_common_recipe'] = validate_declared_recipe(
                json.loads(args.recipe_file.read_text()), json.loads(args.candidate_counts.read_text()))
        print(json.dumps(result, indent=2, allow_nan=False))
        return 2
    except Exception as error:
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__, 'reason': str(error), 'model_jobs': 0}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
