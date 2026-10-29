"""Separately frozen five-fresh-seed confirmation; no modification of seed13."""
from pathlib import Path
import math
import json
from . import rmea_protocol as legacy
from .rmea_protocol import MODES,EXPECTED,identity,read,git,lock,inventory,seal
from .common import atomic_json,digest

ROOT=Path(__file__).resolve().parents[2]
SEEDS=(29,53,71,97,113)
NEW_FILES=('baselines/retacred/rmea_confirmation_protocol.py',
 'baselines/retacred/rmea_confirmation_run.py','baselines/retacred/rmea_confirmation_trace.py',
 'baselines/retacred/rmea_confirmation_values.py','baselines/retacred/rmea_confirmation_handoff.py',
 'baselines/retacred/rmea_confirmation_exchange.py','experiments/q_rmea_confirmation_handoff.py',
 'scripts/run_q_rmea_confirmation_handoff.sh','scripts/check_q_rmea_confirmation_handoff.sh',
 'scripts/export_q_rmea_confirmation_handoff.sh','scripts/validate_sample_trace_v2.py',
 'scripts/view_rmea_confirmation_trace.py','docs/contracts/sample-trace.v2.schema.json',
 'tests/test_rmea_confirmation.py')
FILES=legacy.FILES+NEW_FILES

def sources():
    if any((ROOT/n).is_symlink() for n in FILES):raise ValueError('Symlinked source')
    return {n:digest(ROOT/n) for n in FILES}

def contract():
    return dict(schema='q-rmea.confirmation.v1',seeds=list(SEEDS),discovery_seed13_excluded=True,
      scientific_reference='7ca1b015c7900ad45f7968d0c1e56db5c7b3300809c2bd7de496f166a5b0615c',
      modes=list(MODES),seed_configs=[config('discovery',s) for s in SEEDS],
      cache_policy='Independent identical ordered extraction per seed; no new cache sharing or precision change.',
      utility_gate='Q-minus-uniform test full40 Macro-F1: mean>0, at least4/5 positive, paired-t two-sided95% CI lower>0.',
      mechanism_gate='Q fixed-selected checkpoint all-test mean CE(shuffled)-CE(actual): mean>0, at least4/5 positive, paired-t two-sided95% CI lower>0; every seed uniform CE delta abs<=2e-5 and identity residual<=2e-5.',
      statistics=dict(unit='fresh paired seed',n=5,t_critical_df4=2.7764451051977987,
        exact_sign_flip_two_sided_min_p=.0625,distribution_assumption='Seed deltas treated as independent approximately normal; n=5 is limited.',
        reporting='All5 seeds and5 conditions, failures included in status; no successful-seed dropping, rescue tuning or optional stopping. Exact sign-flip p descriptive, never claim p<.05 from five pairs.'),
      controls='Three classical/dephased contrasts descriptive attribution; unequal parameter counts; no quantum uniqueness or novelty conclusion.',
      trace=dict(schema='sample-trace.v2',selection='first2 SHA256(rmea-sample-v1:split:original_id) ordered samples per train/valid/test',
        lanes=['replay'],checkpoints=['initial','selected'],all_modes=True,
        numeric='Complete detached head input/output, selected encoder output, tokenization, classical scorer states and quantum circuit states; private loopback arbitrary-coordinate reader.',
        boundaries='Inference replay only; no per-sample training trajectory or internal encoder-layer history claim. First-update batch parameters/gradients remain separate private evidence.'),
      lifecycle='One isolated environment/GPU preflight; one bounded canary seed29 then all5 full children serially. No partial training resume. Export-only retries use all immutable closed phases, no training.',
      report_files=['summary.json','manifest.json','trace_validation.json','receipt.json','run_summary.json','data.sha256','data_counts.txt'],
      claim='Pending owner audit; no automatic L2/L3, novelty, quantum advantage or checkpoint-resume certification.')

def config(profile,seed):
    if seed not in SEEDS:raise ValueError('Fresh confirmation seed required')
    cfg=legacy.config(profile)
    cfg.update(schema='q-rmea.protocol.confirmation.v1',seed=seed,
       replication='Frozen independent confirmation29/53/71/97/113; discovery13 excluded.',trace_schema='sample-trace.v2')
    return cfg

def review_gate():
    r=read(ROOT/'docs/validation/q_rmea_confirmation_review.json')
    if r.get('review_scope')!='both' or r.get('must_fix') or r.get('source_sha256')!=identity(sources()) or r.get('contract_sha256')!=identity(contract()) or r.get('verdict')!='PASS':raise ValueError('Exact-source scope-both confirmation review PASS required')
    return r

def audit(root):
    root=Path(root)
    if root.is_symlink() or not (root/'RUN_COMPLETE').is_file():raise ValueError('Run incomplete')
    if inventory(root)!=read(root/'RUN_COMPLETE')['artifacts']:raise ValueError('Closed artifact inventory differs')
    m=read(root/'manifest.json');cfg=m['config'];s=read(root/'summary.json')
    if m['source_sha256']!=identity(sources()) or cfg!=config(cfg['profile'],cfg['seed']):raise ValueError('Wrong source/config')
    if set(s['results'])!=set(MODES) or s['common_initialization_equal'] is not True or s['seed']!=cfg['seed']:raise ValueError('Missing/mismatched seed conditions')
    t=read(root/'trace_validation.json')
    if t.get('schema')!='q-rmea.trace-readiness.v2' or any(t.get(k)!='pass' for k in ('structural','semantic','replay','coordinate_reader')):raise ValueError('Trace v2 readiness incomplete')
    return m,s

def paired(values):
    from itertools import product
    values=[float(v) for v in values]
    if len(values)!=5 or not all(math.isfinite(v) for v in values):raise ValueError('Exactly5 finite paired seed deltas required')
    mean=sum(values)/5;sd=math.sqrt(sum((v-mean)**2 for v in values)/4);half=2.7764451051977987*sd/math.sqrt(5)
    observed=abs(sum(values));extreme=sum(abs(sum(v*s for v,s in zip(values,signs)))>=observed-1e-14 for signs in product((-1,1),repeat=5))
    positive=sum(v>0 for v in values)
    return dict(deltas=values,mean=mean,sd=sd,ci95=[mean-half,mean+half],positive_seeds=positive,
      paired_t_gate=mean>0 and positive>=4 and mean-half>0,exact_two_sided_sign_flip_p=extreme/32,n=5)
