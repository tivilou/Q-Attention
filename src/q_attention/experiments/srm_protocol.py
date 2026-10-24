"""Frozen bounded-only Q-SRM protocol. There is deliberately no full-data mode."""
from __future__ import annotations
import platform
import subprocess
from pathlib import Path
from q_attention.experiments.cfra_protocol import read,sha,identity,atomic_json,exclusive,epoch_order

ROOT=Path(__file__).resolve().parents[3]
MODES=('disabled','quantum','fidelity','diagonal','key_mismatch')
PAUSED_EXIT=85
SOURCE_FILES=(
    'src/q_attention/plugins/q_srm.py','src/q_attention/adapters/q_srm.py','src/q_attention/analysis/srm_trace.py',
    'src/q_attention/experiments/srm_protocol.py','src/q_attention/experiments/srm_runner.py',
    'src/q_attention/models/relation_transformer.py','src/q_attention/tasks/relation.py','src/q_attention/tasks/cfra_data.py',
    'src/q_attention/metrics.py','src/q_attention/tasks/ecpa_toy_data.py','src/q_attention/analysis/ecpa_trace.py','src/q_attention/analysis/ecpa_contract_validator.py',
    'src/q_attention/analysis/cfra_trace.py','src/q_attention/experiments/ecpa_protocol.py',
    'src/q_attention/experiments/cfra_protocol.py','src/q_attention/experiments/cfra_checkpoint.py',
    'experiments/run_srm_screen.py','scripts/run_q_srm_screen.sh','configs/q_srm_screen.json',
    'tests/test_srm.py','tests/test_srm_lifecycle.py')


def source_contract():
    names=set(SOURCE_FILES)
    for name in SOURCE_FILES:
        for parent in (ROOT/name).parents:
            if parent==ROOT:break
            init=parent/'__init__.py'
            if init.exists():names.add(str(init.relative_to(ROOT).as_posix()))
    return {name:sha(ROOT/name) for name in sorted(names)}


def config(profile):
    value=read(ROOT/'configs/q_srm_screen.json')
    if profile=='canary':
        value.update(profile='synthetic_canary',seeds=[13],expected_selected={'train':12,'valid':6,'test':6},source_hashes=None)
        value['model'].update(dim=16,num_heads=2,ff_dim=32)
        value['warmup'].update(epochs=2,batch_size=4)
        value['screen'].update(epochs=2,batch_size=4)
    elif profile!='screen':raise ValueError('only canary or bounded screen profiles are supported')
    if value['modes']!=list(MODES) or value['loss']!='factual_cross_entropy_only':raise ValueError('unfrozen modes/loss')
    return value


def environment(device):
    import numpy,torch
    return dict(python=platform.python_version(),torch=torch.__version__,numpy=numpy.__version__,
        cuda_build=torch.version.cuda,device=device,deterministic=True,torch_threads=1,
        gpu_name=torch.cuda.get_device_name(0) if device=='cuda' and torch.cuda.is_available() else None)


def validate_review():
    path=ROOT/'docs/validation/q_srm_review.json'
    if not path.exists():raise ValueError('missing exact-code mechanism/lifecycle review')
    review=read(path)
    if (review.get('review_scope')!='both' or review.get('verdict') not in ('PASS','PASS_WITH_RISK')
        or review.get('must_fix') or review.get('source_sha256')!=identity(source_contract())):
        raise ValueError('review scope/verdict/source differs from this implementation')
    return review


def load_data(directory,cfg):
    from q_attention.tasks import cfra_data
    from q_attention.tasks.relation import load_relation_jsonl,build_vocab,build_label_map,RelationDataset
    if cfg['profile']=='synthetic_canary':
        from q_attention.experiments.cfra_protocol import synthetic_data
        synthetic_data(directory)
        rows={s:load_relation_jsonl(Path(directory)/(s+'.jsonl')) for s in ('train','valid','test')}
        manifest={s:dict(source_sha256=sha(Path(directory)/(s+'.jsonl')),source_count=len(r),
            selected_count=len(r),selected_indexes=list(range(len(r)))) for s,r in rows.items()}
    else:
        for split,digest in cfg['source_hashes'].items():
            if sha(Path(directory)/(split+'.jsonl'))!=digest:raise ValueError('source data differs: '+split)
        rows,_,_,_,manifest=cfra_data.prepare(directory,cfg)
    if {s:len(r) for s,r in rows.items()}!=cfg['expected_selected']:raise ValueError('bounded population count differs')
    # Rebuild the common factual vocabulary; no unused neutralization tokens.
    vocab,labels=build_vocab(rows['train']),build_label_map(rows['train'])
    if any(r.label not in labels for group in rows.values() for r in group):raise ValueError('unseen held-out label')
    manifest['vocab_sha256'],manifest['labels_sha256']=identity(vocab),identity(labels)
    datasets={s:RelationDataset(r,vocab,labels) for s,r in rows.items()}
    return rows,datasets,vocab,labels,manifest


def update_indexes(count,batch_size,update,seed):
    import math
    epoch,index=divmod(update,math.ceil(count/batch_size))
    return epoch_order(count,seed,epoch)[index*batch_size:(index+1)*batch_size]


def revision():return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
