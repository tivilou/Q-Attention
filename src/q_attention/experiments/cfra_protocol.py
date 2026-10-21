"""Frozen seed13 contract; parent-side functions use only the standard library."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[3]
EXPERIMENT='retacred_q_cfra_formal_single_seed'
MODES=('disabled','quantum','classical_cosine','classical_rbf','diagonal','shuffled','entity_permuted')
PAUSED_EXIT=85

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def checkpoint_file(directory,entry):
    name=entry['file']
    if Path(name).name!=name or not name.endswith('.pt'):raise ValueError('unsafe checkpoint generation')
    path=Path(directory)/'checkpoints'/name
    if path.is_symlink() or not path.resolve().is_relative_to(Path(directory).resolve()):raise ValueError('unsafe checkpoint path')
    if sha(path)!=entry['sha256']:raise ValueError('checkpoint generation hash mismatch')
    return path

def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def syncdir(path):
    descriptor=os.open(Path(path),os.O_RDONLY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.'+path.name+'-',dir=path.parent)
    temporary=Path(name)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as stream:
            json.dump(value,stream,ensure_ascii=False,indent=2)
            stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path);syncdir(path.parent)
    finally:temporary.unlink(missing_ok=True)

@contextmanager
def exclusive(path):
    import fcntl
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise RuntimeError('another process owns '+str(path)) from exc
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)

def revision():
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()

def source_contract():
    paths=sorted((ROOT/'src/q_attention').rglob('*.py'))
    paths.extend(ROOT/name for name in (
        'experiments/run_cfra_formal.py','experiments/validate_cfra_handoff.py',
        'scripts/run_retacred_q_cfra_formal_single_seed.sh','scripts/check_retacred_q_cfra_formal_single_seed.sh',
        'scripts/collab/lib/exchange_upload.py','tests/test_cfra_formal.py',
        'configs/retacred_q_cfra_formal_single_seed.json','configs/q_cfra_trace.json'))
    return {str(path.relative_to(ROOT)):sha(path) for path in paths}

def load_config(canary=False):
    value=read(ROOT/'configs/retacred_q_cfra_formal_single_seed.json')
    if canary:
        value.update(profile='synthetic_canary',expected_records=dict(train=12,valid=6,test=6),expected_sha256=None)
        value['baseline'].update(epochs=2,batch_size=4)
        value['kernel'].update(epochs=2,batch_size=4)
        value['model'].update(dim=8,num_heads=2,ff_dim=16)
    if value['seed']!=13 or value['selectors']!=list(MODES):raise ValueError('unfrozen seed/selectors')
    return value

def inspect_data(directory,config):
    """Counts/hashes/shapes only; never fit a vocabulary on held-out records."""
    result={}
    for split,count in config['expected_records'].items():
        path=Path(directory)/(split+'.jsonl');supports=Counter();maximum=0;actual=0
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                if not line.strip():continue
                row=json.loads(line);n=len(row['tokens']);s,o=row['subject'],row['object']
                if not n or any(not 0<=a<b<=n for a,b in (s,o)) or max(s[0],o[0])<min(s[1],o[1]):
                    raise ValueError('invalid or overlapping entity spans')
                actual+=1;maximum=max(maximum,n);supports[str(row['label'])]+=1
        if actual!=count:raise ValueError(f'{split}: expected all{count} records, got{actual}')
        digest=sha(path)
        if config['expected_sha256'] and digest!=config['expected_sha256'][split]:
            raise ValueError('data source identity differs: '+split)
        result[split]=dict(file=path.name,records=actual,sha256=digest,max_length=maximum,supports=dict(sorted(supports.items())))
    if any(set(d['supports'])-set(result['train']['supports']) for d in result.values()):
        raise ValueError('held-out relation absent from train')
    return result

def load_data(directory,config):
    from q_attention.tasks.relation import load_relation_jsonl,build_vocab,build_label_map,RelationDataset
    manifest=inspect_data(directory,config)
    rows={split:load_relation_jsonl(Path(directory)/(split+'.jsonl')) for split in manifest}
    vocab,labels=build_vocab(rows['train']),build_label_map(rows['train'])
    for role,field in (('subject','subj_type'),('object','obj_type')):
        for kind in sorted({'UNKNOWN'}|{str(r.metadata.get(field,'UNKNOWN')) for r in rows['train']}):
            vocab[f'<cfra:{role}:{kind}>']=len(vocab)
    datasets={split:RelationDataset(items,vocab,labels) for split,items in rows.items()}
    return rows,datasets,vocab,labels,max(d['max_length'] for d in manifest.values()),manifest

def synthetic_data(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    for split,count in (('train',12),('valid',6),('test',6)):
        rows=[dict(tokens=[f'person_{split}_{i}','joined' if i%2 else 'left',f'org_{split}_{i}',*['today']*(1+i%3)],
            subject=[0,1],object=[2,3],label='member' if i%2 else 'former',
            metadata=dict(subj_type='PERSON',obj_type='ORG')) for i in range(count)]
        text=''.join(json.dumps(row)+'\n' for row in rows);path=directory/(split+'.jsonl')
        if path.exists() and path.read_text()!=text:raise ValueError('synthetic data changed')
        path.write_text(text)

def seed_all(seed):
    from q_attention.experiments.ecpa_protocol import seed_all as operation
    operation(seed)

def rng_state():
    from q_attention.experiments.ecpa_protocol import rng_state as operation
    return operation()

def restore_rng(value):
    from q_attention.experiments.ecpa_protocol import restore_rng as operation
    operation(value)

def metrics(*args):
    from q_attention.experiments.ecpa_protocol import metrics as operation
    return operation(*args)

def epoch_order(count,seed,epoch):
    import torch
    order=torch.randperm(count,generator=torch.Generator().manual_seed(seed+epoch*10007)).tolist()
    if epoch==0:order.remove(0);order.insert(0,0)
    return order

def update_indexes(count,logical,update,seed):
    import math
    epoch,position=divmod(update,math.ceil(count/logical))
    return epoch_order(count,seed,epoch)[position*logical:(position+1)*logical]

def validate_release():
    release=read(ROOT/'docs/validation/q_cfra_formal_release.json')
    if release['source_sha256']!=identity(source_contract()):raise ValueError('source differs from reviewed release')
    for name,digest in release['evidence_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('release evidence changed: '+name)
    return release
