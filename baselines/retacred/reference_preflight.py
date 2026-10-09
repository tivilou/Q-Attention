"""Read-only environment/disk/auth checks, before any formal model updates."""
from pathlib import Path
import argparse,json,os,ssl,shutil,importlib.metadata
from urllib.request import Request,urlopen
from urllib.parse import quote
from .common import ROOT,PACKAGE

def disk_requirement(names,seeds):
    # Full FP32 weights, Adam moments, atomic latest staging, dev best and
    # selected train/test numerical events. Curriculum retains three reviewers.
    gib=sum(60 if n=='curriculum' else 20 for n in names)*len(seeds)+12
    return gib*1024**3

def check(names,seeds,data_dir,gpu,*,network=True,review_receipt=None,resume=None):
    from .reference_complete import require_review
    from .orchestrate import gpu_ids
    if review_receipt is not None:require_review(review_receipt)
    for s in ('train','valid','test'):
        p=Path(data_dir)/(s+'.jsonl')
        if not p.is_file() or p.is_symlink():raise ValueError('missing/symlinked canonical data '+s)
    versions={}
    for n,v in [('transformers','4.44.0'),('huggingface-hub','0.36.2'),('numpy','1.26.4'),('ujson','5.11.0'),('spacy','3.7.5'),('click','8.1.8')]:
        versions[n]=importlib.metadata.version(n)
        if versions[n]!=v:raise ValueError('locked environment mismatch '+n+'; install requirements-reference-complete.txt')
    import torch
    if not torch.cuda.is_available():raise ValueError('CUDA GPU required')
    devices=gpu_ids(gpu)
    if not devices:raise ValueError('no selected GPUs')
    from .luke import require_runtime_memory
    require_runtime_memory()
    remaining=list(names)
    if resume:
        remaining=[n for n in names if any(not (Path(resume)/'tasks'/f'{n}_seed{s}'/'RUN_COMPLETE').exists() for s in seeds)]
    required=disk_requirement(remaining,seeds);free=shutil.disk_usage(ROOT).free
    if free<required:raise ValueError(f'insufficient free disk: require {required/1024**3:.0f}GiB conservative reserve, available {free/1024**3:.1f}GiB; preserve existing evidence')
    if network:
        from .assets import CA
        url=os.environ.get('PROJECT_EXCHANGE_URL','');token=os.environ.get('PROJECT_EXCHANGE_TOKEN','')
        if not url.startswith('https://') or not token:raise ValueError('set project HTTPS exchange URL and token before training')
        context=ssl.create_default_context(cafile=CA)
        with urlopen(url.rstrip('/')+'/healthz',context=context,timeout=20) as r:health=json.load(r)
        if health.get('status')!='ok' or health.get('service')!='q-attention-artifact-exchange-v1':raise ValueError('wrong exchange service')
        lock=json.loads((PACKAGE/'reference_resources.lock.json').read_text())
        rows=[lock['source_bundle']]
        if any(n in ('curriculum','luke_cfie') for n in names):rows.append(lock['parser_asset'])
        for v,n in [('sup','ctl_drp_supervised'),('unsup','ctl_drp_unsupervised')]:
            if n in names:rows.append(lock['ctl_weights'][v])
        for filename in ('assets.lock.json','luke_assets.lock.json','metaentail_assets.lock.json'):
            spec=json.loads((PACKAGE/filename).read_text())
            rows.extend(dict(row,namespace=spec['exchange_namespace']) for row in spec['files'])
        for row in rows:
            for part in row['parts']:
                request=Request(url.rstrip('/')+'/files/'+row['namespace']+'/'+quote(part['name']),method='HEAD',headers={'Authorization':'Bearer '+token})
                with urlopen(request,context=context,timeout=25) as response:
                    if int(response.headers['Content-Length'])!=part['size_bytes']:raise ValueError('locked resource transport size drift')
    return {'status':'passed','versions':versions,'devices':devices,'disk_reserve_bytes':required,'free_bytes':free,'network_authenticated':network,'formal_updates':0}

def main(argv=None):
    from .reference_complete import model_ids
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',default='all');p.add_argument('--seeds',default='13');p.add_argument('--gpu',default='auto')
    p.add_argument('--data-dir',type=Path,default=ROOT/'data/relation/retacred');p.add_argument('--resume',type=Path)
    p.add_argument('--review-receipt',type=Path,default=PACKAGE/'reference_review.json')
    a=p.parse_args(argv);print(json.dumps(check(model_ids(a.models),[int(s) for s in a.seeds.split(',')],a.data_dir,a.gpu,review_receipt=a.review_receipt,resume=a.resume),indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
