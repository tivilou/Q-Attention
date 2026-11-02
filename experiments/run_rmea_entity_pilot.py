"""Pinned owner train2048/valid1024 cached-feature learning diagnostic; no test."""
import argparse
import hashlib
import heapq
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baselines.retacred import rmea_entity_study_protocol as p


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--cache-sha256',required=True)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    p.review_gate()
    if p.git('status','--porcelain'):raise ValueError('Clean source required')
    if p.digest(args.cache)!=args.cache_sha256:raise ValueError('Pinned old feature hash differs')
    output=args.output.resolve()
    if output.exists() or output.is_relative_to(p.ROOT):raise ValueError('Fresh private output required')
    import numpy as np
    from baselines.retacred.rmea_entity_study import train
    cfg=p.config('pilot');metadata={};labels=p.read(p.ROOT/'docs/contracts/q_rmea_entity_labels.json')
    output.mkdir(parents=True);(output/'cache').mkdir()
    for split,n in cfg['counts'].items():
        if p.digest(args.data_dir/f'{split}.jsonl')!=p.EXPECTED[split][1]:raise ValueError('Train/valid dataset differs')
        def rows():
            with (args.data_dir/f'{split}.jsonl').open(encoding='utf8') as f:
                for line in f:
                    row=json.loads(line)
                    rank=hashlib.sha256(('rmea-readiness-v1:'+split+':'+str(row['metadata']['id'])).encode()).hexdigest()
                    yield rank,row
        selected=[row for _,row in heapq.nsmallest(n,rows(),key=lambda pair:pair[0])]
        with np.load(args.cache,allow_pickle=False) as old:
            ids=old[split+'_ids'].astype(str).tolist()
            expected=[hashlib.sha256(str(row['metadata']['id']).encode()).hexdigest() for row in selected]
            if ids!=expected:raise ValueError('Historical cache sample selection differs')
            targets=old[split+'_targets']
            if targets.tolist()!=[labels.index(row['label']) for row in selected]:raise ValueError('Historical target mapping differs')
            for oldname,newname in [('features','h'),('roles','roles'),('offsets','offsets'),('targets','targets')]:
                np.save(output/'cache'/f'{split}-{newname}.npy',old[split+'_'+oldname])
        metadata[split]=[dict(id_hash=sid,words=len(row['tokens'])) for sid,row in zip(ids,selected)]
    p.atomic_json(output/'cache/metadata.json',metadata)
    p.atomic_json(output/'manifest.json',dict(schema='q-rmea.entity.run.v1',config=cfg,source_sha256=p.identity(p.sources()),git_revision=p.git('rev-parse','HEAD'),
        data={s:dict(source_sha256=p.EXPECTED[s][1],source_count=p.EXPECTED[s][0],selected_count=n) for s,n in cfg['counts'].items()},
        environment=dict(device='CPU',cache_sha256=args.cache_sha256,test_loaded=False)))
    train(cfg,output,labels,device='cpu',trace=False)
    p.seal(output)
    print(json.dumps(dict(pilot_complete=True,test_loaded=False,formal_runs=0)))


if __name__=='__main__':main()
