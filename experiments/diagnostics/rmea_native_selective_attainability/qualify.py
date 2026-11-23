"""TRAIN-only input geometry coverage; no model, outcomes or raw-text output."""
from pathlib import Path
import hashlib,json
import numpy as np


def regions(length,subject,object_):
    s0,s1=map(int,subject);o0,o1=map(int,object_)
    assert 0<=s0<s1<=length and 0<=o0<o1<=length
    entity=np.zeros(length,bool);entity[s0:s1]=True;entity[o0:o1]=True
    bridge=np.zeros(length,bool)
    if s1<=o0:bridge[s1:o0]=True
    elif o1<=s0:bridge[o1:s0]=True
    outer=~entity&~bridge
    eligible=bool(bridge.any() and outer.any() and not (s0<o1 and o0<s1))
    assert not np.any(bridge&outer) and not np.any((bridge|outer)&entity)
    assert np.array_equal(bridge|outer,~entity)
    return bridge,outer,eligible


def main():
    root=Path('/hy-tmp/Q-Attention/inputs/native-word-metadata-20261123')
    cache=Path('/hy-tmp/Q-Attention/inputs/native-baseline/discovery/cache')
    path=root/'train.private.json';assert hashlib.sha256(path.read_bytes()).hexdigest()=='f5eafb7d51bca46988313c204b6fa5ef19e82fabc32dc9feee9faa70f2285053'
    rows=json.loads(path.read_bytes());assert len(rows)==8192
    data={k:np.load(cache/('train-'+k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ('roles','offsets','targets')}
    counts=np.zeros((40,2),np.int64);sizes=[];eligible_ids=[];roles_match=True
    for i,row in enumerate(rows):
        n=len(row['tokens']);lo,hi=map(int,data['offsets'][i:i+2]);assert hi-lo==n
        b,o,e=regions(n,row['subject'],row['object']);r=np.asarray(data['roles'][lo:hi])
        expected=[]
        for span,col in ((row['subject'],2),(row['object'],3)):
            left,right=map(int,span);m=np.zeros(n,bool);m[left:right]=True
            assert np.array_equal(m,r[:,col].astype(bool));expected.append(m)
        counts[int(data['targets'][i]),int(e)]+=1
        sizes.append((int(b.sum()),int(o.sum())))
        if e:eligible_ids.append(i)
    eligible=int(counts[:,1].sum());positive=int(counts[1:,1].sum());classes=int((counts[1:,1]>=16).sum())
    gates=dict(eligible_rows=eligible>=2048,positive_rows=positive>=1000,positive_classes=classes>=10)
    print(json.dumps(dict(status='PASS_SUPPORT_ONLY' if all(gates.values()) else 'FAIL_CLOSE_GEOMETRY',gates=gates,
        train_count=len(rows),eligible_rows=eligible,eligible_positive=positive,positive_classes_ge16=classes,
        per_class_ineligible_eligible=counts.tolist(),region_size_median=np.median(np.asarray(sizes),axis=0).tolist(),
        train_word_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        eligible_ids_sha256=hashlib.sha256(np.array(eligible_ids,dtype='<i8').tobytes()).hexdigest(),
        all_span_membership_matches_cache=True,model_calls=0,optimizer_updates=0,valid_test_used=False)))


if __name__=='__main__':main()
