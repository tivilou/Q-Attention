
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,time
t=time.monotonic()
p=Path('/hy-tmp/Q-Attention/inputs/native-baseline/recovery-inputs/train.jsonl')
raw=p.read_bytes()
sha=hashlib.sha256(raw).hexdigest()
assert sha=='f1051d574a44df3ecbd036c1de4fbbe0a372c768f6ec990330e2cea4a657269f'
fields=Counter(); types=defaultdict(Counter); meta=Counter(); meta_types=defaultdict(Counter)
arrays=Counter(); structures=Counter(); rows=0
for line in raw.splitlines():
    if not line.strip():continue
    r=json.loads(line);rows+=1
    for k,v in r.items():
        fields[k]+=1;types[k][type(v).__name__]+=1
        if isinstance(v,list):arrays[k]+=1
        if isinstance(v,dict):
            for kk,vv in v.items():
                structures[k+'.'+kk]+=1
                if k=='metadata':
                    meta[kk]+=1;meta_types[kk][type(vv).__name__]+=1
    assert time.monotonic()-t<30,'CPU metadata cap exhausted'
assert rows==58465
print(json.dumps({'status':'PASS_EXACT_CANONICAL_TRAIN_TOP_LEVEL_AND_ONE_NESTED_LEVEL',
 'source_sha256':sha,'rows':rows,'top_level_fields':dict(sorted(fields.items())),
 'top_level_types':dict(sorted(types.items())),'top_level_array_fields':dict(sorted(arrays.items())),
 'nested_field_coverage':dict(sorted(structures.items())),
 'metadata_field_types':dict(sorted(meta_types.items())),
 'new_fits':0,'model_forwards':0,'gpu_initializations':0,'valid_test_read':False,
 'remote_file_writes':0,'seconds':time.monotonic()-t},sort_keys=True))
