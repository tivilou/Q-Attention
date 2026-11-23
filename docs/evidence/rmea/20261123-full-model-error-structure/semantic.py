"""Reuse the accepted text-reader transport on a distinct complete-model error cohort."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

r=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('existing_semantic_reader',r.parent/'semantic-problem-audit-20261123/audit.py')
reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
reader.ROOT=r/'semantic';reader.HOST='/hy-tmp/Q-Attention/control/full-model-error-structure-20261123/semantic'
reader.ROOT.mkdir(exist_ok=True)
SELECT=r'''
from pathlib import Path
from collections import defaultdict,Counter
import hashlib,json,sys
import numpy as np
cfg=json.load(sys.stdin);r=Path('/hy-tmp/Q-Attention');out=Path(cfg['host']);assert not out.exists(),'Reuse sealed selection'
sha=lambda x:hashlib.sha256(x).hexdigest()
old=r/'control/historical-type-replay-20261123/output';cache=r/'inputs/native-baseline/discovery/cache'
labels=json.loads((r/'inputs/native-baseline/recovery-inputs/labels.json').read_bytes())
words=json.loads((r/'inputs/native-word-metadata-20261123/valid.private.json').read_bytes())
assert sha((r/'inputs/native-word-metadata-20261123/valid.private.json').read_bytes())=='59822bfc80e6a590e2a50f9e0f4c569a3be8945b303da0d319966a6e3c62a06c'
vm=json.loads((r/'control/task-bottleneck-discovery-20261123/validation-cases.private.json').read_bytes())['metadata']
tm=json.loads((r/'control/task-bottleneck-discovery-20261123/cases.private.json').read_bytes())['type_metadata']
y=np.load(cache/'valid-targets.npy',allow_pickle=False);assert len(y)==len(words)==len(vm)==19584
with np.load(r/'control/native-ordered-mass-value-20261123/output/diagnostic.private.npz',allow_pickle=False) as f:fit=f['calibration'].copy()
support=defaultdict(set);seen=set()
for i in np.flatnonzero(fit):g,st,ot=tm[i];support[st,ot].add(g);seen.add(g)
support_sha=sha(json.dumps(sorted((list(k),sorted(v)) for k,v in support.items()),separators=(',',':')).encode())
assert support_sha=='d4c1100df6d306a3fae48d66f7907dfbe2a39d1b30843abb359ba32de7053186'
allow=np.ones((len(y),40),bool)
for i,(g,st,ot) in enumerate(vm):
 assert g==y[i]
 if (st,ot) in support:
  allow[i]=False;allow[i,0]=True
  for k in support[st,ot]|(set(range(40))-seen):allow[i,k]=True
modes=['relation_softmax','real_overlap','quantum','dephased'];zs={m:np.load(old/(m+'-logits.private.npy'),allow_pickle=False) for m in modes}
pred={m:np.where(allow,z,-np.inf).argmax(1) for m,z in zs.items()};q=zs['quantum'];qp=pred['quantum']
posallow=allow.copy();posallow[:,0]=False;top={m:np.where(posallow,z,-np.inf).argmax(1) for m,z in zs.items()}
common=(y>0)&np.logical_and.reduce([pred[m]==0 for m in modes]);legal=allow[np.arange(len(y)),y]
target=common&legal&(top['quantum']==y);assert common.sum()==1155 and target.sum()==815
detail={}
for m,z in zs.items():
 ids=np.flatnonzero(common&legal);gz=z[ids,y[ids]].astype(float);nz=z[ids,0].astype(float)
 detail[m]=dict(eligible_common_fn=len(ids),gold_top_positive=int((top[m][ids]==y[ids]).sum()),gold_minus_null_quantiles=np.quantile(gz-nz,[0,.25,.5,.75,1]).tolist())
detail['all_four_gold_top_positive']=int((common&legal&np.logical_and.reduce([top[m]==y for m in modes])).sum())
population={'positive_missed':np.flatnonzero(target),'positive_correct':np.flatnonzero((y>0)&(qp==y)&legal),
 'null_false_positive':np.flatnonzero((y==0)&(qp>0)),'null_correct':np.flatnonzero((y==0)&(qp==0))}
chosen=[];used=set()
for family in cfg['stratum_order']:
 candidates=[]
 for i in population[family]:
  w=words[int(i)];key=sha(json.dumps([w['tokens'],w['subject'],w['object']],ensure_ascii=False,separators=(',',':')).encode());sentence=sha(json.dumps(w['tokens'],ensure_ascii=False,separators=(',',':')).encode())
  candidates.append((sha(('full-model-semantic-v1:'+key).encode()),key,sentence,int(i)))
 take=[]
 for _,key,sentence,i in sorted(candidates):
  if sentence in used:continue
  take.append(dict(family=family,input_sha256=key,sentence_sha256=sentence,row=i,case_id=sha(('full-case:'+key).encode())[:24]));used.add(sentence)
  if len(take)==8:break
 assert len(take)==8,family
 chosen.extend(take)
seal=sha(json.dumps(chosen,sort_keys=True,separators=(',',':')).encode());mapping=[];packet=[]
for rec in chosen:
 i=rec['row'];w=words[i];g,st,ot=vm[i]
 assert 0<=w['subject'][0]<w['subject'][1]<=len(w['tokens']) and 0<=w['object'][0]<w['object'][1]<=len(w['tokens'])
 mapping.append(dict(**rec,gold=labels[g],predicted=labels[int(qp[i])],gold_allowed=bool(legal[i])))
 packet.append(dict(case_id=rec['case_id'],tokens=w['tokens'],subject=w['subject'],object=w['object'],subject_type=st,object_type=ot))
out.mkdir(parents=True)
report=dict(selection_sha256=seal,selected=32,per_stratum=8,distinct_sentences=len(used),population={k:len(v) for k,v in population.items()},
 scope='Outcome-stratified descriptive sample of reused VALID from original complete models on current cache. No population prevalence, independent validation, tuning, new hypothesis test or quantum attribution.',
 model_forwards=0,optimizer_updates=0,test_rows_evaluated=0,cos_writes=0)
values={'design.json':cfg,'selection.private.json':mapping,'judge-packet.private.json':dict(inventory=labels,items=packet,selection_sha256=seal),'selection.json':report,'detection-diagnostic.json':detail}
for n,v in values.items():(out/n).write_bytes((json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode())
print(json.dumps(dict(selection=report,detection=detail)))
'''

def prepare():
    assert not (reader.ROOT/'selection.json').exists()
    cfg=dict(profile='complete-model-VALID-detection-error-semantics32-v1',host=reader.HOST,
        stratum_order=['positive_missed','positive_correct','null_false_positive','null_correct'],per_stratum=8,
        selection='Fixed whole1155sharedFN cohort; primary815quantum gold-top-positive/gold-type-allowed. Hash-select8 then8each quantum correctpositive/nullFP/nullcorrect, globally distinct exactsentences. No replacement or prompt/model rescue.',
        protocol='Existing complete20epoch checkpoint cohort on rebuilt cache; reused VALID is exploratory, not independent heldout. Distinct from old TRAIN fourepoch semantic32.',
        annotation='Two isolated gpt-6.1-sol/medium contexts; all labels/predictions/strata and other reader hidden. Reuse accepted reader/parser unchanged; same-model judgments not independent humans.',
        stop='Report all32 and missing/uncertain as such. No post-valid-output redispatch, favorable cohort switch, label overwrite or threshold correction. No new method/training or L5 acceptance.',
        storage='Raw text/mappings/answers remain private on compute. API packet RAM-only relay; local aggregate receipts only.',
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),reader_sha256=hashlib.sha256(Path(reader.__file__).read_bytes()).hexdigest())
    reader.save('design.json',cfg)
    v=json.loads(reader.ops.remote([reader.ops.PY,'-B','-c',SELECT],data=json.dumps(cfg).encode(),timeout=35))
    reader.save('selection.json',v['selection']);reader.save('detection-diagnostic.json',v['detection']);print(json.dumps(v))

if __name__=='__main__':
    {'prepare':prepare,'run':reader.run}[sys.argv[1]]()
