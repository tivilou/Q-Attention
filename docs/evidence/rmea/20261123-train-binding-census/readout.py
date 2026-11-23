"""R0: fixed-prediction TRAIN census of same-type competing object mentions."""
from pathlib import Path
from collections import defaultdict
import argparse,hashlib,json,time
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
 q=p.with_suffix('.pending');q.write_bytes((json.dumps(v,indent=2,allow_nan=False)+'\n').encode());q.replace(p)
def main():
 ap=argparse.ArgumentParser()
 for n in ('root','design','output','progress'):ap.add_argument('--'+n,type=Path,required=True)
 ap.add_argument('--attempt',required=True);a=ap.parse_args();start=time.monotonic();r=a.root
 cfg=json.loads(a.design.read_bytes());assert sha(__file__)==cfg['code_sha256']
 for name,digest in cfg['input_hashes'].items():assert sha(r/name)==digest,name
 assert not a.output.exists();a.output.mkdir(parents=True)
 labels=json.loads((r/'inputs/native-baseline/recovery-inputs/labels.json').read_bytes())
 original=[json.loads(s) for s in (r/'inputs/native-baseline/recovery-inputs/train.jsonl').read_bytes().splitlines() if s.strip()]
 metadata=json.loads((r/'inputs/native-baseline/discovery/cache/metadata.json').read_bytes())['train']
 index={hashlib.sha256(str(row['metadata']['id']).encode()).hexdigest():i for i,row in enumerate(original)}
 assert len(index)==len(original)==len(metadata)==58465
 assert set(index)=={v['id_hash'] for v in metadata}
 order=np.array([index[v['id_hash']] for v in metadata],dtype=np.int64)
 rows=[original[i] for i in order]
 assert all(len(row['tokens'])==v['words'] for row,v in zip(rows,metadata))
 assert all(row[k]==v[k] for row,v in zip(rows[:2],metadata[:2]) for k in ('tokens','subject','object'))
 y=np.load(r/'inputs/native-baseline/discovery/cache/train-targets.npy',allow_pickle=False);n=len(rows)
 assert n==58465==len(y) and labels[0]=='no_relation' and len(labels)==40
 assert all(labels[int(k)]==row['label'] for k,row in zip(y,rows))
 groups=defaultdict(set);keys=[];length=np.zeros(n,int);types=[];sentence_ids=[]
 for i,row in enumerate(rows):
  words=tuple(row['tokens']);s=tuple(row['subject']);o=tuple(row['object']);t=row['metadata']['obj_type']
  assert 0<=s[0]<s[1]<=len(words) and 0<=o[0]<o[1]<=len(words)
  key=(words,s,t);keys.append(key);groups[key].add(o);length[i]=len(words)
  types.append((row['metadata']['subj_type'],t))
  sentence_ids.append(hashlib.sha256(json.dumps(words,ensure_ascii=False,separators=(',',':')).encode()).hexdigest())
 competing=np.array([any(other!=tuple(row['object']) and (other[1]<=row['object'][0] or row['object'][1]<=other[0]) for other in groups[key]) for row,key in zip(rows,keys)],bool)
 bins=np.digitize(length,np.array(cfg['length_bin_upper_bounds']),right=True)
 allpositive=y>0
 pred=[];tops=[]
 for mode in cfg['modes']:
  z=np.load(r/('control/full-model-error-structure-20261123/output/'+mode+'-logits.private.npy'),mmap_mode='r',allow_pickle=False)
  assert z.shape==(n,40) and np.isfinite(z).all();pred.append(z.argmax(1));tops.append(z[:,1:].argmax(1)+1)
 pred=np.stack(pred);tops=np.stack(tops)
 # All overlap decisions depend on inputs/labels only, never predictions.
 eligible=[];byclass=defaultdict(list)
 for k in range(1,40):
  for b in range(len(cfg['length_bin_upper_bounds'])+1):
   left=np.flatnonzero((y==k)&(bins==b)&competing);right=np.flatnonzero((y==k)&(bins==b)&~competing)
   if min(len(left),len(right))>=cfg['minimum_per_arm_cell']:
    cell=(k,b,left,right);eligible.append(cell);byclass[k].append(cell)
 assert sum(len(g) for g in byclass.values())==len(eligible)
 report={}
 for m,mode in enumerate(cfg['modes']):
  pp=pred[m];fn=pp==0;classes=[]
  for k,cells in sorted(byclass.items()):
   w=np.array([min(len(l),len(rr)) for _,_,l,rr in cells],float);w/=w.sum()
   fa=np.array([fn[l].mean() for _,_,l,_ in cells]);fb=np.array([fn[rr].mean() for _,_,_,rr in cells])
   classes.append(dict(label=labels[k],overlap_cells=len(cells),competing_rows=sum(len(l) for _,_,l,_ in cells),simple_rows=sum(len(rr) for _,_,_,rr in cells),
    competing_standardized_fn_rate=float(w@fa),simple_standardized_fn_rate=float(w@fb),delta=float(w@(fa-fb))))
  standard=float(np.mean([v['delta'] for v in classes])) if classes else None
  strata={}
  for name,mask in [('positive_competing',allpositive&competing),('positive_simple',allpositive&~competing),('null_competing',~allpositive&competing),('null_simple',~allpositive&~competing)]:
   ids=np.flatnonzero(mask);den=len(ids);pos=bool(allpositive[ids[0]]) if den else name.startswith('positive')
   fids=ids[fn[ids]] if pos else np.array([],int)
   strata[name]=dict(rows=den,correct=int((pp[ids]==y[ids]).sum()),null_predictions=int(fn[ids].sum()),
    positive_false_negatives=int(fn[ids].sum()) if pos else 0,null_false_positives=int((pp[ids]!=0).sum()) if not pos else 0,
    wrong_positive=int(((pp[ids]>0)&(pp[ids]!=y[ids])).sum()) if pos else 0,
    fn_gold_top_positive=int((tops[m,fids]==y[fids]).sum()),fn_rate=float(fn[ids].mean()) if pos and den else None)
  report[mode]=dict(strata=strata,standardized_positive_fn_delta=standard,positive_delta_classes=sum(v['delta']>0 for v in classes),per_class=classes)
 # Negative controls standardized by ordered input type pair and length.
 nullstrata=defaultdict(lambda:[[],[]])
 for i in np.flatnonzero(~allpositive):nullstrata[(types[i],int(bins[i]))][0 if competing[i] else 1].append(int(i))
 nullcells=[(np.array(aa,int),np.array(bb,int)) for aa,bb in nullstrata.values() if min(len(aa),len(bb))>=cfg['minimum_per_arm_cell']]
 nw=np.array([min(len(aa),len(bb)) for aa,bb in nullcells],float)
 if len(nw):nw/=nw.sum()
 for m,mode in enumerate(cfg['modes']):
  fp=pred[m]!=0
  report[mode]['standardized_null_fp_delta']=float(nw@np.array([fp[aa].mean()-fp[bb].mean() for aa,bb in nullcells])) if len(nw) else None
 counts=dict(rows=n,unique_sentences=len(set(sentence_ids)),competing_rows=int(competing.sum()),
  competing_positive_rows=int((competing&allpositive).sum()),simple_positive_rows=int((~competing&allpositive).sum()),
  overlap_positive_classes=len(byclass),overlap_class_length_cells=len(eligible),overlap_null_type_length_cells=len(nullcells))
 gate=dict(class_coverage=len(byclass)>=cfg['minimum_positive_classes'],
  row_coverage=min(counts['competing_positive_rows'],counts['simple_positive_rows'])>=cfg['minimum_positive_rows_per_arm'])
 for mode,row in report.items():gate[mode+'_association']=row['standardized_positive_fn_delta'] is not None and row['standardized_positive_fn_delta']>=cfg['minimum_absolute_fn_delta']
 passed=all(gate.values())
 np.savez(a.output/'census.private.npz',source_order=order,labels=y,competing=competing,length_bin=bins,predictions=pred,gold_top_positive=tops,sentence_id=np.array(sentence_ids,dtype='U64'))
 summary=dict(profile=cfg['profile'],attempt=a.attempt,counts=counts,results=report,gates=gate,
  identity_alignment=dict(rows_reordered=int((order!=np.arange(n)).sum()),all_ids_unique=True,all_labels_match=True,all_word_counts_match=True,source_order_sha256=hashlib.sha256(order.tobytes()).hexdigest()),
  decision='ASSOCIATION_SUPPORTED_PROCEED_TO_BLIND_SEMANTIC_CHECK' if passed else 'DO_NOT_PROMOTE_THIS_STRUCTURAL_EXPLANATION',
  private_census_sha256=sha(a.output/'census.private.npz'),seconds=time.monotonic()-start,
  model_forwards=0,optimizer_updates=0,valid_rows=0,test_rows=0,cos_writes=0,
  scope='Full reused TRAIN descriptive class/length-standardized association, not causal, independent utility, quantum necessity or a rescue/reselection of the closed40-pair sample.')
 save(a.output/'summary.json',summary);(a.output/'RUN_COMPLETE').write_bytes(b'completed\n')
 save(a.progress,dict(schema='research.progress.v1',attempt=a.attempt,phase='complete',current=n,total=n,unit='example',observed_at=time.time(),heartbeat_at=time.time()))
 print(json.dumps(summary))
if __name__=='__main__':main()
