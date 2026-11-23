
from pathlib import Path
import hashlib,json,time
import numpy as np
start=time.monotonic();root=Path('/hy-tmp/Q-Attention/control')
old=root/'train-binding-census-aligned-20261123';out=root/'context-rejection-reference-20261123'
assert not (out/'interval-census.json').exists()
summary=json.loads((old/'output/summary.json').read_bytes());cfg=json.loads((old/'design.json').read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
census=old/'output/census.private.npz';assert sha(census)==summary['private_census_sha256']
pair_path=old/'output/positive-null-pairs.private.npy'
pair_summary=json.loads((old/'output/paired-collapse.json').read_bytes())
assert sha(pair_path)==pair_summary['private_pair_sha256']
with np.load(census,allow_pickle=False) as a:
 y=a['labels'];saved_pred=a['predictions'];sentence=a['sentence_id']
sent,group=np.unique(sentence,return_inverse=True);n=len(y);ng=len(sent)
pair=np.load(pair_path,allow_pickle=False);pos,neg=pair.T
assert np.all(group[pos]==group[neg]) and np.all(y[pos]>0) and np.all(y[neg]==0)
def macro(pred):
 f=[]
 for k in range(1,40):
  tp=int(((y==k)&(pred==k)).sum());fp=int(((y!=k)&(pred==k)).sum());fn=int(((y==k)&(pred!=k)).sum())
  f.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
 return float(np.mean(f))
results={};each_null=[];each_repairable=[];private={}
for m,mode in enumerate(cfg['modes']):
 path=root/('full-model-error-structure-20261123/output/'+mode+'-logits.private.npy')
 assert sha(path)==cfg['input_hashes'][str(path.relative_to(root.parent))]
 z=np.load(path,mmap_mode='r',allow_pickle=False);assert z.shape==(n,40) and np.isfinite(z).all()
 top=z[:,1:].argmax(1)+1;d=z[np.arange(n),top].astype(float)-z[:,0].astype(float)
 pred=np.where(d>0,top,0);assert np.array_equal(pred,saved_pred[m])
 correct=pred==y;null_correct=(y==0)&correct;miss=(y>0)&(pred==0)
 # A nonnegative sentence offset preserves every correct positive automatically.
 # All correct null rows in that sentence impose t <= min_i[-d_i].
 upper=np.full(ng,np.inf);np.minimum.at(upper,group[null_correct],-d[null_correct])
 assert np.all(upper>=0)
 required=np.zeros(ng);np.maximum.at(required,group,-d)
 shift=np.where(np.isfinite(upper),upper,required+1.)
 new=np.where(d>-shift[group],top,0)
 assert np.all(new[correct]==y[correct])
 repaired=miss&(top==y)&(upper[group]>-d)
 assert np.array_equal((new==y)&~correct,repaired)
 assert not np.any((y==0)&(pred==0)&(new!=0))
 collapsed=(pred[pos]==0)&(pred[neg]==0)
 gold_order=z[pos,y[pos]].astype(float)-z[pos,0].astype(float)>z[neg,y[pos]].astype(float)-z[neg,0].astype(float)
 pair_local=collapsed&(top[pos]==y[pos])&(d[pos]>d[neg])
 pair_safe=collapsed&repaired[pos]
 assert np.all(~pair_safe|pair_local)
 # Independently evaluate no-harm upper bounds with explicit sentence groups.
 upper_check=np.full(ng,np.inf)
 for j in np.flatnonzero(null_correct):
  g=group[j];upper_check[g]=min(upper_check[g],-d[j])
 assert np.array_equal(upper,upper_check)
 assert int(repaired.sum())==sum(int(top[j]==y[j] and -d[j]<upper[group[j]]) for j in np.flatnonzero(miss))
 private[mode+'_repairable']=repaired;each_repairable.append(repaired);each_null.append(collapsed)
 results[mode]=dict(rows=n,positive_rows=int((y>0).sum()),native_positive_fn=int(miss.sum()),
  native_fn_gold_top_positive=int((miss&(top==y)).sum()),
  damage_free_sentence_offset_repairable_positive_rows=int(repaired.sum()),
  repairable_positive_rows_with_an_annotated_correct_null_in_same_sentence=int((repaired&np.isfinite(upper[group])).sum()),
  repairable_positive_relation_classes=len(set(map(int,y[repaired]))),
  repairable_sentences=len(set(map(int,group[repaired]))),
  natural_pairs_both_null=int(collapsed.sum()),natural_pairs_both_null_gold_margin_ordered=int((collapsed&gold_order).sum()),
  natural_pairs_both_null_locally_threshold_repairable=int(pair_local.sum()),
  natural_pairs_both_null_threshold_repairable_without_harming_any_correct_TRAIN_row=int(pair_safe.sum()),
  natural_paired_unique_positive_repairable=len(set(map(int,pos[pair_safe]))),
  native_train_macro_f1=macro(pred),label_aware_sentence_raise_oracle_train_macro_f1=macro(new),
  native_correct_predictions_harmed=int((correct&(new!=y)).sum()),
  native_correct_positive_to_null=int(((y>0)&correct&(new==0)).sum()),
  new_null_false_positives=int((null_correct&(new!=0)).sum()),
  native_null_fp=int(((y==0)&(pred!=0)).sum()),oracle_null_fp=int(((y==0)&(new!=0)).sum()))
shared_pairs=np.logical_and.reduce(each_null);shared_ids=np.unique(pos[shared_pairs]);all_repairable=np.logical_and.reduce(each_repairable)
out.mkdir(parents=True,exist_ok=True)
np.savez(out/'interval-census.private.npz',**private)
report=dict(profile='R0-posthoc-TRAIN-damage-free-common-sentence-offset-headroom-v1',
 rows=n,unique_sentences=ng,results=results,
 all_four_both_null_unique_positive_rows=len(shared_ids),
 all_four_both_null_positive_rows_repairable_in_each_heads_own_oracle=int(all_repairable[shared_ids].sum()),
 input_census_sha256=sha(census),input_pairs_sha256=sha(pair_path),private_result_sha256=sha(out/'interval-census.private.npz'),
 exact_condition='For gold-top-positive missed row i, nonnegative sentence shift repairs it iff -d_i < min_{j: same sentence and native-correct null}(-d_j); empty minimum is infinity. All40 native-correct rows are protected. Gold-aware diagnostic only, never an inference rule.',
 unit='Unique TRAIN rows and exact sentences; natural pairs share rows and are not independent.',
 verification='Vectorized upper bounds equal explicit sentence minima; all old correct predictions preserved, all new correct predictions equal the predicted repairable set, null FP counts unchanged.',
 model_forwards=0,optimizer_updates=0,valid_rows=0,test_rows=0,cos_writes=0,
 seconds=time.monotonic()-start,
 limitations='Posthoc TRAIN label-aware conditional headroom, not expected gain, learned threshold, causal mechanism, semantic correctness, independent utility, novelty or L5. A sentence offset can be classical. No quantum plugin or experiment launched.')
(out/'interval-census.json').write_bytes((json.dumps(report,indent=2)+'\n').encode());print(json.dumps(report))
