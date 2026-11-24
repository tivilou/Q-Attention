"""R0 paired content selection from existing TRAIN predictions; no model forward."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,time
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def main():
 ap=argparse.ArgumentParser()
 for n in ('root','design','output','progress'):ap.add_argument('--'+n,type=Path,required=True)
 ap.add_argument('--attempt',required=True);a=ap.parse_args();start=time.monotonic()
 cfg=json.loads(a.design.read_bytes());assert sha(__file__)==cfg['code_sha256']
 r=a.root;old=r/'control/full-train-error-contrast-20261124-cpu'
 base=r/'worker/attempts/6a2bf35b8d9d4ac4a6b4fe2acb6bc116/output/20261121T034306Z_seed13'
 cache=base/'discovery/cache';source=base/'recovery-inputs/train.jsonl'
 assert sha(source)==cfg['source_sha256'] and sha(cache/'metadata.json')==cfg['metadata_sha256']
 labels=json.loads((old/'inputs/labels.json').read_bytes());y=np.load(cache/'train-targets.npy',allow_pickle=False)
 previous=json.loads((old/'output/summary.json').read_bytes());pred={}
 for mode in ('quantum','dephased','softmax12','real12'):
  path=old/'output'/(mode+'-logits.private.npy');assert sha(path)==previous['results'][mode]['logits_sha256']
  z=np.load(path,mmap_mode='r',allow_pickle=False);assert z.shape==(58465,40)
  pred[mode]=z.argmax(1)
 q=pred['quantum'];d=pred['real12']
 peers_correct=np.logical_and.reduce([pred[m]==y for m in ('dephased','softmax12','real12')])
 peers_wrong=np.logical_and.reduce([pred[m]!=y for m in ('dephased','softmax12','real12')])
 metadata=json.loads((cache/'metadata.json').read_bytes())['train']
 original=[json.loads(line) for line in source.read_bytes().splitlines() if line.strip()]
 lookup={hashlib.sha256(str(row['metadata']['id']).encode()).hexdigest():row for row in original}
 assert len(lookup)==len(original)==len(metadata)==len(y)==58465
 assert set(lookup)=={row['id_hash'] for row in metadata}
 rows=[lookup[row['id_hash']] for row in metadata]
 assert all(labels[int(k)]==row['label'] for k,row in zip(y,rows))
 masks=dict(q_only_positive=(y>0)&(q==y)&peers_wrong,q_only_null=(y==0)&(q==y)&peers_wrong,
  peer_only_positive=(y>0)&(q!=y)&peers_correct,peer_only_null=(y==0)&(q!=y)&peers_correct)
 assert {k:int(v.sum()) for k,v in masks.items()}==dict(q_only_positive=159,q_only_null=136,peer_only_positive=462,peer_only_null=405)
 assert all(int(v.sum())>=8 for v in masks.values())
 selected=[];truth={}
 for group,mask in masks.items():
  population=np.flatnonzero(mask)
  indices=sorted(population,key=lambda i:hashlib.sha256((cfg['selection_salt']+metadata[i]['id_hash']).encode()).digest())[:8]
  for i in indices:
   row=rows[i];case=hashlib.sha256(('blind-case:'+metadata[i]['id_hash']).encode()).hexdigest()[:12]
   flip=int(case[-1],16)%2;pair=[int(q[i]),int(d[i])];pair=pair[::-1] if flip else pair
   options=dict(A=labels[pair[0]],B=labels[pair[1]])
   tokens=row['tokens'];subject=row['subject'];obj=row['object']
   selected.append(dict(case_id=case,tokens=tokens,subject_span=subject,object_span=obj,
    subject_text=' '.join(tokens[subject[0]:subject[1]]),object_text=' '.join(tokens[obj[0]:obj[1]]),options=options))
   truth[case]=dict(group=group,index=int(i),id_hash=metadata[i]['id_hash'],gold=row['label'],
    quantum_label=labels[int(q[i])],real12_label=labels[int(d[i])],gold_option='A' if pair[0]==y[i] else 'B')
 selected.sort(key=lambda row:hashlib.sha256(('packet-order:'+row['case_id']).encode()).digest())
 packet=dict(task='Choose which of two proposed relation labels is supported for the MARKED subject/object by this sentence only. This is a blind content judgment, not model evaluation.',
  instructions=['Do not infer labels from topic/co-occurrence or outside knowledge. Spans use zero-based half-open token indices.',
  'no_relation means neither any listed positive relation nor the proposed positive option is established for this exact marked pair. Evidence about a different pair is insufficient.',
  'Choose A or B only when one is clearly supported; choose both,neither,uncertain otherwise. Do not invent missing words or mention source models.',
  'For every case choose a category and give one short sentence explaining argument linkage. Quote only a few exact token words; do not rewrite gold labels.'],
  category_options=['same_pair_explicit','other_pair_only','negated_or_hypothetical','argument_or_type_mismatch','implicit_or_external_knowledge','underspecified','ambiguous'],
  response_schema=dict(rows=[dict(case_id='id',choice='A|B|both|neither|uncertain',category='one category',reason='short sentence',evidence_structure='single_cue|multi_cue|no_pair_evidence|uncertain',distractor_relation='present|absent|uncertain')]),cases=selected)
 packet['instructions'].append('For each case also classify evidence_structure: one local cue, multiple cues needed jointly, no pair evidence, or uncertain. Classify distractor_relation only when a positive relation about a DIFFERENT entity pair is expressed in the sentence. Preserve the fixed enumerated output fields.')
 assert not a.output.exists();a.output.mkdir(parents=True)
 save(a.output/'blind-packet.private.json',packet);save(a.output/'truth.private.json',truth)
 summary=dict(profile=cfg['profile'],rows=58465,selected_rows=len(selected),group_populations={k:int(v.sum()) for k,v in masks.items()},
  selected_per_group=8,selection='Fixed hash within four outcome-defined groups; exploratory case sample, no prevalence or independent utility claim.',
  packet_sha256=sha(a.output/'blind-packet.private.json'),truth_sha256=sha(a.output/'truth.private.json'),
  source_sha256=cfg['source_sha256'],seconds=time.monotonic()-start,model_forwards=0,training_updates=0,new_valid=0,new_test=0,cos_writes=0)
 save(a.output/'summary.json',summary);save(a.output/'RUN_COMPLETE',dict(attempt=a.attempt,summary_sha256=sha(a.output/'summary.json')))
 save(a.progress,dict(schema='research.progress.v1',attempt=a.attempt,phase='complete_content_packet',current=32,total=32,unit='case',observed_at=time.time(),heartbeat_at=time.time()))
 print(json.dumps(summary))

if __name__=='__main__':main()
