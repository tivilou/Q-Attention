"""Six frozen hypothetical inputs, immutable selected heads, forward-only CPU probe.

Private arrays include final encoder states and every applicable head field.
No optimizer, training, test-data loading, checkpoint write or automatic rerun.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

from trace_tools import sha, pinned, ref, display, numeric, axes, owner, read_value
from tokenization import marked_words
from scripts.validate_sample_trace_v2 import STAGES, validate
from legacy.rmea import RMEAHead
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
EXPECTED_ENV={'torch':'2.5.1+cpu','numpy':'1.26.4','transformers':'4.44.2','tokenizers':'0.19.1'}
EXTRA_AXES={'input_ids':['sample','piece'],'attention_mask':['sample','piece'],
    'piece_to_word':['piece'],'piece_hidden':['sample','piece','hidden'],
    'word_mean':['sample','word','hidden'],'word_half':['sample','word','hidden']}

def save(path,obj):
    with path.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n')

def require(condition,message):
    if not condition:raise ValueError(message)

def artifact(path,item):
    require(not path.is_symlink() and path.is_file(),'Missing or linked dependency')
    require(path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],'Dependency identity differs')
    return path

def check(contract,inputs,annotations):
    require(contract['schema']=='qattn.semantic-forward.contract.v1','Contract version differs')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU execution must hide GPUs')
    actual={k:importlib.metadata.version(k) for k in EXPECTED_ENV}
    require(actual==EXPECTED_ENV and sys.version_info[:2]==(3,10),'Pinned runtime differs')
    for name,digest in contract['source_files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'Invalid source path')
        require(not (HERE/name).is_symlink() and sha(HERE/name)==digest,'Source identity differs: '+name)
    require(sha(inputs)==contract['inputs_sha256'] and sha(annotations)==contract['annotations_sha256'],'Probe release differs')
    rows=json.loads(inputs.read_bytes())['samples'];ann=json.loads(annotations.read_bytes())['samples']
    require(len(rows)==6 and len(ann)==6,'Exactly six probe inputs required')
    require([r['sample_id'] for r in rows]==contract['sample_ids'],'Input ordering differs')
    require([r['sample_id'] for r in ann]==contract['sample_ids'],'Annotation ordering differs')
    require([m['variant'] for m in contract['models']]==['quantum','dephased','relation_softmax'],'Model set differs')
    require(len({r['sample_id'] for r in rows})==6,'Repeated input')
    for row in rows:
        require(set(row)=={'sample_id','case','kind','tokens','subject','object'},'Input whitelist differs')
        require(all(isinstance(t,str) and t for t in row['tokens']),'Invalid tokens')
        for role in ('subject','object'):
            a,b=row[role];require(type(a) is int and type(b) is int and 0<=a<b<=len(row['tokens']),'Invalid half-open span')
        require(set(range(*row['subject'])).isdisjoint(range(*row['object'])),'Overlapping entity spans unsupported')
    for case in ('pool-03','pool-08'):
        triplet=[r for r in rows if r['case']==case]
        require([r['kind'] for r in triplet]==['original','name_preserving','relation_changing'],'Triplet order differs')
        require(len({len(r['tokens']) for r in triplet})==1,'Word alignment differs')
        require(all(r['subject']==triplet[0]['subject'] and r['object']==triplet[0]['object'] for r in triplet),'Ordered roles differ')
    labels=json.loads((HERE/'labels.json').read_bytes())
    require(len(labels)==40 and len(set(labels))==40,'Forty train-derived labels required')
    require(all(a['intended_label'] in labels for a in ann),'Unknown annotation')
    assets=Path(contract['assets_directory']);lock=json.loads((HERE/'assets.lock.json').read_bytes())
    require(not assets.is_symlink() and {p.name for p in assets.iterdir()}=={f['name'] for f in lock['files']},'Asset inventory differs')
    for f in lock['files']:artifact(assets/f['name'],dict(bytes=f['size_bytes'],sha256=f['sha256']))
    for m in contract['models']:artifact(Path(m['path']),m)
    for a in contract['anchors']:artifact(Path(a['path']),a)
    cgroup=Path('/sys/fs/cgroup');v2=(cgroup/'memory.max').is_file()
    limit=(cgroup/('memory.max' if v2 else 'memory/memory.limit_in_bytes')).read_text().strip()
    used=int((cgroup/('memory.current' if v2 else 'memory/memory.usage_in_bytes')).read_text())
    require(limit=='max' or int(limit)-used>=10*1024**3,'Less than 10 GiB cgroup headroom')
    require(shutil.disk_usage(inputs.parent).free>=2*1024**3,'Less than 2 GiB free disk')
    events=cgroup/('memory.events' if v2 else 'memory/memory.failcnt')
    return rows,ann,labels,dict(environment=actual,python=sys.version.split()[0],cgroup_version=2 if v2 else 1,memory_max=limit,memory_current=used,memory_events=events.read_text())

def load_head(path,variant):
    state=torch.load(path,map_location='cpu',weights_only=True)
    require(state['value.weight'].dtype==torch.float32,'Selected head dtype differs')
    head=RMEAHead(variant);head.load_state_dict(state,strict=True)
    return head.eval().requires_grad_(False)

def role_features(row):
    positions=np.arange(len(row['tokens']),dtype=np.float32);s,o=row['subject'],row['object']
    return np.stack([np.clip((positions-(s[0]+s[1]-1)/2)/32,-1,1),np.clip((positions-(o[0]+o[1]-1)/2)/32,-1,1),
        ((positions>=s[0])&(positions<s[1])).astype(np.float32),((positions>=o[0])&(positions<o[1])).astype(np.float32)],axis=-1)[None]

def encoding(row,tokenizer,encoder):
    # Only these three fields enter preprocessing, not case IDs or annotations.
    model_input={k:row[k] for k in ('tokens','subject','object')}
    words,original=marked_words(model_input)
    batch=tokenizer(words,is_split_into_words=True,padding=False,truncation=False,return_tensors='pt')
    require(set(batch)=={'input_ids','attention_mask'} and batch['input_ids'].shape[1]<=512,'Unexpected encoder input/length')
    piece_map=np.array([-1 if w is None or original[w] is None else original[w] for w in batch.word_ids()],dtype=np.int64)
    hidden=encoder(**batch).last_hidden_state.float()
    means=[]
    for word in range(len(row['tokens'])):
        ix=np.flatnonzero(piece_map==word);require(len(ix)>0,'Original word disappeared')
        means.append(hidden[0,ix.tolist()].mean(0))
    mean=torch.stack(means)[None];half=mean.half();h=half.float();roles=role_features(row)
    arrays={k:v.detach().cpu().numpy().copy() for k,v in batch.items()}
    arrays.update(piece_to_word=piece_map,piece_hidden=hidden.detach().cpu().numpy().copy(),
        word_mean=mean.detach().cpu().numpy().copy(),word_half=half.cpu().numpy().copy())
    return arrays,(h,torch.from_numpy(roles),torch.ones(h.shape[:2],dtype=torch.bool))

def build_trace(out,contract,records,source_revision):
    samples=[]
    for record in records:
        merged={**record['encoder'],**record['head']};reps=[]
        for name,array in sorted(merged.items()):
            stage=('tokenization' if name in ('input_ids','attention_mask','piece_to_word') else 'encoding') if name in EXTRA_AXES else ('preparation' if name in ('psi','phi','effect_angles') else owner(name))
            ax=EXTRA_AXES[name] if name in EXTRA_AXES else axes(name,array)
            require(len(ax)==array.ndim and np.isfinite(array).all(),'Invalid representation '+name)
            path=record['encoder_path'] if name in EXTRA_AXES else record['head_path'];art=ref(out,path)
            reps.append(dict(id=name,stage_id=stage,status='observed',meaning=name+' at the actual forward-only selected-model boundary',
                shape=list(array.shape),dtype=str(array.dtype),axis_semantics=ax,artifact_refs=[art],display=display(name),
                value_capture=dict(status='complete',shape=list(array.shape),logical_element_count=int(array.size),stored_element_count=int(array.size),artifact_refs=[art],
                    reader_projection=dict(status='available',kind='coordinate_inspector',coordinate_access='arbitrary',
                        reader_ref='worker.py value --run-dir RUN --sample '+record['id']+' --representation '+name,
                        source_sha256=[art['sha256']],implementation_sha256=[sha(HERE/'worker.py'),sha(HERE/'trace_tools.py')]))))
        incoming={'tokenization':['external:licensed_input','external:tokenizer_assets'],
            'encoding':['input_ids','attention_mask','external:encoder_assets'],
            'inputs':['piece_hidden','piece_to_word','word_mean','word_half','external:ordered_spans'],
            'preparation':['features','external:selected_checkpoint'],'values':['h','external:selected_checkpoint'],
            'measurement':['mask','external:selected_checkpoint']+(['psi','phi','utilities'] if 'psi' in merged else ['key','query']),
            'prediction':['alpha','utilities','bias']}
        if not record['encoder']:incoming['inputs']=['external:pinned_anchor']
        # Preparation produces angles/key; measurement also consumes every applicable preparation output.
        stages=[]
        for stage in incoming:
            names=[r['id'] for r in reps if r['stage_id']==stage]
            if not names:continue
            canonical={'tokenization':'preprocess','encoding':'context','inputs':'data','prediction':'generation'}.get(stage,'scoring')
            stages.append(dict(stage_id=stage,stage=stage,canonical_stage=canonical,lane='replay',status='observed',purpose=stage+' in the fixed-input forward diagnostic',
                input_refs=incoming[stage],output_refs=names,artifact_refs=[ref(out,record['encoder_path']),ref(out,record['head_path'])],display=display(stage),target_access='not_available'))
        samples.append(dict(sample_id=record['id'],lane='replay',split=record.get('split','train'),checkpoint='selected',
            roles=dict(source_sample=dict(id=record['input_id']),sequence_or_features=dict(boundary=record.get('boundary','full raw-input reencoding then word-mean/fp16/fp32')),
                task_objects=dict(subject_role=2,object_role=3),prediction=dict(value=int(record['head']['logits'].argmax())),
                provenance=dict(condition=record['model'],checkpoint=ref(out,out/'checkpoints'/(record['model']+'.pt')),encoder_lock_sha256=sha(HERE/'assets.lock.json'),
                    input_release_sha256=contract['inputs_sha256'],selection_rule=contract['selection_rule'],scope=contract['scope'])),representations=reps,stages=stages))
    common=sorted(set.intersection(*(set(r['id'] for r in s['representations']) for s in samples)))
    payload=dict(schema_version='sample-trace.v2',trace_id=out.name,trace_completeness='complete',
        experiment=dict(run_id=out.name,dataset=contract['scope'],code_revision=source_revision,config_sha256=sha(HERE/'contract.json'),model_identity='Three fixed selected fresh5 seed29 heads'),
        sample_selection=dict(rule=contract['selection_rule'],population_scope=contract['scope'],selected_count=len(samples),selected_sample_ids=[s['sample_id'] for s in samples]),
        task_contract=dict(task_type='exploratory hypothetical relation classification',required_roles=['source_sample','sequence_or_features','task_objects','prediction','provenance'],
            required_representations=common,required_value_representations=common,required_splits=sorted({s['split'] for s in samples}),required_checkpoints=['selected']),
        coverage={s:'observed' if s in {g['canonical_stage'] for sample in samples for g in sample['stages']} else 'not_applicable' for s in STAGES},samples=samples)
    errors,warnings=validate(payload,require_complete=True);require(not errors,str(errors));save(out/'sample-trace.json',payload)
    return payload,warnings

def inspect_encoder(arrays,h):
    require(np.array_equal(arrays['word_half'].astype(np.float32),h),'Half-roundtrip features differ')
    hidden=torch.from_numpy(arrays['piece_hidden']);mapping=arrays['piece_to_word'];reconstructed=[]
    for j in range(h.shape[1]):reconstructed.append(hidden[0,np.flatnonzero(mapping==j).tolist()].mean(0))
    actual=torch.stack(reconstructed)[None].numpy()
    require(np.array_equal(actual,arrays['word_mean']),'Word-mean replay differs')
    require(np.array_equal(actual.astype(np.float16),arrays['word_half']),'Half capture differs')

def canonical(arrays):
    a=np.array(arrays['alpha'],dtype=np.float64,order='C');u=np.array(arrays['utilities'],dtype=np.float64,order='C')
    return (a*u).sum(-1,dtype=np.float64)+arrays['bias'].astype(np.float64)

def paired(first,second):
    a,b=[np.array(x['alpha'],dtype=np.float64,order='C') for x in (first,second)]
    u,v=[np.array(x['utilities'],dtype=np.float64,order='C') for x in (first,second)]
    change=canonical(second)-canonical(first)
    alpha_term=((b-a)*(u+v)/2).sum(-1,dtype=np.float64)
    utility_term=((a+b)*(v-u)/2).sum(-1,dtype=np.float64)
    residual=float(np.max(np.abs(change-alpha_term-utility_term)))
    require(np.array_equal(first['bias'],second['bias']) and residual<1e-9,'Product accounting differs')
    result=dict(alpha_total_variation_mean=float(np.mean(np.abs(b-a).sum(-1)/2)),utility_rms_change=float(np.sqrt(np.mean((v-u)**2))),
        h_rms_change=float(np.sqrt(np.mean((second['h'].astype(float)-first['h'].astype(float))**2))),
        alpha_term_l2=float(np.linalg.norm(alpha_term)),utility_term_l2=float(np.linalg.norm(utility_term)),identity_residual_max=residual,
        native_decision_changed=bool(first['logits'].argmax()!=second['logits'].argmax()))
    if 'psi' in first:
        overlap=np.abs(np.sum(first['psi'].astype(np.complex128).conj()*second['psi'].astype(np.complex128),axis=-1))**2
        result['token_projective_distance_mean']=float(np.sqrt(np.maximum(0,1-np.minimum(1,overlap))).mean())
    return result

@torch.inference_mode()
def run(args):
    started=time.monotonic();contract=json.loads((HERE/'contract.json').read_bytes())
    rows,annotations,labels,ready=check(contract,args.inputs,args.annotations)
    out=args.run_dir.resolve();require(not out.exists(),'Output exists: inspect it, never automatically rerun')
    repo=HERE.parents[1]
    revision=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    require(revision==args.source_revision and not subprocess.check_output(['git','-C',str(repo),'status','--porcelain=v1'],text=True),'Clean exact scheduler source required')
    require(not out.is_relative_to(repo),'Private output must stay outside the code worktree')
    out.mkdir();save(out/'readiness.json',ready);(out/'checkpoints').mkdir();(out/'arrays').mkdir()
    for model in contract['models']:
        cp=out/'checkpoints'/(model['variant']+'.pt');shutil.copyfile(model['path'],cp);artifact(cp,model)
    for name,path in [('licensed-inputs.json',args.inputs),('probe-annotations.json',args.annotations),('contract.json',HERE/'contract.json')]:shutil.copyfile(path,out/name)
    torch.set_num_threads(4);torch.manual_seed(13);np.random.seed(13);torch.use_deterministic_algorithms(True)
    calls=dict(encoder=0,head_capture=0,head_replay=0,anchor=0)
    q=load_head(Path(contract['models'][0]['path']),'quantum');anchors=[]
    for anchor_index,item in enumerate(contract['anchors']):
        with np.load(item['path'],allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        captured=numeric(q,tuple(torch.from_numpy(arrays[k]) for k in ('h','roles','mask')));actual=captured['logits'];calls['anchor']+=1
        require(np.allclose(actual,arrays['logits'],rtol=3e-4,atol=5e-5),'Selected Q head anchor failed')
        anchors.append(dict(sha256=item['sha256'],max_abs=float(np.max(np.abs(actual-arrays['logits'])))))
        if anchor_index==0:
            fixture=out/'fixture';fixture.mkdir();(fixture/'checkpoints').mkdir();shutil.copyfile(out/'checkpoints/quantum.pt',fixture/'checkpoints/quantum.pt')
            path=fixture/'anchor.npz';np.savez_compressed(path,**captured)
            fixture_contract=dict(contract,scope='Previously exposed validation anchor, numeric trace preflight only',selection_rule='First exact disclosed Q anchor; unrelated to training probe selection')
            fixture_record=dict(id='quantum:numeric-anchor',input_id='fixed-anchor-0',model='quantum',split='valid',boundary='frozen-feature anchor; no reconstructed encoder states',encoder={},head=captured,encoder_path=path,head_path=path)
            fixture_payload,_=build_trace(fixture,fixture_contract,[fixture_record],args.source_revision)
            for name,value in captured.items():
                coord=[0]*value.ndim;seen=read_value(fixture,fixture_payload['samples'][0],name,[coord])[0];expected=value[tuple(coord)].item()
                if isinstance(expected,complex):expected=dict(real=expected.real,imag=expected.imag)
                require(seen==expected,'Real-valued fixture reader failed')
    del q
    from transformers import RobertaModel,RobertaTokenizerFast
    tokenizer=RobertaTokenizerFast.from_pretrained(contract['assets_directory'],add_prefix_space=True,local_files_only=True)
    encoder=RobertaModel.from_pretrained(contract['assets_directory'],local_files_only=True,add_pooling_layer=False,torch_dtype=torch.float32,attn_implementation='eager').eval().requires_grad_(False)
    encodings=[]
    for index,row in enumerate(rows):
        arrays,inputs=encoding(row,tokenizer,encoder);calls['encoder']+=1
        inspect_encoder(arrays,inputs[0].numpy());path=out/'arrays'/('encoder-'+str(index)+'.npz');np.savez_compressed(path,**arrays)
        encodings.append((arrays,inputs,path))
    del encoder
    records=[];reports={};reader_checks=0;array_checks=0
    annotations_by_id={r['sample_id']:r['intended_label'] for r in annotations}
    for model in contract['models']:
        mode=model['variant'];cp=out/'checkpoints'/(mode+'.pt');artifact(cp,model)
        head=load_head(cp,mode);model_records=[];rounding_disagreements=0;native_compose_error=0.
        for index,(row,(enc,inputs,encoder_path)) in enumerate(zip(rows,encodings)):
            captured=numeric(head,inputs);calls['head_capture']+=1
            path=out/'arrays'/(mode+'-'+str(index)+'.npz');np.savez_compressed(path,**captured)
            model_records.append(captured);rounding_disagreements+=int(captured['logits'].argmax()!=canonical(captured).argmax())
            native_compose_error=max(native_compose_error,float(np.max(np.abs(captured['logits']-canonical(captured)))))
            records.append(dict(id=mode+':'+row['sample_id'],input_id=row['sample_id'],model=mode,encoder=enc,head=captured,encoder_path=encoder_path,head_path=path))
        # A separately reconstructed head replays every applicable numeric field.
        replay_head=load_head(cp,mode)
        for captured,(_,inputs,_) in zip(model_records,encodings):
            repeat=numeric(replay_head,inputs);calls['head_replay']+=1
            require(set(repeat)==set(captured),'Applicable field inventory differs')
            for name,value in repeat.items():
                require(value.shape==captured[name].shape and value.dtype==captured[name].dtype and np.array_equal(value,captured[name]),'Fresh full-value replay differs: '+name);array_checks+=1
        correct={r['kind']:[] for r in rows};pair_results=[]
        for case in ('pool-03','pool-08'):
            ix=[i for i,r in enumerate(rows) if r['case']==case];require(len(ix)==3,'Missing triplet')
            for i in ix:correct[rows[i]['kind']].append(labels[int(model_records[i]['logits'].argmax())]==annotations_by_id[rows[i]['sample_id']])
            pair_results.append(dict(case=case,name_preserving=paired(model_records[ix[0]],model_records[ix[1]]),relation_changing=paired(model_records[ix[0]],model_records[ix[2]])))
        reports[mode]=dict(inputs=6,bases=2,correct_counts={k:sum(v) for k,v in correct.items()},all_three_correct=sum(all(labels[int(model_records[i]['logits'].argmax())]==annotations_by_id[rows[i]['sample_id']] for i,r in enumerate(rows) if r['case']==case) for case in ('pool-03','pool-08')),
            pairs=pair_results,native_vs_compose64_argmax_disagreements=rounding_disagreements,native_vs_compose64_max_abs=native_compose_error)
    payload,warnings=build_trace(out,contract,records,args.source_revision)
    for sample,record in zip(payload['samples'],records):
        require(sample['sample_id']==record['id'],'Trace record identity differs')
        actual={**record['encoder'],**record['head']}
        require({r['id'] for r in sample['representations']}==set(actual),'Trace applicable field inventory differs')
        for name,value in actual.items():
            projection=next(r for r in sample['representations'] if r['id']==name)['value_capture']['reader_projection']
            require(projection['implementation_sha256']==[sha(HERE/'worker.py'),sha(HERE/'trace_tools.py')],'Reader source identity differs')
            coordinates=[[0]*value.ndim,[n-1 for n in value.shape]]
            seen=read_value(out,sample,name,coordinates)
            expected=[value[tuple(c)].item() for c in coordinates]
            expected=[dict(real=v.real,imag=v.imag) if isinstance(v,complex) else v for v in expected]
            require(seen==expected,'Coordinate reader differs');reader_checks+=2
    require(calls==dict(encoder=6,head_capture=18,head_replay=18,anchor=2),'Forward budget differs')
    size=sum(p.stat().st_size for p in out.rglob('*') if p.is_file());require(size<256*1024**2,'Private output budget exceeded')
    report=dict(schema='qattn.semantic-forward.result.v1',status='COMPLETE_EXPLORATORY',scope=contract['scope'],contract_sha256=sha(HERE/'contract.json'),source_revision=args.source_revision,
        inputs_sha256=sha(args.inputs),annotations_sha256=sha(args.annotations),models=reports,forward_calls=calls,anchors=anchors,
        validation=dict(fresh_head_array_comparisons=array_checks,coordinate_reads=reader_checks,encoder_word_mean_and_cast_replay=True,structural_trace_errors=0,trace_warnings=warnings),
        seconds=time.monotonic()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,private_bytes_before_report=size,
        training=False,test_loaded=False,l5_accepted=False,limits=contract['limits'])
    save(out/'private-report.json',report)
    # Only compact aggregates may be proposed for Git after a separate export audit.
    compact={k:v for k,v in report.items() if k not in ('models','anchors')}
    compact['models']={m:{k:v for k,v in r.items() if k!='pairs'} for m,r in reports.items()}
    save(out/'compact-report.json',compact)
    files={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}
    save(out/'RUN_COMPLETE',dict(schema='qattn.semantic-forward.closed.v1',files=files))
    print(json.dumps(dict(status='COMPLETE_EXPLORATORY',forward_calls=calls,models=compact['models'],seconds=report['seconds'],l5_accepted=False)),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    for action in ('run','preflight'):
        q=sub.add_parser(action);q.add_argument('--inputs',type=Path,required=True);q.add_argument('--annotations',type=Path,required=True)
        if action=='run':q.add_argument('--run-dir',type=Path,required=True);q.add_argument('--source-revision',required=True)
    q=sub.add_parser('value');q.add_argument('--run-dir',type=Path,required=True);q.add_argument('--sample',required=True);q.add_argument('--representation',required=True);q.add_argument('--coords',required=True)
    args=p.parse_args()
    if args.action=='run':run(args)
    elif args.action=='preflight':
        _,_,_,ready=check(json.loads((HERE/'contract.json').read_bytes()),args.inputs,args.annotations);print(json.dumps(dict(status='PREFLIGHT_PASS_NO_MODEL_FORWARD',**ready)))
    else:
        root=args.run_dir.resolve();payload=json.loads((root/'sample-trace.json').read_bytes());sample=next(s for s in payload['samples'] if s['sample_id']==args.sample)
        print(json.dumps(read_value(root,sample,args.representation,json.loads(args.coords))))
if __name__=='__main__':main()
