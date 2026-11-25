from pathlib import Path
import hashlib,json,sys
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def display(s):return dict(summary=s,default_view='human_summary',visible_fields=[],visualizations=[])
def write_case(root,name,arrays):
    root=Path(root);p=root/(name+'.values.json')
    payload={k:dict(shape=list(a.shape),dtype=str(a.dtype),values=a.tolist()) for k,a in arrays.items()}
    p.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    ref=dict(path=p.name,sha256=sha(p),bytes=p.stat().st_size,visibility='portal')
    axes=dict(inputs=['step','unit'],preliminary=['step','unit'],memory_before=['step','row','column'],
        preLN=['step','unit'],centered=['step','unit'],variance=['step'],hidden=['step','unit'],memory_after=['step','row','column'],
        gamma=['unit'],beta=['unit'],initial_hidden=['unit'],initial_memory=['row','column'],final_readout=['singleton'])
    groups=dict(data=[],scoring=[],generation=[],diagnosis=[]);reps=[]
    for k,a in arrays.items():
        g='data' if k in ('inputs','gamma','beta','initial_hidden','initial_memory') else 'generation' if k=='hidden' else 'diagnosis' if k=='final_readout' else 'scoring'
        groups[g].append(k)
        reps.append(dict(id=k,stage_id=g,status='observed',meaning=k+' complete Eq1/Eq5 equation-simulation value',shape=list(a.shape),dtype=str(a.dtype),axis_semantics=axes[k],
            artifact_refs=[ref],display=display(k),value_capture=dict(status='complete',shape=list(a.shape),logical_element_count=int(a.size),stored_element_count=int(a.size),artifact_refs=[ref],
            reader_projection=dict(status='available',kind='coordinate_inspector',coordinate_access='arbitrary',reader_ref='fw_trace.py read_value '+name+' '+k,source_sha256=[ref['sha256']]))))
    links=dict(data=['source:frozen_synthetic_sequence','external:conference_Eq1_Eq5'],scoring=['inputs','gamma','beta','initial_hidden','initial_memory'],generation=['preLN','centered','variance','gamma','beta'],diagnosis=['hidden'])
    stages=[dict(stage_id=g,stage=g,canonical_stage=g,lane='replay',status='observed',step=j,purpose=g+' for declared synthetic equation instantiation',
        input_refs=links[g],output_refs=keys,artifact_refs=[ref],display=display(g),target_access='not_available') for j,(g,keys) in enumerate(groups.items())]
    return dict(sample_id=name,lane='replay',split='synthetic',checkpoint='fixed_equations',roles=dict(
        source_sample=dict(sequence=name,origin='Frozen symbolic boundary vectors, not NLP data'),
        protocol=dict(memory_eta=.5,memory_decay=.95,inner_steps=1,layernorm_epsilon=1e-5,W='zero',C='identity',gain='sqrt(1+epsilon)/sqrt7',bias=0),
        output=dict(readout='shared e2-e0',value=float(arrays['final_readout'][0])),
        provenance=dict(reader_source_sha256=sha(__file__),source='Conference SHA028b1b47 p3 Eq1/2/3/4 p4 Eq5; not original code or paper results')),
        stages=stages,representations=reps)

def read_value(root,sample,key,coords):
    root=Path(root).resolve();rep=next(r for r in sample['representations'] if r['id']==key);ref=rep['artifact_refs'][0];p=root/ref['path']
    assert p.resolve().is_relative_to(root) and not p.is_symlink() and sha(p)==ref['sha256']
    data=json.loads(p.read_bytes())[key];a=np.asarray(data['values'],dtype=data['dtype']);assert list(a.shape)==rep['shape'] and np.isfinite(a).all()
    assert isinstance(coords,list) and 1<=len(coords)<=64
    out=[]
    for c in coords:
        assert len(c)==a.ndim and all(type(i) is int and 0<=i<n for i,n in zip(c,a.shape));out.append(a[tuple(c)].item())
    return out

def close(root,samples,scope_sha):
    root=Path(root);sys.path.insert(0,str(root/'source'));from validate_sample_trace_v2 import validate,STAGES
    required=[r['id'] for r in samples[0]['representations']]
    payload=dict(schema_version='sample-trace.v2',trace_id='source-bound-fastweights-equation-witness',trace_completeness='complete',
        experiment=dict(run_id='native-fastweights-class-audit-20261126',dataset='Synthetic equation instantiation only',
            code_revision=sha(Path(__file__).with_name('audit.py')),config_sha256=scope_sha,model_identity='Conference Eq1/Eq5 fixed parameters; no trained NLP model',seed=None),
        sample_selection=dict(rule='Both complete two-write/one-read sequences frozen before execution',population_scope='Two synthetic sequences; not task population or trained-model comparison',selected_count=2,selected_sample_ids=[s['sample_id'] for s in samples]),
        task_contract=dict(task_type='FastWeights equation simulation with nonnegative self-consistent history',required_roles=['source_sample','protocol','output','provenance'],
            required_representations=required,required_value_representations=required,required_splits=['synthetic'],required_checkpoints=['fixed_equations'],required_lanes=['replay'],checkpoint_policy='Fixed symbolic parameters; no fitting/training checkpoint stage'),
        coverage={s:'observed' if s in ('data','scoring','generation','diagnosis') else 'not_applicable' for s in STAGES},samples=samples)
    errors,warnings=validate(payload,require_complete=True);assert not errors and not warnings,(errors,warnings)
    reads=elements=0
    for sample in samples:
        for rep in sample['representations']:
            data=json.loads((root/rep['artifact_refs'][0]['path']).read_bytes())[rep['id']];a=np.asarray(data['values'],dtype=data['dtype'])
            coords=[list(map(int,np.unravel_index(i,a.shape))) for i in sorted({0,a.size//2,a.size-1})]
            for coord,value in zip(coords,read_value(root,sample,rep['id'],coords)):assert value==a[tuple(coord)].item();reads+=1
            elements+=a.size
    p=root/'sample-trace.v2.json';p.write_text(json.dumps(payload,indent=2)+'\n')
    return dict(schema='sample-trace.v2',sha256=sha(p),cases=2,representations=2*len(required),elements=int(elements),coordinate_reads=reads,errors=errors,warnings=warnings,
                scope='Complete declared synthetic Eq1/Eq5 dynamics only; original paper code/training/NLP evaluation not performed')
