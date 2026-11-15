"""Complete numeric DAG for an intermediate attention transplant, not a model."""
import copy,hashlib,json
from pathlib import Path
import numpy as np
import allocation_trace as base

def capture(root,payload,expected):
    root=Path(root);lookup={(s['roles']['provenance']['condition'],s['roles']['source_sample']['id']):s for s in payload['samples']}
    ids=sorted({sid for _,sid in lookup});models=sorted({m for m,_ in lookup});records=[];checked=0;maximum=0.
    directory=root/'interventions';directory.mkdir(exist_ok=False)
    for sid in ids:
        for control in [m for m in models if m!='quantum']:
            for donor,recipient in [(control,'quantum'),('quantum',control)]:
                d=lookup[(donor,sid)];r=lookup[(recipient,sid)]
                dreps={a['id']:a for a in d['representations']};rreps={a['id']:a for a in r['representations']}
                def array(rep):
                    with np.load(base.pinned(root,rep['artifact_refs'][0]),allow_pickle=False) as f:return f[rep['id']].copy()
                # The producer function receives neither labels nor outcomes.
                alpha=array(dreps['alpha']);u=array(rreps['utilities']);bias=array(rreps['bias'])
                logits=(np.ascontiguousarray(alpha,dtype=np.float64)*np.ascontiguousarray(u,dtype=np.float64)).sum(-1)+bias
                wanted=expected[(donor,recipient,sid)]
                maximum=max(maximum,float(np.max(np.abs(logits[0]-wanted))))
                np.testing.assert_array_equal(logits[0],wanted)
                assert int(logits.argmax())==int(wanted.argmax())
                name=donor+'-into-'+recipient+'-'+sid+'.npz';path=directory/name
                np.savez(path,logits=logits);ref=base.ref(root,path)
                reps=[]
                for key in ['h','roles','mask','alpha','utilities','bias']:
                    rep=copy.deepcopy((dreps if key=='alpha' else rreps)[key]);rep['stage_id']='donor' if key=='alpha' else ('inputs' if key in ['h','roles','mask'] else 'recipient')
                    rep['value_capture']['reader_projection']['reader_ref']='experiments/rmea_allocation/runner.py value --trace-kind intervention --sample '+donor+'->'+recipient+':'+sid+' --representation '+key
                    reps.append(rep)
                reps.append(dict(id='logits',stage_id='prediction',status='observed',meaning='Donor attention with fixed recipient utility and bias',
                    shape=list(logits.shape),dtype=str(logits.dtype),axis_semantics=['sample','relation'],artifact_refs=[ref],display=base.display('Transplanted logits'),
                    value_capture=dict(status='complete',shape=list(logits.shape),logical_element_count=int(logits.size),stored_element_count=int(logits.size),artifact_refs=[ref],
                        reader_projection=dict(status='available',kind='coordinate_inspector',coordinate_access='arbitrary',reader_ref='experiments/rmea_allocation/runner.py value --trace-kind intervention --sample '+donor+'->'+recipient+':'+sid+' --representation logits',source_sha256=[ref['sha256']]))))
                stages=[]
                for group,canonical,inputs,outputs in [('inputs','data',['source:'+r['sample_id']],['h','roles','mask']),('donor','scoring',['source:'+d['sample_id']],['alpha']),('recipient','scoring',['source:'+r['sample_id']],['utilities','bias']),('prediction','generation',['alpha','utilities','bias'],['logits'])]:
                    stages.append(dict(stage_id=group,stage=group,canonical_stage=canonical,lane='replay',status='observed',purpose='Frozen-variable transplant '+group,input_refs=inputs,output_refs=outputs,artifact_refs=[ref] if group=='prediction' else list({x['path']:x for a in reps if a['id'] in outputs for x in a['artifact_refs']}.values()),target_access='not_available',display=base.display(group)))
                record=dict(sample_id=donor+'->'+recipient+':'+sid,lane='replay',split='valid',checkpoint='selected',roles=dict(source_sample=dict(id=sid),sequence_or_features=dict(boundary='frozen-feature transplant'),task_objects=dict(subject_role=2,object_role=3),prediction=dict(value=int(logits.argmax())),provenance=dict(donor_case=d['sample_id'],recipient_case=r['sample_id'],donor_checkpoint=d['roles']['provenance']['checkpoint'],recipient_checkpoint=r['roles']['provenance']['checkpoint'])),representations=reps,stages=stages)
                # Independent reader and reconstruction check against the stored
                # original-head sources, not a detached summary number.
                sampled={}
                for rep in reps:
                    actual=array(rep);assert list(actual.shape)==rep['shape'] and str(actual.dtype)==rep['dtype']
                    coords=[list(map(int,np.unravel_index(k,actual.shape))) for k in sorted({0,actual.size//2,actual.size-1})]
                    values=base.read_value(root,record,rep['id'],coords)
                    assert all(v==actual[tuple(c)].item() for c,v in zip(coords,values))
                    sampled[rep['id']]=actual;checked+=1
                reconstructed=np.einsum('brt,brt->br',sampled['alpha'].astype(np.float64),sampled['utilities'].astype(np.float64))+sampled['bias']
                np.testing.assert_allclose(reconstructed,sampled['logits'],atol=1e-12,rtol=1e-12)
                records.append(record)
    result=dict(schema_version='sample-trace.v2',trace_id='allocation-transplants',trace_completeness='complete',experiment=dict(payload['experiment'],model_identity='Intermediate-variable transplants of pinned original heads'),
        sample_selection=dict(selected_count=len(records),selected_sample_ids=[s['sample_id'] for s in records],rule=payload['sample_selection']['rule'],population_scope=payload['sample_selection']['population_scope']),
        task_contract=dict(task_type='frozen-feature intermediate allocation replay',required_roles=['source_sample','sequence_or_features','task_objects','prediction','provenance'],required_representations=['h','roles','mask','alpha','utilities','bias','logits'],required_value_representations=['h','roles','mask','alpha','utilities','bias','logits'],required_splits=['valid'],required_checkpoints=['selected']),coverage=dict(payload['coverage']),samples=records)
    errors,warnings=base.validate(result,require_complete=True);assert not errors,errors
    (root/'intervention-trace.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
    return dict(structural='pass',source_lineage='pass',independent_formula='pass',coordinate_reader='pass',model_cases=len(records),arrays=checked,cohort_logit_max_residual=maximum,warnings=warnings,scope='Existing original-head traces plus explicitly executed transplants; no target input or causal training claim.')
