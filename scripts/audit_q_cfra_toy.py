"""Private-run audit; export only aggregate metrics, counts and integrity receipts."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import statistics
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('run')
parser.add_argument('output')
parser.add_argument('--project-root',required=True)
parser.add_argument('--recovery-inspection')
args=parser.parse_args()
project=Path(args.project_root).resolve()
sys.path.insert(0,str(project/'src'))
from q_attention.analysis.cfra_trace import validate_trace,sha
root=Path(args.run).resolve()
manifest=json.loads((root/'run_manifest.json').read_text())
summary=json.loads((root/'summary.json').read_text())
complete=json.loads((root/'RUN_COMPLETE.json').read_text())
assert complete['contract']==manifest['contract']==summary['contract']
assert complete['summary_sha256']==sha(root/'summary.json')
recovery_integrity=dict(status='not_provided')
if args.recovery_inspection:
    recovery=json.loads(Path(args.recovery_inspection).read_text())
    assert recovery['contract']==manifest['contract']
    for relative,expected in recovery['completion_marker_sha256'].items():assert sha(root/relative)==expected
    recovery_integrity=dict(status='passed',preserved_completed_jobs=len(recovery['completion_marker_sha256']),
        receipt_sha256=sha(Path(args.recovery_inspection)))
spec=importlib.util.spec_from_file_location('cfra_runner',project/'experiments/run_cfra_toy.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
runner.torch.set_num_threads(1)
assert runner.source_identity()==manifest['contract']['source_sha256']
assert runner.d.digest(manifest['config'])==manifest['contract']['config_sha256']
assert runner.d.digest(manifest['subset'])==manifest['contract']['subset_sha256']
data_identity={}
for split in ('train','valid','test'):
    identity=manifest['subset'][split]
    source=Path(manifest['data_dir'])/(split+'.jsonl')
    assert sha(source)==identity['source_sha256']
    with source.open() as stream:
        assert sum(bool(line.strip()) for line in stream)==identity['source_count']
    data_identity[split]={key:identity[key] for key in ('source_sha256','source_count','selected_count','supports','unavailable_labels','selected_ids_sha256')}
receipts=[]
matched_inputs={}
case_observability=[]
geometry_diagnostics=[]
mechanism_diagnostics=[]
for seed in manifest['config']['seeds']:
    base_state=None
    warm=runner.torch.load(root/f'seed{seed}'/'warmup.pt',map_location='cpu',weights_only=False)
    assert warm['phase_identity']==dict(phase='warmup',seed=seed,mode='shared')
    assert warm['contract']==manifest['contract']
    for mode in manifest['config']['modes']:
        directory=root/f'seed{seed}'/mode
        marker=json.loads((directory/'RUN_COMPLETE.json').read_text())
        job=json.loads((directory/'summary.json').read_text())
        root_row=next(row for row in summary['conditions'] if row['seed']==seed and row['mode']==mode)
        assert root_row==job
        assert marker['contract']==job['contract']==manifest['contract']
        assert marker['summary_sha256']==sha(directory/'summary.json')
        assert marker['trace_sha256']==sha(directory/'sample_trace.json')
        phase=job['phase_identity']
        assert phase==marker['phase_identity']
        assert (phase['phase'],phase['seed'],phase['mode'])==('screen',seed,mode)
        base_state=base_state or phase['base_state_sha256']
        assert phase['base_state_sha256']==base_state
        checkpoint=runner.torch.load(directory/'latest.pt',map_location='cpu',weights_only=False)
        assert checkpoint['phase_identity']==phase
        assert checkpoint['contract']==manifest['contract']
        assert checkpoint['next_epoch']==manifest['config']['screen']['epochs']
        assert checkpoint['best_metric']==job['valid']['metrics']['macro_f1']
        assert checkpoint['best_epoch']==job['best_epoch']
        assert checkpoint['updates']==job['updates']
        expected_updates=math.ceil(manifest['subset']['train']['selected_count']/manifest['config']['screen']['batch_size'])*manifest['config']['screen']['epochs']
        assert job['updates']==expected_updates
        assert job['plugin_parameters']==(0 if mode=='disabled' else 4*(manifest['config']['model']['dim']+1)+6+12)
        if mode=='disabled':classifier_parameters=job['trainable_parameters']
        assert job['trainable_parameters']==classifier_parameters+job['plugin_parameters']
        for split in ('valid','test'):
            report=job[split]
            classes=list(report['per_class'].values())
            # Reports use the fixed train label vocabulary, including classes
            # with zero support after the outcome-blind held-out filtering.
            expected_supports={label:manifest['subset'][split]['supports'].get(label,0)
                for label in manifest['subset']['train']['supports']}
            assert len(classes)==len(expected_supports)
            assert sum(c['support'] for c in classes)==manifest['subset'][split]['selected_count']
            assert sum(c['predicted_support'] for c in classes)==manifest['subset'][split]['selected_count']
            assert {c['label']:c['support'] for c in classes}==expected_supports
            for c in classes:
                tp,fp,fn=c['true_positive'],c['false_positive'],c['false_negative']
                assert c['support']==tp+fn and c['predicted_support']==tp+fp
                precision=tp/(tp+fp) if tp+fp else 0.
                recall=tp/(tp+fn) if tp+fn else 0.
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
                for field,value in (('precision',precision),('recall',recall),('f1',f1)):
                    assert math.isclose(c[field],value,abs_tol=1e-14)
            assert math.isclose(report['metrics']['macro_f1'],statistics.mean(c['f1'] for c in classes),abs_tol=1e-14)
            assert math.isclose(report['metrics']['accuracy'],sum(c['true_positive'] for c in classes)/sum(c['support'] for c in classes),abs_tol=1e-14)
        readiness=validate_trace(directory/'sample_trace.json')
        assert readiness['status']=='passed'
        trace=json.loads((directory/'sample_trace.json').read_text())
        for sample in trace['samples']:
            roles={key:sample['roles'][key] for key in ('sentence','tokens','subject','object','correct_relation')}
            input_hash=runner.d.digest(roles)
            key=sample['sample_id']
            assert matched_inputs.setdefault(key,input_hash)==input_hash,'Unmatched source cases'
            reps={rep['id']:rep for rep in sample['representations']}
            if seed==13 and mode in ('disabled','quantum') and (sample['checkpoint'],sample['lane']) in (('update0','training'),('best','replay')):
                case_observability.append(dict(seed=seed,mode=mode,sample_id=key,split=sample['split'],
                    checkpoint=sample['checkpoint'],lane=sample['lane'],input_role_sha256=input_hash,
                    token_count=len(roles['tokens']),trace_sha256=marker['trace_sha256'],
                    representations={name:dict(shape=rep['shape'],dtype=rep['dtype']) for name,rep in reps.items()},
                    raw_input_and_prediction='captured privately; public projection intentionally absent'))
            if mode=='quantum':
                def array(name):
                    return np.load(directory/reps[name]['artifact_refs'][0]['path'],allow_pickle=False)
                mask=array('attention_mask').astype(bool)
                f=array('plugin_fidelity')[mask]
                c=array('plugin_covariance')[mask]
                commutator=array('plugin_commutator')[mask]
                geometry_diagnostics.append(dict(seed=seed,sample_id=key,fidelity_min=float(f.min()),fidelity_max=float(f.max()),
                    fidelity_mean=float(f.mean()),fidelity_clamped_upper_fraction=float((f>=1-1.01e-5).mean()),
                    covariance_abs_mean=float(np.abs(c).mean()),commutator_mean=float(commutator.mean())))
                if sample['lane']=='training':
                    gradient=array('parameter_gradient')
                    groups={'projection':[],'channel':[],'readout':[]}
                    cursor=0
                    for item in trace['parameter_order']:
                        part=gradient[cursor:cursor+item['numel']];cursor+=item['numel']
                        for group in groups:
                            if item['name'].startswith('kernel.'+group):groups[group].extend(part.tolist())
                    assert cursor==gradient.size and all(groups.values())
                    norms={name:float(np.linalg.norm(values)) for name,values in groups.items()}
                    raw=array('base_scores').copy();raw[...,~mask]=-1e30
                    reference=runner.torch.from_numpy(raw).softmax(-1).numpy()
                    movement=float(np.abs(array('attention')-reference)[...,mask,:].sum(-1).max())
                    mechanism_diagnostics.append(dict(seed=seed,plugin_group_gradient_norms=norms,max_valid_attention_l1_shift=movement))
        for label,state in (('best',checkpoint['best']),('final',checkpoint['model'])):
            digest=hashlib.sha256()
            for name,tensor in sorted(state.items()):
                digest.update(name.encode());digest.update(str((tuple(tensor.shape),tensor.dtype)).encode())
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            for sample in trace['samples']:
                if sample['checkpoint']==label:
                    assert sample['roles']['checkpoint_identity']['model_state_sha256']==digest.hexdigest()
        receipts.append(dict(seed=seed,mode=mode,trace_sha256=marker['trace_sha256'],
            summary_sha256=marker['summary_sha256'],**readiness))
assert len(summary['conditions'])==len(receipts)
aggregates={}
base={row['seed']:row for row in summary['conditions'] if row['mode']=='disabled'}
def stat(values):
    return dict(n=len(values),values=values,mean=statistics.mean(values),std=statistics.stdev(values) if len(values)>1 else 0.)
for mode in manifest['config']['modes']:
    rows=[row for row in summary['conditions'] if row['mode']==mode]
    aggregates[mode]=dict(n=len(rows),metrics={name:stat([row['test']['metrics'][name] for row in rows]) for name in ('macro_f1','accuracy','loss')},
        delta_vs_disabled={name:stat([row['test']['metrics'][name]-base[row['seed']]['test']['metrics'][name] for row in rows]) for name in ('macro_f1','accuracy','loss')})
if summary['decision']['stage']=='bounded real-data toy only':
    declared=summary['decision'];paired=aggregates['quantum']['delta_vs_disabled']['macro_f1'];gates=manifest['config']['gates']
    assert declared['paired_deltas']==paired['values']
    assert declared['mean_delta']==paired['mean'] and declared['delta_sample_sd']==paired['std']
    positive=sum(value>0 for value in paired['values'])
    assert positive==declared['positive_seeds']
    utility=paired['mean']>=gates['mean_delta'] and positive>=gates['positive_seeds'] and (paired['std']<=gates['delta_sd'] or paired['std']<=aggregates['disabled']['metrics']['macro_f1']['std'])
    parity=paired['mean']>=gates['mean_delta'] and aggregates['shuffled']['metrics']['macro_f1']['mean']>=aggregates['quantum']['metrics']['macro_f1']['mean']-gates['shuffled_parity_tolerance']
    assert parity==declared['shuffled_parity']
    assert len(mechanism_diagnostics)==len(manifest['config']['seeds'])
    assert mechanism_diagnostics==declared['mechanism_diagnostics']
    collapsed=any(min(x['plugin_group_gradient_norms'].values())<=1e-12 or x['max_valid_attention_l1_shift']<=1e-8 for x in mechanism_diagnostics)
    assert collapsed==declared['collapsed_plugin_or_intervention']
    assert declared['classification']==('kill' if positive<=1 or parity or collapsed else 'promote_to_formal_preparation' if utility else 'inconclusive')
artifact=dict(schema_version='q-attention.cfra-private-run-audit.v1',status='passed',
    audit_script_sha256=sha(Path(__file__)),recovery_integrity=recovery_integrity,
    run_directory=str(root),contract=manifest['contract'],config=manifest['config'],
    implementation_commit=manifest['git_revision'],data_identity=data_identity,
    source_file_sha256={name:sha(project/name) for name in runner.SOURCE_FILES},
    summary_sha256=sha(root/'summary.json'),run_manifest_sha256=sha(root/'run_manifest.json'),
    completion_sha256=sha(root/'RUN_COMPLETE.json'),
    jobs=len(receipts),sample_trace_records=sum(x['samples'] for x in receipts),
    full_value_artifacts=sum(x['full_value_artifacts'] for x in receipts),receipts=receipts,
    decision=summary['decision'],aggregates=aggregates,
    matched_case_inputs=dict(status='passed',case_ids=len(matched_inputs),input_role_sha256=matched_inputs),
    case_observability=case_observability,geometry_diagnostics=geometry_diagnostics,
    diagnostic_scope='Predeclared selected-case captures only; repeated split/checkpoint observations are not independent population evidence.',
    conditions=[{k:row[k] for k in ('seed','mode','best_epoch','updates','trainable_parameters','plugin_parameters','valid','test')} for row in summary['conditions']],
    privacy='No raw sample text, token IDs, activations, predictions, gradients or checkpoints exported.',
    claim_ceiling='Numerical canary or bounded toy only; no full-data L1/L2 or quantum advantage.')
Path(args.output).write_text(json.dumps(artifact,indent=2),encoding='utf-8')
print(json.dumps({k:artifact[k] for k in ('status','jobs','sample_trace_records','full_value_artifacts','decision')}))
