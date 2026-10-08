"""Lightweight driver/trace contracts; not a successful neural runtime claim."""
import ast
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from baselines.retacred import metaentail_runtime_fidelity as r
from baselines.retacred.common import select_models

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('mode,minimum',[('tiny',768*1024**2),('full',24*1024**3)])
def test_declared_budget_and_precision_not_author_recipe(mode,minimum):
    value=r.description(mode,'cpu')
    assert value['minimum_host_headroom_bytes']==minimum
    assert value['optimizer_updates']==2 and value['replayed_updates']==1
    assert value['pairs_per_selected_group']==3
    assert value['formal_training'] is False and value['formal_comparison_admitted'] is False
    assert value['optimizer_claim']=='diagnostic_only_not_author_recipe'
    assert value['rtol']==value['atol']==0 and value['precision']=='FP32'


@pytest.mark.parametrize('mode,device',[('fast','cpu'),('full','gpu'),(None,'cpu')])
def test_invalid_description(mode,device):
    with pytest.raises(ValueError): r.description(mode,device)


def test_synthetic_train_filter_and_evaluation_identity_are_distinct():
    from baselines.retacred.metaentail import fit_type_filter,inference_group,training_group
    rows,test=r.synthetic_records()
    assert len({row['id'] for row in rows+[test]})==4
    filter=fit_type_filter(rows,split='train')
    assert len(inference_group(test,filter)['candidates'])==3
    assert [row['nli_target'] for row in training_group(rows[1],filter)['candidates']]==[
        'contradiction','entailment','neutral']


def metadata(phase='train',pairs=3,layers=2):
    keys=['task_input.input_ids','task_input.attention_mask','task_output.logits','task_output.probabilities',
          'functional.metaentail_cls_readout.input','functional.metaentail_cls_readout.output',
          'functional.metaentail_classifier_tanh.input','functional.metaentail_classifier_tanh.output',
          'classifier.dense.call0.input.0','classifier.dense.call0.output.tensor',
          'classifier.out_proj.call0.input.0','classifier.out_proj.call0.output.tensor']
    keys += ['model.hidden_states.layer'+str(i) for i in range(layers+1)]
    keys += ['model.attentions.layer'+str(i) for i in range(layers)]
    keys += [f'functional.attention_scores_to_probabilities.call{i}.{suffix}'
             for i in range(layers) for suffix in ('input','output')]
    result={key:{'shape':[pairs,3],'dtype':'torch.float32','numel':pairs*3,
                 'value_storage':'complete_numeric_tensor'} for key in keys}
    if phase=='train':
        result.update({'task_input.labels':{'shape':[pairs]},'task_output.loss':{'shape':[]},
                       'activation.gradient':{'shape':[pairs,3]},
                       'classifier.parameter_gradient.out_proj.weight':{'shape':[3,32]}})
    return {'tensors':result}


@pytest.mark.parametrize('phase',['train','test'])
def test_trace_validator_requires_all_candidate_pairs_and_states(phase):
    r.validate_trace(metadata(phase),phase=phase,pairs=3,layers=2)


@pytest.mark.parametrize('key',['model.hidden_states.layer0','model.hidden_states.layer2','model.attentions.layer1',
                               'functional.metaentail_cls_readout.output','functional.metaentail_classifier_tanh.output',
                               'classifier.out_proj.call0.output.tensor','functional.attention_scores_to_probabilities.call0.input'])
def test_missing_embedding_attention_readout_or_head_not_complete(key):
    value=metadata(); value['tensors'].pop(key)
    with pytest.raises(AssertionError,match='absent'): r.validate_trace(value,phase='train',pairs=3,layers=2)


@pytest.mark.parametrize('kind',['summary','empty','only_one_pair','wrong_head','no_target','no_loss','no_activation_grad','no_parameter_grad'])
def test_placeholder_and_partial_group_training_trace_rejected(kind):
    value=metadata(); fields=value['tensors']
    if kind=='summary': fields['model.hidden_states.layer0']['value_storage']='shape_only'
    elif kind=='empty': fields['model.attentions.layer0']['numel']=0
    elif kind=='only_one_pair': fields['task_output.logits']['shape']=[1,3]
    elif kind=='wrong_head': fields['task_output.logits']['shape']=[3,40]
    else:
        key={'no_target':'task_input.labels','no_loss':'task_output.loss','no_activation_grad':'activation.gradient',
             'no_parameter_grad':'classifier.parameter_gradient.out_proj.weight'}[kind]
        fields.pop(key)
    with pytest.raises(AssertionError): r.validate_trace(value,phase='train',pairs=3,layers=2)


@pytest.mark.parametrize('field',['task_input.labels','task_output.loss'])
def test_evaluation_gold_target_tensor_rejected(field):
    value=metadata('test'); value['tensors'][field]={'shape':[3]}
    with pytest.raises(AssertionError,match='gold-dependent'): r.validate_trace(value,phase='test',pairs=3,layers=2)


def test_blocked_memory_writes_honest_evidence_before_backend_or_weights(tmp_path,monkeypatch,capsys):
    def gate(minimum):
        assert minimum==24*1024**3
        raise RuntimeError('fixture memory headroom insufficient')
    monkeypatch.setattr(r,'require_runtime_memory',gate)
    monkeypatch.setattr(r,'make_adapter',lambda *args:pytest.fail('adapter entered despite memory gate'))
    output=tmp_path/'blocked'
    assert r.main(['--mode','full','--output-dir',str(output)])==2
    value=json.loads((output/'runtime-evidence.json').read_text())
    assert value['status']=='pending_memory_gate' and value['backend_imported'] is False and value['weights_read'] is False
    assert json.loads(capsys.readouterr().out)==value


def test_existing_and_symlinked_output_is_preserved(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'require_runtime_memory',lambda *args:pytest.fail('output validation did not run first'))
    output=tmp_path/'keep'; output.mkdir(); data=output/'original'; data.write_text('keep')
    with pytest.raises(SystemExit): r.main(['--output-dir',str(output)])
    assert data.read_text()=='keep'
    linked=tmp_path/'alias'; linked.symlink_to(output,target_is_directory=True)
    with pytest.raises(ValueError,match='symlink'): r.main(['--output-dir',str(linked)])


def test_read_only_describe_no_backend_network_weights_or_output(tmp_path):
    script=r'''
import pathlib,sys
def audit(event,args):
 if event=='import' and args[0].split('.')[0] in ('torch','tensorflow','jax','numpy'): raise AssertionError('backend imported')
 if event.startswith('socket.'): raise AssertionError('network used')
 if event=='open' and isinstance(args[0],str) and pathlib.Path(args[0]).suffix in ('.bin','.safetensors','.pt'): raise AssertionError('weights opened')
sys.addaudithook(audit)
from baselines.retacred.metaentail_runtime_fidelity import main
raise SystemExit(main(['--mode','full','--describe','--output-dir',sys.argv[1]]))
'''
    output=tmp_path/'must-not-exist'
    done=subprocess.run([sys.executable,'-c',script,str(output)],cwd=ROOT,text=True,capture_output=True,
                        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),timeout=20)
    assert done.returncode==0,done.stderr
    assert json.loads(done.stdout)['optimizer_updates']==2 and not output.exists()


def test_portable_shell_exact_entry_from_unrelated_cwd_and_relocated_repo(tmp_path):
    script=ROOT/'scripts/check_retacred_metaentail_runtime.sh'
    env=dict(os.environ,PYTHON_BIN=sys.executable,PYTHONDONTWRITEBYTECODE='1')
    done=subprocess.run(['bash',str(script),'--mode','tiny','--describe'],cwd=tmp_path,env=env,
                        text=True,capture_output=True,timeout=20)
    assert done.returncode==0,done.stderr
    assert json.loads(done.stdout)['mode']=='tiny'
    relocated=tmp_path/'moved-project'; shutil.copytree(ROOT/'baselines',relocated/'baselines',ignore=shutil.ignore_patterns('__pycache__'))
    (relocated/'scripts').mkdir(); shutil.copyfile(script,relocated/'scripts'/script.name)
    done=subprocess.run(['bash',str(relocated/'scripts'/script.name),'--describe'],cwd=tmp_path,env=env,
                        text=True,capture_output=True,timeout=20)
    assert done.returncode==0,done.stderr
    assert json.loads(done.stdout)['mode']=='full'


def test_actual_driver_has_native_and_resume_comparisons_without_formal_admission():
    source=Path(r.__file__).read_text(); tree=ast.parse(source)
    assert all(marker in source for marker in ('restore_checkpoint(checkpoint,rejected',
        "'next_expanded_pair'",'resumed!=uninterrupted','gradient_identity(model)!=expected_gradients',
        'capture_group(adapter,len(evaluated',"weights_only=True",'state_identity(reloaded)!=state_identity(capture.values)'))
    assert not any(isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
                   and node.func.attr in ('kill','killpg','download') for node in ast.walk(tree))
    with pytest.raises(ValueError): select_models('metaentail')
    assert len(select_models('ready'))==5
