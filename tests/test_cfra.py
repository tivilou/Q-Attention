import copy
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from q_attention.adapters.q_cfra import CFRAModel,neutralize,objective
from q_attention.models.relation_transformer import RelationExtractionModel,RelationTransformerConfig
from q_attention.plugins.q_cfra import CounterfactualFidelityAttention,MODES,fidelity,rotation
from q_attention.analysis.cfra_trace import validate_trace,inventory,Trace,values,vector,state_sha
from q_attention.tasks.relation import RelationRecord


def fixture():
    torch.manual_seed(27)
    base = RelationExtractionModel(RelationTransformerConfig(vocab_size=24,num_labels=3,dim=8,
        num_heads=2,num_layers=2,ff_dim=16,dropout=0,max_length=8))
    ids = torch.tensor([[2,3,4,5,0],[6,7,8,0,0]])
    mask = ids != 0
    subject = torch.tensor([[1,0,0,0,0],[1,0,0,0,0]],dtype=torch.bool)
    obj = torch.tensor([[0,0,1,0,0],[0,1,0,0,0]],dtype=torch.bool)
    return base,dict(input_ids=ids,attention_mask=mask,subject_mask=subject,object_mask=obj,
                     neutral_ids=torch.tensor([[20,21],[20,22]]))


def test_aligned_type_views_are_target_free():
    _,batch = fixture()
    v = neutralize(batch['input_ids'],batch['attention_mask'],batch['subject_mask'],batch['object_mask'],batch['neutral_ids'])
    assert torch.equal(v[:,0],batch['input_ids'])
    assert torch.equal(v[:,1][~batch['subject_mask']],batch['input_ids'][~batch['subject_mask']])
    assert torch.equal(v[:,2][~batch['object_mask']],batch['input_ids'][~batch['object_mask']])
    bad = dict(batch); bad['subject_mask']=batch['object_mask']
    with pytest.raises(ValueError,match='overlapping'):
        neutralize(bad['input_ids'],bad['attention_mask'],bad['subject_mask'],bad['object_mask'],bad['neutral_ids'])


def test_disabled_is_exact_unmodified_model():
    base,batch = fixture()
    expected = base(**{k:v for k,v in batch.items() if k!='neutral_ids'})
    disabled = CFRAModel(copy.deepcopy(base),'disabled')
    actual = disabled(**batch)[:,0]
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


@pytest.mark.parametrize('mode',MODES[1:])
def test_modes_have_active_finite_parameters_and_masked_full_attention(mode):
    base,batch = fixture()
    torch.manual_seed(101)
    model = CFRAModel(base,mode)
    logits = model(**batch,capture=True)
    loss,_ = objective(logits,model.attention,batch['attention_mask'],torch.tensor([0,1]))
    loss.backward()
    for name,p in model.kernel.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
        assert p.grad.norm()>0,name
    assert int(torch.count_nonzero(model.kernel.readout.grad))==12
    assert model.kernel.channel.grad.abs().min()>1e-10
    assert torch.allclose(model.attention.sum(-1),torch.ones_like(model.attention.sum(-1)))
    mask = batch['attention_mask'][:,None,None,None,:].expand_as(model.attention)
    assert torch.count_nonzero(model.attention[~mask])==0
    assert not torch.allclose(model.observed['attention_logits'],model.observed['base_scores'])


def test_controls_match_trainable_budget():
    counts=[]
    for mode in MODES[1:]:
        base,_=fixture(); model=CFRAModel(base,mode)
        counts.append(sum(p.numel() for p in model.parameters() if p.requires_grad))
    assert len(set(counts))==1


@pytest.mark.parametrize('mode',['quantum','diagonal','shuffled','entity_permuted'])
def test_cptp_fidelity_and_circuit_against_independent_spectral_reference(mode):
    base,batch=fixture(); model=CFRAModel(base,mode)
    model(**batch,capture=True)
    state = model.observed['plugin_density']
    torch.testing.assert_close(state,state.mH)
    torch.testing.assert_close(state.diagonal(dim1=-2,dim2=-1).sum(-1),torch.ones(state.shape[:-2],dtype=state.dtype))
    assert torch.linalg.eigvalsh(state).min()>0
    factual,s,o = model.observed['plugin_paired_density'].unbind(1)
    eigen,vectors = torch.linalg.eigh(factual.double() if not factual.is_complex() else factual.to(torch.complex128))
    sqrt = vectors @ torch.diag_embed(eigen.clamp_min(0).sqrt().to(vectors.dtype)) @ vectors.mH
    for i,other in enumerate((s,o)):
        inner = sqrt @ other.to(sqrt.dtype) @ sqrt
        reference = torch.linalg.eigvalsh(inner).clamp_min(0).sqrt().sum(-1).square().clamp(1e-5,1-1e-5)
        torch.testing.assert_close(model.observed['plugin_fidelity'][...,i].double(),reference,atol=2e-5,rtol=2e-5)
    states=model.observed['plugin_circuit_states']
    torch.testing.assert_close(states.abs().square().sum(-1),torch.ones(states.shape[:-1]),atol=1e-6,rtol=1e-6)
    if mode=='diagonal':
        assert torch.count_nonzero(model.observed['plugin_commutator'])==0


@pytest.mark.parametrize('mode',['quantum','shuffled','entity_permuted'])
def test_sentence_local_controls_and_replay_ignore_batch_partition(mode):
    base,batch=fixture(); model=CFRAModel(base,mode)
    both=model(**batch).detach()
    separate=torch.cat([model(**{k:v[i:i+1] for k,v in batch.items()}) for i in range(2)])
    torch.testing.assert_close(both,separate,atol=2e-7,rtol=1e-6)


def test_zero_projection_still_produces_valid_channel():
    kernel=CounterfactualFidelityAttention(8)
    with torch.no_grad():
        kernel.projection.weight.zero_(); kernel.projection.bias.zero_()
    kernel.geometry(torch.zeros(1,3,4,8),torch.ones(1,4,dtype=torch.bool),True)
    torch.testing.assert_close(kernel.last['density'].diagonal(dim1=-2,dim2=-1).sum(-1),torch.ones(1,3,4,dtype=torch.complex64))


def test_compiled_circuit_matches_gate_reference_states_and_gradients():
    import math
    torch.manual_seed(19)
    kernel=CounterfactualFidelityAttention(8).double()
    scores=torch.randn(1,3,2,3,4,dtype=torch.float64,requires_grad=True)
    geometry=torch.randn(1,4,3,dtype=torch.float64,requires_grad=True)
    actual=kernel(scores,geometry,capture=True)
    actual_states=kernel.last['circuit_states']
    torch.testing.assert_close(kernel(scores,geometry,capture=False),actual,atol=2e-14,rtol=2e-14)
    inputs=torch.cat((scores[...,None],geometry[:,None,None,None,:,:].expand(*scores.shape,3)),-1)
    angles=math.pi*torch.tanh(inputs*inputs.new_tensor([1,.25,.25,1]))
    state=torch.zeros((*scores.shape,2,2),dtype=torch.complex128);state[...,0,0]=1
    states=[state.reshape(*scores.shape,4)]
    def apply(gate,qubit):
        nonlocal state
        state=gate @ state if qubit==0 else state @ gate.transpose(-1,-2)
    for axis,qubit,index in ((1,0,0),(1,1,1),(2,0,2),(2,1,3)):
        apply(rotation(angles[...,index],axis),qubit)
    states.append(state.reshape(*scores.shape,4))
    for depth in range(2):
        for qubit in range(2):
            for axis in (2,1,0):
                apply(rotation(kernel.readout[depth,qubit,axis],axis),qubit)
        state=state*state.real.new_tensor([[1,1],[1,-1]])
        states.append(state.reshape(*scores.shape,4))
    reference=(state.abs().square().reshape(*scores.shape,4)*scores.new_tensor([3,-1,-1,-1])).sum(-1)/.7
    torch.testing.assert_close(actual_states,torch.stack(states,-2),atol=2e-14,rtol=2e-14)
    torch.testing.assert_close(actual,reference,atol=2e-14,rtol=2e-14)
    weights=torch.randn_like(actual)
    for observed,expected in zip(torch.autograd.grad((actual*weights).sum(),(scores,geometry,kernel.readout),retain_graph=True),
                                 torch.autograd.grad((reference*weights).sum(),(scores,geometry,kernel.readout))):
        torch.testing.assert_close(observed,expected,atol=5e-14,rtol=5e-13)


def test_trace_inventory_is_mode_explicit():
    assert not any(x.startswith('plugin_') for x in inventory('disabled'))
    assert 'plugin_density' in inventory('quantum')
    assert 'plugin_density' not in inventory('classical_cosine')


def test_real_partial_trace_structure_and_numeric_lineage(tmp_path):
    base,batch=fixture(); model=CFRAModel(base,'quantum')
    batch={k:v[:1] for k,v in batch.items()}
    before=vector(model)
    logits=model(**batch,capture=True)
    gold=torch.tensor([0])
    loss,parts=objective(logits,model.attention,batch['attention_mask'],gold)
    loss.backward()
    experiment=dict(run_id='unit',seed=13,model_identity='quantum',config_sha256='a'*64,
        code_revision='b'*64,dataset='public unit fixture',dataset_identity='c'*64)
    writer=Trace(tmp_path,'quantum',experiment,{'relation0':0,'relation1':1,'relation2':2},
                 [dict(name=n,shape=list(p.shape),numel=p.numel()) for n,p in model.named_parameters() if p.requires_grad])
    row=RelationRecord(tokens=('alice','joined','company','today'),subject=(0,1),object=(2,3),label='relation0')
    writer.add(row,'train','initial','replay',values(model,gold,parts,before,vector(model,True),before),0,state_sha(model))
    validate_trace(tmp_path/'sample_trace.json',complete=False)


def load_runner():
    path=Path(__file__).resolve().parents[1]/'experiments/run_cfra_toy.py'
    spec=importlib.util.spec_from_file_location('cfra_runner',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('mode',MODES)
def test_microbatch_preserves_logical_objective_and_gradients(mode):
    runner=load_runner()
    base,batch=fixture()
    batch['labels']=torch.tensor([0,1])
    torch.manual_seed(101)
    whole=CFRAModel(copy.deepcopy(base),mode)
    # Construct each adapter independently because the hook owns an explicit ref.
    torch.manual_seed(101)
    micro=CFRAModel(copy.deepcopy(base),mode)
    logits=whole(**runner.inputs(batch))
    loss,parts=objective(logits,whole.attention,batch['attention_mask'],batch['labels'])
    loss.backward()
    actual=runner.logical_backward(micro,batch,dict(prediction_js=.05,attention_js=.05),1,capture=True)
    torch.testing.assert_close(actual,parts,atol=2e-7,rtol=2e-6)
    torch.testing.assert_close(vector(micro,True),vector(whole,True),atol=3e-7,rtol=2e-5)


def test_canary_resume_exact_and_private_value_tampering_rejected(tmp_path):
    runner=load_runner()
    data=tmp_path/'data'; data.mkdir()
    for split in ('train','valid','test'):
        rows=[]
        for i in range(12):
            rows.append(dict(tokens=['alice',('joined','lives','founded')[i%3],f'org{i}',split],subject=[0,1],object=[2,3],label=str(i%3),metadata=dict(subj_type='PERSON',obj_type='ORGANIZATION',id=f'{split}-{i}')))
        (data/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    a,b=tmp_path/'uninterrupted',tmp_path/'resumed'
    runner.main(['--data-dir',str(data),'--output-dir',str(a),'--canary'])
    for index,mode in enumerate(MODES):
        command=['--data-dir',str(data),'--output-dir',str(b),'--canary','--stop-after-epoch','1']
        if index:
            command.append('--resume')
        runner.main(command)
        partial=b/'seed13'/mode/'latest.pt'
        state=torch.load(partial,weights_only=False)
        assert state['next_epoch']==1
        assert not (partial.parent/'RUN_COMPLETE.json').exists()
        # Active plugin Adam moments must exist before the interruption.
        if mode!='disabled':
            assert len(state['optimizer']['state'])>2
        original=partial.read_bytes()
        state['phase_identity']['mode']='wrong-condition'
        torch.save(state,partial)
        with pytest.raises(ValueError,match='phase identity'):
            runner.main(['--data-dir',str(data),'--output-dir',str(b),'--canary','--resume'])
        partial.write_bytes(original)
    runner.main(['--data-dir',str(data),'--output-dir',str(b),'--canary','--resume'])
    def exact(left,right):
        if isinstance(left,torch.Tensor):
            torch.testing.assert_close(left,right,rtol=0,atol=0)
        elif isinstance(left,dict):
            assert left.keys()==right.keys()
            for key in left:
                exact(left[key],right[key])
        elif isinstance(left,(tuple,list)):
            assert len(left)==len(right)
            for x,y in zip(left,right):
                exact(x,y)
        else:
            assert left==right
    for mode in MODES:
        x=torch.load(a/'seed13'/mode/'latest.pt',weights_only=False)
        y=torch.load(b/'seed13'/mode/'latest.pt',weights_only=False)
        exact(x['model'],y['model'])
        exact(x['optimizer'],y['optimizer'])
        exact(x['rng'],y['rng'])
        assert x['updates']==y['updates']
        validate_trace(b/'seed13'/mode/'sample_trace.json')
    path=b/'seed13/quantum/sample_trace.json'
    doc=json.loads(path.read_text())
    array_file=path.parent/doc['samples'][0]['representations'][0]['artifact_refs'][0]['path']
    original=array_file.read_bytes(); array_file.write_bytes(original+b'changed')
    with pytest.raises(ValueError,match='mutated'):
        validate_trace(path)
