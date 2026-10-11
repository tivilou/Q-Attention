"""Execution-memory equivalence and actual forward evidence, no formal data."""
import copy

import pytest

from baselines.retacred.ctl_drp_neural import joint_model
from baselines.retacred.ctl_drp_trace import CandidateCapture, inventory
from baselines.retacred.ctl_drp_task import LABELS


def encoder():
    import torch
    from transformers import RobertaConfig, RobertaModel
    torch.manual_seed(13)
    config=RobertaConfig(vocab_size=48,hidden_size=24,num_hidden_layers=3,
        num_attention_heads=4,intermediate_size=48,max_position_embeddings=514,
        hidden_dropout_prob=.17,attention_probs_dropout_prob=.13,pad_token_id=1)
    config._attn_implementation='eager'
    return RobertaModel(config,add_pooling_layer=False)


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_checkpointed_full_candidate_loss_gradient_optimizer_and_rng_match(device):
    import torch
    if device.startswith('cuda') and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.set_num_threads(1)
    source=encoder()
    full=joint_model(copy.deepcopy(source),memory_efficient=False).to(device).train()
    lean=joint_model(copy.deepcopy(source),memory_efficient=True).to(device).train()
    assert lean.encoder.is_gradient_checkpointing
    groups=[LABELS[:5],LABELS[:3]]
    ids=torch.tensor([[0,9,12,7,6,5,4,3,2]]*8,device=device)
    mask=torch.ones_like(ids)
    results=[]
    for model in (full,lean):
        torch.manual_seed(71)
        if device.startswith('cuda'):torch.cuda.manual_seed_all(71)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,foreach=False,fused=False)
        optimizer.zero_grad(set_to_none=True)
        out=model(ids,mask,groups,gold=[LABELS[2],LABELS[1]])
        out['loss'].backward()
        grads={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
        rng=torch.cuda.get_rng_state().clone() if device.startswith('cuda') else torch.get_rng_state().clone()
        optimizer.step()
        results.append((out['scores'].detach().cpu(),out['loss'].detach().cpu(),grads,rng,
                        {n:p.detach().cpu().clone() for n,p in model.named_parameters()}))
        if model is lean:assert out['hidden_states'] is None and out['attentions'] is None
    a,b=results
    torch.testing.assert_close(a[0],b[0],rtol=0,atol=0)
    torch.testing.assert_close(a[1],b[1],rtol=0,atol=0)
    assert torch.equal(a[3],b[3]),'checkpoint recomputation consumed new dropout RNG'
    assert a[2].keys()==b[2].keys()
    for name in a[2]:torch.testing.assert_close(a[2][name],b[2][name],rtol=1e-5,atol=1e-7)
    for name in a[4]:torch.testing.assert_close(a[4][name],b[4][name],rtol=1e-5,atol=1e-7)


def test_checkpointed_forward_still_records_all_layer_values_and_actual_gradients():
    import torch
    model=joint_model(encoder()).train()
    names=LABELS[:4]
    ids=torch.tensor([[0,9,12,7,2]]*len(names))
    groups=[{'candidates':names}]
    capture=CandidateCapture(model,groups,0)
    with capture:
        capture.save('task.input_ids',ids)
        capture.save('task.attention_mask',torch.ones_like(ids))
        result=model(ids,torch.ones_like(ids),[names],gold=[names[1]])
    calls=dict(capture.counts)
    result['loss'].backward()
    assert capture.counts==calls,'backward recomputation was recorded as another input event'
    for _,key,*_ in inventory(3,'test'):
        if key.startswith('task.'):continue
        assert key in capture.values,key
        assert torch.isfinite(capture.values[key]).all()
    assert any(k.endswith('.gradient') for k in capture.values)
    assert capture.values['functional.attention_scores_to_probabilities.call2.output'].shape==(4,4,5,5)
