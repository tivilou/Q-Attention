import copy
import numpy as np
import pytest
import torch
from q_attention.plugins.q_srm import MODES, ThreeQubitMap, inverse_square_root, measure
from q_attention.adapters.q_srm import SRMModel, objective
from q_attention.models.relation_transformer import RelationExtractionModel, RelationTransformerConfig


def fixture():
    torch.manual_seed(27)
    base = RelationExtractionModel(RelationTransformerConfig(vocab_size=24, num_labels=3, dim=16,
        num_heads=2, num_layers=2, ff_dim=32, dropout=0, max_length=8))
    ids = torch.tensor([[2,3,4,5,0], [6,7,8,0,0]])
    mask = ids != 0
    subject = torch.tensor([[1,0,0,0,0], [1,0,0,0,0]], dtype=torch.bool)
    obj = torch.tensor([[0,0,1,0,0], [0,1,0,0,0]], dtype=torch.bool)
    return base, dict(input_ids=ids, attention_mask=mask, subject_mask=subject, object_mask=obj)


def states(n=5):
    torch.manual_seed(11)
    q = torch.randn(2, 2, n, 8, dtype=torch.complex128)
    k = torch.randn_like(q)
    q = q / q.norm(dim=-1, keepdim=True)
    k = k / k.norm(dim=-1, keepdim=True)
    mask = torch.ones((2,n), dtype=torch.bool)
    mask[1,-1] = False
    return q,k,mask


def test_disabled_exact_base_output_and_classifier_gradient():
    base,batch = fixture()
    expected = base(**batch)
    expected.sum().backward()
    actual_model = SRMModel(copy.deepcopy(base), 'disabled')
    actual_model.zero_grad(set_to_none=True)
    actual = actual_model(**batch, capture=True)
    actual.sum().backward()
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    for a,b in zip(base.classifier.parameters(),actual_model.base.classifier.parameters()):
        torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
    assert not any(p.requires_grad for p in actual_model.base.encoder.parameters())


def test_144_nominal_parameters_and_unitary_state_lineage():
    base,batch = fixture()
    for mode in MODES[1:]:
        model = SRMModel(copy.deepcopy(base),mode)
        assert sum(p.numel() for p in model.kernel.parameters()) == 144
        model(**batch,capture=True)
        for prefix in ('query','key'):
            trace=model.kernel.last[prefix+'_circuit_states']
            torch.testing.assert_close(trace.abs().square().sum(-1),torch.ones_like(trace.real[...,0]),atol=1e-12,rtol=1e-12)


@pytest.mark.parametrize('mode',MODES[1:])
def test_complete_probabilities_mask_and_parameter_gradient(mode):
    base,batch=fixture(); model=SRMModel(base,mode)
    logits=model(**batch,capture=True)
    loss,_=objective(logits,torch.tensor([0,1]));loss.backward()
    a,null=model.attention,model.null_probability
    torch.testing.assert_close(a.sum(-1)+null,torch.ones_like(null),atol=1e-11,rtol=1e-11)
    assert a.min()>=0 and null.min()>=0 and null.max()<=1+1e-11
    assert not a.masked_select(~batch['attention_mask'][:,None,None,:]).any()
    for name,p in model.kernel.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(),name
        assert p.grad.norm()>1e-12,name
    if mode!='fidelity':
        t=model.kernel.last
        assert torch.linalg.eigvalsh(t['effects']).min()>-1e-11
        assert t['completeness'].abs().max()<1e-11
    with pytest.raises(ValueError,match='stacked'):
        SRMModel(base,mode)


def test_inverse_root_gradcheck_at_repeated_eigenvalues():
    # Degenerate spectra occur at initialization and with duplicate key states.
    x=torch.zeros((8,8),dtype=torch.complex128,requires_grad=True)
    def operation(z):
        return inverse_square_root((z+z.mH)/2+0.01*torch.eye(8,dtype=z.dtype))
    assert torch.autograd.gradcheck(operation,(x,),eps=1e-7,atol=3e-5,rtol=1e-4)


def test_inverse_root_gradcheck_unequal_spectrum_and_batch():
    torch.manual_seed(8)
    x=torch.randn(2,3,3,dtype=torch.complex128,requires_grad=True)
    def operation(z):return inverse_square_root(z@z.mH+0.01*torch.eye(3,dtype=z.dtype))
    assert torch.autograd.gradcheck(operation,(x,),eps=1e-6,atol=2e-5,rtol=1e-4)


def test_analytic_duplicate_group_and_null():
    q=torch.zeros(1,1,4,8,dtype=torch.complex128)
    q[...,0]=np.sqrt(.8);q[...,1]=np.sqrt(.2)
    k=torch.zeros_like(q);k[...,:3,0]=1;k[...,3,1]=1
    mask=torch.ones(1,4,dtype=torch.bool)
    a,null,_=measure(q,k,mask)
    torch.testing.assert_close(a[0,0,0,:3].sum(),torch.tensor(.8*3/3.01,dtype=torch.float64))
    torch.testing.assert_close(a[0,0,0,3],torch.tensor(.2/1.01,dtype=torch.float64))
    torch.testing.assert_close(null[0,0,0],1-a[0,0,0].sum())
    assert null[0,0,0]>0  # Dropping null and renormalizing must fail this oracle.


def test_query_outside_key_span_is_null():
    q=torch.zeros(1,1,3,8,dtype=torch.complex128);q[...,1]=1
    k=torch.zeros_like(q);k[...,0]=1
    a,null,_=measure(q,k,torch.ones(1,3,dtype=torch.bool))
    assert not a.any()
    torch.testing.assert_close(null,torch.ones_like(null),atol=1e-12,rtol=1e-12)


@pytest.mark.parametrize('mode',['quantum','fidelity','diagonal'])
def test_token_permutation_and_padding_invariance(mode):
    q,k,mask=states()
    a,null,_=measure(q,k,mask,mode=mode)
    perm=torch.tensor([4,2,0,3,1])
    ap,np_,_=measure(q[:,:,perm],k[:,:,perm],mask[:,perm],mode=mode)
    torch.testing.assert_close(ap,a[:,:,perm][:,:,:,perm],atol=1e-11,rtol=1e-11)
    torch.testing.assert_close(np_,null[:,:,perm],atol=1e-11,rtol=1e-11)
    qp=torch.cat([q,q[:,:,:2]],2);kp=torch.cat([k,k[:,:,:2]],2)
    mp=torch.cat([mask,torch.zeros(2,2,dtype=torch.bool)],1)
    ap,np_,_=measure(qp,kp,mp,mode=mode)
    torch.testing.assert_close(ap[:,:,:5,:5],a,atol=1e-11,rtol=1e-11)
    torch.testing.assert_close(np_[:,:,:5],null,atol=1e-11,rtol=1e-11)
    assert not ap[:,:,5:].any() and torch.equal(np_[:,:,5:],torch.ones_like(np_[:,:,5:]))


def test_exact_classical_expression_values_and_gradients():
    q,k,mask=states();q.requires_grad_();k.requires_grad_()
    a,null,t=measure(q,k,mask,capture=True)
    # Explicit POVM quadratic form is independent of the optimized pure-state path.
    explicit=torch.einsum('bhqi,bhkij,bhqj->bhqk',q.conj(),t['effects'],q).real
    explicit=explicit*mask[:,None,:,None]
    torch.testing.assert_close(explicit,a,atol=1e-11,rtol=1e-11)
    weights=torch.linspace(-1,1,a.numel(),dtype=a.dtype).reshape_as(a)
    g1=torch.autograd.grad((a*weights).sum(),(q,k),retain_graph=True)
    g2=torch.autograd.grad((explicit*weights).sum(),(q,k))
    for x,y in zip(g1,g2):torch.testing.assert_close(x,y,atol=1e-10,rtol=1e-9)


def test_mismatch_permutation_preserves_support_and_changes_context():
    q,k,mask=states()
    a,_,_=measure(q,k,mask)
    other,_,t=measure(q,k,mask,mode='key_mismatch',capture=True)
    assert not torch.allclose(a,other)
    for row in range(2):
        ids=mask[row].nonzero(as_tuple=True)[0]
        assert torch.equal(t['permutation'][row,ids],ids.roll(1))
    torch.testing.assert_close(a.sum(-1),other.sum(-1),atol=1e-11,rtol=1e-11)


def test_common_unitary_gauge_cancels_as_documented():
    q,k,mask=states();u,_=torch.linalg.qr(torch.randn(8,8,dtype=torch.complex128))
    a,null,_=measure(q,k,mask)
    b,nb,_=measure(q@u.T,k@u.T,mask)
    torch.testing.assert_close(a,b,atol=1e-11,rtol=1e-11)
    torch.testing.assert_close(null,nb,atol=1e-11,rtol=1e-11)


def test_context_really_bypasses_softmax_and_preserves_null():
    base,batch=fixture();model=SRMModel(base)
    model(**batch,capture=True)
    o=model.observed
    expected=(o['attention']@o['value'].double()).float()
    torch.testing.assert_close(o['context'],expected,rtol=0,atol=0)
    assert o['null_probability'].max()>0
    assert not torch.allclose(o['attention'].sum(-1),torch.ones_like(o['null_probability']),atol=1e-6,rtol=1e-6)


def test_masks_modes_and_invalid_inputs_fail_closed():
    q,k,mask=states()
    with pytest.raises(ValueError,match='empty'):measure(q,k,torch.zeros_like(mask))
    with pytest.raises(ValueError,match='unit norm'):measure(q,k*2,mask)
    with pytest.raises(ValueError,match='frozen'):measure(q,k,mask,ridge=.1)
    with pytest.raises(ValueError,match='mode'):measure(q,k,mask,mode='oops')
    with pytest.raises(ValueError,match='positive-definite'):inverse_square_root(-torch.eye(8))
    base,batch=fixture();model=SRMModel(base)
    bad=dict(batch,object_mask=batch['subject_mask'])
    with pytest.raises(ValueError,match='overlapping'):model(**bad)


def test_microbatch_gradients_match_logical_factual_ce():
    base,batch=fixture();torch.manual_seed(101);whole=SRMModel(base)
    # Deepcopy of owner hooks is not the model-copy contract: recreate/load state.
    base2,_=fixture();small=SRMModel(base2);small.load_state_dict(whole.state_dict())
    labels=torch.tensor([0,1])
    objective(whole(**batch),labels)[0].backward()
    for i in range(2):
        objective(small(**{k:v[i:i+1] for k,v in batch.items()}),labels[i:i+1])[0].div(2).backward()
    for (name,a),(_,b) in zip(whole.named_parameters(),small.named_parameters()):
        if a.requires_grad:torch.testing.assert_close(a.grad,b.grad,atol=1e-7,rtol=2e-4,msg=name)
