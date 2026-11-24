"""Pure logit-space paired ranking term; no inference or I/O changes."""
import torch
from torch.nn import functional as F

def pair_loss(logits,labels):
    if len(labels)%2 or not torch.all(labels[::2]>0) or not torch.all(labels[1::2]==0):
        raise ValueError('Adjacent positive/null pairs required')
    relation=labels[::2];positive=logits[::2];negative=logits[1::2]
    i=torch.arange(len(relation),device=labels.device)
    d=(positive[i,relation]-positive[:,0])-(negative[i,relation]-negative[:,0])
    return F.softplus(-d).mean()

def reference_checks():
    z=torch.tensor([[2.,1.,3.],[1.,2.,4.]],dtype=torch.float64,requires_grad=True)
    y=torch.tensor([1,0]);loss=pair_loss(z,y);expected=torch.log1p(torch.exp(torch.tensor(2.,dtype=z.dtype)))
    error=float(abs(loss-expected));assert error<1e-12
    row=torch.tensor([[1.2],[-.7]],dtype=z.dtype);column=torch.tensor([[.8,-2.,3.]],dtype=z.dtype)
    invariant=float(abs(pair_loss(z+row+column,y)-loss));assert invariant<1e-12
    g=torch.autograd.grad(loss,z)[0];bias_residual=float(g.sum(0).abs().max());assert bias_residual<1e-12
    assert torch.isfinite(g).all() and g.abs().sum()>0
    # A third class wins both rows despite a very favorable paired margin.
    witness=torch.tensor([[0.,8.,9.],[0.,-8.,9.]],dtype=z.dtype)
    assert pair_loss(witness,y)<1e-6 and witness.argmax(1).tolist()==[2,2]
    return dict(formula_error=error,row_column_shift_error=invariant,bias_gradient_error=bias_residual,
                finite_nonzero_gradient=True,third_class_counterexample_margin=16.,
                third_class_counterexample_predictions=witness.argmax(1).tolist())
