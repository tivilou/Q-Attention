"""Ordered distinct marked-subsequence head; scientific plugin only.

Path D removes subset-path coherence, not coherence within a selected path.
No runner, scheduler, data access or optimizer is owned by this module.
"""
import math
import torch
from rmea_history_posterior import HistoryHead,GaussianCovarianceHead,valid_inputs
from baselines.retacred.rmea_matched_routers import initialize_head
from ordered_subsequence import coherent_masses,dephased_masses,readout,SignedRealOperatorHead

MODES=('quantum_subsequence','path_dephased_subsequence','signed_real16_subsequence','gaussian_covariance')
COMMON=('value.weight','value.bias','decoder','bias')

class QuantumSubsequenceHead(HistoryHead):
    def __init__(self,mode):
        if mode not in MODES[:2]:raise ValueError('Unknown subsequence mode')
        # Preserve original Q constructor/state names and native initialization.
        super().__init__('quantum_history')
        self.mode=mode

    def forward(self,h,roles,mask,details=False,same_state_dephased=False,terminal_dephase=False):
        from rmea_history_posterior import native_unitary,circuit
        h,roles=valid_inputs(h,roles,mask)
        angles=(math.pi*torch.tanh(self.prepare(torch.cat((h,roles),-1)))).reshape(h.shape[:2]+(2,3,2))
        U=native_unitary(angles)
        phi=circuit(math.pi*torch.tanh(self.effect))
        E=(phi[:,:,None]*phi[:,None,:].conj()+torch.eye(8,dtype=U.dtype,device=U.device)[None]/8)/2
        if terminal_dephase:
            # Dephase both final memory state and query effect in this fixed
            # computational basis; Tr(Delta(rho)Delta(E))=Tr(rho Delta(E)).
            E=torch.diag_embed(E.diagonal(dim1=-2,dim2=-1))
        f=dephased_masses if self.mode==MODES[1] or same_state_dephased else coherent_masses
        output=readout(f(U,E,mask),h,mask,self.value,self.decoder,self.bias)
        return output if details else output['logits']

    def same_state_dephased_logits(self,h,roles,mask):
        if self.mode!=MODES[0]:raise ValueError('Requires fixed Q')
        return self(h,roles,mask,same_state_dephased=True)

    def same_state_terminal_dephased_logits(self,h,roles,mask):
        if self.mode!=MODES[0]:raise ValueError('Requires fixed Q')
        return self(h,roles,mask,terminal_dephase=True)

class ClassicalSubsequenceHead(SignedRealOperatorHead):
    def forward(self,h,roles,mask,details=False):
        h,roles=valid_inputs(h,roles,mask)
        A,E,h,roles=self.components(h,roles,mask)
        output=readout(coherent_masses(A,E,mask),h,mask,self.value,self.decoder,self.bias)
        return output if details else output['logits']

def make_models(seed=13,initial=None):
    if initial is None:raise ValueError('Pinned original native initial state required')
    result={}
    with torch.random.fork_rng(devices=[]):
        for mode in MODES:
            torch.manual_seed(seed)
            model=(QuantumSubsequenceHead(mode) if mode in MODES[:2] else
                   ClassicalSubsequenceHead() if mode==MODES[2] else GaussianCovarianceHead())
            state=model.state_dict()
            if mode in MODES[:2]:
                state={k:initial[k].detach().clone() for k in state}
            elif mode==MODES[2]:
                for k in COMMON+('prepare.weight','prepare.bias'):
                    state[k]=initial[k].detach().clone()
            else:
                legacy=initialize_head('real12',initial,dict(seed=seed,training_dtype='float32')).head.state_dict()
                for k in legacy:state[k]=legacy[k].detach().clone()
            model.load_state_dict(state,strict=True);result[mode]=model
    return result

def real_parameters(model):
    return sum(p.numel()*(2 if p.is_complex() else 1) for p in model.parameters() if p.requires_grad)
