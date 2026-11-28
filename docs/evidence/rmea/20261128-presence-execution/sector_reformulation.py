"""Same phase-presence function evaluated in the basis before the last CNOT ring."""
import math

import torch

from presence_phase import MODES, PresenceHead


def ring_gather():
    """Output-to-input index map for the three native CNOTs, wire 0 is MSB."""
    gather = list(range(8))
    for control, target in ((0, 1), (1, 2), (2, 0)):
        gather = [gather[i ^ (1 << (2 - target))] if i & (1 << (2 - control)) else gather[i] for i in range(8)]
    return gather


def rotate_y(state, cosine, sine, wire):
    shape = state.shape
    pair = state.reshape(*shape[:-1], 2**wire, 2, 2**(2-wire))
    a, b = pair.unbind(-2)
    c, s = cosine[..., None, None], sine[..., None, None]
    return torch.stack((c*a-s*b, s*a+c*b), dim=-2).reshape(shape)


class SectorHead(PresenceHead):
    """Frozen sectors refer to wire 0 before the final ring, not an output wire."""

    def __init__(self, mode, circuit, eps=1e-6):
        super().__init__(mode, circuit, eps)
        first_ring = torch.tensor(ring_gather())
        self.register_buffer('first_ring', first_ring, persistent=False)
        self.register_buffer('sector_indices', first_ring.argsort(), persistent=False)
        signs = [[1 if i & (1<<(2-w)) else -1 for w in range(3)] for i in range(8)]
        self.register_buffer('phase_signs', torch.tensor(signs,dtype=torch.float32)/2, persistent=False)

    def pre_final_state(self, angles):
        half_y = angles[..., 0]/2
        cosine, sine = half_y.cos(), half_y.sin()
        phase = angles[..., 1] @ self.phase_signs.T
        complex_phase = torch.complex(phase.cos(), phase.sin())
        # The first layer acts on |000>; construct the signed separable amplitudes.
        first = torch.stack((cosine[..., 0, :], sine[..., 0, :]), dim=-1)
        x, y, z = first.unbind(dim=-2)
        initial_amp = (x[..., :, None, None]*y[..., None, :, None]*z[..., None, None, :]).flatten(-3)
        p = (initial_amp*complex_phase[..., 0, :]).index_select(-1, self.first_ring)
        # RZ on one wire commutes with RY on the other two. All last-layer
        # phase factors can be applied together after the real rotations.
        for wire in range(3):
            p = rotate_y(p, cosine[..., 1, wire], sine[..., 1, wire], wire)
        return p*complex_phase[..., 1, :]

    def masses(self, h, roles, dephase):
        x = self.prepare(torch.cat((h, roles), dim=-1))
        if self.mode == MODES[2]:
            wu, wv = self.real_state(x).split(8, dim=-1)
            qu, qv = self.real_state(self.effect).split(8, dim=-1)
            re = []
            im = []
            for sector in (0,1):
                # C owns its embedding and uses the same original coordinates.
                ids = self.sector_indices[sector*4:(sector+1)*4]
                u,v=qu.index_select(-1,ids).T,qv.index_select(-1,ids).T
                words=torch.cat((wu.index_select(-1,ids),wv.index_select(-1,ids)),dim=-1)
                query=torch.cat((torch.cat((u,-v),dim=-1),torch.cat((v,u),dim=-1)),dim=0)
                a,b=(words@query).split(self.effect.shape[0],dim=-1)
                re.append(a);im.append(b)
            block_mass=sum(a.square()+b.square() for a,b in zip(re,im))
            presence=(re[0][...,0]+re[1][...,0]).square()+(im[0][...,0]+im[1][...,0]).square()
        else:
            word = self.pre_final_state(math.pi*torch.tanh(x).reshape(*x.shape[:-1],2,3,2))
            query = self.pre_final_state(math.pi*torch.tanh(self.effect))
            # Both states use the same final ring. Pull that common permutation
            # out of the overlap; the wire-0 sectors are now actual contiguous slices.
            z0 = word[..., :4] @ query[..., :4].T.conj()
            z1 = word[..., 4:] @ query[..., 4:].T.conj()
            block_mass = z0.abs().square()+z1.abs().square()
            presence = block_mass[...,0] if dephase or self.mode == MODES[1] else (z0[...,0]+z1[...,0]).abs().square()
        return .25+.5*presence,block_mass[...,1:]
