"""Exercise the staged factories through their batched forward and state entry points."""
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys

import torch

from check_presence_phase import dense_check, original
from presence_phase import COMMON, MODES, PresenceHead, make_models


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def diff(a, b):
    return float((a-b).detach().abs().max())


def test(source):
    circuit = original(source)
    torch.manual_seed(20261128)
    initial = PresenceHead(MODES[0], circuit).state_dict()
    models = make_models(circuit, initial)
    q, d, c = models.values()
    assert all(torch.equal(q.state_dict()[k], d.state_dict()[k]) for k in initial)
    assert all(torch.equal(c.state_dict()[k], q.state_dict()[k]) for k in COMMON)
    counts = {mode:sum(p.numel() for p in model.parameters()) for mode, model in models.items()}
    assert list(counts.values()) == [81028,81028,84235], counts
    h = torch.randn(3,9,1024)
    roles = torch.randn(3,9,4)
    mask = torch.arange(9)[None,:] < torch.tensor([1,5,8])[:,None]
    gold = torch.tensor([0,21,39])
    errors = {}
    stats = {}
    for dtype in (torch.float32, torch.float64):
        with torch.no_grad():
            for m in models.values():m.to(dtype)
        ht, rt = h.to(dtype), roles.to(dtype)
        bad = ht.masked_fill(~mask[..., None], float('nan'))
        for mode, m in models.items():
            out = m(ht, rt, mask, details=True)
            logits = out['logits']
            assert logits.shape == (3,40) and torch.isfinite(logits).all()
            assert out['presence_pool'].shape == (3,2,64) and out['identity_pool'].shape == (3,39,64)
            assert not out['attention'][~mask].any()
            assert torch.equal(logits, m(bad, rt, mask))
            tol = 5e-6 if dtype==torch.float32 else 3e-12
            assert diff(out['attention'].sum(dim=1), torch.ones(3,41,dtype=dtype)) < tol
            assert diff(out['presence_pool'][0,0], out['presence_pool'][0,1]) < tol
            m.zero_grad(set_to_none=True)
            torch.nn.functional.cross_entropy(logits,gold).backward()
            grad = {k:float(p.grad.norm()) for k,p in m.named_parameters() if p.grad is not None}
            assert set(grad) == set(dict(m.named_parameters())) and all(0<x<float('inf') for x in grad.values())
            stats[f'{mode}:{dtype}'] = grad
        with torch.no_grad():
            fq = q(ht,rt,mask,details=True,dephase=True)
            plain = q(ht,rt,mask,details=True)
            direct = d(ht,rt,mask)
            assert torch.equal(fq['logits'], direct)
            assert torch.equal(fq['positive_scores'], plain['positive_scores'])
            assert diff(fq['presence_logodds'], plain['presence_logodds'])>1e-6
            zz = c(ht,rt,mask,details=True)
            i=1
            coord=(c.real_state(c.prepare(torch.cat((ht,rt),dim=-1)[i])),c.real_state(c.effect))
            wp, qp = [torch.complex(x[...,:8],x[...,8:]) for x in coord]
            rr = dense_check(wp,qp[0],qp[1:],c.value(ht[i]),mask[i],c.decoder[1:],c.bias[1:],c.decoder[0])
            native_scores = torch.as_tensor(rr[-1], dtype=dtype)
            native_scores[1:] += c.bias[0]
            ec = diff(zz['logits'][i], native_scores)
            assert ec<tol,ec
            errors[str(dtype)] = ec
            for mode,m in models.items():
                buf = BytesIO();torch.save(m.state_dict(),buf);buf.seek(0)
                reloaded = PresenceHead(mode,circuit).to(dtype)
                reloaded.load_state_dict(torch.load(buf,weights_only=True),strict=True)
                assert torch.equal(reloaded(ht,rt,mask),m(ht,rt,mask))
    return {'status':'PASS_STAGED_BATCHED_Q_D_REAL16','parameter_counts':counts,'reference_max_errors':errors,
            'gradient_norms':stats,'shared_initial_state_exact':True,'frozen_intervention_preserves_identity':True,
            'roundtrip_exact':True,'optimizer_updates':0,'task_rows':0,'task_forwards':0,'gpu_allocations':0,
            'scope':'QT CPU artificial input. Loaded supplied synthetic initial state. Gcov and production registry/runner still pending.',
            'plugin_sha256':sha(Path(__file__).with_name('presence_phase.py')),'check_sha256':sha(__file__)}


if __name__ == '__main__':
    torch.set_num_threads(1)
    print(json.dumps(test(sys.argv[1]),indent=2,allow_nan=False))
