"""Frozen-head development probe; no training or formal-ready certificate."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from baselines.retacred.rmea import EPS, RMEAHead

RIDGE = 1 / 8
CUTOFF = 1e-10
CHECKPOINT = '2da52b1ce826046a6f443269f47f49e4e9fa63e10f22f9ebdcde6f890e12ce51'
IDENTITIES_SHA = '5dc89003494f6b782663759ce12b6292041904548e6ce80dd5651144603e87bc'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def covariance(psi, mention):
    z = psi[~mention]
    if not len(z):
        return torch.zeros((8, 8), dtype=psi.dtype)
    return z.T @ z.conj() / len(z)


def transform(psi, phi, mention, cov):
    """Inference only: protect retained entity span, audit discarded directions."""
    eye = torch.eye(8, dtype=psi.dtype)
    left, singular, _ = torch.linalg.svd(psi[mention].T, full_matrices=False)
    rank = int((singular > singular[0] * CUTOFF).sum())
    basis = left[:, :rank]
    perp = eye - basis @ basis.mH
    a = perp @ cov @ perp + RIDGE * eye
    correction = torch.linalg.solve(a, -perp @ cov @ phi.T).T
    corrected = phi + correction
    amplitude = phi.conj() @ psi.T
    new_amplitude = corrected.conj() @ psi.T
    raw = new_amplitude.abs().square() + EPS
    alpha = raw / raw.sum(-1, keepdim=True)
    residual = (new_amplitude[:, mention] - amplitude[:, mention]).abs().max()
    dropped = float(singular[rank]) if rank < len(singular) else 0.0
    # Roundoff allowance plus the explicitly truncated-span error bound.
    bound = dropped * float(correction.norm(dim=-1).max()) + 1e-9
    if not torch.isfinite(alpha).all() or float(residual) > bound:
        raise ValueError('Protected amplitude numerical bound violated')
    return dict(alpha=alpha, raw=raw, corrected=corrected,
                correction=correction, perp=perp, covariance=cov,
                rank=rank, residual=float(residual), residual_bound=bound,
                singular_values=singular)


def unit_check():
    eye = torch.eye(8, dtype=torch.complex128)
    psi = torch.stack((eye[0], eye[1], (eye[0]+eye[2])/2**.5,
                       (eye[1]+1j*eye[3])/2**.5))
    mention = torch.tensor([True, True, False, False])
    phi = torch.stack([(eye[0]+eye[2])/2**.5]*40)
    cov = covariance(psi, mention)
    t = transform(psi, phi, mention, cov)
    before = (phi.conj() @ psi[~mention].T).abs().square().mean(-1)
    after = (t['corrected'].conj() @ psi[~mention].T).abs().square().mean(-1)
    if not (after < before).all() or t['residual'] > 1e-12:
        raise ValueError('Exact synthetic protected-span fixture failed')
    # Degenerate full-span case is operationally inert.
    full = transform(eye, phi, torch.ones(8, dtype=torch.bool), cov)
    if float(full['correction'].abs().max()) > 1e-12:
        raise ValueError('Full protected-span limit failed')
    return dict(protected_synthetic=True, full_span_inert=True)


def decomposition(alpha0, alpha1, utilities, mention):
    m0 = alpha0[:, mention].sum(-1)
    m1 = alpha1[:, mention].sum(-1)
    n0 = alpha0[:, ~mention].sum(-1)
    n1 = alpha1[:, ~mention].sum(-1)
    um = (alpha0[:, mention]*utilities[:, mention]).sum(-1) / m0
    uc0 = (alpha0[:, ~mention]*utilities[:, ~mention]).sum(-1) / n0.clamp_min(1e-300)
    uc1 = (alpha1[:, ~mention]*utilities[:, ~mention]).sum(-1) / n1.clamp_min(1e-300)
    transfer = (m1-m0)*(um-uc0)
    redistribution = n1*(uc1-uc0)
    actual = ((alpha1-alpha0)*utilities).sum(-1)
    error = (actual-transfer-redistribution).abs().max()
    # Truncated span preserves mention amplitudes only to its recorded bound.
    if float(error) > 1e-7:
        raise ValueError('Utility decomposition numerical residual exceeded')
    return um-uc0, transfer, redistribution, float(error)


@torch.no_grad()
def run(args):
    if not __debug__:
        raise RuntimeError('Validation requires assertions enabled')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    unit = unit_check()
    cache = args.baseline_run/'cache'
    assert digest(args.identities) == IDENTITIES_SHA
    identities = json.loads(args.identities.read_bytes())
    assert set(identities) == {'metadata.json', 'valid-h.npy', 'valid-roles.npy',
                               'valid-offsets.npy', 'valid-targets.npy'}
    for name, value in identities.items():
        assert (cache/name).stat().st_size == value['bytes']
        assert digest(cache/name) == value['sha256'], name
    cp = args.baseline_run/'checkpoints/native-final.pt'
    assert digest(cp) == CHECKPOINT
    state = torch.load(cp, map_location='cpu', weights_only=True)
    head = RMEAHead('quantum').double().eval()
    head.load_state_dict(state, strict=True)
    data = {k: np.load(cache/f'valid-{k}.npy', mmap_mode='r', allow_pickle=False)
            for k in ('h', 'roles', 'offsets', 'targets')}
    metadata = json.loads((cache/'metadata.json').read_bytes())['valid']
    assert len(metadata) == len(data['targets']) == 19584
    selected = []
    for label in range(40):
        candidates = np.where(data['targets'] == label)[0].tolist()
        candidates.sort(key=lambda i: metadata[i]['id_hash'])
        n = 18 if label == 0 else 2
        assert len(candidates) >= n
        selected += candidates[:n]
    assert len(selected) == 96
    cases = []
    for i in selected:
        lo, hi = map(int, data['offsets'][i:i+2])
        h = torch.from_numpy(np.array(data['h'][lo:hi], dtype=np.float64))[None]
        roles = torch.from_numpy(np.array(data['roles'][lo:hi], dtype=np.float64))[None]
        mask = torch.ones(h.shape[:2], dtype=torch.bool)
        d = head(h, roles, mask, details=True)
        plain = head(h, roles, mask)
        assert torch.equal(d['alpha'], plain[1]) and torch.equal(d['logits'], plain[0])
        mention = (roles[0, :, 2] != 0) | (roles[0, :, 3] != 0)
        assert mention.any()
        psi = d['psi'][0]
        cases.append(dict(index=i, id_hash=metadata[i]['id_hash'],
                          target=int(data['targets'][i]), h=h, roles=roles,
                          mask=mask, d=d, mention=mention,
                          cov=covariance(psi, mention), ncontext=int((~mention).sum())))
    # Label-blind deterministic recipient/donor mapping; singleton bins use
    # nearest context count, then input identity. No outcome is consulted.
    bins = {}
    for j, case in enumerate(cases):
        bins.setdefault(case['ncontext']//8, []).append(j)
    donor = {}
    singleton = 0
    for group in bins.values():
        group.sort(key=lambda j: cases[j]['id_hash'])
        if len(group)>1:
            donor.update({j:group[(k+1)%len(group)] for k,j in enumerate(group)})
        else:
            singleton += 1
            j = group[0]
            donor[j] = min((k for k in range(96) if k != j),
                           key=lambda k:(abs(cases[k]['ncontext']-cases[j]['ncontext']),cases[k]['id_hash']))
    rows, fixtures = [], []
    for j, case in enumerate(cases):
        d, m = case['d'], case['mention']
        psi, phi, u = d['psi'][0], d['phi'], d['utilities'][0]
        alpha0 = d['alpha'][0]
        a = transform(psi, phi, m, case['cov'])
        b = transform(psi, phi, m, cases[donor[j]]['cov'])
        logits = [d['logits'][0]] + [(v['alpha']*u).sum(-1)+head.bias for v in (a,b)]
        powers = [(v.conj() @ psi[~m].T).abs().square().sum(-1) for v in (phi,a['corrected'])]
        power_increase = float((powers[1]-powers[0]).max())
        assert power_increase < 1e-8
        gap, transfer, redistribution, err = decomposition(alpha0,a['alpha'],u,m)
        _, _, _, donor_err = decomposition(alpha0,b['alpha'],u,m)
        y=case['target']
        competitor = int(logits[0].masked_fill(torch.arange(40)==y,-torch.inf).argmax())
        margins = [float(v[y]-v[competitor]) for v in logits]
        row = dict(id_hash=case['id_hash'], target=y, competitor=competitor,
                   donor_id_hash=cases[donor[j]]['id_hash'], ncontext=case['ncontext'],
                   protected_rank=a['rank'], protected_residual=a['residual'],
                   protected_bound=a['residual_bound'], context_power_max_increase=power_increase,
                   correction_norm=float(a['correction'].norm()), donor_correction_norm=float(b['correction'].norm()),
                   native_utility_gap=float(gap[y]), margins=margins,
                   matched_margin_change=margins[1]-margins[0],
                   donor_margin_change=margins[2]-margins[0],
                   target_transfer=float(transfer[y]), target_within_context=float(redistribution[y]),
                   decomposition_max_error=max(err,donor_err),
                   predictions=[int(v.argmax()) for v in logits])
        rows.append(row)
        if j in (0, 18):
            fixtures.append(dict(h=case['h'],roles=case['roles'],mask=case['mask'],
                                 psi=psi,phi=phi,mention=m,utilities=u,bias=head.bias.detach(),
                                 native_alpha=alpha0,native_logits=logits[0],
                                 matched=a,donor=b,logits=torch.stack(logits)))
    eligible = [v for v in rows if v['target']>0 and v['ncontext']>0]
    def mean(v):return float(np.mean(v)) if len(v) else None
    strata={}
    for name, sign in [('mentions_higher',1),('context_higher',-1)]:
        group=[v for v in eligible if sign*v['native_utility_gap']>0]
        strata[name]=dict(n=len(group),matched=mean([v['matched_margin_change'] for v in group]),
                          donor=mean([v['donor_margin_change'] for v in group]),
                          matched_minus_donor=mean([v['matched_margin_change']-v['donor_margin_change'] for v in group]))
    args.output.mkdir(parents=True,exist_ok=True)
    assert not any((args.output/name).exists() for name in ('probe-summary.json','private-fixture.pt','RUN_COMPLETE'))
    fixture=args.output/'private-fixture.pt'
    torch.save(dict(state=state,cases=fixtures),fixture)
    loaded=torch.load(fixture,map_location='cpu',weights_only=True)
    restored=RMEAHead('quantum').double().eval()
    restored.load_state_dict(loaded['state'],strict=True)
    for f in loaded['cases']:
        assert not f['bias'].requires_grad
        assert torch.equal(f['bias'],restored.bias.detach())
        v=restored(f['h'],f['roles'],f['mask'],details=True)
        assert torch.equal(v['alpha'][0],f['native_alpha'])
        assert torch.equal(v['logits'][0],f['native_logits'])
        replayed=[v['logits'][0]]
        for name in ('matched','donor'):
            t=transform(v['psi'][0],v['phi'],f['mention'],f[name]['covariance'])
            assert torch.equal(t['alpha'],f[name]['alpha'])
            replayed.append((t['alpha']*v['utilities'][0]).sum(-1)+restored.bias)
        assert torch.equal(torch.stack(replayed),f['logits'])
    contrast=None
    if all(strata[name]['n']>0 for name in ('mentions_higher','context_higher')):
        contrast=strata['mentions_higher']['matched_minus_donor']-strata['context_higher']['matched_minus_donor']
    summary=dict(schema='q-rmea.protected-context.probe.v1',status='complete',
                 stage='frozen96case_discovery_only',unit_checks=unit,optimizer_steps=0,
                 checkpoint_sha256=CHECKPOINT,cache_identities=identities,
                 selected_ids=[v['id_hash'] for v in cases],donor_singleton_bins=singleton,
                 protected_rank_counts={str(k):sum(v['protected_rank']==k for v in rows) for k in range(9)},
                 positive_mean_matched_margin_change=mean([v['matched_margin_change'] for v in eligible]),
                 positive_mean_donor_margin_change=mean([v['donor_margin_change'] for v in eligible]),
                 strata=strata,primary_stratum_contrast=contrast,
                 rows=rows,fixture_sha256=digest(fixture),fixture_bytes=fixture.stat().st_size,
                 fixture_replay='bitwise_pass',sample_trace_v2=False,formal_ready=False,L5='NOT_ACCEPTED')
    data=(json.dumps(summary,indent=2,allow_nan=False)+'\n').encode()
    (args.output/'probe-summary.json').write_bytes(data)
    (args.output/'RUN_COMPLETE').write_text(hashlib.sha256(data).hexdigest()+'\n')
    print(json.dumps({k:summary[k] for k in ['status','protected_rank_counts','strata','positive_mean_matched_margin_change','positive_mean_donor_margin_change','optimizer_steps']}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline-run',type=Path,required=True)
    parser.add_argument('--identities',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
