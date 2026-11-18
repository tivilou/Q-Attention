"""Source-bound synthetic implementation checks, not a task experiment."""
import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from experiments.rmea_original_scores.common import complete, guard, read, sha


def unit(c):
    import torch
    from torch.nn import functional as F
    from experiments.rmea_context_geometry.legacy.rmea import RMEAHead
    from experiments.rmea_ecqp.head import ECQPHead, ROUTINGS, cohort_derangement, projection_sign_fingerprint
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(c['fixture_seed'])
    checks = []
    h = torch.randn(3, 6, 1024, dtype=torch.float64) * .1
    roles = torch.zeros(3, 6, 4, dtype=torch.float64)
    roles[:, 0, 2] = 1
    roles[:, 1:3, 3] = 1
    mask = torch.ones(3, 6, dtype=torch.bool)
    mask[:, -1] = False
    h[:, -1] = 0
    targets = torch.tensor([1, 2, 0])
    parity = {}
    for dtype in [torch.float32, torch.float64]:
        native = RMEAHead('quantum').to(dtype=dtype)
        head = ECQPHead().to(dtype=dtype)
        head.load_native(native)
        assert sum(p.numel() for p in native.parameters()) == 81028
        assert sum(p.numel() for p in head.parameters()) == 81220
        assert not head.projection.requires_grad
        a = native(h.to(dtype), roles.to(dtype), mask, details=True)
        b = head(h.to(dtype), roles.to(dtype), mask, details=True)
        for key in ['logits', 'alpha', 'values', 'utilities', 'angles', 'effect_angles', 'psi', 'phi', 'probability', 'normalizer']:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        wrong = head(h.to(dtype), roles.to(dtype), mask, details=True, anchor_override=b['entity_summary'].roll(1, 0))
        torch.testing.assert_close(wrong['logits'], b['logits'], rtol=0, atol=0)
        parity[str(dtype)] = float((a['logits'] - b['logits']).abs().max())
    checks.append('original_zero_B_and_mismatch_initialization_exact_float32_float64')
    assert projection_sign_fingerprint(head.projection) == c['projection_sign_sha256']
    checks.append('fixed_projection_sign_hash_and_parameter_count')
    base = RMEAHead('quantum').double()
    model = ECQPHead().double()
    model.load_native(base)
    loss = F.cross_entropy(model(h, roles, mask)[0], targets)
    loss.backward()
    assert model.B.grad is not None and torch.isfinite(model.B.grad).all()
    assert float(model.B.grad.abs().max()) > 1e-12
    for p in model.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    grad = model.B.grad.detach().clone()
    index = tuple(int(i) for i in torch.unravel_index(grad.abs().argmax(), grad.shape))
    delta = 1e-5
    with torch.no_grad():
        model.B[index] += delta
        plus = float(F.cross_entropy(model(h, roles, mask)[0], targets))
        model.B[index] -= 2 * delta
        minus = float(F.cross_entropy(model(h, roles, mask)[0], targets))
        model.B[index] += delta
    numeric = (plus - minus) / (2 * delta)
    analytic = float(grad[index])
    assert abs(numeric - analytic) < 1e-7 + 2e-3 * abs(analytic)
    checks.append('finite_gradients_and_B_central_difference')
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    optimizer.step()
    assert not torch.equal(model.B, torch.zeros_like(model.B))
    correct = model(h, roles, mask, details=True)
    wrong = model(h, roles, mask, details=True, anchor_override=correct['entity_summary'].roll(1, 0))
    post_step_angle_effect = float((correct['prepare_raw'] - wrong['prepare_raw']).abs().max())
    assert post_step_angle_effect > 1e-12
    checks.append('one_SGD_step_and_post_update_summary_effect')
    # Deterministic nonzero branch for padding, permutation and control tests.
    with torch.no_grad():
        model.B.copy_(torch.sin(torch.arange(192, dtype=torch.float64).reshape(12, 16)) * .01)
    expected = model(h, roles, mask, details=True)
    padded = h.clone()
    padded_roles = roles.clone()
    padded[~mask] = torch.nan
    padded_roles[~mask] = torch.inf
    observed = model(padded, padded_roles, mask, details=True)
    torch.testing.assert_close(observed['logits'], expected['logits'], rtol=0, atol=0)
    assert torch.isfinite(observed['logits']).all() and torch.equal(observed['alpha'][..., -1], torch.zeros(3, 40, dtype=torch.float64))
    differentiable = padded.clone().requires_grad_(True)
    model(differentiable, padded_roles, mask)[0].sum().backward()
    assert torch.isfinite(differentiable.grad).all() and torch.equal(differentiable.grad[~mask], torch.zeros_like(differentiable.grad[~mask]))
    checks.append('padding_NaN_ignored_zero_attention_and_zero_gradient')
    permutation = torch.tensor([3, 1, 5, 0, 4, 2])
    permuted = model(h[:, permutation], roles[:, permutation], mask[:, permutation], details=True)
    torch.testing.assert_close(permuted['logits'], expected['logits'], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(permuted['alpha'], expected['alpha'][..., permutation], rtol=1e-12, atol=1e-12)
    reordered = model(h.flip(0), roles.flip(0), mask.flip(0))[0]
    torch.testing.assert_close(reordered, expected['logits'].flip(0), rtol=1e-12, atol=1e-12)
    checks.append('word_permutation_equivariance_and_sample_order_independence')
    errors = 0
    bad_cases = []
    missing = roles.clone(); missing[..., 3] = 0
    bad_cases.append((h, missing, mask))
    bad_cases.append((h, roles, mask.float()))
    bad_cases.append((h, roles, torch.zeros_like(mask)))
    nonfinite = h.clone(); nonfinite[0, 0, 0] = torch.nan
    bad_cases.append((nonfinite, roles, mask))
    nonbinary = roles.clone(); nonbinary[0, 0, 2] = .5
    bad_cases.append((h, nonbinary, mask))
    for inputs in bad_cases:
        try:
            model(*inputs)
        except ValueError:
            errors += 1
    assert errors == len(bad_cases)
    checks.append('missing_entity_invalid_mask_nonfinite_and_role_rejections')
    control_reports = []
    for routing in ROUTINGS:
        for feature in ['entity', 'token_square']:
            head = ECQPHead(routing=routing, feature=feature).double()
            head.load_native(base)
            with torch.no_grad():
                head.B.copy_(model.B)
            d = head(h, roles, mask, details=True)
            assert d['logits'].shape == (3, 40) and d['alpha'].shape == (3, 40, 6)
            assert torch.isfinite(d['logits']).all() and (d['alpha'] >= 0).all()
            torch.testing.assert_close(d['alpha'].sum(-1), torch.ones(3, 40, dtype=torch.float64), rtol=1e-12, atol=1e-12)
            F.cross_entropy(d['logits'], targets).backward()
            assert head.B.grad is not None and torch.isfinite(head.B.grad).all() and float(head.B.grad.abs().max()) > 1e-12
            control_reports.append(dict(routing=routing, feature=feature, parameters=sum(p.numel() for p in head.parameters())))
    checks.append('same_parameter_classical12_and_token_square_controls_forward_backward')
    linear = ECQPHead(feature='token_linear').double(); linear.load_native(base)
    with torch.no_grad():
        linear.B.copy_(model.B)
        folded = RMEAHead('quantum').double(); folded.load_state_dict(base.state_dict())
        folded.prepare.weight[:, :1024] += linear.B @ linear.projection
    a = linear(h, roles, mask, details=True); b = folded(h, roles, mask, details=True)
    torch.testing.assert_close(a['angles'], b['angles'], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(a['logits'], b['logits'], rtol=1e-12, atol=1e-12)
    checks.append('token_linear_control_folds_into_native_affine_map')
    donors = cohort_derangement(['a','b','c','d','e','f'], ['x','x','y','z','z','w'], 'fit')
    reverse = cohort_derangement(['f','e','d','c','b','a'], ['w','z','z','y','x','x'], 'fit')
    assert donors == reverse and set(donors) == set(donors.values())
    groups = dict(zip(['a','b','c','d','e','f'], ['x','x','y','z','z','w']))
    assert all(groups[s] != groups[d] for s, d in donors.items())
    rejected = 0
    for ids, gs in [(['a','a'],['x','y']), (['a','b','c'],['x','x','y'])]:
        try: cohort_derangement(ids, gs, 'fit')
        except ValueError: rejected += 1
    assert rejected == 2
    checks.append('whole_cohort_group_excluding_bijection_order_invariant_and_infeasible_rejected')
    buffer = io.BytesIO(); torch.save(model.state_dict(), buffer); buffer.seek(0)
    restored = ECQPHead().double(); restored.load_state_dict(torch.load(buffer, weights_only=True))
    torch.testing.assert_close(restored(h, roles, mask)[0], expected['logits'], rtol=0, atol=0)
    corrupt = {k:v.clone() for k,v in model.state_dict().items()}; corrupt['projection'][0,0] *= -1
    try:
        restored.load_state_dict(corrupt)
    except ValueError:
        pass
    else:
        raise AssertionError('Corrupt projection accepted')
    checks.append('state_roundtrip_and_corrupt_projection_rejected')
    return dict(status='PASS_ECQP_IMPLEMENTATION_UNIT', checks=checks, parity_max_abs=parity,
        projection_sign_sha256=c['projection_sign_sha256'], parameters=81220,
        gradient_check=dict(index=list(index), analytic=analytic, central_difference=numeric, step=delta),
        post_update_prepare_raw_summary_effect=post_step_angle_effect, controls=control_reports,
        unit_optimizer_steps=1, task_training=False, real_data_reads=0, COS_requests=0,
        utility='NOT_RUN', novelty='UNESTABLISHED', L5='NOT_ACCEPTED',
        limitations=['Synthetic source/gradient checks only, no task utility, real encoder reachability or family separation.',
            'Equal trainable parameter counts do not imply equal active capacity or computational cost.',
            'Clinical, signal-processing and bilinear precedents supply ideas, not transferred performance or novelty.'])


def main():
    guard()
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', required=True, choices=['unit'])
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    c = read(HERE / 'contract.json')
    for name, digest in c['dependencies'].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError('Dependency revision mismatch: ' + name)
    env = {name:importlib.metadata.version(name) for name in c['environment']}
    if env != c['environment'] or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Expected frozen CPU environment and scheduler-empty CUDA mask')
    started = time.monotonic()
    result = unit(c)
    result.update(contract_sha256=sha(HERE / 'contract.json'), environment=env,
        python_execution=dict(optimize=sys.flags.optimize, PYTHONOPTIMIZE_present='PYTHONOPTIMIZE' in os.environ),
        elapsed_seconds=time.monotonic() - started)
    complete(args.output, args.action, result)
    print(json.dumps(dict(status=result['status'], action=args.action, task_training=False, utility='NOT_RUN', L5='NOT_ACCEPTED')))


if __name__ == '__main__':
    main()
