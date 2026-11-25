"""Frozen native-Q observable-span diagnostic. No gold labels or backprop."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import signal
import sys
import time
import numpy as np
from span_math import coordinates_torch, solve_gram, observables


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    temp = path.with_suffix('.pending')
    temp.write_bytes((json.dumps(value, indent=2, allow_nan=False) + '\n').encode())
    temp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    for key in ('root', 'design', 'output', 'progress'):
        ap.add_argument('--' + key, type=Path, required=True)
    ap.add_argument('--attempt', required=True)
    args = ap.parse_args()
    began = time.monotonic()
    cfg = json.loads(args.design.read_bytes())
    assert cfg['rcond'] == 1e-10 and cfg['batch_size'] == 64 and cfg['wall_seconds'] == 600

    def timeout(signum, frame):
        raise TimeoutError('Fixed 600-second diagnostic budget expired; no automatic retry')

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(cfg['wall_seconds'])
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    cursor, forwards = 0, 0

    def progress(phase):
        assert time.monotonic() - began < cfg['wall_seconds']
        save(args.progress, {'schema': 'research.progress.v1', 'attempt': args.attempt, 'phase': phase,
                            'current': cursor, 'total': 20480, 'unit': 'example_pass',
                            'observed_at': time.time(), 'heartbeat_at': time.time()})

    progress('verify_fixed_inputs')
    for name, h in cfg['code_hashes'].items():
        assert digest(args.design.parent / name) == h
    for name, h in cfg['input_hashes'].items():
        assert Path(name).resolve().is_relative_to(args.root.resolve()) and digest(name) == h, name
    f = cfg['fixed_inputs']
    cohort = json.loads(Path(f['cohort_path']).read_bytes())
    fit, assessment = cohort['fit'], cohort['assessment']
    assert (len(fit), len(assessment)) == (8192, 4096)
    assert len(set(fit + assessment)) == 12288
    assert not {cohort['selected_groups'][str(i)] for i in fit} & {cohort['selected_groups'][str(i)] for i in assessment}
    assert cohort['metadata_sha256'] == f['cohort_metadata_sha256']
    assert cohort['target_sha256'] == f['train_cache']['train-targets.npy']
    completed = args.root / f['completed_relative']
    cache = completed / 'cache'
    data = {key: np.load(cache / ('train-' + key + '.npy'), mmap_mode='r', allow_pickle=False)
            for key in ('h', 'roles', 'offsets')}
    assert len(data['offsets']) == 58466 and data['offsets'][-1] == len(data['h']) == len(data['roles'])
    assert min(fit + assessment) >= 0 and max(fit + assessment) < 58465
    sys.path.insert(0, str(args.root / f['source_relative']))
    import torch
    from baselines.retacred.rmea_matched_routers import Head
    from baselines.retacred.rmea_token_transport_study import state_identity
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.device_count() == 1
    device = torch.device('cuda', 0)
    state = torch.load(completed / f['checkpoint_relative'], map_location='cpu', weights_only=True)
    assert state_identity(state) == f['state_sha256']
    model = Head('quantum').double().to(device)
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    effective_before = state_identity(model.state_dict())
    head = model.head
    assert head.mode == 'quantum'
    G = torch.zeros((64, 64), dtype=torch.float64, device=device)
    B = torch.zeros((64, 40), dtype=torch.float64, device=device)
    ys = torch.zeros(40, dtype=torch.float64, device=device)
    y2 = torch.zeros(40, dtype=torch.float64, device=device)
    token_counts = {'fit_statistics': 0, 'fit_replay': 0, 'assessment': 0}
    coefficients = None

    def batches(indices, phase):
        nonlocal cursor, forwards
        for lo in range(0, len(indices), cfg['batch_size']):
            idx = indices[lo:lo + cfg['batch_size']]
            lengths = [int(data['offsets'][i + 1] - data['offsets'][i]) for i in idx]
            n = max(lengths)
            hh = np.zeros((len(idx), n, 1024), np.float64)
            rr = np.zeros((len(idx), n, 4), np.float64)
            mm = np.zeros((len(idx), n), bool)
            for j, i in enumerate(idx):
                a, b = map(int, data['offsets'][i:i + 2])
                hh[j, :b - a] = data['h'][a:b]
                rr[j, :b - a] = data['roles'][a:b]
                mm[j, :b - a] = True
            h, roles, mask = (torch.from_numpy(x).to(device) for x in (hh, rr, mm))
            detail = head(h, roles, mask, True)
            forwards += 1
            X = coordinates_torch(detail['psi'])
            U = detail['utilities'].transpose(1, 2)
            w = mask.to(torch.float64) / mask.sum(-1, keepdim=True)
            assert X.shape == (len(idx), n, 64) and U.shape == (len(idx), n, 40)
            assert torch.isfinite(U).all() and torch.isfinite(X).all()
            token_counts[phase] += sum(lengths)
            yield X, U, w, mask, detail, h
            cursor += len(idx)
            progress(phase)
            if lo % (16 * cfg['batch_size']) == 0:
                print(json.dumps({'phase': phase, 'current': cursor, 'total': 20480}), flush=True)

    with torch.inference_mode():
        for X, U, w, mask, detail, h in batches(fit, 'fit_statistics'):
            x, u, ww = X[mask], U[mask], w[mask]
            G += x.T @ (ww[:, None] * x)
            B += x.T @ (ww[:, None] * u)
            ys += (ww[:, None] * u).sum(0)
            y2 += (ww[:, None] * u.square()).sum(0)
    g, b = G.cpu().numpy(), B.cpu().numpy()
    mu = ys.cpu().numpy() / len(fit)
    variance = y2.cpu().numpy() / len(fit) - mu ** 2
    assert variance.min() >= -1e-10
    variance = np.maximum(variance, 0)
    aq, q_info = solve_gram(g, b, cfg['rcond'])
    ad, d_info = solve_gram(g[:8, :8], b[:8], cfg['rcond'])
    V = observables(aq)
    norms = np.max(np.abs(np.linalg.eigvalsh(V)), axis=-1)
    private = args.output / 'coefficients.private.npz'
    np.savez(private, quantum=aq, diagonal=ad, constant=mu, fit_variance=variance)
    coefficient_hash = digest(private)
    np.savez(args.output / 'sufficient.private.npz', G=g, B=b, sum_u=ys.cpu().numpy(), sum_u2=y2.cpu().numpy())
    coefficients = {'quantum_span': torch.from_numpy(aq).to(device), 'diagonal_span': torch.from_numpy(ad).to(device),
                    'constant': torch.from_numpy(mu).to(device)}
    save(args.output / 'fit-lock.json', {'coefficient_sha256': coefficient_hash, 'fit_examples': 8192,
                                       'assessment_forward_count_at_lock': 0, 'quantum_solver': q_info, 'diagonal_solver': d_info})
    reference_errors = {'numpy_operator_utility': 0., 'numpy_teacher_probability': 0., 'numpy_teacher_logits': 0., 'numpy_span_logits': 0.}
    checked_examples = 0
    metrics = {}
    with torch.inference_mode():
        for split, indices, phase in [('fit', fit, 'fit_replay'), ('assessment', assessment, 'assessment')]:
            assert digest(private) == coefficient_hash
            sums = {k: np.zeros(40) for k in coefficients}
            agree = {k: 0 for k in coefficients}
            total = 0
            for X, U, w, mask, detail, h in batches(indices, phase):
                teacher = detail['logits']
                pred = teacher.argmax(-1)
                approximation = {'quantum_span': X @ coefficients['quantum_span'],
                                 'diagonal_span': X[..., :8] @ coefficients['diagonal_span'],
                                 'constant': coefficients['constant'].expand_as(U)}
                logits = {}
                for kind, value in approximation.items():
                    sums[kind] += (w[..., None] * (value - U).square()).sum((0, 1)).cpu().numpy()
                    logits[kind] = (detail['alpha'] * value.transpose(1, 2)).sum(-1) + head.bias
                    agree[kind] += int((logits[kind].argmax(-1) == pred).sum())
                if split == 'assessment' and checked_examples == 0:
                    count = min(2, len(U))
                    psi = detail['psi'][:count].cpu().numpy()
                    phi = detail['phi'].cpu().numpy()
                    values = np.einsum('bti,cij,btj->btc', psi.conj(), V, psi).real
                    qhat = approximation['quantum_span'][:count].cpu().numpy()
                    probability = np.abs(np.einsum('ci,bti->bct', phi.conj(), psi)) ** 2
                    m = mask[:count].cpu().numpy()
                    raw = (probability + 1e-6) * m[:, None]
                    alpha = raw / raw.sum(-1, keepdims=True)
                    hcpu = h[:count].cpu().numpy()
                    teacher_value = (hcpu @ head.value.weight.cpu().numpy().T + head.value.bias.cpu().numpy()) @ head.decoder.cpu().numpy().T
                    teacher_logits = (alpha * teacher_value.transpose(0, 2, 1)).sum(-1) + head.bias.cpu().numpy()
                    span_logits = (alpha * values.transpose(0, 2, 1)).sum(-1) + head.bias.cpu().numpy()
                    reference_errors['numpy_operator_utility'] = float(np.max(np.abs(values - qhat)))
                    reference_errors['numpy_teacher_probability'] = float(np.max(np.abs(probability - detail['probability'][:count].cpu().numpy())))
                    reference_errors['numpy_teacher_logits'] = float(np.max(np.abs(teacher_logits - teacher[:count].cpu().numpy())))
                    reference_errors['numpy_span_logits'] = float(np.max(np.abs(span_logits - logits['quantum_span'][:count].cpu().numpy())))
                    assert max(reference_errors.values()) < 1e-7, reference_errors
                    checked_examples = count
                total += len(U)
            assert total == len(indices)
            metrics[split] = {}
            for kind in coefficients:
                mse = sums[kind] / total
                normalized = mse / (variance + 1e-12)
                metrics[split][kind] = {'MSE_all40': mse.tolist(), 'normalized_MSE_all40': normalized.tolist(),
                                       'mean_normalized_MSE': float(normalized.mean()), 'max_normalized_MSE': float(normalized.max()),
                                       'teacher_argmax_agreement': agree[kind] / total, 'agreement_count': agree[kind], 'examples': total}
    assert forwards == 320 and cursor == 20480 and checked_examples == 2
    assert state_identity(model.state_dict()) == effective_before
    assert digest(private) == coefficient_hash
    result = metrics['assessment']['quantum_span']
    gates = {'mean_MSE': result['mean_normalized_MSE'] <= .01, 'each_class_MSE': result['max_normalized_MSE'] <= .04,
             'teacher_agreement': result['teacher_argmax_agreement'] >= .99}
    verification = {'status': 'PASS_NUMERICAL_REPLAY_VERIFICATION', 'numpy_matrix_replay_examples': checked_examples,
                    'max_errors': reference_errors, 'coefficient_hash_unchanged': True, 'effective_teacher_state_unchanged': True,
                    'stored_state_sha256': f['state_sha256'], 'float64_effective_state_sha256': effective_before,
                    'assessment_forward_count_at_fit_lock': 0, 'decoded_gold_labels': 0,
                    'scope': 'Independent NumPy operator/overlap/pooling on two reused examples; not independent circuit implementation, task confirmation or external research review.'}
    summary = {'schema': 'qattn.observable-value-span.v1', 'status': 'COMPLETE',
               'decision': 'RETAIN_INFORMATION_QUESTION_ONLY' if all(gates.values()) else 'STOP_FIXED_REPRESENTATION_VALUE_REPLACEMENT',
               'gates': gates, 'metrics': metrics, 'fit_teacher_variance_all40': variance.tolist(),
               'quantum_solver': q_info, 'diagonal_solver': d_info, 'quantum_operator_norm_all40': norms.tolist(),
               'native_classical_value_reference': 'Exact teacher by definition; not a fitted utility result or substitute mainline.',
               'private_coefficient_sha256': coefficient_hash, 'token_counts_by_pass': token_counts,
               'unique_TRAIN_feature_examples': 12288, 'example_passes': cursor, 'native_head_forwards': forwards,
               'native_circuit_executions': 2 * forwards, 'optimizer_updates': 0, 'linear_solutions': 2,
               'gold_label_decodes': 0, 'new_official_VALID_TEST_examples': 0, 'GPU_count': 1,
               'elapsed_seconds': time.monotonic() - began, 'float_dtype': 'float64_replay_of_fixed_stored_values',
               'limitations': ['Teacher was trained on full TRAIN; split isolation applies only to this linear solution.',
                               'Unconstrained Hermitian relaxation does not establish bounded-norm feasibility.',
                               'Failure concerns this fixed representation only; no universal joint-training impossibility.',
                               'No new task utility, new theory, external review or independent confirmation.'],
               'goal': 'ACTIVE', 'newL3': 'NOT_ACCEPTED', 'newL4T': 'NOT_ACCEPTED'}
    assert summary['elapsed_seconds'] < 600
    save(args.output / 'verification.json', verification)
    save(args.output / 'summary.json', summary)
    save(args.output / 'RUN_COMPLETE', {'summary_sha256': digest(args.output / 'summary.json'),
                                       'verification_sha256': digest(args.output / 'verification.json'), 'attempt': args.attempt})
    progress('completed_observable_span')
    signal.alarm(0)
    print(json.dumps({'status': 'COMPLETE', 'gates': gates, 'assessment': result,
                      'elapsed_seconds': summary['elapsed_seconds'], 'native_forwards': forwards}), flush=True)


if __name__ == '__main__':
    main()
