"""One deterministic synthetic witness using the unmodified original RMEA head.

No training, data, checkpoint, encoder, GPU, search or formal task evaluation.
Run only after the exact reviewed contract and shared CPU admission are recorded.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

SOURCES = {
    'experiments/rmea_semantic_probe/legacy/rmea.py': '9f4bfc57a1a966c3383c1a2e82fe406ac3d115b653b062290fcf42d3c4683ee0',
    'experiments/rmea_semantic_probe/legacy/frozen_feature_head.py': 'f4e63a43994d2e241c9ae30497dcbcd4bc3a98f04ec6b16746b1e721371b6af9',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.source_root.resolve(strict=True)
    for name, expected in SOURCES.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    # Fail on an existing output. A failed/completed attempt is inspected, never silently repeated.
    out = args.out_dir.resolve()
    assert out != root and not out.is_relative_to(root), 'Private fixture output must be outside the Git worktree'
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(root))
    import torch
    from experiments.rmea_semantic_probe.legacy.rmea import RMEAHead, EPS

    torch.set_num_threads(1)
    torch.manual_seed(0)  # Constructors initialize randomly; every parameter is overwritten below.
    assert EPS == 1e-6
    fractions = [-.99, -.75, -.5, -.25, 0., .25, .5, .75, .99]
    x = [math.atanh(v) for v in fractions]
    h = torch.zeros((9, 2, 1024), dtype=torch.float64, device='cpu')
    h[:, 0, 0] = torch.tensor(x, dtype=torch.float64)
    h[:, 0, 1] = 1
    h[:, 1, 0] = math.atanh(1 / 8)
    roles = torch.zeros((9, 2, 4), dtype=torch.float64, device='cpu')
    mask = torch.ones((9, 2), dtype=torch.bool, device='cpu')
    head = RMEAHead('quantum').double().cpu().eval()
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
        for layer in range(2):
            for wire in range(3):
                head.prepare.weight[layer * 6 + wire * 2 + 1, 0] = 1
                if layer == 0:
                    head.prepare.bias[wire * 2] = math.atanh(.5)
        head.effect[0, 1, 2, 0] = math.atanh(.5)
        head.effect[1, 1, 2, 0] = math.atanh(-.5)
        head.value.weight[0, 1] = 1
        head.decoder[0, 0] = head.decoder[1, 0] = 1
        head.bias.fill_(-2)
        head.bias[:2] = 0
        details = head(h, roles, mask, details=True)  # Exactly one constructed-head batch forward.
    cosine = torch.tensor([math.cos(4 * math.pi * v) for v in fractions], dtype=torch.float64)
    q_expected = torch.empty((9, 2, 2), dtype=torch.float64)
    q_expected[:, 0, 0] = (1 + cosine) / 8
    q_expected[:, 1, 0] = (1 - cosine) / 8
    q_expected[:, :, 1] = 1 / 8
    probability = details['probability'][:, :2]
    assert torch.isfinite(probability).all()
    q_error = float((probability - q_expected).abs().max())
    assert q_error < 1e-12, q_error
    expected_logits = (q_expected[:, :, 0] + EPS) / (q_expected[:, :, 0] + 1 / 8 + 2 * EPS)
    logits = details['logits']
    logit_error = float((logits[:, :2] - expected_logits).abs().max())
    assert logit_error < 1e-12, logit_error
    expected_labels = [0, 1, 0, 1, 0, 1, 0, 1, 0]
    assert logits.argmax(-1).tolist() == expected_labels
    difference = logits[:, 0] - logits[:, 1]
    signed_margin = difference * torch.tensor([1 if r == 0 else -1 for r in expected_labels], dtype=torch.float64)
    assert float(signed_margin.min()) > .49
    assert int((difference[:-1] * difference[1:] < 0).sum()) == 8
    result = dict(schema='qattn.affine-curve.synthetic-fixture.v1', status='PASS',
        worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), source_sha256=SOURCES,
        source_scope='Two modules byte-identical to original formal f3f22660b609811e6fb824f7d19e9cafd0c1e446; not whole-pipeline equivalence.',
        theta_over_pi=fractions, x=x, epsilon=EPS,
        probability=probability.tolist(), expected_probability=q_expected.tolist(), probability_max_error=q_error,
        logits_first_two=logits[:, :2].tolist(), expected_logits=expected_logits.tolist(), logit_max_error=logit_error,
        pairwise_score_difference=difference.tolist(), signed_margin=signed_margin.tolist(), minimum_signed_margin=float(signed_margin.min()),
        expected_labels=expected_labels, constructed_head_labels=logits.argmax(-1).tolist(), strict_sign_changes=8,
        classical_bound=dict(scope='Any existing relation_softmax parameters on this fixed two-token affine feature interface, including learned value map/decoder/bias.',
                             nonzero_pairwise_score_difference_zero_limit=7, proof_executed_as_numeric_test=False,
                             consequence='No nine alternating strict signed margins; no claim of trained-classifier task accuracy.'),
        model_forwards=1, constructed_inputs=9, training_runs=0, checkpoint_reads=0, real_encoder_forwards=0,
        dataset_reads=0, gpu_use=False, novelty_established=False, natural_feature_support_established=False,
        l5_accepted=False)
    target = out / 'affine-curve-result.json'
    pending = out / 'affine-curve-result.pending'
    pending.write_bytes((json.dumps(result, indent=2) + '\n').encode())
    pending.replace(target)
    assert json.loads(target.read_bytes()) == result
    print(json.dumps(dict(status='PASS', result=str(target), probability_max_error=q_error,
                         logit_max_error=logit_error, minimum_signed_margin=float(signed_margin.min()),
                         strict_sign_changes=8, model_forwards=1, training_runs=0, l5_accepted=False)))


if __name__ == '__main__':
    main()
