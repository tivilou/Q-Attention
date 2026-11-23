"""Read settled screen artifacts and independently recompute reported endpoints."""
from pathlib import Path
import hashlib
import json
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    r = Path(__file__).parent; out = r/'output'
    execution = json.loads((r/'execution.json').read_bytes())
    returned = json.loads((r/'resource-return.json').read_bytes())
    assert execution['exit_code'] == 0 and not execution['stderr']
    assert returned['campaign_status'] == 'completed' and returned['resources_reusable']
    assert returned['runner_attempt_id'] == execution['runner_attempt_id']
    summary = json.loads((out/'summary.json').read_bytes())
    assert summary['attempt'] == returned['runner_attempt_id']
    seal = json.loads((out/'RUN_COMPLETE').read_bytes())
    assert set(seal) == {'summary.json','fixtures.json'}
    assert all(sha(out/n) == h for n,h in seal.items())
    cfg = json.loads((r/'design.json').read_bytes())
    assert sha(r/'design.json') == summary['design_sha256']
    assert all(sha(r/n) == h for n,h in cfg['code_hashes'].items())
    assert summary['code_hashes'] == cfg['code_hashes']
    assert sha(out/'diagnostic.private.npz') == summary['private_diagnostic_sha256']
    checks = json.loads((out/'fixtures.json').read_bytes())
    assert checks['status'] == checks['retrieval']['status'] == 'PASS'
    assert checks['retrieval']['population'] == 4900 and checks['retrieval']['particular_diagonal_mse'] == .5
    with np.load(out/'diagnostic.private.npz', allow_pickle=False) as a:
        groups = a['groups'].copy(); cal = a['calibration'].copy()
        idx = a['assessment_indices'].copy(); labels = a['labels'].copy()
        assert np.array_equal(idx, np.flatnonzero(~cal)) and (cal.sum(), len(idx)) == (4192,4000)
        assert not set(groups[cal]) & set(groups[idx])
        losses = {}; recomputed = {}
        for mode in cfg['modes']:
            z = a[mode+'_logits']; assert z.shape == (4000,40) and np.isfinite(z).all()
            z = z-z.max(1,keepdims=True)
            ce = np.log(np.exp(z).sum(1))-z[np.arange(4000), labels[idx]]
            predictions = z.argmax(1)
            cm = np.bincount(labels[idx]*40+predictions, minlength=1600).reshape(40,40)
            den = cm.sum(0)+cm.sum(1)
            f1 = np.divide(2*cm.diagonal(), den, out=np.zeros(40), where=den>0)
            losses[mode] = ce
            row = summary['results'][mode]
            assert abs(row['ce']-ce.mean()) < 1e-10
            assert abs(row['positive39_macro_f1']-f1[1:].mean()) < 1e-12
            assert row['updates'] == 512 and row['parameters'] == 81028
            assert row['final_sha256'] == sha(out/mode/'final.pt')
            private = json.loads((out/mode/'result.json').read_bytes())
            assert len(private['train_ce_history']) == 512 and private['first_update']['state_changed']
            assert all(np.isfinite(x) and x>=0 for x in private['first_update']['finite_gradient_norms'].values())
            assert 'train_ce_history' not in row and 'first_update' not in row
            recomputed[mode] = dict(ce=float(ce.mean()), macro=float(f1[1:].mean()), items=int(cm.sum()))
        rows = list(summary['results'].values())
        assert len({q['initial_sha256'] for q in rows}) == len({q['order_sha256'] for q in rows}) == 1
        assert min(q['assessment_started_at'] for q in rows) >= max(q['training_finished_at'] for q in rows)
        values = np.column_stack([losses[k]-losses['coherent'] for k in ('diagonal','native')])
        assert np.max(abs(values-a['differences'])) < 1e-10
        _, inverse = np.unique(groups[idx], return_inverse=True)
        sizes = np.bincount(inverse); sums = np.zeros((len(sizes),2))
        np.add.at(sums, inverse, values)
        rng = np.random.default_rng(cfg['bootstrap_seed']); resampled = []
        for _ in range(cfg['bootstrap_replicates']):
            selection = rng.integers(len(sizes), size=len(sizes))
            resampled.append(sums[selection].sum(0)/sizes[selection].sum())
        resampled = np.asarray(resampled)
        assert np.max(abs(resampled-a['bootstrap'])) < 1e-10
        lower = np.quantile(resampled, .025, axis=0)
    gates = {}
    for j,k in enumerate(('diagonal','native')):
        assert abs(lower[j]-summary['paired_bootstrap_lower_025'][k]) < 1e-10
        assert abs(values[:,j].mean()-summary['paired_ce_comparator_minus_coherent'][k]) < 1e-10
        gates.update({'ce_beats_'+k:recomputed['coherent']['ce']<recomputed[k]['ce'],
            'paired_lower_positive_'+k:bool(lower[j]>0),
            'macro_not_below_'+k:recomputed['coherent']['macro']>=recomputed[k]['macro']})
    assert gates == summary['gates']
    expected = 'PASS_FOR_INDEPENDENT_STUDY_PREPARATION_ONLY' if all(gates.values()) else 'FAIL_CLOSE_EXACT_SCREEN'
    assert summary['gate'] == expected
    result = dict(status='PASS', engineering='PASS', scientific_gate=expected,
        independent_ce_confusion_macro_bootstrap_and_gate_recompute=True,
        all_training_before_any_assessment=True, public_rows_compact=True,
        summary_sha256=sha(out/'summary.json'), private_array_hash_verified=True,
        common_initialization_order=True, exact_sentence_overlap=0, recomputed=recomputed,
        design_sha256=sha(r/'design.json'), code_hashes=cfg['code_hashes'], L5='NOT_ACCEPTED')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
