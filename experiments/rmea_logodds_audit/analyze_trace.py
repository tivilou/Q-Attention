"""One label-free, forward-free analysis of an immutable selected-head trace.

This is a posthoc descriptive oracle, not a trained low-rank replacement.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import numpy as np

CONDITIONS = ('uniform','relation_softmax','real_overlap','quantum','dephased')
CLOSED_SHA = 'b567555cd4ce7fe98539b369ef624e8f9f26e696fbdff658a03d10b7104bf603'
TRACE_SHA = '5ebfda50e59ca9dbe0ddab1ae3a55498f337f4dadfef304a44cb6a4fd539ff62'
READ_LIMIT = 20 * 1024 * 1024
RANK = 8
FEATURE_RCOND = 1e-6
PARITY_ATOL = 1e-4


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()


def softmax_by_word(logits):
    shifted = logits - logits.max(axis=1,keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1,keepdims=True)


def score(alpha,utility,bias):
    return np.sum(np.ascontiguousarray(alpha*utility),axis=1,dtype=np.float64)+bias


def load_closed(root):
    root = root.resolve(strict=True)
    def pinned(name,digest):
        path = root/name
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
            raise ValueError('Artifact containment failed')
        raw = path.read_bytes()
        if sha(raw) != digest:
            raise ValueError('Immutable artifact hash mismatch')
        return path,raw
    _,raw = pinned('RUN_COMPLETE',CLOSED_SHA)
    closed = json.loads(raw)
    assert closed['schema'] == 'qattn.allocation-audit.closed.v1'
    trace_name = 'private/sample-trace.json'
    assert closed['files'][trace_name] == TRACE_SHA
    _,raw = pinned(trace_name,TRACE_SHA)
    trace = json.loads(raw)
    assert trace['schema_version'] == 'sample-trace.v2' and len(trace['samples']) == 35
    groups = {v:{} for v in CONDITIONS}
    refs = {}
    for sample in trace['samples']:
        variant = sample['roles']['provenance']['condition']
        assert variant in groups and sample['split']=='valid' and sample['lane']=='replay' and sample['checkpoint']=='selected'
        case = sample['roles']['source_sample']['id']
        assert case not in groups[variant]
        reps = {r['id']:r for r in sample['representations']}
        required = {'alpha','utilities','bias','mask','logits','features'}
        if variant == 'relation_softmax':
            required |= {'key','query'}
        assert required <= reps.keys()
        artifacts = {r['artifact_refs'][0]['path'] for name,r in reps.items() if name in required}
        assert len(artifacts) == 1
        relative = 'private/'+artifacts.pop()
        assert relative.startswith('private/trace/') and relative.endswith('.npz')
        for name in required:
            rep = reps[name]
            assert rep['status']=='observed' and rep['value_capture']['status']=='complete'
            assert rep['artifact_refs'][0]['sha256'] == closed['files'][relative]
        path = root/relative
        assert not path.is_symlink() and path.resolve(strict=True).is_relative_to(root)
        refs[relative] = dict(bytes=path.stat().st_size,sha256=closed['files'][relative])
        groups[variant][case] = (relative,reps,required)
    cases = sorted(groups['quantum'])
    assert len(cases)==7 and all(sorted(g)==cases for g in groups.values()) and len(refs)==35
    assert sum(x['bytes'] for x in refs.values()) <= READ_LIMIT
    arrays = {v:[] for v in CONDITIONS}
    for variant in CONDITIONS:
        for case in cases:
            relative,reps,required = groups[variant][case]
            path,_ = pinned(relative,refs[relative]['sha256'])
            with np.load(path,allow_pickle=False) as archive:
                item = {}
                for name in required:
                    a = archive[name]
                    assert list(a.shape)==reps[name]['shape'] and str(a.dtype)==reps[name]['dtype']
                    assert np.isfinite(a).all()
                    item[name] = np.asarray(a,dtype=np.float64)
            t = item['mask'].shape[-1]
            assert item['mask'].shape==(1,t) and item['mask'].all()
            assert item['alpha'].shape==item['utilities'].shape==(1,40,t)
            assert item['bias'].shape==(40,) and item['logits'].shape==(1,40)
            assert item['features'].shape==(1,t,1028)
            a = item['alpha'][0]
            assert (a>np.finfo(np.float32).tiny).all() and np.max(abs(a.sum(axis=1)-1))<5e-6
            arrays[variant].append(item)
    return arrays,dict(packages=35,bytes=sum(x['bytes'] for x in refs.values()),closure_sha256=CLOSED_SHA,
                       trace_sha256=TRACE_SHA,matched_cases=7,conditions=5,
                       cases_identity_sha256=sha(encoded(cases)),read_limit=READ_LIMIT)


def analyze(arrays,identity):
    reference = arrays['quantum']
    feature_rows = []
    lengths = []
    for i,item in enumerate(reference):
        f = item['features'][0]
        lengths.append(len(f))
        feature_rows.append(f[1:]-f[0])
        assert all(np.array_equal(arrays[v][i]['features'],item['features']) for v in CONDITIONS)
    d = np.concatenate(feature_rows,axis=0)
    ud,sd,vhd = np.linalg.svd(d,full_matrices=False)
    keep = sd>FEATURE_RCOND*sd[0]
    full_row_rank = bool(keep.all())
    dplus = (vhd[keep].T/sd[keep]) @ ud[:,keep].T
    results = {}
    for variant in CONDITIONS:
        items = arrays[variant]
        l = np.concatenate([(np.log(v['alpha'][0])-np.log(v['alpha'][0,:,0:1])).T[1:] for v in items],axis=0)
        left,singular,right = np.linalg.svd(l,full_matrices=False)
        z = left[:,:RANK]*singular[:RANK]
        q = np.sqrt(RANK)*right[:RANK].T
        l8 = z @ q.T/np.sqrt(RANK)
        key_coefficients = dplus @ z
        compiled = d @ key_coefficients @ q.T/np.sqrt(RANK)
        compilation_error = float(np.max(abs(compiled-l8)))
        # Compilability is conditional; no fallback, tuning, or case exclusion.
        compiled_ok = bool(full_row_rank and compilation_error<PARITY_ATOL)
        native_error,normalization_error,projection_error = [],[],[]
        original_margins,changes,near_ties,l1 = [],0,0,[]
        row = 0
        for item,t in zip(items,lengths):
            alpha,utility,bias = item['alpha'][0],item['utilities'][0],item['bias']
            normalized = alpha/alpha.sum(axis=1,keepdims=True)
            target = np.zeros((40,t),dtype=np.float64)
            target[:,1:] = l8[row:row+t-1].T
            row += t-1
            projected = softmax_by_word(target)
            orig_score = score(alpha,utility,bias)
            norm_score = score(normalized,utility,bias)
            projected_score = score(projected,utility,bias)
            err = float(np.max(abs(orig_score-item['logits'][0])))
            assert err<PARITY_ATOL
            native_error.append(err)
            normalization_error.append(float(np.max(abs(norm_score-orig_score))))
            deviation = float(np.max(abs(projected_score-norm_score)))
            projection_error.append(deviation)
            top = np.sort(norm_score)[-2:]
            margin = float(top[1]-top[0])
            original_margins.append(margin)
            changes += int(np.argmax(projected_score)!=np.argmax(norm_score))
            near_ties += int(margin<=2*deviation)
            l1.append(float(np.mean(np.sum(abs(projected-normalized),axis=1))))
        assert row==len(l)
        denom = float(np.linalg.norm(l,'fro'))
        tail = float(np.linalg.norm(singular[RANK:]))
        control = None
        if variant=='relation_softmax':
            exact = np.concatenate([((v['key'][0,1:]-v['key'][0,0]) @ v['query'].T)/np.sqrt(RANK) for v in items],axis=0)
            residual = float(np.max(abs(exact-l)))
            assert residual<PARITY_ATOL
            control = dict(max_log_ratio_vs_stored_key_query_error=residual,
                           tolerance=PARITY_ATOL,interpretation='Stored fp32 attention calibration against exact real-arithmetic rank<=8 factorization; not a certified circuit rank bound.')
        results[variant] = dict(log_ratio_shape=list(l.shape),singular_values=singular.tolist(),
            rank8_tail_frobenius=tail,rank8_tail_relative_frobenius=tail/denom if denom else 0.0,
            max_native_logit_recomposition_error=max(native_error),
            max_alpha_normalization_logit_error=max(normalization_error),
            max_rank8_projection_logit_change=max(projection_error),mean_rank8_projection_logit_change=float(np.mean(projection_error)),
            mean_attention_l1_change=float(np.mean(l1)),prediction_changes=changes,
            cases_not_certified_by_simple_2delta_margin_bound=near_ties,
            min_original_normalized_margin=min(original_margins),
            compiled_into_actual_affine_key_parameterization_on_same_cases=compiled_ok,
            compilation_max_log_ratio_residual=compilation_error,
            least_squares_key_norm=float(np.linalg.norm(key_coefficients,'fro')),
            query_norm=float(np.linalg.norm(q,'fro')),classical_calibration=control)
    return dict(schema='qattn.logodds-static-audit.v1',status='PASS',input_identity=identity,
        worker_sha256=sha(Path(__file__).read_bytes()),numpy_version=np.__version__,
        design='All seven pre-existing posthoc cases and all five fixed selected seed29 heads; all valid words, anchor=first valid word; rank8 SVD fit once to each full stacked log-ratio matrix, no label inputs.',
        features=dict(shape=list(d.shape),rcond=FEATURE_RCOND,rank_at_rcond=int(keep.sum()),
                      full_row_rank_at_rcond=full_row_rank,sigma_max=float(sd[0]),sigma_min=float(sd[-1]),
                      all_five_conditions_identical=True),
        results=results,rank=RANK,parity_tolerance=PARITY_ATOL,
        new_model_forwards=0,checkpoint_loads=0,encoder_forwards=0,gradient_training_runs=0,
        deterministic_matrix_fits=5,test_examples=0,target_labels_read=0,cos_reads=0,gpus=0,
        scope='Posthoc descriptive static-array transformation and same-case least-squares oracle. Not a trained replacement, new utility result, exact circuit rank certification, or independent confirmation.',
        limits=['Spectrum alone cannot identify task necessity. Projection damage does not exclude other output-preserving low-rank scores.',
                'A small or zero change on these outcome-selected cases is not a population-level null result.',
                'Full-row-rank compilation uses teacher attention on these same cases; does not establish optimization, bounded-norm generalization, or unseen-input fidelity.',
                'Fixed utilities/bias are from each recipient head; no cross-head transplant or label-based fitting.',
                'No exact deployed-head forward was performed; actual-head parameterization correspondence is algebraic on these finite features.'],
        novelty_established=False,l5_accepted=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root',type=Path,required=True)
    parser.add_argument('--out-dir',type=Path,required=True)
    args = parser.parse_args()
    assert not args.out_dir.exists(), 'Fresh output child required; no overwrite/replay'
    arrays,identity = load_closed(args.input_root)
    result = analyze(arrays,identity)
    args.out_dir.mkdir(parents=True,exist_ok=False)
    raw = encoded(result)
    (args.out_dir/'summary.json').write_bytes(raw)
    complete = dict(schema='qattn.logodds-static-audit.closed.v1',files={'summary.json':sha(raw)},worker_sha256=result['worker_sha256'],input_closure_sha256=CLOSED_SHA,new_model_forwards=0)
    (args.out_dir/'RUN_COMPLETE').write_bytes(encoded(complete))
    print(json.dumps(dict(status='PASS',cases=7,conditions=5,new_model_forwards=0,gradient_training_runs=0,deterministic_matrix_fits=5)))


if __name__=='__main__':
    main()
