
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,time
import numpy as np
t=time.monotonic();root=Path('/hy-tmp/Q-Attention');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
raw=root/'inputs/native-baseline/recovery-inputs/train.jsonl'
assert sha(raw)=='f1051d574a44df3ecbd036c1de4fbbe0a372c768f6ec990330e2cea4a657269f'
rows=[json.loads(s) for s in raw.read_bytes().splitlines() if s.strip()]
cache=root/'inputs/native-baseline/discovery/cache'
assert sha(cache/'metadata.json')=='9e6eac9ad582a4ef4e1a8a12190bfef3b8f0a18c5efbd9072f91f31563656c0e'
assert sha(cache/'train-targets.npy')=='6ce409b586a421995f7d70906f062a7e91fe9d35a04be5359a2f58a3485ee604'
metadata=json.loads((cache/'metadata.json').read_bytes())['train'];labels=np.load(cache/'train-targets.npy',allow_pickle=False)
lookup={hashlib.sha256(str(r['metadata']['id']).encode()).hexdigest():r for r in rows}
assert len(rows)==len(lookup)==len(metadata)==len(labels)==58465
ordered=[lookup[m['id_hash']] for m in metadata]
groups=[hashlib.sha256(json.dumps(r['tokens'],ensure_ascii=False,separators=(',',':')).encode()).hexdigest() for r in ordered]
assert len(set(groups))==33477
label_mapping={}
for r,y in zip(ordered,labels):
    if r['label'] in label_mapping:assert label_mapping[r['label']]==int(y)
    else:label_mapping[r['label']]=int(y)
assert len(label_mapping)==len(set(label_mapping.values()))==40
cp=root/'control/staged-bottleneck-design-20261125/cohort.private.json'
assert sha(cp)=='8fb8393bbe2241929748283f2bbaa219682696409857ec15678585bcd0b88ff0'
cohort=json.loads(cp.read_bytes());fit=cohort['fit'];assessment=cohort['assessment']
assert len(fit)==8192 and len(assessment)==4096 and not set(fit)&set(assessment)
assert not {groups[i] for i in fit}&{groups[i] for i in assessment}
assert all(groups[i]==cohort['selected_groups'][str(i)] for i in fit+assessment)

def analyze(indices):
    grouped=defaultdict(list)
    for i in indices:
        r=ordered[i];s=tuple(r['subject']);o=tuple(r['object'])
        assert len(s)==len(o)==2 and all(isinstance(v,int) for v in s+o)
        assert 0<=s[0]<=s[1]<=len(r['tokens']) and 0<=o[0]<=o[1]<=len(r['tokens'])
        assert s!=o
        grouped[groups[i]].append((s,o,int(labels[i])))
    row_hist=Counter();edge_hist=Counter();node_hist=Counter()
    multiple=connected3=cycle_groups=shared_rows=recip_groups=recip_pairs=ambig_recip=duplicate_rows=conflicted_edges=0
    matrix=np.zeros((40,40),dtype=np.int64);self_loops=0
    for items in grouped.values():
        by_edge=defaultdict(set)
        for s,o,y in items:by_edge[(s,o)].add(y)
        nodes={n for e in by_edge for n in e};adj={n:set() for n in nodes}
        undirected={frozenset(e) for e in by_edge}
        for e in undirected:
            assert len(e)==2
            a,b=tuple(e);adj[a].add(b);adj[b].add(a)
        components=[];visited=set()
        for n in nodes:
            if n in visited:continue
            stack=[n];size=0
            while stack:
                v=stack.pop()
                if v in visited:continue
                visited.add(v);size+=1;stack.extend(adj[v]-visited)
            components.append(size)
        row_hist[len(items)]+=1;edge_hist[len(by_edge)]+=1;node_hist[len(nodes)]+=1
        multiple+=len(by_edge)>1;connected3+=max(components)>=3
        cycle_groups+=len(undirected)-len(nodes)+len(components)>0
        incident=Counter()
        for s,o in by_edge:incident[s]+=1;incident[o]+=1
        for s,o,y in items:
            # A reversed query counts as a distinct coedge, duplicate same query does not.
            n_other=incident[s]+incident[o]-2-(1 if (o,s) in by_edge else 0)
            shared_rows+=n_other>0
        duplicate_rows+=len(items)-len(by_edge)
        conflicted_edges+=sum(len(v)>1 for v in by_edge.values())
        seen=set();this_recip=0
        for (s,o),ys in by_edge.items():
            if (o,s) not in by_edge or frozenset((s,o)) in seen:continue
            seen.add(frozenset((s,o)));recip_pairs+=1;this_recip+=1
            zs=by_edge[(o,s)]
            if len(ys)!=1 or len(zs)!=1:ambig_recip+=1;continue
            a=next(iter(ys));b=next(iter(zs));matrix[a,b]+=1;matrix[b,a]+=1
        recip_groups+=this_recip>0
        assert time.monotonic()-t<45,'CPU cap'
    assert sum(k*v for k,v in row_hist.items())==len(indices)
    assert int(matrix.sum())==2*(recip_pairs-ambig_recip)
    assert np.array_equal(matrix,matrix.T)
    byclass=[]
    for i,row in enumerate(matrix):
        nz=np.flatnonzero(row)
        byclass.append({'class_id':i,'directed_observed_pairs':int(row.sum()),'distinct_observed_reverse_labels':len(nz),
                        'reverse_labels':[{'class_id':int(j),'count':int(row[j])} for j in nz]})
    return {'rows':len(indices),'sentence_groups':len(grouped),'groups_with_multiple_directed_edges':int(multiple),
       'groups_with_connected3plus_entities':int(connected3),'groups_with_undirected_cycle':int(cycle_groups),
       'rows_with_distinct_sharing_coedge':int(shared_rows),'groups_with_reciprocal_target_pair':int(recip_groups),
       'reciprocal_unordered_span_pairs':int(recip_pairs),'ambiguous_label_reciprocal_pairs':int(ambig_recip),
       'duplicate_same_edge_rows':int(duplicate_rows),'same_edge_multiple_distinct_labels':int(conflicted_edges),
       'rows_per_group_histogram':dict(sorted(row_hist.items())),
       'unique_directed_edges_per_group_histogram':dict(sorted(edge_hist.items())),
       'entity_spans_per_group_histogram':dict(sorted(node_hist.items())),
       'observed_reverse_label_matrix':matrix.tolist(),'observed_reverse_by_class':byclass}

results={'canonical_train':analyze(range(len(ordered))),'original_fit8192':analyze(fit)}
native=root/'owner/attempts/9a28320fd4174c8f95544c9ba96eb1ae/source/baselines/retacred/rmea.py'
assert sha(native)=='9f4bfc57a1a966c3383c1a2e82fe406ac3d115b653b062290fcf42d3c4683ee0'
print(json.dumps({'status':'PASS_TRAIN_STRUCTURAL_AVAILABILITY_NOT_INFERENCE_QUALIFICATION','source_sha256':sha(raw),
 'cohort_sha256':sha(cp),'native_source_sha256':sha(native),'results':results,'seconds':time.monotonic()-t,
 'new_fits':0,'new_model_forwards':0,'gpu_initializations':0,'parser_runs':0,'valid_test_read':False,
 'remote_file_writes':0,'raw_rows_or_graphs_exported':False,
 'limits':['TRAIN annotated graphs require other target-row span annotations absent from the qualified single-pair query interface.',
           'Token-hash group is exact same token list, not established document/entity linkage.',
           'Missing target edge is unobserved, not null; inverse table is empirical not logical constraint.',
           'FullTRAIN and oldFIT are historically exposed development data, not independent evidence.']}))
