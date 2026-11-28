"""Read-only contrasting entity and context variations on the original frozen FIT."""
from collections import defaultdict
from hashlib import sha256
from itertools import combinations
from pathlib import Path
import json
import sys


def checked(pin):
    raw=Path(pin['path']).read_bytes()
    assert sha256(raw).hexdigest()==pin['sha256']
    return raw


def census(rows):
    entity=defaultdict(list)
    context=defaultdict(list)
    labels={}
    for i,(tokens,s,o,types,label) in rows.items():
        assert all(0<=a<b<=len(tokens) for a,b in (s,o))
        assert s[1]<=o[0] or o[1]<=s[0]
        base=(s,o,types)
        labels[i]=label
        for role,(a,b) in enumerate((s,o)):
            entity[(base,role,tokens[:a],tokens[b:])].append((i,tokens[a:b]))
        for j,token in enumerate(tokens):
            if s[0]<=j<s[1] or o[0]<=j<o[1]:
                continue
            context[(base,j,tokens[:j],tokens[j+1:])].append((i,token))

    def compare(groups):
        same=set()
        changed=set()
        support_groups=[0,0]
        for candidates in groups.values():
            a_group=False
            b_group=False
            for (i,a),(j,b) in combinations(candidates,2):
                if a==b:
                    continue
                pair=(min(i,j),max(i,j))
                if labels[i]==labels[j]:
                    same.add(pair)
                    a_group=True
                else:
                    changed.add(pair)
                    b_group=True
            support_groups[0]+=int(a_group)
            support_groups[1]+=int(b_group)
        assert not same & changed
        positive_same={p for p in same if labels[p[0]]!=0}
        positive_changed={p for p in changed if labels[p[0]] and labels[p[1]]}
        metrics=dict(same_label_pairs=len(same),different_label_pairs=len(changed),
                     same_positive_pairs=len(positive_same),different_positive_pairs=len(positive_changed),
                     null_positive_pairs=len(changed)-len(positive_changed),
                     groups_with_same_label_pairs=support_groups[0],groups_with_different_label_pairs=support_groups[1])
        return metrics,positive_same,changed

    entity_stats,e_same,e_diff=compare(entity)
    context_stats,c_same,c_diff=compare(context)
    assert not (e_same|e_diff)&(c_same|c_diff)
    e_bases={i for p in e_same for i in p}
    c_bases={i for p in c_diff for i in p}
    return dict(entity_one_span_exact_type_match=entity_stats,
                context_one_word_same_entity_pair=context_stats,
                base_rows_supporting_same_positive_entity_and_changed_label_context=len(e_bases&c_bases))


def fixture():
    # Same surrounding words/fixed ordered spans, changing one named entity;
    # a context change preserves both surfaces. Include a changed-name label
    # to ensure annotation conflicts aren't silently counted as invariances.
    s,o=(0,1),(2,3)
    types=('X','Y')
    rows={0:(('A','joined','B'),s,o,types,1),1:(('C','joined','B'),s,o,types,1),
          2:(('A','left','B'),s,o,types,2),3:(('D','joined','B'),s,o,types,0),
          4:(('Q','joined','B'),s,o,('Z','Y'),1)}
    result=census(rows)
    assert result['entity_one_span_exact_type_match']['same_positive_pairs']==1
    assert result['entity_one_span_exact_type_match']['different_label_pairs']==2
    assert result['context_one_word_same_entity_pair']['different_positive_pairs']==1
    assert result['base_rows_supporting_same_positive_entity_and_changed_label_context']==1


def run(c):
    cohort=json.loads(checked(c['cohort']))
    selected=set(cohort['indices']['fit'])
    hold=set(cohort['indices']['selection'])|set(cohort['indices']['qualification'])
    assert len(selected)==4096 and not selected&hold
    metadata=json.loads(checked(c['metadata']))['train']
    assert len(metadata)==len({m['id_hash'] for m in metadata})==58465
    identity={metadata[i]['id_hash']:i for i in selected}
    names=json.loads(checked(c['label_contract']))
    assert len(names)==len(set(names))==40 and names[0]=='no_relation'
    known={name:i for i,name in enumerate(names)}
    rows={}
    for line in checked(c['source']).splitlines():
        if not line.strip():
            continue
        z=json.loads(line)
        md=z['metadata']
        i=identity.get(sha256(str(md['id']).encode()).hexdigest())
        if i is None:
            continue
        assert i not in rows and len(z['tokens'])==metadata[i]['words']
        st,ot=md['subj_type'],md['obj_type']
        assert st and ot
        rows[i]=tuple(z['tokens']),tuple(z['subject']),tuple(z['object']),(st,ot),known[z['label']]
    assert rows.keys()==selected
    fixture()
    return dict(status='PASS_EXACT_FIT_VARIANT_GEOMETRY_CENSUS',fit_rows=len(rows),**census(rows),
                source_hashes={k:p['sha256'] for k,p in c.items()},
                scope='Natural annotated contrasts only, not counterfactual ground truth. Equal mention type does not establish semantic invariance.',
                no_variants_generated=True,selection_qualification_labels_indexed=0,
                model_forwards=0,updates=0,gpu_allocations=0,rows_exported=0)


if __name__=='__main__':
    print(json.dumps(run(json.load(sys.stdin)),allow_nan=False))