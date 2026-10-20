"""Outcome-blind bounded Re-TACRED population and aligned typed views."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import torch
from q_attention.tasks.relation import (load_relation_jsonl,build_vocab,build_label_map,
                                       RelationDataset,collate_relation_batch)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def prepare(directory,config):
    data,manifest = {},{}
    for split in ('train','valid','test'):
        path = Path(directory)/f'{split}.jsonl'
        records = load_relation_jsonl(path)
        groups = {}
        for index,row in enumerate(records):
            if len(row.tokens)>config['max_length'] or max(row.subject[0],row.object[0])<min(row.subject[1],row.object[1]):
                continue
            groups.setdefault(row.label,[]).append(index)
        indexes = []
        for label in sorted(groups):
            ordered = sorted(groups[label],key=lambda i:digest([config['selection_seed'],split,i]))
            indexes.extend(ordered[:config['per_class']])
        indexes.sort(key=lambda i:digest([config['selection_seed'],split,i]))
        if not indexes or len(indexes)>=len(records):
            raise ValueError('bounded screen requires a nonempty proper subset')
        data[split] = [records[i] for i in indexes]
        manifest[split] = dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            source_count=len(records),selected_indexes=indexes,selected_count=len(indexes),
            supports={label:sum(records[i].label==label for i in indexes) for label in sorted(groups)},
            unavailable_labels=sorted({r.label for r in records}-set(groups)),
            selected_ids_sha256=digest([records[i].metadata.get('id',str(i)) for i in indexes]))
    vocab,labels = build_vocab(data['train']),build_label_map(data['train'])
    if any(r.label not in labels for rows in data.values() for r in rows):
        raise ValueError('held-out class unsupported by selected train population')
    for role,field in (('subject','subj_type'),('object','obj_type')):
        types = sorted({'UNKNOWN'} | {str(row.metadata.get(field,'UNKNOWN')) for row in data['train']})
        for kind in types:
            vocab[f'<cfra:{role}:{kind}>'] = len(vocab)
    datasets = {split:RelationDataset(rows,vocab,labels) for split,rows in data.items()}
    manifest['vocab_sha256'],manifest['labels_sha256'] = digest(vocab),digest(labels)
    return data,datasets,vocab,labels,manifest


def batch(rows,dataset,vocab,indexes):
    value = collate_relation_batch([dataset[int(i)] for i in indexes])
    neutral = []
    for index in indexes:
        row = rows[int(index)]
        ids = []
        for role,field in (('subject','subj_type'),('object','obj_type')):
            key = f'<cfra:{role}:{row.metadata.get(field,"UNKNOWN")}>'
            ids.append(vocab.get(key,vocab[f'<cfra:{role}:UNKNOWN>']))
        neutral.append(ids)
    value['neutral_ids'] = torch.tensor(neutral,dtype=torch.long)
    return value


def initialize_neutral_embeddings(base,rows,vocab):
    with torch.no_grad():
        embedding = base.encoder.token_embedding.weight
        for role,field in (('subject','subj_type'),('object','obj_type')):
            groups = {}
            for row in rows:
                kind = str(row.metadata.get(field,'UNKNOWN'))
                span = getattr(row,role)
                indexes = [vocab.get(t.lower(),1) for t in row.tokens[span[0]:span[1]]]
                groups.setdefault(kind,[]).extend(indexes)
            fallback = sorted({i for indexes in groups.values() for i in indexes})
            for token,index in vocab.items():
                if token.startswith(f'<cfra:{role}:'):
                    kind = token[len(f'<cfra:{role}:'):-1]
                    selected = groups.get(kind,fallback)
                    embedding[index].copy_(embedding[selected].mean(0))
