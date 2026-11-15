"""Verbatim selected Round3 numeric/reader helpers; provenance in contract."""
import hashlib
from pathlib import Path
import numpy as np
import torch

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def pinned(root, ref):
    root = Path(root).resolve()
    name = ref['path']
    if not isinstance(name, str) or not name or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('Expected a package-relative artifact')
    p = root/name
    if p.is_symlink() or not p.resolve().is_relative_to(root) or sha(p) != ref['sha256']:
        raise ValueError('Artifact path or hash mismatch')
    return p

def ref(root, path):
    path = Path(path)
    return dict(path=path.relative_to(root).as_posix(), sha256=sha(path),
                bytes=path.stat().st_size, visibility='private')

def display(text):
    return dict(summary=text, default_view='human_summary', visible_fields=[], visualizations=[])

@torch.no_grad()
def numeric(head, inputs):
    d = dict(head(*inputs, details=True))
    for name, value in list(d.items()):
        if isinstance(value, dict):
            del d[name]
            d.update({name+'_'+k: v for k, v in value.items()})
    h, roles, mask = inputs
    features=torch.cat((h,roles),-1)
    if head.mode in ('quantum','dephased'):
        d.update(prepare_raw=head.prepare(features),effect_raw=head.effect)
    elif head.mode!='uniform':
        key=head.key(features);query=head.query
        d.update(key=key,query=query)
        if head.mode=='relation_softmax': d['scores']=torch.einsum('btd,rd->brt',key,query)/(8**.5)
        else:
            nk=torch.nn.functional.normalize(key,dim=-1);nq=torch.nn.functional.normalize(query,dim=-1)
            d.update(normalized_key=nk,normalized_query=nq,scores=torch.einsum('btd,rd->brt',nk,nq).square())
    d.update(h=h, roles=roles, mask=mask, features=torch.cat((h, roles), -1),
             decoder=head.decoder, bias=head.bias)
    return {k: v.detach().cpu().numpy().copy() for k, v in d.items()}

AXES = {
    'h': ['sample','word','hidden'], 'roles': ['sample','word','role'],
    'mask': ['sample','word'], 'features': ['sample','word','feature'],
    'entity_inputs': ['sample','subject_or_object','feature'],
    'word_angles': ['sample','word','flat_rotation'],
    'subject_angles': ['sample','flat_rotation'], 'object_angles': ['sample','flat_rotation'],
    'angles': ['sample','word','layer','wire','rotation'],
    'subject': ['sample','basis'], 'object': ['sample','basis'],
    'psi': ['sample','word','basis'], 'phi': ['relation','basis'],
    'conditional': ['sample','relation','basis'],
    'values': ['sample','word','value'], 'decoder': ['relation','value'],
    'bias': ['relation'], 'logits': ['sample','relation'],
    'cp': ['relation','rank','role','flat_rotation'],
}

for _name in ('probability','alpha','utilities','sensitivity','amplitude','separable_null_q','separable_null_direction','separable_null_alpha'):
    AXES[_name] = ['sample','relation','word']

for _name in ('normalizer','separable_null_c','separable_null_q_mass','separable_null_mixture'):
    AXES[_name] = ['sample','relation','singleton']

AXES.update(prepare_raw=['sample','word','flat_rotation'],effect_raw=['relation','layer','wire','rotation'],
            key=['sample','word','key'],query=['relation','key'],normalized_key=['sample','word','key'],
            normalized_query=['relation','key'],scores=['sample','relation','word'])

def axes(name, value):
    if name == 'effect_angles':
        return ['relation','layer','wire','rotation'] if value.ndim == 4 else ['relation','wire','rotation']
    return AXES[name]

def owner(name):
    if name in ('h','roles','mask','features'): return 'inputs'
    if name in ('entity_inputs','word_angles','subject_angles','object_angles','angles','prepare_raw','effect_raw','key','query','normalized_key','normalized_query'): return 'preparation'
    if name in ('values','decoder','utilities','bias'): return 'values'
    if name == 'logits': return 'prediction'
    return 'measurement'

def read_value(root, sample, name, coords):
    rep = next(x for x in sample['representations'] if x['id'] == name)
    if not isinstance(coords, list) or not 1 <= len(coords) <= 64:
        raise ValueError('Expected 1..64 explicit coordinates')
    with np.load(pinned(root, rep['artifact_refs'][0]), allow_pickle=False) as f:
        a = f[name]
    if list(a.shape) != rep['shape'] or str(a.dtype) != rep['dtype'] or not np.isfinite(a).all():
        raise ValueError('Array metadata/nonfinite mismatch')
    result = []
    for c in coords:
        if not isinstance(c, list) or len(c) != a.ndim or any(type(i) is not int or not 0 <= i < n for i,n in zip(c,a.shape)):
            raise ValueError('Coordinate outside declared axes')
        v = a[tuple(c)].item()
        result.append(dict(real=v.real,imag=v.imag) if isinstance(v,complex) else v)
    return result
