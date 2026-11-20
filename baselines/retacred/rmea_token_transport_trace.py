"""Final-state sample-trace.v2 with complete private arrays and semantic replay."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from . import rmea_token_transport_protocol as p
from .rmea_token_transport import TrainableTokenTransportHead, route
from scripts.validate_sample_trace_v2 import STAGES, validate

AXES = dict(
    h=['sample','word','hidden'], roles=['sample','word','role'], mask=['sample','word'],
    features=['sample','word','feature'], prepare_raw=['sample','word','flat_angle'],
    angles=['sample','word','layer','wire','rotation'], effect_raw=['relation','layer','wire','rotation'],
    effect_angles=['relation','layer','wire','rotation'], psi=['sample','word','basis'], phi=['relation','basis'],
    alpha=['sample','relation','word'], native_alpha=['sample','relation','word'],
    normalizer=['sample','relation'], values=['sample','word','value'],
    utilities=['sample','relation','word'], decoder=['relation','value'], bias=['relation'], logits=['sample','relation'],
    gram=['sample','word','word2'], graph=['sample','word','word2'], generator=['sample','word','word2'],
    unitary=['sample','word','word2'], eigenvalues=['sample','word'], spectral_scale=['sample'],
    amplitudes=['sample','relation','word'], route_swap_alpha=['route','sample','relation','word'],
    route_swap_logits=['route','sample','relation'], input_ids=['subword'], attention_mask=['subword'],
    encoder_hidden=['subword','hidden'], target=['singleton'])
GROUPS = ('tokenization','encoder','word_input','values','weights','prediction','diagnosis','evaluation')


def display(text):
    return dict(summary=text, default_view='human_summary', visible_fields=[], visualizations=[])


def owner(name):
    if name in ('input_ids','attention_mask'): return 'tokenization'
    if name=='encoder_hidden': return 'encoder'
    if name in ('h','roles','mask','features'): return 'word_input'
    if name in ('values','utilities','decoder','bias'): return 'values'
    if name=='logits': return 'prediction'
    if name=='target': return 'evaluation'
    if name.startswith('route_swap_'): return 'diagnosis'
    return 'weights'


@torch.no_grad()
def details(head, h, roles, mask):
    d = head(h, roles, mask, details=True)
    arrays = {k:v for k,v in d.items() if torch.is_tensor(v)}
    arrays.update({k:v for k,v in d['transport'].items() if k not in ('alpha','native_alpha','normalizer')})
    features = torch.cat((h, roles), -1)
    arrays.update(h=h, roles=roles, mask=mask, features=features, prepare_raw=head.prepare(features),
                  effect_raw=head.effect, decoder=head.decoder, bias=head.bias)
    swaps = torch.stack([route(d['psi'], d['phi'], mask, kind=k)['alpha'] for k in p.MODES])
    arrays.update(route_swap_alpha=swaps, route_swap_logits=(swaps*d['utilities'][None]).sum(-1)+head.bias)
    return {k:v.detach().cpu().numpy().copy() for k,v in arrays.items()}


def capture(root, head, kind, split, batch, metadata):
    head.eval(); samples=[]; root=Path(root)
    cp=root/'checkpoints'/(kind+'-final.pt')
    for i in range(2):
        inputs,y=batch(split,[i]); meta=metadata[split][i]
        arrays=details(head,*inputs)
        arrays.update(input_ids=np.asarray(meta['input_ids'],np.int64),
            attention_mask=np.asarray(meta['attention_mask'],np.int64),
            encoder_hidden=np.load(root/'cache'/f'{split}-encoder-{i}.npy',allow_pickle=False),
            target=np.asarray([int(y[0])],np.int64))
        name=f'{kind}-final-{split}-{i}.npz'; path=root/'trace'/name; path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path,**arrays)
        ref=dict(path='trace/'+name,sha256=p.digest(path),bytes=path.stat().st_size,visibility='private')
        sid=f'{kind}:final:{split}:{i}'; reps=[]; groups={g:[] for g in GROUPS}
        for key,a in sorted(arrays.items()):
            if key not in AXES or len(AXES[key])!=a.ndim or not np.isfinite(a).all():
                raise ValueError('Unspecified or invalid complete representation '+key)
            g=owner(key); groups[g].append(key)
            reps.append(dict(id=key,stage_id=g,status='observed',meaning=key+' complete detached producer value',
                shape=list(a.shape),dtype=str(a.dtype),axis_semantics=AXES[key],artifact_refs=[ref],display=display(key),
                portal_preview=dict(array_key=key,owner_stage=g,direction='output',visibility='private'),
                value_capture=dict(status='complete',shape=list(a.shape),logical_element_count=int(a.size),stored_element_count=int(a.size),artifact_refs=[ref],
                    reader_projection=dict(status='available',kind='coordinate_inspector',coordinate_access='arbitrary',
                        reader_ref='scripts/view_rmea_token_transport_trace.py --sample '+sid+' --representation '+key,
                        source_sha256=[p.digest(p.ROOT/'scripts/view_rmea_token_transport_trace.py')]))))
        links=dict(tokenization=['source:tokens','source:entity_spans'],
            encoder=['input_ids','attention_mask','external:pinned_roberta_large'],
            word_input=['encoder_hidden','source:word_pieces','source:entity_spans'],
            values=['h','external:checkpoint_value_decoder'],
            weights=['features','mask','external:checkpoint_prepare_effect'],
            prediction=['alpha','utilities','bias'],
            diagnosis=['psi','phi','mask','utilities','bias'], evaluation=['logits','source:held_out_target'])
        canonical=dict(tokenization='data',encoder='preprocess',word_input='preprocess',values='scoring',weights='scoring',
                       prediction='generation',diagnosis='diagnosis',evaluation='evaluation')
        stages=[dict(stage_id=g,stage=g,canonical_stage=canonical[g],lane='replay',status='observed',step=j,checkpoint='final',
            purpose=g+' from frozen final state; gold enters posthoc evaluation only',input_refs=links[g],output_refs=groups[g],
            artifact_refs=[ref],display=display(g),target_access='post_generation_only' if g=='evaluation' else 'not_available')
            for j,g in enumerate(GROUPS) if groups[g]]
        samples.append(dict(sample_id=sid,lane='replay',split=split,checkpoint='final',
            roles=dict(source_sample=dict(tokens=meta['tokens'],id_hash=meta['id_hash']),
                sequence_or_features=dict(word_pieces=meta['word_pieces']),task_objects=dict(subject=meta['subject'],object=meta['object']),
                target_or_gold=dict(value=int(y[0]),boundary='post_generation_only'),prediction=dict(value=int(arrays['logits'].argmax())),
                provenance=dict(condition=kind,cache_index=i,checkpoint_file=cp.relative_to(root).as_posix(),checkpoint_sha256=p.digest(cp))),
            stages=stages,representations=reps))
    return samples


def read_value(root, sample, representation, coords):
    root=Path(root).resolve(); rep=next(r for r in sample['representations'] if r['id']==representation)
    ref=rep['value_capture']['artifact_refs'][0]; path=root/ref['path']
    if path.is_symlink() or not path.resolve().is_relative_to(root) or p.digest(path)!=ref['sha256']:
        raise ValueError('Private value path or digest differs')
    if not isinstance(coords,list) or not 1<=len(coords)<=64:
        raise ValueError('Expected1..64 coordinates')
    with np.load(path,allow_pickle=False) as f: a=f[representation].copy()
    if list(a.shape)!=rep['shape'] or str(a.dtype)!=rep['dtype'] or not np.isfinite(a).all():
        raise ValueError('Complete numeric representation differs')
    result=[]
    for c in coords:
        if len(c)!=a.ndim or any(type(v)!=int or not 0<=v<n for v,n in zip(c,a.shape)):
            raise ValueError('Coordinate outside declared axes')
        v=a[tuple(c)].item(); result.append(dict(real=v.real,imag=v.imag) if isinstance(v,complex) else v)
    return result


def close(root, samples, cfg, metadata):
    root=Path(root); manifest=p.read(root/'manifest.json')
    required=['input_ids','attention_mask','encoder_hidden','h','roles','mask','features',
              'psi','phi','alpha','native_alpha','values','utilities','logits','route_swap_logits','target']
    payload=dict(schema_version='sample-trace.v2',trace_id='rmea-token-transport-final-seed13',trace_completeness='complete',
        experiment=dict(run_id=root.parent.name,dataset='Re-TACRED; tiny pinned cache only when environment.fixture_sha256 is set',
            code_revision=manifest['git_revision'],config_sha256=p.identity(cfg),model_identity='Four final64bit transport heads; frozen RoBERTa-large word inputs',seed=13),
        sample_selection=dict(rule='First2 outcome-blind hash-ordered units per split, every condition at final only',
            population_scope='24 fixed post-training inference replays; no trajectory/internal-encoder claim',selected_count=len(samples),selected_sample_ids=[s['sample_id'] for s in samples]),
        task_contract=dict(task_type='relation token transport final-state inference',
            required_roles=['source_sample','sequence_or_features','task_objects','prediction','target_or_gold'],
            required_representations=required,required_value_representations=required,required_splits=list(cfg['counts']),
            required_checkpoints=['final'],required_lanes=['replay'],checkpoint_policy=cfg['selection']),
        coverage={s:'observed' if s in ('data','preprocess','scoring','generation','diagnosis','evaluation') else 'not_applicable' for s in STAGES},samples=samples)
    errors,warnings=validate(payload,require_complete=True)
    if errors: raise ValueError(errors)
    seen=set(); parity={}; checked=coordinates=elements=0
    for sample in samples:
        prov=sample['roles']['provenance']; kind=prov['condition']; i=prov['cache_index']; split=sample['split']
        key=(kind,split,i)
        if key in seen: raise ValueError('Duplicate trace sample')
        seen.add(key); cp=root/prov['checkpoint_file']
        if p.digest(cp)!=prov['checkpoint_sha256']: raise ValueError('Final checkpoint changed')
        head=TrainableTokenTransportHead(kind).double()
        head.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True)); head.eval()
        ref=sample['representations'][0]['artifact_refs'][0]
        if p.digest(root/ref['path'])!=ref['sha256']: raise ValueError('Trace array artifact changed')
        with np.load(root/ref['path'],allow_pickle=False) as f: saved={k:f[k].copy() for k in f.files}
        meta=metadata[split][i]; n=len(meta['tokens']); pieces=meta['word_pieces']
        if (sample['roles']['source_sample']!=dict(tokens=meta['tokens'],id_hash=meta['id_hash'])
                or sample['roles']['task_objects']!=dict(subject=meta['subject'],object=meta['object'])
                or sample['roles']['sequence_or_features']['word_pieces']!=pieces):
            raise ValueError('Original sample lineage differs')
        np.testing.assert_array_equal(saved['input_ids'],meta['input_ids'])
        np.testing.assert_array_equal(saved['attention_mask'],meta['attention_mask'])
        np.testing.assert_array_equal(saved['encoder_hidden'],np.load(root/'cache'/f'{split}-encoder-{i}.npy',allow_pickle=False))
        if len(pieces)!=n or len(saved['encoder_hidden'])!=len(saved['input_ids']) or not np.isin(saved['attention_mask'],[0,1]).all():
            raise ValueError('Subword coordinate coverage differs')
        for row in pieces:
            if not row or any(type(k)!=int or not 0<=k<len(saved['input_ids']) or saved['attention_mask'][k]!=1 for k in row):
                raise ValueError('Invalid original word mapping')
        off=np.load(root/'cache'/f'{split}-offsets.npy',mmap_mode='r'); lo,hi=map(int,off[i:i+2])
        cached=np.load(root/'cache'/f'{split}-h.npy',mmap_mode='r')[lo:hi].astype(np.float64)
        if n!=hi-lo or saved['h'].shape!=(1,n,1024) or saved['mask'].shape!=(1,n) or not saved['mask'].all():
            raise ValueError('Complete unpadded word input differs')
        recomputed=np.stack([saved['encoder_hidden'][row].mean(0) for row in pieces]).astype(np.float16).astype(np.float64)
        np.testing.assert_allclose(recomputed,cached,atol=.002,rtol=.002)
        np.testing.assert_array_equal(saved['h'][0],cached)
        np.testing.assert_array_equal(saved['roles'][0],np.load(root/'cache'/f'{split}-roles.npy',mmap_mode='r')[lo:hi])
        positions=np.arange(n,dtype=np.float32); s,o=meta['subject'],meta['object']
        expected_roles=np.stack([np.clip((positions-(s[0]+s[1]-1)/2)/32,-1,1),np.clip((positions-(o[0]+o[1]-1)/2)/32,-1,1),
            ((positions>=s[0])&(positions<s[1])).astype(np.float32),((positions>=o[0])&(positions<o[1])).astype(np.float32)],-1)
        np.testing.assert_array_equal(saved['roles'][0],expected_roles)
        target=int(np.load(root/'cache'/f'{split}-targets.npy',mmap_mode='r')[i])
        if int(saved['target'][0])!=target or sample['roles']['target_or_gold']['value']!=target:
            raise ValueError('Posthoc gold differs')
        actual=details(head,*(torch.from_numpy(saved[k]) for k in ('h','roles','mask')))
        if set(saved)!=set(actual)|{'input_ids','attention_mask','encoder_hidden','target'}:
            raise ValueError('Missing or extra producer numeric state')
        for name,a in actual.items():
            np.testing.assert_allclose(a,saved[name],atol=1e-8,rtol=1e-8,err_msg=name)
        if int(saved['logits'].argmax())!=sample['roles']['prediction']['value']:
            raise ValueError('Prediction replay differs')
        for rep in sample['representations']:
            a=saved[rep['id']]
            if list(a.shape)!=rep['shape'] or int(a.size)!=rep['value_capture']['stored_element_count']:
                raise ValueError('Incomplete numeric coordinate coverage')
            coords=[list(map(int,np.unravel_index(k,a.shape))) for k in sorted({0,a.size//2,a.size-1})]
            for coord,value in zip(coords,read_value(root,sample,rep['id'],coords)):
                v=complex(value['real'],value['imag']) if isinstance(value,dict) else value
                if v!=a[tuple(coord)].item(): raise ValueError('Source-linked coordinate reader differs')
                coordinates+=1
            checked+=1; elements+=int(a.size)
        parity.setdefault((split,i),set()).add(p.identity({k:hashlib.sha256(saved[k].tobytes()).hexdigest()
            for k in ('h','roles','mask','input_ids','attention_mask','encoder_hidden')}))
    if seen!={(k,s,i) for k in p.MODES for s in cfg['counts'] for i in range(2)} or any(len(v)!=1 for v in parity.values()):
        raise ValueError('Trace selection/condition parity differs')
    p.atomic_json(root/'sample-trace.json',payload)
    report=dict(schema='q-rmea.token-transport.trace-readiness.v2',structural='pass',semantic='pass',replay='pass',coordinate_reader='pass',
        samples=len(samples),complete_numeric_arrays_checked=checked,complete_numeric_elements_checked=elements,
        source_linked_coordinates_checked=coordinates,warnings=warnings,
        boundary='Saved final subword encoder output to original words and final heads; no live encoder/internal layer/trajectory claim')
    p.atomic_json(root/'trace_validation.json',report)
    return report
