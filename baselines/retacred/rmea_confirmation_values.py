"""Private source-linked bounded coordinate reads, never public tensor export."""
from pathlib import Path
import numpy as np
from . import rmea_confirmation_protocol as p

MEANINGS={
 'h':('Frozen original-word vectors','word_input',['sample','original_word','hidden_component']),
 'roles':('Ordered subject/object distance and membership features','word_input',['sample','original_word','role_feature']),
 'mask':('Valid original-word mask including entity words','word_input',['sample','original_word']),
 'input_ids':('Tokenizer IDs including entity markers and padding','tokenization',['subword_position']),
 'attention_mask':('Encoder valid-subword mask','tokenization',['subword_position']),
 'encoder_hidden':('Frozen encoder final output before original-word means','encoder',['subword_position','hidden_component']),
 'features':('Concatenated frozen word vector and ordered role features','word_input',['sample','original_word','feature_component']),
 'values':('Shared linear value projection of each word','values',['sample','original_word','value_component']),
 'utilities':('Relation decoder dot product with each word value','values',['sample','relation','original_word']),
 'decoder':('Shared learned relation decoder parameters at this checkpoint','values',['relation','value_component']),
 'bias':('Relation classifier bias at this checkpoint','values',['relation']),
 'alpha':('Actual normalized relation-specific word weights','weights',['sample','relation','original_word']),
 'logits':('Relation scores before posthoc label comparison','prediction',['sample','relation']),
 'angles':('Bounded RY/RZ preparation angles in radians','weights',['sample','original_word','layer','wire','rotation']),
 'effect_angles':('Bounded relation-effect angles in radians','weights',['relation','layer','wire','rotation']),
 'psi':('Complex prepared three-qubit word states','weights',['sample','original_word','basis_state']),
 'phi':('Complex rank-one relation-effect states','weights',['relation','basis_state']),
 'probability':('Binary measurement probability per relation and word','weights',['sample','relation','original_word']),
 'normalizer':('Sum of masked probability plus epsilon over words','weights',['sample','relation','singleton']),
 'sensitivity':('Analytic score sensitivity to word measurement probability','diagnosis',['sample','relation','original_word']),
 'prepare_raw':('Linear preparation output before tanh and pi scaling','weights',['sample','original_word','flattened_angle']),
 'effect_raw':('Learned effect parameters before tanh and pi scaling','weights',['relation','layer','wire','rotation']),
 'key':('Linear classical word keys before scoring','weights',['sample','original_word','key_component']),
 'query':('Learned classical relation queries before scoring','weights',['relation','key_component']),
 'normalized_key':('Unit-normalized real word keys','weights',['sample','original_word','key_component']),
 'normalized_query':('Unit-normalized real relation queries','weights',['relation','key_component']),
 'scores':('Classical scores before softmax or overlap normalization','weights',['sample','relation','original_word']),
 'pooled':('Classical relation-weighted word-value averages','prediction',['sample','relation','value_component']),
 'shuffled_alpha':('Weights after deterministic nonentity-context permutation','diagnosis',['sample','relation','original_word']),
 'shuffled_logits':('Relation scores with shuffled context weights','diagnosis',['sample','relation']),
 'tied_alpha':('Relation-mean weights broadcast to every relation','diagnosis',['sample','relation','original_word']),
 'tied_logits':('Relation scores with tied weights','diagnosis',['sample','relation']),
 'permutation':('Source-word index for each shuffled destination word','diagnosis',['sample','original_word']),
 'identity_residual':('Residual of exact logit-difference weighted-utility identity','diagnosis',['sample','relation']),
 'eligible_context_mask':('Valid nonentity words eligible for permutation','diagnosis',['sample','original_word'])}
BASE={'h','roles','mask','values','utilities','decoder','bias','alpha','logits','shuffled_alpha','shuffled_logits','tied_alpha','tied_logits','permutation','identity_residual','eligible_context_mask'}
EXTRA={'input_ids','attention_mask','encoder_hidden','features'}
def expected(mode):
    names=BASE|EXTRA
    if mode in ('quantum','dephased'):return names|{'angles','effect_angles','psi','phi','probability','normalizer','sensitivity','prepare_raw','effect_raw'}
    if mode!='uniform':names=names|{'key','query','scores'}
    if mode=='real_overlap':names=names|{'normalized_key','normalized_query'}
    return names|{'pooled'}

def artifact(root,name):
    path=Path(root)/name
    if path.is_symlink() or not path.resolve().is_relative_to(Path(root).resolve()):raise ValueError('Unsafe value artifact')
    return dict(path=name,sha256=p.digest(path),bytes=path.stat().st_size,visibility='exchange_only')

def load(root,representation):
    projection=representation['portal_preview'];ref=representation['value_capture']['artifact_refs'][0]
    actual=artifact(root,ref['path'])
    if actual!=ref:raise ValueError('Reader source hash differs')
    with np.load(Path(root)/ref['path'],allow_pickle=False) as data:value=data[projection['array_key']].copy()
    if list(value.shape)!=representation['shape'] or str(value.dtype)!=representation['dtype'] or not np.isfinite(value).all():raise ValueError('Reader source shape/dtype/values differ')
    return value

def coordinates(root,sample,rep_id,indices):
    rep=next(r for r in sample['representations'] if r['id']==rep_id)
    if not isinstance(indices,list) or not 1<=len(indices)<=64:raise ValueError('Request1..64 coordinates')
    value=load(root,rep);result=[]
    for coord in indices:
        if not isinstance(coord,list) or len(coord)!=value.ndim or any(type(v)!=int or v<0 or v>=s for v,s in zip(coord,value.shape)):raise ValueError('Coordinate outside declared axes')
        v=value[tuple(coord)].item()
        result.append(dict(real=float(v.real),imag=float(v.imag)) if isinstance(v,complex) else v)
    return dict(representation=rep_id,coordinates=indices,values=result,source_sha256=rep['value_capture']['reader_projection']['source_sha256'])
