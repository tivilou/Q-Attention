import copy
import pytest
from baselines.retacred.metaentail_neural import loading_keys_checked,UNUSED_POOLER_SHAPES

def report():
    return dict(missing_keys=[],unexpected_keys=list(UNUSED_POOLER_SHAPES),mismatched_keys=[],error_msgs=[])

def evidence():
    return {n:dict(shape=shape,dtype='F32',sha256='a'*64) for n,shape in UNUSED_POOLER_SHAPES.items()}

def test_only_exact_verified_official_dead_pooler_pair_is_accepted():
    r=loading_keys_checked(report(),verified_unused_pooler=evidence())
    assert r == dict(ignored_unused_pooler=sorted(UNUSED_POOLER_SHAPES),active_parameters_exact=True)
    r=report();r['unexpected_keys']=[]
    assert loading_keys_checked(r)['ignored_unused_pooler']==[]

@pytest.mark.parametrize('kind',['no_evidence','partial_evidence','shape','dtype','hash','head_missing','head_mismatch','encoder_unexpected','partial_pair','duplicate','error'])
def test_no_broader_loading_exception(kind):
    r=report();e=evidence();n=next(iter(e))
    if kind=='no_evidence':e=None
    elif kind=='partial_evidence':e.pop(n)
    elif kind=='shape':e[n]['shape']=[1]
    elif kind=='dtype':e[n]['dtype']='F16'
    elif kind=='hash':e[n]['sha256']='z'*64
    elif kind=='head_missing':r['missing_keys']=['classifier.out_proj.weight']
    elif kind=='head_mismatch':r['mismatched_keys']=[('classifier.out_proj.weight',[3,1024],[40,1024])]
    elif kind=='encoder_unexpected':r['unexpected_keys'].append('roberta.encoder.layer.24.weight')
    elif kind=='partial_pair':r['unexpected_keys'].pop()
    elif kind=='duplicate':r['unexpected_keys']=[n,n]
    else:r['error_msgs']=['corrupt tensor']
    with pytest.raises(ValueError):loading_keys_checked(r,verified_unused_pooler=e)
