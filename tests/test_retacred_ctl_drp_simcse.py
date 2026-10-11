"""Source/candidate identity and synthetic real-BPE checks; never weights/train."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from baselines.retacred import ctl_drp_simcse as s

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ['CTL_SIMCSE_EVIDENCE_CACHE'])

def test_official_identity_and_honest_weights():
    lock = s.load_lock()
    assert lock['paper_checkpoint_identity_bound'] is False
    assert lock['formal_comparison_admitted'] is False
    assert lock['weights_downloaded'] is False
    assert lock['facts']['models']['sup']['weight']['bytes'] == 1421571527
    assert lock['facts']['models']['sup']['weight']['sha256'] != lock['facts']['models']['unsup']['weight']['sha256']
    assert lock['facts']['models']['unsup']['config']['_name_or_path'].find('lr3e-5') != -1
    assert lock['facts']['models']['sup']['config']['_name_or_path'].find('lr1e-5') != -1

def test_actual_24_originals_and_source_blobs():
    assert s.verify_cache(CACHE, s.load_lock()) == {
        'originals_verified': 24, 'source_executed': False, 'weights_loaded': False}

def test_pooler_normalization_is_not_ctl_readout():
    lock = s.load_lock()
    assert lock['facts']['models']['sup']['upstream_sentence_embedding_pooling'] == 'cls_with_linear_tanh'
    assert lock['facts']['models']['unsup']['upstream_sentence_embedding_pooling'] == 'cls_before_pooler'
    assert lock['facts']['upstream_tool_l2_normalization_default'] is True
    assert lock['ctl_readout'] == 'last_hidden_state[:,0,:]_without_pooler_or_l2_normalization'

@pytest.mark.parametrize('field,value',[
    ('paper_checkpoint_identity_bound', True), ('weights_downloaded', True),
    ('formal_comparison_admitted', True), ('weights_numerically_verified', True),
    ('paper_literals',['sup-simcse-roberta-large','unsup-simcse-roberta-large']),
    ('ctl_readout','pooler_output')])
def test_admission_and_readout_tampering(tmp_path, field, value):
    lock = s.load_lock()
    lock[field] = value
    p = tmp_path/'lock.json'
    p.write_text(json.dumps(lock))
    with pytest.raises(ValueError):
        s.load_lock(p)

def test_revision_tampering(tmp_path):
    lock = s.load_lock()
    lock['facts']['models']['sup']['revision'] = 'main'
    p = tmp_path/'lock.json'
    p.write_text(json.dumps(lock))
    with pytest.raises(ValueError):
        s.load_lock(p)

def test_receipt_and_blob_tampering(tmp_path):
    lock = s.load_lock()
    (tmp_path/'collection.json').write_bytes(b'[]')
    with pytest.raises(ValueError, match='receipt'):
        s.verify_cache(tmp_path,lock)
    records = json.loads((CACHE/'collection.json').read_text())
    contents = {r['url']:(CACHE/r['path']).read_bytes() for r in records}
    u = next(k for k in contents if k.endswith('/config.json'))
    contents[u] += b' '
    with pytest.raises(ValueError,match='blob'):
        s.derive_facts(contents)

def test_real_byte_bpe():
    result = s.tokenizer_fixture(CACHE,s.load_lock())
    assert result['neural_model_executed'] is False
    assert result['added_entity_markers'] is False and result['truncation_chosen'] is False
    assert result['variants']['sup']['pair_ids'] == [0,31414,232,2,2,31414,2]
    assert result['variants']['sup']['overflow_length'] == 513
    assert result['variants']['sup'] == result['variants']['unsup']

def test_cohort_and_old_paper_boundary_preserved():
    from baselines.retacred.common import cohort, select_models
    from baselines.retacred.ctl_drp_paper import load_lock
    registry = cohort()
    assert len(registry['default_models']) == 5
    assert load_lock()['checkpoint_identity_bound'] is False
    for name in ('ctl_drp_supervised','ctl_drp_unsupervised'):
        row = next(x for x in registry['models'] if x['id']==name)
        assert row['status']=='blocked' and row['checkpoint_identity_bound'] is False
        assert row['reported_score_rank_eligible'] is False
        with pytest.raises(ValueError):
            select_models(name)

def test_no_neural_backend_or_network_from_actual_cli():
    code = '''
import sys
class Guard:
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in ('torch','transformers','requests','socket','http') or name=='urllib.request':
            raise AssertionError('backend/network import: '+name)
sys.meta_path.insert(0,Guard())
from baselines.retacred import ctl_drp_simcse as s
raise SystemExit(s.main())
'''
    result=subprocess.run([sys.executable,'-c',code,'--verify-cache',str(CACHE)],
                          cwd=ROOT,text=True,capture_output=True)
    assert result.returncode==2,result.stderr
    data=json.loads(result.stdout)
    assert data['cache_evidence']['originals_verified']==24
    assert data['network_requests']==0 and data['model_jobs']==0

def test_portable_descriptor_and_cli_pending():
    env=dict(os.environ,PYTHON_BIN=sys.executable)
    result=subprocess.run(['bash','scripts/check_retacred_ctl_drp_simcse.sh','--describe'],
                          cwd=ROOT,env=dict(env,PYTHON_BIN='/no/python'),text=True,capture_output=True)
    assert result.returncode==0 and json.loads(result.stdout)['weights'] is False
    result=subprocess.run(['bash','scripts/check_retacred_ctl_drp_simcse.sh'],cwd=ROOT,
                          env=env,text=True,capture_output=True)
    assert result.returncode==2,result.stderr
    assert json.loads(result.stdout)['cache_evidence']['originals_verified']==0

def test_fixture_requires_cache_and_regular_evidence():
    result=subprocess.run([sys.executable,'-m','baselines.retacred.ctl_drp_simcse','--tokenizer-fixture'],
                          cwd=ROOT,text=True,capture_output=True)
    assert result.returncode==1
    assert json.loads(result.stdout)['status']=='invalid_evidence'
