import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from baselines.retacred import ctl_drp_candidates as c
from baselines.retacred.ctl_drp_paper import candidate_reference

CACHE = Path(os.environ['CTL_CANDIDATE_EVIDENCE_CACHE'])
ROOT = Path(__file__).resolve().parents[1]
ROWS = [
    {'subj_type':'PERSON','obj_type':'ORGANIZATION','relation':'per:employee_of'},
    {'subj_type':'PERSON','obj_type':'ORGANIZATION','relation':'per:schools_attended'},
    {'subj_type':'PERSON','obj_type':'ORGANIZATION','relation':'no_relation'},
    {'subj_type':'PERSON','obj_type':'PERSON','relation':'per:spouse'},
    {'subj_type':'PERSON','obj_type':'URL','relation':'no_relation'},
]

def test_actual_source_paper_and_pure_fragment():
    result=c.verify_cache(CACHE,c.load_lock())
    assert result=={'originals_verified':12,'source_aggregation_fragment_executed':True,
                    'whole_upstream_script_executed':False,'neural_model_executed':False}

def test_source_tacred_constants_are_not_complete_canonical_H():
    facts=c.load_lock()['facts']
    assert facts['constant_multiclass_groups']==13 and facts['constant_positive_labels']==33
    assert len(facts['constant_canonical_missing'])==12 and len(facts['constant_extra_labels'])==6
    assert facts['heldout_gold_aggregation_and_direct_semantic_shortcut_observed'] is True
    assert facts['ctl_exact_H_author_mapping_recovered'] is False

def test_actual_aggregation_subgraph_matches_all_supports():
    records=json.loads((CACHE/'collection.json').read_text())
    for name in c.PREPARES:
        item=next(r for r in records if r['url']==c.RAW+name)
        original=copy.deepcopy(ROWS)
        result=c.source_fragment_reference((CACHE/item['path']).read_bytes(),ROWS)
        assert ROWS==original
        assert result[('PERSON','ORGANIZATION')]=={'per:employee_of','per:schools_attended','no_relation'}
        assert result[('PERSON','URL')]=={'no_relation'}

def test_fingerprint_excludes_version_dependent_ast_dump_fields():
    lock=c.load_lock()
    normalized="r = d['relation']\ns = d['subj_type']\no = d['obj_type']\nt = (s, o)\nif t not in so2reltion:\n    so2reltion[t] = set((r,))\nelse:\n    so2reltion[t].add(r)"
    assert lock['facts']['aggregation_normalized_source_sha256']==hashlib.sha256(normalized.encode()).hexdigest()

@pytest.mark.parametrize('split',['dev','test','validation','Train','all'])
def test_reject_heldout_and_ambiguous_fit(split):
    with pytest.raises(ValueError,match='heldout fitting'):
        c.fit_train_pairs(ROWS,source_split=split)

def test_deterministic_hash_order_and_roundtrip():
    table=c.fit_train_pairs(ROWS,source_split='train')
    assert table==c.fit_train_pairs(list(reversed(ROWS)),source_split='train')
    table=json.loads(json.dumps(table))
    assert c.candidate_labels(table,'PERSON','ORGANIZATION')==['no_relation','per:employee_of','per:schools_attended']
    assert table['physical_split_identity_verified'] is False
    assert table['formal_comparison_admitted'] is False

def test_lookup_has_no_gold_and_does_not_mutate():
    assert list(inspect.signature(c.candidate_labels).parameters)==['table','subject_type','object_type']
    table=c.fit_train_pairs(ROWS,source_split='train');before=copy.deepcopy(table)
    one=copy.deepcopy(ROWS);two=copy.deepcopy(ROWS)
    one[0]['relation']='per:employee_of';two[0]['relation']='per:spouse'
    assert c.coverage_diagnostic(table,one)['excluded_gold_indices']==[]
    assert c.coverage_diagnostic(table,two)['excluded_gold_indices']==[0]
    assert table==before
    assert c.candidate_labels(table,'PERSON','ORGANIZATION')==['no_relation','per:employee_of','per:schools_attended']

def test_unknown_and_gold_excluded_are_not_repaired():
    table=c.fit_train_pairs(ROWS,source_split='train')
    with pytest.raises(LookupError,match='explicit H policy'):
        c.candidate_labels(table,'ORGANIZATION','PERSON')
    diagnostic=c.coverage_diagnostic(table,[
        {'subj_type':'ORGANIZATION','obj_type':'PERSON','relation':'org:founded_by'},
        {'subj_type':'PERSON','obj_type':'ORGANIZATION','relation':'per:spouse'}])
    assert diagnostic['unknown_pair_indices']==[0] and diagnostic['excluded_gold_indices']==[1]
    assert diagnostic['gold_reinserted'] is False and diagnostic['candidates_modified'] is False

def test_single_positive_competes_with_NA_not_direct_prediction():
    table=c.fit_train_pairs([ROWS[3]],source_split='train')
    allowed=c.candidate_labels(table,'PERSON','PERSON')
    assert allowed==['no_relation','per:spouse']
    result=candidate_reference({r:[0.] for r in allowed},[1.],0.,allowed,'per:spouse')
    assert result['probabilities']=={'no_relation':.5,'per:spouse':.5}
    assert len(result['maximizers'])==2
    only=c.candidate_labels(c.fit_train_pairs([ROWS[4]],source_split='train'),'PERSON','URL')
    assert only==['no_relation']

def test_exact_pair_support_does_not_invent_cartesian_combinations():
    rows=[{'subj_type':'PERSON','obj_type':'ORGANIZATION','relation':'per:employee_of'},
          {'subj_type':'ORGANIZATION','obj_type':'PERSON','relation':'per:employee_of'}]
    table=c.fit_train_pairs(rows,source_split='train')
    with pytest.raises(LookupError):
        c.candidate_labels(table,'PERSON','PERSON')

@pytest.mark.parametrize('row',[
    {'subj_type':'PERSON','obj_type':'ORGANIZATION','relation':'org:parents'},
    {'subj_type':' PERSON','obj_type':'ORGANIZATION','relation':'per:employee_of'},
    {'subj_type':'PERSON','obj_type':None,'relation':'per:employee_of'},
    {'subj_type':'PERSON','obj_type':'ORGANIZATION','relation':True},
])
def test_invalid_and_noncanonical_training_rows_rejected(row):
    with pytest.raises(ValueError):
        c.fit_train_pairs([row],source_split='train')

def test_table_and_admission_tampering_rejected(tmp_path):
    table=c.fit_train_pairs(ROWS,source_split='train')
    table['pairs'][0]['relations'].append('per:spouse')
    with pytest.raises(ValueError,match='contract'):
        c.candidate_labels(table,'PERSON','ORGANIZATION')
    for field in ('formal_comparison_admitted','ctl_exact_H_author_mapping_recovered'):
        lock=c.load_lock();lock[field]=True;p=tmp_path/'lock.json';p.write_text(json.dumps(lock))
        with pytest.raises(ValueError):c.load_lock(p)

def test_source_window_and_receipt_tampering(tmp_path):
    records=json.loads((CACHE/'collection.json').read_text())
    item=next(r for r in records if r['url']==c.RAW+c.PREPARES[0])
    data=(CACHE/item['path']).read_bytes().replace(b"so2reltion[t].add(r)",b"so2reltion[t].add('per:spouse')")
    with pytest.raises(ValueError,match='algorithm'):
        c.source_fragment_reference(data,ROWS)
    (tmp_path/'collection.json').write_bytes(b'[]')
    with pytest.raises(ValueError,match='receipt'):
        c.verify_cache(tmp_path,c.load_lock())

def test_portable_descriptor_and_actual_offline_tripwire():
    result=subprocess.run(['bash','scripts/check_retacred_ctl_drp_candidates.sh','--describe'],cwd=ROOT,
                          env=dict(os.environ,PYTHON_BIN='/no/python'),text=True,capture_output=True)
    assert result.returncode==0 and json.loads(result.stdout)['training'] is False
    code='''
import builtins,runpy,socket,sys
original=builtins.__import__
def guard(name,*args,**kwargs):
    if name.split('.')[0] in {'torch','transformers','requests'} or name in {'urllib.request','http.client'}:
        raise AssertionError('backend/network import: '+name)
    return original(name,*args,**kwargs)
builtins.__import__=guard
def deny(*args,**kwargs):raise AssertionError('network used')
socket.socket=deny
sys.argv=['audit','--verify-cache',sys.argv[1]]
runpy.run_module('baselines.retacred.ctl_drp_candidates',run_name='__main__')
'''
    result=subprocess.run([sys.executable,'-c',code,str(CACHE)],cwd=ROOT,text=True,capture_output=True)
    assert result.returncode==2,result.stderr
    value=json.loads(result.stdout)
    assert value['cache_evidence']['originals_verified']==12
    assert value['model_jobs']==0 and value['network_requests']==0

def test_existing_cohort_and_exact_H_gates_preserved():
    from baselines.retacred.common import cohort,select_models
    from baselines.retacred.ctl_drp_paper import load_lock
    assert len(cohort()['default_models'])==5 and load_lock()['type_predicate_recovered'] is False
    for name in ('ctl_drp_supervised','ctl_drp_unsupervised'):
        row=next(r for r in cohort()['models'] if r['id']==name)
        assert row['status']=='blocked' and row['checkpoint_identity_bound'] is False
        with pytest.raises(ValueError):select_models(name)
