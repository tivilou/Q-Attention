"""Asset bytes/transport and preparation boundaries, not neural certification."""
import ast
import copy
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import pytest

from baselines.retacred import mnli_assets as a
from baselines.retacred import metaentail_neural as n
from baselines.retacred.common import select_models

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'.external/assets/mnli'/a.REVISION


def spec(): return json.loads(a.LOCK.read_text())


def test_seven_pinned_files_nine_shards_and_actual_head_bytes():
    locked=a.validate_lock(spec())
    assert len(locked['files'])==7 and sum(len(row['parts']) for row in locked['files'])==9
    result=a.verify_checkpoint(CACHE/'model')
    assert result['weights_locally_verified'] is True
    assert result['neural_runtime_verified'] is False
    header=n.inspect_safetensors(CACHE/'model/model.safetensors')
    assert header['selected_tensors']['classifier.out_proj.weight']['shape']==[3,1024]
    assert header['selected_tensors']['roberta.embeddings.word_embeddings.weight']['shape']==[50265,1024]
    assert all(row['dtype']=='F32' and len(row['sha256'])==64 for row in header['selected_tensors'].values())
    assert header['neural_runtime_verified'] is False


@pytest.mark.parametrize('kind',['repo','revision','namespace','missing','extra','hash','size','git_blob',
                               'part_order','part_size','part_hash','model_load'])
def test_asset_lock_rejects_identity_or_shard_drift(kind):
    value=spec()
    if kind in ('repo','revision'): value[kind]='wrong'
    elif kind=='namespace': value['exchange_namespace']='other/project/run'
    elif kind=='missing': value['files'].pop()
    elif kind=='extra': value['files'].append(copy.deepcopy(value['files'][0]))
    elif kind in ('hash','size','git_blob'):
        key={'hash':'sha256','size':'size_bytes','git_blob':'git_blob'}[kind]; value['files'][0][key]='wrong'
    elif kind=='part_order': value['files'][-1]['parts'].reverse()
    elif kind=='part_size': value['files'][-1]['parts'][0]['size_bytes']-=1
    elif kind=='part_hash': value['files'][-1]['parts'][0]['sha256']='x'*64
    else: value['model_load_certified']=True
    with pytest.raises(ValueError): a.validate_lock(value)


def test_offline_complete_cache_needs_no_credentials_or_transport(monkeypatch):
    monkeypatch.delenv('PROJECT_EXCHANGE_TOKEN',raising=False)
    monkeypatch.delenv('PROJECT_EXCHANGE_URL',raising=False)
    monkeypatch.setattr(a,'stream_part',lambda *args,**kwargs:pytest.fail('cache hit used network'))
    assert a.fetch(cache_dir=CACHE)==CACHE/'model'
    receipt=json.loads((CACHE/'fetch-verification.json').read_text())
    assert receipt['network_downloads']==0 and receipt['weights_locally_verified'] is True


def test_missing_cache_requires_configuration_before_download(tmp_path,monkeypatch):
    monkeypatch.delenv('PROJECT_EXCHANGE_TOKEN',raising=False)
    monkeypatch.delenv('PROJECT_EXCHANGE_URL',raising=False)
    monkeypatch.setattr(a,'stream_part',lambda *args,**kwargs:pytest.fail('unconfigured download used transport'))
    with pytest.raises(ValueError): a.fetch(cache_dir=tmp_path/'new')


def test_corrupt_existing_cache_preserved_not_replaced(tmp_path,monkeypatch):
    cache=tmp_path/'corrupt'; model=cache/'model'; model.mkdir(parents=True)
    path=model/'README.md'; path.write_bytes(b'broken')
    monkeypatch.setattr(a,'stream_part',lambda *args,**kwargs:pytest.fail('corrupt cache was downloaded over'))
    with pytest.raises(ValueError,match='corrupt'): a.fetch(cache_dir=cache)
    assert path.read_bytes()==b'broken'


@pytest.mark.parametrize('kind',['missing_keys','unexpected_keys','mismatched_keys','error_msgs','absent','nonlist'])
def test_require_explicit_clean_pretrained_loading_report(kind):
    report=dict(missing_keys=[],unexpected_keys=[],mismatched_keys=[],error_msgs=[])
    n.loading_keys_checked(report)
    if kind=='absent': report.pop('missing_keys')
    elif kind=='nonlist': report['missing_keys']=None
    else: report[kind]=['classifier.out_proj.weight']
    with pytest.raises(ValueError): n.loading_keys_checked(report)


def test_constructor_memory_gate_precedes_weights_backend_and_tokenizer(monkeypatch):
    def reject(): raise RuntimeError('fixture insufficient memory')
    monkeypatch.setattr(n,'require_runtime_memory',reject)
    monkeypatch.setattr(n,'verify_checkpoint',lambda *args:pytest.fail('weights touched before memory gate'))
    with pytest.raises(RuntimeError,match='MNLI neural runtime pending at pre-import memory gate'):
        n.MetaEntailFoundation('/not-read')


def test_loading_call_preserves_mnli_head_and_no_formal_dispatch():
    tree=ast.parse(Path(n.__file__).read_text())
    calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call)
           and isinstance(node.func,ast.Attribute) and node.func.attr=='from_pretrained'
           and isinstance(node.func.value,ast.Name) and node.func.value.id=='RobertaForSequenceClassification']
    assert len(calls)==1
    kwargs={key.arg:ast.literal_eval(key.value) for key in calls[0].keywords}
    assert kwargs==dict(local_files_only=True,output_loading_info=True,use_safetensors=True,
                        ignore_mismatched_sizes=False,attn_implementation='eager')
    assert not any(isinstance(node,ast.Attribute) and node.attr in ('reset_parameters','resize_token_embeddings') for node in ast.walk(tree))
    with pytest.raises(ValueError): select_models('metaentail')
    assert len(select_models('ready'))==5


def feature(identity='s',length=6,label=2,split='train'):
    row=dict(sample_id=identity,candidate_relation='per:employee_of',candidate_index=0,
             input_ids=[0]+[10]*(length-2)+[2],attention_mask=[1]*length,
             filter_identity='filterA',split=split)
    if split=='train': row['labels']=label
    return row


def test_padding_attention_labels_and_input_immutability():
    rows=[feature(length=6),feature(identity='t',length=9,label=0)]; before=copy.deepcopy(rows)
    batch=n.collate_pairs(rows)
    assert batch['input_ids'][0][-3:]==[1,1,1] and batch['attention_mask'][0][-3:]==[0,0,0]
    assert batch['labels']==[2,0] and rows==before
    evaluated=n.collate_pairs([feature(split='test')])
    assert set(evaluated)=={'input_ids','attention_mask'}


@pytest.mark.parametrize('kind',['empty','mixed','bad_label','bool_label','filter','mask','vocab','overlong'])
def test_invalid_pair_batches_rejected(kind):
    rows=[feature()]
    if kind=='empty': rows=[]
    elif kind=='mixed': rows.append(feature(split='test'))
    elif kind in ('bad_label','bool_label'): rows[0]['labels']=40 if kind=='bad_label' else True
    elif kind=='filter': rows.append(dict(feature(),filter_identity='other'))
    elif kind=='mask': rows[0]['attention_mask'][1]=0
    elif kind=='vocab': rows[0]['input_ids'][1]=50265
    else: rows=[feature(length=513)]
    with pytest.raises(ValueError): n.collate_pairs(rows)


def tiny_safetensor(path,*,shape=[2,3],dtype='F32',end=24):
    value={'classifier.out_proj.weight':{'dtype':dtype,'shape':shape,'data_offsets':[0,end]}}
    header=json.dumps(value).encode(); path.write_bytes(struct.pack('<Q',len(header))+header+b'\0'*24)


def test_header_parser_bounds_and_shape_extent(tmp_path,monkeypatch):
    path=tmp_path/'model.safetensors'; tiny_safetensor(path)
    monkeypatch.setattr(n,'HEAD_SHAPES',{'classifier.out_proj.weight':[2,3]})
    assert n.inspect_safetensors(path)['selected_tensors']['classifier.out_proj.weight']['shape']==[2,3]
    tiny_safetensor(path,shape=[3,2])
    with pytest.raises(ValueError,match='shape mismatch'): n.inspect_safetensors(path)
    tiny_safetensor(path,end=23)
    with pytest.raises(ValueError,match='extent'): n.inspect_safetensors(path)
    path.write_bytes(struct.pack('<Q',2**40)+b'{}')
    with pytest.raises(ValueError,match='header size'): n.inspect_safetensors(path)


def test_real_pair_expansion_target_isolation_and_order_mapping_subprocess(tmp_path):
    script=r'''
import itertools,json,pathlib,sys
def audit(event,args):
 if event=='import' and args[0].split('.')[0] in ('torch','tensorflow','jax'): raise AssertionError('backend imported')
 if event.startswith('socket.'): raise AssertionError('network used')
 if event=='open' and isinstance(args[0],str) and pathlib.Path(args[0]).suffix in ('.bin','.safetensors','.pt'): raise AssertionError('weights/tensors opened')
sys.addaudithook(audit)
from transformers import RobertaTokenizerFast
from baselines.retacred.metaentail import fit_type_filter,NLI_LABELS,grouped_prediction
from baselines.retacred.metaentail_neural import encoded_group,collate_pairs
cache=pathlib.Path(sys.argv[1]); tokenizer=RobertaTokenizerFast.from_pretrained(cache,local_files_only=True,add_prefix_space=False)
raw={'id':'synthetic-A','token':['Alice','joined','Acme','.'],'subj_start':0,'subj_end':0,
 'obj_start':2,'obj_end':2,'subj_type':'PERSON','obj_type':'ORGANIZATION','relation':'per:employee_of'}
records=[dict(raw,id=str(i),relation=label) for i,label in enumerate(('no_relation','per:employee_of','per:schools_attended'))]
filter=fit_type_filter(records,split='train')
class NoGold(dict):
 def __getitem__(self,key):
  if key=='relation': raise AssertionError('evaluation read gold')
  return super().__getitem__(key)
 def get(self,key,*args):
  if key=='relation': raise AssertionError('evaluation read gold')
  return super().get(key,*args)
test=encoded_group(NoGold(raw),filter,tokenizer,NLI_LABELS,split='test',max_length=512)
assert len(test['features'])==3 and all('labels' not in row for row in test['features'])
assert len(collate_pairs(test['features'])['input_ids'])==3
for order in itertools.permutations(NLI_LABELS):
 train=encoded_group(raw,filter,tokenizer,order,split='train',max_length=512)
 assert [row['labels'] for row in train['features']]==[order.index(x) for x in ('contradiction','entailment','neutral')]
 assert [row['input_ids'] for row in train['features']]==[row['input_ids'] for row in test['features']]
assert [row['candidate_index'] for row in test['features']]==[0,1,2]
assert all(row['sample_id']=='synthetic-A' for row in test['features'])
changed=encoded_group(NoGold(dict(raw,relation='per:spouse')),filter,tokenizer,NLI_LABELS,split='test',max_length=512)
assert changed==test
empty=encoded_group(NoGold(dict(raw,subj_type='NEW')),filter,tokenizer,NLI_LABELS,split='test',max_length=512)
assert not empty['features']
decision=grouped_prediction(empty['group'],[],nli_order=NLI_LABELS)
assert decision['relation'] is None and decision['reason']=='no_feasible_hypotheses'
result={'expanded_test_group':test,'abstention':decision,'neural_runtime_verified':False,'model_jobs':0}
pathlib.Path(sys.argv[2]).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'pairs':len(test['features']),'target_access':'none','model_jobs':0}))
'''
    env=dict(os.environ,USE_TORCH='0',USE_TF='0',USE_FLAX='0',HF_HUB_OFFLINE='1',
             TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
    output=tmp_path/'expanded-pairs.json'
    done=subprocess.run([sys.executable,'-c',script,str(CACHE/'model'),str(output)],cwd=ROOT,
                        env=env,text=True,capture_output=True,timeout=30)
    assert done.returncode==0,done.stderr
    result=json.loads(output.read_text())
    assert len(result['expanded_test_group']['features'])==3
    assert json.loads(done.stdout)['pairs']==3
