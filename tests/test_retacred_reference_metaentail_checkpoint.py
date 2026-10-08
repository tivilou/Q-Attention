"""Small official tokenizer/identity and budget checks, never a neural job."""
import copy
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from baselines.retacred import metaentail_checkpoint as c
from baselines.retacred.common import select_models

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT/".external/retacred/metaentail-checkpoint-20261008B"


def actual_config():
    return json.loads((CACHE/"config.json").read_text()), json.loads((CACHE/"tokenizer_config.json").read_text())


def test_real_official_small_files_and_provider_only_weight_identity():
    result = c.verify_metadata(CACHE)
    assert result["revision"]=="2a8f12d27941090092df78e4ba6f0928eb5eac98"
    assert len(result["files"])==6 and result["max_pair_tokens_including_specials"]==512
    assert result["content_token_budget"]==508
    assert result["nli_order"]==["contradiction","neutral","entailment"]
    assert result["weight_hash_evidence"]=="provider_metadata_only"
    assert result["weights_locally_verified"] is False and result["neural_runtime_verified"] is False
    assert result["model_jobs"]==0 and result["network_requests"]==0
    assert not (CACHE/"model.safetensors").exists()
    assert not (CACHE/"pytorch_model.bin").exists()
    metadata = json.loads((CACHE/"identity.json").read_text())
    locked = c.checkpoint_spec()["checkpoint"]
    assert locked["files"]==metadata["files"] and locked["weights"]==metadata["weights"]
    collection = json.loads((CACHE/"collection.json").read_text())
    assert len(collection)==7 and all(row["status"]==200 for row in collection)
    assert all("2a8f12d27941090092df78e4ba6f0928eb5eac98" in row["url"] for row in collection[1:])


@pytest.mark.parametrize("order",list(itertools.permutations(("CONTRADICTION","NEUTRAL","ENTAILMENT"))))
def test_config_contract_reads_actual_column_order_not_constant(order):
    config, tokenizer = actual_config()
    config["id2label"]={str(i):name for i,name in enumerate(order)}
    config["label2id"]={name:i for i,name in enumerate(order)}
    result = c.config_contract(config,tokenizer)
    assert result["nli_order"]==[name.lower() for name in order]
    assert result["nli_label_ids"]=={name.lower():i for i,name in enumerate(order)}


@pytest.mark.parametrize("change",[
    {"_num_labels":40}, {"architectures":["RobertaModel"]}, {"model_type":"bert"},
    {"hidden_size":768}, {"max_position_embeddings":1026}, {"pad_token_id":0},
    {"id2label":{"0":"LABEL_0","1":"LABEL_1","2":"LABEL_2"}},
    {"label2id":{"ENTAILMENT":0,"NEUTRAL":1,"CONTRADICTION":2}}
])
def test_reject_generic_or_modified_head_architecture(change):
    config, tokenizer = actual_config()
    with pytest.raises(ValueError):
        c.config_contract(dict(config,**change),tokenizer)


@pytest.mark.parametrize("limit",[1024,514,True,1000000000000])
def test_reject_mismatched_tokenizer_length(limit):
    config, tokenizer = actual_config()
    with pytest.raises(ValueError,match="length mismatch"):
        c.config_contract(config,{"model_max_length":limit})


def test_metadata_refuses_tamper_and_symlink_without_weight_reads(tmp_path):
    fixture = tmp_path/"small"; fixture.mkdir()
    for name in c.checkpoint_spec()["checkpoint"]["files"]:
        shutil.copyfile(CACHE/name,fixture/name)
    file = fixture/"config.json"
    file.write_bytes(file.read_bytes().replace(b'"CONTRADICTION"',b'"CONTRADICTI0N"'))
    with pytest.raises(ValueError,match="identity mismatch"):
        c.verify_metadata(fixture)
    shutil.copyfile(CACHE/"config.json",file)
    link = tmp_path/"alias"; link.symlink_to(fixture,target_is_directory=True)
    with pytest.raises(ValueError,match="symlinked"):
        c.verify_metadata(link)


def test_real_fast_tokenizer_complete_pair_boundary_and_no_backend(tmp_path):
    # Separate subprocess disables every neural backend before any HF import.
    script = r'''
import json
import pathlib
import sys

def audit(event,args):
    if event == 'import' and args[0].split('.')[0] in ('torch','tensorflow','jax'):
        raise AssertionError('neural backend imported')
    if event.startswith('socket.'):
        raise AssertionError('network used')
    if event == 'open' and isinstance(args[0],str) and pathlib.Path(args[0]).suffix in ('.bin','.safetensors','.pt'):
        raise AssertionError('weights or numeric tensor opened')

sys.addaudithook(audit)
from transformers import RobertaTokenizerFast
from baselines.retacred.metaentail_checkpoint import verify_metadata,encode_pair
from baselines.retacred import metaentail as m
cache = pathlib.Path(sys.argv[1]); output=pathlib.Path(sys.argv[2])
verified=verify_metadata(cache)
tokenizer=RobertaTokenizerFast.from_pretrained(str(cache),local_files_only=True,add_prefix_space=False)
raw={'id':'synthetic-A','token':['Alice','joined','Acme','.'],'subj_start':0,'subj_end':0,
     'obj_start':2,'obj_end':2,'subj_type':'PERSON','obj_type':'ORGANIZATION','relation':'per:employee_of'}
filter=m.fit_type_filter([raw],split='train')
group=m.inference_group(raw,filter)
pair=encode_pair(group['premise'],group['candidates'][0]['hypothesis'],tokenizer,max_length=512)
assert pair['premise']=='Alice joined Acme .' and pair['hypothesis']=='Alice is an employee of Acme.'
assert pair['sequence_ids'].count(None)==4
assert [token for token,mark in zip(pair['tokens'],pair['special_tokens_mask']) if mark]==['<s>','</s>','</s>','</s>']
assert all(seq is None if mark else seq in (0,1) for seq,mark in zip(pair['sequence_ids'],pair['special_tokens_mask']))
assert 'labels' not in pair and pair['target_access']=='none'
for idx,(seq,(start,end)) in enumerate(zip(pair['sequence_ids'],pair['offset_mapping'])):
    if seq is not None:
        assert 0<=start<end<=len(pair['premise'] if seq==0 else pair['hypothesis'])
first=[id for id,seq in zip(pair['input_ids'],pair['sequence_ids']) if seq==0]
second=[id for id,seq in zip(pair['input_ids'],pair['sequence_ids']) if seq==1]
assert pair['input_ids']==[0]+first+[2,2]+second+[2]
reverse=encode_pair(pair['hypothesis'],pair['premise'],tokenizer,max_length=512)
assert reverse['input_ids']!=pair['input_ids']
hyp='Bob'
hyp_count=len(tokenizer(hyp,add_special_tokens=False)['input_ids'])
content_count=512-4-hyp_count
premise=' '.join(['x']*content_count)
boundary=encode_pair(premise,hyp,tokenizer,max_length=512)
assert boundary['pair_tokens']==512 and boundary['truncated'] is False
try:
    encode_pair(premise+' x',hyp,tokenizer,max_length=512)
except ValueError as error:
    assert 'actual=513 limit=512' in str(error)
else:
    raise AssertionError('513-token pair accepted')
try:
    encode_pair(premise,hyp,tokenizer,max_length=511)
except ValueError as error:
    assert 'actual=512 limit=511' in str(error)
else:
    raise AssertionError('explicit shorter limit ignored')
assert not any(name.split('.')[0] in ('torch','tensorflow','jax') for name in sys.modules)
result={'metadata':verified,'synthetic_input':raw,'pair':pair,'boundary_length':512,
        'overflow_rejected':513,'neural_runtime_verified':False,'model_jobs':0}
output.write_text(json.dumps(result,indent=2)+'\n')
assert json.loads(output.read_text())==result
print(json.dumps({'status':'real_tokenizer_verified','pair':pair,'boundary_length':512,
                  'overflow_rejected':513,'neural_runtime_verified':False,'model_jobs':0}))
'''
    env = dict(os.environ,USE_TORCH="0",USE_TF="0",USE_FLAX="0",HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1",TOKENIZERS_PARALLELISM="false",PYTHONDONTWRITEBYTECODE="1")
    output = tmp_path/"actual-tokenizer.json"
    done = subprocess.run([sys.executable,"-c",script,str(CACHE),str(output)],cwd=ROOT,
                          env=env,text=True,capture_output=True,timeout=30)
    assert done.returncode==0,done.stderr
    result = json.loads(output.read_text())
    assert result["pair"]["input_ids"]==json.loads(done.stdout)["pair"]["input_ids"]
    assert result["boundary_length"]==512 and result["overflow_rejected"]==513


def test_expanded_pair_not_original_example_budget_and_exact_tail():
    result = c.expanded_pair_budget([3,2],epochs=2,pair_batch_size=2,accumulation_steps=2)
    assert result["original_examples"]==2 and result["pairs_per_epoch"]==5
    assert result["mini_batches_per_epoch"]==3 and result["optimizer_updates_per_epoch"]==2
    assert result["total_pair_presentations"]==10 and result["total_optimizer_updates"]==4
    assert result["last_mini_batch_pairs"]==1 and result["last_update_pairs"]==1
    assert result["last_update_mini_batches"]==1 and result["drop_last"] is False
    assert result["loss_denominator"]=="actual_pairs_in_optimizer_window"
    assert result["policy"]=="proposed_adapter_not_paper_recipe"
    assert result["formal_admission"] is False


@pytest.mark.parametrize("pairs,batch,accum",[(1,32,4),(40,8,5),(17,8,2),(15,8,2),(40,7,3)])
def test_budget_matches_enumerated_windows(pairs,batch,accum):
    result = c.expanded_pair_budget([pairs],epochs=3,pair_batch_size=batch,accumulation_steps=accum)
    mini = [min(batch,pairs-start) for start in range(0,pairs,batch)]
    windows = [mini[i:i+accum] for i in range(0,len(mini),accum)]
    assert result["mini_batches_per_epoch"]==len(mini)
    assert result["optimizer_updates_per_epoch"]==len(windows)
    assert result["last_mini_batch_pairs"]==mini[-1]
    assert result["last_update_pairs"]==sum(windows[-1])
    assert result["last_update_mini_batches"]==len(windows[-1])
    assert result["total_optimizer_updates"]==3*len(windows)


@pytest.mark.parametrize("counts",[[],[0],[41],[True],[1.5]])
def test_invalid_training_counts_never_silently_dropped(counts):
    with pytest.raises(ValueError):
        c.expanded_pair_budget(counts,epochs=1,pair_batch_size=32,accumulation_steps=1)


@pytest.mark.parametrize("field,value",[("epochs",0),("pair_batch_size",True),("accumulation_steps",-1)])
def test_invalid_explicit_budget_parameter(field,value):
    args=dict(epochs=1,pair_batch_size=32,accumulation_steps=1); args[field]=value
    with pytest.raises(ValueError):
        c.expanded_pair_budget([3],**args)


def test_preparation_does_not_admit_metaentail_or_replace_defaults():
    with pytest.raises(ValueError):
        select_models("metaentail")
    assert len(select_models("ready"))==5
    spec=c.checkpoint_spec()
    assert spec["planned_neural_loading"]["num_labels_override"] is False
    assert spec["planned_neural_loading"]["classifier_reinitialization"] is False
    assert spec["budget"]["training_epochs_batch_lr_optimizer_scheduler"]=="unresolved_for_general_domain"
