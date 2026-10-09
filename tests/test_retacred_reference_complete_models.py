import copy,json,zipfile,types
from pathlib import Path
import pytest

from baselines.retacred.common import ROOT,PACKAGE,digest
from baselines.retacred.reference_models import SequenceRouting,SyntaxCurriculum,LukeVariants,DependencyParser,LABELS
from baselines.retacred.reference_worker import seed_all
from baselines.retacred.reference_trace import Capture,emit,ValueReader

EVIDENCE=ROOT/'.external/retacred/reference-cohort-completion-20261009T'

def record(i=0):return dict(id='synthetic-'+str(i),token=['Alice','joined','Acme','.'],subj_start=0,subj_end=0,obj_start=2,obj_end=2,subj_type='PERSON',obj_type='ORGANIZATION',relation='per:employee_of')

def small_config(kind='roberta',labels=40):
    from transformers import RobertaConfig,LukeConfig
    if kind=='luke':return LukeConfig(vocab_size=50267,entity_vocab_size=4,entity_emb_size=8,hidden_size=16,num_hidden_layers=2,num_attention_heads=2,intermediate_size=32,num_labels=40,use_entity_aware_attention=True,classifier_bias=True)
    return RobertaConfig(vocab_size=50265,hidden_size=16,num_hidden_layers=2,num_attention_heads=2,intermediate_size=32,num_labels=labels,max_position_embeddings=514,pad_token_id=1)

def parser():
    p=EVIDENCE/'resources/parser'
    if not p.exists():
        with zipfile.ZipFile(EVIDENCE/'resources/en_core_web_sm-3.7.1-py3-none-any.whl') as z:z.extractall(p)
    return DependencyParser(p/'en_core_web_sm/en_core_web_sm-3.7.1')

def tiny_luke(device):
    import torch
    from transformers import LukeTokenizer,LukeForEntityPairClassification
    from baselines.retacred.luke import encode_records,collate_features
    source=ROOT/'.external/retacred/luke-foundation-20261008'
    from baselines.retacred.reference_mechanisms import source_classes
    code=source/'examples--relation_classification--luke_for_entity_pair_classification.py'
    ns={'__name__':'pinned_tiny_luke'};exec(compile(code.read_bytes(),str(code),'exec'),ns)
    config=small_config('luke');model=ns['LukeForEntityPairClassification'](config).to(device)
    checkpoint=EVIDENCE/'luke-tokenizer'
    tokenizer=LukeTokenizer.from_pretrained(checkpoint,local_files_only=True,additional_special_tokens=['<ent>','<ent2>'],task=None)
    f=types.SimpleNamespace(model=model,tokenizer=tokenizer)
    f.encode=lambda rows:encode_records(source,tokenizer,rows,LABELS)
    f.batch=lambda rows:{k:torch.tensor(v,device=device) for k,v in collate_features(rows,tokenizer.pad_token_id).items()}
    f.finalize_vocabulary=lambda:None
    return f

@pytest.mark.parametrize('name',['sequence_routing_h3','curriculum','luke_core','luke_cfie','luke_resample','luke_focal'])
def test_actual_cuda_mechanism_and_full_numeric_trace(tmp_path,name):
    import torch
    if not torch.cuda.is_available():pytest.skip('GPU unavailable')
    seed_all(13);device='cuda:0';records=[record()]
    if name=='sequence_routing_h3':
        adapter=SequenceRouting(EVIDENCE/'tokenizer',ROOT/'.external/reference_sources/sequence_routing',EVIDENCE/'resources',device=device,config=small_config())
    elif name=='curriculum':adapter=SyntaxCurriculum(EVIDENCE/'tokenizer',parser(),device=device,config=small_config())
    else:adapter=LukeVariants(name,None,None,records,device=device,parser=parser() if name=='luke_cfie' else None,foundation=tiny_luke(device))
    adapter.model_id=name;features=adapter.encode(records);batch=adapter.batch(features);adapter.model.train()
    optimizer=torch.optim.AdamW(adapter.model.parameters(),lr=1e-4)
    optimizer.zero_grad();cap=Capture(adapter.model,[0],1)
    before={n:p.detach().cpu().clone() for n,p in adapter.model.named_parameters() if 'classifier' in n or 'bilinear' in n}
    with cap:
        loss,logits=adapter.forward(batch);cap.snapshot_mechanism();loss.backward()
    assert torch.isfinite(loss) and any(p.grad is not None and p.grad.abs().sum()>0 for p in adapter.model.parameters())
    optimizer.step()
    trace=emit(adapter,batch,records,cap,phase='train',step=1,event=0,directory=tmp_path,loss=loss,logits=logits,parameter_before=before)
    reader=ValueReader(tmp_path/trace['path']);assert reader.values
    if name=='sequence_routing_h3':assert 'mechanism..depth_stack' in cap.values or any('depth_stack' in k for k in cap.values)
    if name=='curriculum':assert any('dependency_adjacency' in k for k in cap.values) and any('attention' in k for k in cap.values)
    adapter.model.eval();cap=Capture(adapter.model,[0],1)
    with torch.no_grad(),cap:
        loss,logits=adapter.forward(batch);cap.snapshot_mechanism()
    emit(adapter,batch,records,cap,phase='test',step=1,event=1,directory=tmp_path,loss=loss,logits=logits)
    if name=='luke_core':assert {'factual_probability','entity_only_probability','blank_probability'}<=set(adapter.model.last)
    if name=='luke_cfie':assert 'counterfactual_logits' in adapter.model.last and any('masked' in k for k in features[0])

def test_parser_pretokenized_alignment_and_label_independence():
    p=parser();a=record();b=dict(a,relation='no_relation');assert p.parse(a)==p.parse(b)
    assert len(p.parse(a)['heads'])==4

def test_nli_real_threeway_head_gradients_and_group_decision(tmp_path):
    import torch
    from transformers import RobertaForSequenceClassification,RobertaTokenizerFast
    from baselines.retacred.metaentail_neural import encoded_group,collate_pairs
    from baselines.retacred.metaentail import fit_type_filter,grouped_prediction
    from baselines.retacred.reference_models import head_gradients
    records=[record(0),dict(record(1),relation='no_relation')];filter_=fit_type_filter(records,split='train')
    tokenizer=RobertaTokenizerFast.from_pretrained(EVIDENCE/'tokenizer',local_files_only=True,model_max_length=512,add_prefix_space=False)
    model=RobertaForSequenceClassification(small_config(labels=3)).cuda();order=('contradiction','neutral','entailment')
    encoded=encoded_group(records[0],filter_,tokenizer,order,split='train',max_length=512)
    batch={k:torch.tensor(v,device='cuda:0') for k,v in collate_pairs(encoded['features']).items()}
    cap=Capture(model,list(range(len(batch['input_ids']))),len(batch['input_ids']))
    with cap:out=model(**batch);out.loss.backward()
    assert model.roberta.embeddings.word_embeddings.weight.grad.abs().sum()>0
    adapter=types.SimpleNamespace(model=model,tokenizer=tokenizer,parameter_trace=lambda:head_gradients(model))
    emit(adapter,batch,[records[0]],cap,phase='train',step=1,event=0,directory=tmp_path,loss=out.loss,logits=out.logits)
    assert grouped_prediction(encoded['group'],out.logits.softmax(-1).tolist(),nli_order=order)['status'] in ('selected','abstained')

def test_sequence_upstream_head_forward_gradient_update_exact():
    import torch
    from baselines.retacred.reference_mechanisms import source_classes
    code=EVIDENCE/'resources/heinsen_routing.py';ns={'__name__':'routing_native'};exec(compile(code.read_bytes(),str(code),'exec'),ns)
    namespace={'torch':torch,'nn':torch.nn,'F':torch.nn.functional,'Routing':ns['EfficientVectorRouting']}
    Head,MLP=source_classes(ROOT/'.external/reference_sources/sequence_routing/RC_Model.py',('RoutingHead','MLP'),namespace)
    config={'d_depth':3};params=[{'n_inp':-1,'n_out':4,'d_inp':16,'d_out':16},{'n_inp':4,'n_out':4,'d_inp':16,'d_out':16},{'n_inp':4,'n_out':8,'d_inp':16,'d_out':1}]
    seed_all(13);a=Head(config,params);b=copy.deepcopy(a);x=torch.randn(2,5,3,16,requires_grad=True);y=x.detach().clone().requires_grad_()
    oa=torch.optim.Adam(a.parameters(),lr=1e-5);ob=torch.optim.Adam(b.parameters(),lr=1e-5)
    # Direct source object is the adapter's active class, not a renamed MLP.
    ra=a(x);rb=b(y);assert torch.equal(ra,rb);ra.square().mean().backward();rb.square().mean().backward()
    assert torch.equal(x.grad,y.grad);oa.step();ob.step();assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())

@pytest.mark.parametrize('name',['typed_marker_punct','typed_marker','entity_marker_punct','entity_marker','typed_marker_comparison_2025','ptr'])
def test_actual_source_marker_and_PTR_two_update_lifecycle(tmp_path,name):
    import torch,shutil
    from transformers import RobertaModel
    from baselines.retacred.reference_models import make
    from baselines.retacred.reference_worker import train_standard,evaluate
    from baselines.retacred.reference_recipes import recipe
    from baselines.retacred.reference_trace import validate
    if not torch.cuda.is_available():pytest.skip('GPU unavailable')
    checkpoint=tmp_path/'tiny-encoder';checkpoint.mkdir()
    RobertaModel(small_config()).save_pretrained(checkpoint)
    for p in (EVIDENCE/'tokenizer').iterdir():
        if p.name!='config.json':shutil.copyfile(p,checkpoint/p.name)
    assets={'roberta':str(checkpoint),'improved_source':str(EVIDENCE/'sources/improved'),'ptr_source':str(EVIDENCE/'sources/ptr'),'ptr_metadata':str(EVIDENCE/'sources/ptr_metadata')}
    seed_all(13);records=[record(i) for i in range(4)]
    adapter=make(name,assets,records,device='cuda:0',checkpointing=True);adapter.model_id=name
    data={s:records for s in ('train','valid','test')};r=dict(recipe(name),epochs=1,logical_batch=2,selection='final_epoch')
    features,state=train_standard(adapter,data,tmp_path/'run',r,13,1)
    metrics,events=evaluate(adapter,features['test'],records,r,1,case_index=0,run=tmp_path/'run',step=state['cursor'],events=state['traces'])
    validate(tmp_path/'run',state['traces']+events)
    assert state['cursor']==2 and [item['label'] for item in metrics['per_class']]==adapter.labels
    trace=json.loads((tmp_path/'run/case_study'/state['traces'][0]['path']).read_text())
    assert any(r['id'].startswith('parameter_before') for r in trace['representations'])
