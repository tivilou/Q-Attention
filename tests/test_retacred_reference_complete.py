"""Bounded mechanism/lifecycle checks. No formal dataset or public write."""
import copy,json
from pathlib import Path
import pytest
from baselines.retacred.reference_mechanisms import (focal_loss,core_record,core_scores,one_hop_intervention,
    resampled_order,reviewer_partition,curriculum_order,graph_attention_module,classification_with_abstention)
from baselines.retacred.reference_recipes import inventory,recipe
from baselines.retacred.reference_complete import model_ids

def row(i=0,label='per:employee_of'):
    return dict(id='synthetic-'+str(i),token=['Alice','joined','Acme','.'],subj_start=0,subj_end=0,
        obj_start=2,obj_end=2,subj_type='PERSON',obj_type='ORGANIZATION',relation=label)

def test_cohort_full_selection_and_exclusion():
    names=model_ids('all');assert len(names)==15;assert 'ptr_reversed' not in names
    assert len(inventory())==15
    with pytest.raises(ValueError):model_ids('ptr_reversed')
    assert recipe('ctl_drp_supervised')['logical_batch']==512
    assert recipe('metaentail')['training_unit']=='expanded_NLI_pair'

def test_masks_never_access_gold():
    a=row();b=dict(a,relation='per:schools_attended')
    for kind in ['entity_only','blank']:
        assert core_record(a,kind)==core_record(b,kind)
        assert 'relation' not in core_record(a,kind)
    assert core_record(a,'entity_only')['token']==['Alice','<mask>','Acme','<mask>']
    cf,mask=one_hop_intervention(a,[1,1,1,1]);assert mask==[0,1,2];assert cf['token'][-1]=='.'

def test_own_subset_partition_and_full_last_stage():
    folds=reviewer_partition(20,13);assert sorted(i for f in folds for i in f)==list(range(20))
    for k,fold in enumerate(folds):assert not set(fold)&set(i for j,f in enumerate(folds) if j!=k for i in f)
    difficulty=[0,0,1,2,2];early=curriculum_order(difficulty,0,6,13);last=curriculum_order(difficulty,5,6,13)
    assert set(early)=={0,1};assert set(last)==set(range(5))

def test_resample_exact_population_and_determinism():
    labels=[0]*90+[1]*10;a=resampled_order(labels,13)
    assert len(a)==100;assert a==resampled_order(labels,13);assert sum(labels[i]==1 for i in a)>30

def test_abstention_never_becomes_null_or_disappears():
    labels=['no_relation','positive'];m=classification_with_abstention(['no_relation','positive'],[None,None],labels)
    assert m['items']==2 and m['accuracy']==0 and m['abstained']==2
    assert m['confusion_matrix']==[[0,0,1],[0,0,1]]

def test_focal_equation_gradient_and_gamma0():
    import torch
    x=torch.tensor([[2.,-1.],[.1,.4]],requires_grad=True);y=torch.tensor([0,1])
    assert torch.equal(focal_loss(x,y,0),torch.nn.functional.cross_entropy(x,y))
    loss=focal_loss(x,y);loss.backward();assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0

def test_core_two_corrections_and_type_mask():
    import torch
    f=torch.tensor([[.8,.2]]);e=torch.tensor([[.9,.1]]);b=torch.tensor([[.6,.4]])
    assert torch.allclose(core_scores(f,e,b,(1.,.5)),f-e-.5*b)
    scores=core_scores(f,e,b,(0.,0.),torch.tensor([[False,True]]));assert scores.argmax(-1).item()==1

@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_gat_graph_non_edges_and_bilinear_gradients(device):
    import torch
    if device.startswith('cuda') and not torch.cuda.is_available():pytest.skip('no CUDA')
    m=graph_attention_module(8,2,4).to(device);x=torch.randn(1,3,8,device=device,requires_grad=True)
    adj=torch.tensor([[[1,1,0],[1,1,1],[0,1,1]]],dtype=torch.bool,device=device)
    out=m(x,adj,torch.ones(1,3,device=device));assert out.shape==(1,3,8)
    assert m.last['attention'][0,:,0,2].eq(0).all()
    head=torch.nn.Bilinear(8,8,40).to(device);loss=head(out[:,0],out[:,2]).square().mean();loss.backward()
    # Softmax removes a query-constant score if all LeakyReLU terms are in one
    # linear region, so a_source can legitimately have zero gradient.
    assert x.grad.abs().sum()>0 and m.a_target.grad.abs().sum()>0

class Tiny:
    def __init__(self,device):
        import torch
        self.labels=['no_relation']+[f'relation{i}' for i in range(1,40)];self.device=device
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.embedding=torch.nn.Embedding(10,8);self.dropout=torch.nn.Dropout(.1);self.classifier=torch.nn.Linear(8,40)
            def forward(self,input_ids,attention_mask,labels):
                h=self.embedding(input_ids);p=torch.softmax(h@h.transpose(1,2),-1);rep=(p@h).mean(1)
                logits=self.classifier(self.dropout(rep));return torch.nn.functional.cross_entropy(logits,labels),logits
        self.model=Model().to(device)
    def encode(self,records):return [dict(input_ids=[0,1,2,3],labels=i%2) for i,_ in enumerate(records)]
    def batch(self,features):
        import torch
        return dict(input_ids=torch.tensor([f['input_ids'] for f in features],device=self.device),
            attention_mask=torch.ones(len(features),4,device=self.device,dtype=torch.long),labels=torch.tensor([f['labels'] for f in features],device=self.device))
    def forward(self,b):return self.model(**b)
    def finalize_vocabulary(self):pass
    def parameter_trace(self):
        return {'parameter_gradient.'+n:p.grad.detach().cpu().clone() for n,p in self.model.classifier.named_parameters() if p.grad is not None}

@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_exact_batch_resume_and_actual_value_trace(tmp_path,device):
    import torch
    from baselines.retacred.reference_worker import seed_all,train_standard,evaluate
    from baselines.retacred.reference_trace import validate
    if device.startswith('cuda') and not torch.cuda.is_available():pytest.skip('no CUDA')
    data={s:[row(i) for i in range(4)] for s in ('train','valid','test')}
    r=dict(recipe('typed_marker_comparison_2025'),epochs=1,logical_batch=2,selection='final_epoch')
    seed_all(13);a=Tiny(device);full=train_standard(a,data,tmp_path/'full',r,13,1)
    full_weights={k:v.detach().cpu().clone() for k,v in a.model.state_dict().items()}
    seed_all(13);b=Tiny(device);assert train_standard(b,data,tmp_path/'resume',r,13,1,pause_after=1) is None
    seed_all(999);c=Tiny(device);resumed=train_standard(c,data,tmp_path/'resume',r,13,1,resume=True)
    assert resumed[1]['cursor']==2 and all(torch.equal(full_weights[k],v.cpu()) for k,v in c.model.state_dict().items())
    features,state=resumed;_,events=evaluate(c,features['test'],data['test'],r,1,case_index=0,run=tmp_path/'resume',step=2,events=state['traces'])
    validate(tmp_path/'resume',state['traces']+events)
    for event in state['traces']+events:
        payload=json.loads((tmp_path/'resume'/'case_study'/event['path']).read_text())
        values=torch.load(tmp_path/'resume'/'case_study'/payload['tensor_artifact'],weights_only=True)
        assert values and all(x['value_status']=='complete_values' for x in payload['representations'])
    changed={**r,'learning_rate':r['learning_rate']*2}
    with pytest.raises(ValueError,match='immutable'):train_standard(c,data,tmp_path/'resume',changed,13,1,resume=True)
