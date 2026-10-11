"""Mechanism-sensitive operations, each separately testable on small tensors."""
from __future__ import annotations
import ast, random
from collections import Counter
from pathlib import Path

def resampled_order(labels, seed):
    counts=Counter(labels)
    if not labels:raise ValueError('nonempty train labels required')
    return random.Random(seed).choices(range(len(labels)),weights=[1/counts[x] for x in labels],k=len(labels))

def reviewer_partition(count,seed,reviewers=3):
    if count<reviewers or reviewers<2:raise ValueError('each reviewer needs an own subset')
    order=list(range(count));random.Random(seed).shuffle(order)
    return [order[i::reviewers] for i in range(reviewers)]

def curriculum_order(difficulty,epoch,epochs,seed):
    if not difficulty or not 0<=epoch<epochs:raise ValueError('invalid curriculum epoch')
    buckets=sorted(set(difficulty));stages=min(3,len(buckets))
    stage=min(stages-1,epoch*stages//epochs)
    allowed=buckets[:max(1,(len(buckets)*(stage+1)+stages-1)//stages)]
    order=[i for i,x in enumerate(difficulty) if x in allowed]
    random.Random(seed+epoch).shuffle(order)
    return order

def focal_loss(logits,labels,gamma=2.,alpha=1.):
    import torch
    if gamma<0 or alpha<=0:raise ValueError('invalid frozen focal constants')
    logp=torch.log_softmax(logits,-1).gather(1,labels[:,None]).squeeze(1)
    return (-alpha*(1-logp.exp()).pow(gamma)*logp).mean()

def core_scores(factual,entity_only,blank,coefficients,mask=None):
    score=factual-coefficients[0]*entity_only-coefficients[1]*blank
    if mask is not None:score=score.masked_fill(~mask,float('-inf'))
    return score

def core_record(record,kind):
    """Mask context in place; preserve role positions/markers, never read gold."""
    result={k:v for k,v in record.items() if k!='relation'}
    spans=set(range(record['subj_start'],record['subj_end']+1))|set(range(record['obj_start'],record['obj_end']+1))
    result['token']=[token if kind=='entity_only' and i in spans else '<mask>' for i,token in enumerate(record['token'])]
    if kind not in ('entity_only','blank'):raise ValueError('unknown counterfactual')
    return result

def one_hop_intervention(record,heads):
    if len(heads)!=len(record['token']) or any(type(h)is not int or not 0<=h<len(heads) for h in heads):
        raise ValueError('aligned zero-based dependency heads required')
    entities=set(range(record['subj_start'],record['subj_end']+1))|set(range(record['obj_start'],record['obj_end']+1))
    mask=entities|{heads[i] for i in entities}|{i for i,h in enumerate(heads) if h in entities}
    result={k:v for k,v in record.items() if k!='relation'}
    result['token']=['<mask>' if i in mask else t for i,t in enumerate(record['token'])]
    return result,sorted(mask)

def source_classes(path,names,namespace):
    """Import exact pinned class ASTs, excluding unrelated module-level I/O."""
    tree=ast.parse(Path(path).read_text(encoding='utf-8'))
    classes=[node for node in tree.body if isinstance(node,ast.ClassDef) and node.name in names]
    if {node.name for node in classes}!=set(names):raise ValueError('missing upstream class')
    exec(compile(ast.Module(body=classes,type_ignores=[]),str(path),'exec'),namespace)
    return [namespace[name] for name in names]

def graph_attention_module(input_dim,heads=4,head_dim=128):
    import torch
    from torch import nn
    class GAT(nn.Module):
        def __init__(self):
            super().__init__();self.projection=nn.Linear(input_dim,heads*head_dim,bias=False)
            self.a_source=nn.Parameter(torch.empty(heads,head_dim));self.a_target=nn.Parameter(torch.empty(heads,head_dim))
            nn.init.xavier_uniform_(self.a_source);nn.init.xavier_uniform_(self.a_target)
            self.dropout=nn.Dropout(.1);self.last={}
        def forward(self,x,adjacency,valid):
            h=self.projection(x).view(*x.shape[:2],heads,head_dim)
            source=(h*self.a_source).sum(-1).transpose(1,2)
            target=(h*self.a_target).sum(-1).transpose(1,2)
            scores=torch.nn.functional.leaky_relu(source[:,:,:,None]+target[:,:,None,:],.2)
            # Padded query rows get a self edge solely to avoid all-inf softmax;
            # they are zeroed and never pooled, so real dependency edges are intact.
            mask=adjacency|torch.eye(x.shape[1],device=x.device,dtype=torch.bool)[None]
            attention=torch.softmax(scores.masked_fill(~mask[:,None],float('-inf')),-1)
            out=torch.einsum('bhij,bjhd->bihd',self.dropout(attention),h).flatten(-2)
            out=torch.nn.functional.elu(out)*valid[:,:,None]
            self.last={'projected':h,'edge_logits':scores,'attention':attention,'output':out}
            return out
    return GAT()

def classification_with_abstention(gold,predicted,labels,loss=None):
    """40 gold classes, plus a separate abstention column; no forced null class."""
    from .common import metrics
    if any(p is not None and p not in labels for p in predicted):raise ValueError('invalid prediction')
    n=len(labels);counts=[[0]*(n+1) for _ in labels]
    for g,p in zip(gold,predicted):counts[labels.index(g)][n if p is None else labels.index(p)]+=1
    if not gold or len(gold)!=len(predicted):raise ValueError('aligned nonempty decisions required')
    classes=[]
    for i,name in enumerate(labels):
        tp=counts[i][i];fp=sum(row[i] for row in counts)-tp;fn=sum(counts[i])-tp
        p=tp/(tp+fp) if tp+fp else 0.;r=tp/(tp+fn) if tp+fn else 0.
        classes.append(dict(label=name,support=sum(counts[i]),precision=p,recall=r,f1=2*p*r/(p+r) if p+r else 0.))
    tp=sum(counts[i][i] for i in range(1,n));guesses=sum(sum(row[1:n]) for row in counts)
    positives=sum(sum(row) for row in counts[1:]);p=tp/guesses if guesses else 0.;r=tp/positives if positives else 0.
    accuracy=sum(counts[i][i] for i in range(n))/len(gold)
    return dict(items=len(gold),loss=loss,accuracy=accuracy,micro_precision=p,micro_recall=r,
        micro_f1=2*p*r/(p+r) if p+r else 0.,micro_convention='benchmark_excluding_no_relation_abstention_is_false_negative',
        macro_precision=sum(x['precision'] for x in classes)/n,macro_recall=sum(x['recall'] for x in classes)/n,
        macro_f1=sum(x['f1'] for x in classes)/n,macro_convention='all_40_labels_zero_division_0',
        per_class=classes,confusion_matrix=counts,confusion_columns=[*labels,'ABSTAIN'],
        abstained=predicted.count(None),coverage=1-predicted.count(None)/len(gold),samples_dropped=0)
