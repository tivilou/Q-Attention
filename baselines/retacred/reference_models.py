"""Separate active paths for routing, syntax curriculum, and LUKE variants.

Unspecified author ports are frozen reconstructions, never score replays.
"""
from __future__ import annotations
import copy, json
from pathlib import Path
from types import SimpleNamespace
from .common import ROOT, PACKAGE, digest, pin_source
from .reference_mechanisms import source_classes, graph_attention_module, focal_loss, core_record, one_hop_intervention
from experiments.prepare_retacred_reference_improved_baseline import UPSTREAM_LABELS

LABELS=list(UPSTREAM_LABELS)

class DependencyParser:
    def __init__(self, path):
        import spacy
        if spacy.__version__!='3.7.5':raise ValueError('parser runtime requires spacy3.7.5')
        self.nlp=spacy.load(str(path),exclude=['ner'])
        if self.nlp.meta['version']!='3.7.1' or 'parser' not in self.nlp.pipe_names:
            raise ValueError('locked dependency parser3.7.1 required')
    def parse(self,record):
        from spacy.tokens import Doc
        doc=Doc(self.nlp.vocab,words=record['token'],spaces=[True]*(len(record['token'])-1)+[False])
        doc=self.nlp(doc)
        if [t.text for t in doc]!=record['token']:raise ValueError('parser changed pretokenized words')
        return {'heads':[t.head.i for t in doc],'pos':[t.pos_ for t in doc], 'dependencies':[t.dep_ for t in doc]}

def padded(features,tokenizer,device):
    import torch
    width=max(len(f['input_ids']) for f in features)
    out={'input_ids':[],'attention_mask':[],'labels':[]}
    for f in features:
        out['input_ids'].append(f['input_ids']+[tokenizer.pad_token_id]*(width-len(f['input_ids'])))
        out['attention_mask'].append([1]*len(f['input_ids'])+[0]*(width-len(f['input_ids'])))
        out['labels'].append(f['labels'])
    return {k:torch.tensor(v,device=device,dtype=torch.long) for k,v in out.items()}

def source_identity(path):
    lock=json.loads((PACKAGE/'sequence_routing_resources.lock.json').read_text())
    for name,sha in lock['active_file_sha256'].items():
        if digest(path/name)!=sha:raise ValueError('sequence source drift: '+name)
    return lock['active_file_sha256']

class SequenceRouting:
    def __init__(self,checkpoint,source,resource,*,device,checkpointing=False,config=None):
        import torch
        from torch import nn
        from torch.nn.functional import normalize
        from transformers import AutoTokenizer,AutoModelForMaskedLM
        self.device=device;self.labels=LABELS;self.source=Path(source)
        self.identity=source_identity(self.source)
        resource=Path(resource);lock=json.loads((PACKAGE/'reference_resources.lock.json').read_text())['routing']
        code=resource/'heinsen_routing.py'
        if digest(code)!=lock['sha256']:raise ValueError('routing dependency source drift')
        ns={'__name__':'locked_heinsen_routing'};exec(compile(code.read_bytes(),str(code),'exec'),ns)
        namespace={'torch':torch,'nn':nn,'F':torch.nn.functional,'Routing':ns['EfficientVectorRouting']}
        RoutingHead,MLP=source_classes(self.source/'RC_Model.py',('RoutingHead','MLP'),namespace)
        self.tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True)
        specials=['[mask]','[p0]','[e11]','[e12]','[e21]','[e22]','[entsep]','[es]','[en]','[CLS]','[SEP]']
        # Canonical label order is an explicit bootstrap choice; unused relation
        # tokens are installed once, never from dev/test first-occurrence order.
        self.tokenizer.add_special_tokens({'additional_special_tokens':specials+['['+r.lower()+']' for r in LABELS]})
        if config is None:
            transformer=AutoModelForMaskedLM.from_pretrained(checkpoint,local_files_only=True,attn_implementation='eager')
        else:
            from transformers import RobertaForMaskedLM
            transformer=RobertaForMaskedLM(config)
        transformer.resize_token_embeddings(len(self.tokenizer));h=transformer.config.hidden_size
        depth=transformer.config.num_hidden_layers+1
        if checkpointing:transformer.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False,'preserve_rng_state':True})
        routing_size=256 if config is None else 4;output_size=512 if config is None else 8
        head=RoutingHead({'d_depth':depth},[
            {'n_inp':-1,'n_out':routing_size,'d_inp':h,'d_out':h},
            {'n_inp':routing_size,'n_out':routing_size,'d_inp':h,'d_out':h},
            {'n_inp':routing_size,'n_out':output_size,'d_inp':h,'d_out':1}])
        classifier=MLP([output_size,output_size,output_size,40])
        for name,p in classifier.named_parameters():
            if 'weight' in name and p.dim()==2:nn.init.xavier_normal_(p)
        class Model(nn.Module):
            def __init__(self):
                super().__init__();self.transformer=transformer;self.head_3=head;self.classifier=classifier;self.last={}
            def forward(self,input_ids,attention_mask,labels=None):
                # Original H3 stacks all hidden depths and routes position+depth.
                # Padding participation is retained (source does not mask H3).
                # The vocabulary prediction head is not consumed by H3. Keep
                # its parameters/source names but execute only the encoder.
                depths=self.transformer.roberta(input_ids=input_ids,attention_mask=attention_mask,output_hidden_states=True).hidden_states
                stack=torch.stack(depths,dim=-2);representation=self.head_3(stack);logits=self.classifier(representation)
                self.last={'depth_stack':stack,'routing_representation':representation,'logits':logits}
                return torch.nn.functional.cross_entropy(logits,labels) if labels is not None else None,logits
        self.model=Model().to(device)
    def encode(self,records):
        out=[]
        for r in records:
            tokens=[]
            for i,t in enumerate(r['token']):
                for role,opening in [('subj','[e11]'),('obj','[e21]')]:
                    if i==r[role+'_start']:tokens.append(opening)
                tokens.append(t)
                for role,closing,punct in [('subj','[e12]',('*','@')),('obj','[e22]',('#','^'))]:
                    if i==r[role+'_end']:tokens.extend([punct[0],*r[role+'_type'].split(),punct[1],closing])
            ids=self.tokenizer.encode(' '.join(tokens),truncation=False)
            if len(ids)>512:raise ValueError('Sequence Mix exceeds512; no silent entity truncation')
            out.append({'input_ids':ids,'labels':LABELS.index(r['relation']),'mix_tokens':tokens})
        return out
    def batch(self,features):return padded(features,self.tokenizer,self.device)
    def forward(self,batch):return self.model(**batch)
    def finalize_vocabulary(self):pass
    def trace_fields(self,batch,row):return {'representation':'all_transformer_depths_then_three_EfficientVectorRouting_stages'}
    def parameter_trace(self):return head_gradients(self.model)

def head_gradients(model):
    return {'parameter_gradient.'+name:p.grad.detach().cpu().clone() for name,p in model.named_parameters()
            if p.grad is not None and ('classifier' in name or 'bilinear' in name or 'head_3' in name)}

class SyntaxCurriculum:
    def __init__(self,checkpoint,parser,*,device,checkpointing=False,config=None):
        import torch
        from torch import nn
        from transformers import AutoTokenizer,AutoModel,RobertaModel
        self.device=device;self.labels=LABELS;self.parser=parser
        self.tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True,use_fast=True,add_prefix_space=True)
        encoder=AutoModel.from_pretrained(checkpoint,local_files_only=True,attn_implementation='eager') if config is None else RobertaModel(config)
        if checkpointing:encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False,'preserve_rng_state':True})
        h=encoder.config.hidden_size;heads=4;hd=128 if config is None else 4
        gat1=graph_attention_module(h,heads,hd);gat2=graph_attention_module(heads*hd,heads,hd)
        class Model(nn.Module):
            def __init__(self):
                super().__init__();self.encoder=encoder;self.gat1=gat1;self.gat2=gat2
                self.bilinear=nn.Bilinear(h+heads*hd,h+heads*hd,40);self.dropout=nn.Dropout(.1);self.last={}
            def forward(self,input_ids,attention_mask,labels,word_positions,word_valid,adjacency,subject_words,object_words,marker_positions):
                hidden=self.encoder(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state
                words=hidden.gather(1,word_positions[:,:,None].expand(-1,-1,h))*word_valid[:,:,None]
                graph=self.gat2(self.gat1(words,adjacency,word_valid),adjacency,word_valid)
                subject=(graph*subject_words[:,:,None]).sum(1)/subject_words.sum(1,keepdim=True)
                object_=(graph*object_words[:,:,None]).sum(1)/object_words.sum(1,keepdim=True)
                b=torch.arange(len(hidden),device=hidden.device)
                s=torch.cat([hidden[b,marker_positions[:,0]],subject],-1);o=torch.cat([hidden[b,marker_positions[:,1]],object_],-1)
                logits=self.bilinear(self.dropout(s),self.dropout(o))
                self.last={'word_hidden':words,'dependency_adjacency':adjacency,'subject_representation':s,'object_representation':o,'logits':logits}
                return torch.nn.functional.cross_entropy(logits,labels),logits
        self.model=Model().to(device)
    def encode(self,records):
        out=[]
        for r in records:
            parsed=self.parser.parse(r);words=[];raw_to_marked=[];markers={}
            for i,t in enumerate(r['token']):
                for role,punct in [('subj','@'),('obj','#')]:
                    if i==r[role+'_start']:markers[role]=len(words);words.append(punct)
                raw_to_marked.append(len(words));words.append(t)
                for role,punct in [('subj','@'),('obj','#')]:
                    if i==r[role+'_end']:words.append(punct)
            encoded=self.tokenizer(words,is_split_into_words=True,truncation=False);word_ids=encoded.word_ids()
            if len(encoded['input_ids'])>512:raise ValueError('dependency model overflow512')
            positions=[word_ids.index(w) for w in raw_to_marked]
            out.append({'input_ids':encoded['input_ids'],'labels':LABELS.index(r['relation']),
                'word_positions':positions,'heads':parsed['heads'],'dependency':parsed,
                'subject_words':[int(r['subj_start']<=i<=r['subj_end']) for i in range(len(positions))],
                'object_words':[int(r['obj_start']<=i<=r['obj_end']) for i in range(len(positions))],
                'marker_positions':[word_ids.index(markers[x]) for x in ('subj','obj')]})
        return out
    def batch(self,features):
        import torch
        batch=padded(features,self.tokenizer,self.device);width=max(len(f['heads']) for f in features)
        adjacency=torch.zeros(len(features),width,width,dtype=torch.bool,device=self.device)
        for b,f in enumerate(features):
            for i,h in enumerate(f['heads']):adjacency[b,i,h]=True;adjacency[b,h,i]=True
        batch['adjacency']=adjacency
        for key in ('word_positions','subject_words','object_words'):
            batch[key]=torch.tensor([f[key]+[0]*(width-len(f[key])) for f in features],device=self.device)
        batch['word_valid']=torch.tensor([[1]*len(f['heads'])+[0]*(width-len(f['heads'])) for f in features],device=self.device)
        batch['marker_positions']=torch.tensor([f['marker_positions'] for f in features],device=self.device)
        return batch
    def forward(self,batch):return self.model(**batch)
    def finalize_vocabulary(self):pass
    def trace_fields(self,batch,row):return {'representation':'marker_plus_dependency_GAT_entity_mean_then_bilinear'}
    def parameter_trace(self):return head_gradients(self.model)

class LukeVariants:
    def __init__(self,model_id,source,checkpoint,train_records,*,device,checkpointing=False,parser=None,foundation=None):
        import torch
        from torch import nn
        from .luke import LUKE
        self.foundation=foundation or LUKE(source,checkpoint,LABELS,device=device,checkpointing=checkpointing)
        self.model_id=model_id;self.device=device;self.labels=LABELS;self.tokenizer=self.foundation.tokenizer;self.parser=parser
        self.model=self.foundation.model;self.coefficients=(0.,0.) if model_id=='luke_core' else 0.
        self.type_pairs={}
        for r in train_records:self.type_pairs.setdefault((r['subj_type'],r['obj_type']),set()).add(LABELS.index(r['relation']))
        if model_id=='luke_cfie':
            types=sorted({r[x+'_type'] for r in train_records for x in ('subj','obj')});self.type_ids={'UNK':0,**{t:i+1 for i,t in enumerate(types)}}
            self.pos_ids={t:i for i,t in enumerate(['UNK','ADJ','ADP','ADV','AUX','CCONJ','DET','INTJ','NOUN','NUM','PART','PRON','PROPN','PUNCT','SCONJ','SYM','VERB','X','SPACE'])}
            backbone=self.model.luke;h=backbone.config.hidden_size
            class Causal(nn.Module):
                def __init__(self):
                    super().__init__();self.luke=backbone;self.dropout=nn.Dropout(.1)
                    self.type_embeddings=nn.Embedding(len(types)+1,32);self.pos_embeddings=nn.Embedding(19,32)
                    self.sentence_classifier=nn.Linear(h,40);self.entity_classifier=nn.Linear(2*h,40)
                    self.tag_classifier=nn.Linear(128,40);self.last={};self.alpha=0.
                def branch(self,batch):
                    output=self.luke(**{k:v for k,v in batch.items() if k in ('input_ids','attention_mask','entity_ids','entity_attention_mask','entity_position_ids')},return_dict=True)
                    sentence=self.dropout(output.last_hidden_state[:,0]);entities=self.dropout(output.entity_last_hidden_state.flatten(1))
                    tags=torch.cat([self.type_embeddings(batch['type_ids']).flatten(1),self.pos_embeddings(batch['pos_ids']).flatten(1)],-1)
                    s=self.sentence_classifier(sentence);x=self.entity_classifier(entities);z=self.tag_classifier(tags)
                    return s+x+z,dict(sentence_representation=sentence,entity_representation=entities,tag_representation=tags,
                                     sentence_logits=s,entity_logits=x,tag_logits=z)
                def forward(self,**batch):
                    labels=batch.pop('labels',None);cf=batch.pop('counterfactual',None)
                    logits,parts=self.branch(batch);self.last=parts
                    if self.training:
                        loss=None if labels is None else torch.nn.functional.cross_entropy(logits,labels)+.9*torch.nn.functional.cross_entropy(parts['sentence_logits'],labels)+.9*torch.nn.functional.cross_entropy(parts['tag_logits'],labels)+torch.nn.functional.cross_entropy(parts['entity_logits'],labels)
                    else:
                        cf_logits,cf_parts=self.branch(cf);self.last.update({'counterfactual.'+k:v for k,v in cf_parts.items()})
                        self.last['factual_logits']=logits;self.last['counterfactual_logits']=cf_logits
                        logits=logits-self.alpha*cf_logits
                        loss=None if labels is None else torch.nn.functional.cross_entropy(logits,labels)
                    self.last['final_logits']=logits
                    return loss,logits
            self.model=Causal().to(device)
    def encode(self,records):
        # LUKE encoder requires labels for its interface; counterfactual inputs
        # are made from a gold-free copy and a constant interface placeholder.
        features=self.foundation.encode(records)
        for r,f in zip(records,features):
            f['raw_input']=r
            if self.model_id=='luke_core':
                for kind in ('entity_only','blank'):
                    cf=core_record(r,kind);cf['relation']='no_relation'
                    f[kind]=self.foundation.encode([cf])[0]
                f['candidate_ids']=sorted(self.type_pairs.get((r['subj_type'],r['obj_type']),set(range(40))))
            elif self.model_id=='luke_cfie':
                p=self.parser.parse(r);cf,mask=one_hop_intervention(r,p['heads']);cf['relation']='no_relation'
                f['counterfactual']=self.foundation.encode([cf])[0];f['dependency']=p;f['masked_word_indices']=mask
                f['type_ids']=[self.type_ids.get(r[x+'_type'],0) for x in ('subj','obj')]
                f['pos_ids']=[self.pos_ids.get(p['pos'][r[x+'_start']],0) for x in ('subj','obj')]
        return features
    def batch(self,features):
        import torch
        batch=self.foundation.batch(features)
        if self.model_id=='luke_core':
            batch['entity_only']=self.foundation.batch([f['entity_only'] for f in features]);batch['blank']=self.foundation.batch([f['blank'] for f in features])
            mask=torch.zeros(len(features),40,dtype=torch.bool,device=self.device)
            for i,f in enumerate(features):mask[i,f['candidate_ids']]=True
            batch['candidate_mask']=mask
        elif self.model_id=='luke_cfie':
            batch['counterfactual']=self.foundation.batch([f['counterfactual'] for f in features])
            batch['counterfactual'].pop('labels')
            for key in ('type_ids','pos_ids'):
                batch[key]=torch.tensor([f[key] for f in features],device=self.device);batch['counterfactual'][key]=batch[key]
        return batch
    def forward(self,batch):
        import torch
        from .reference_mechanisms import core_scores
        if self.model_id=='luke_cfie':return self.model(**dict(batch))
        if self.model_id=='luke_core':
            fact={k:v for k,v in batch.items() if k not in ('entity_only','blank','candidate_mask')}
            output=self.model(**fact,return_dict=True);loss=output.loss;logits=output.logits
            if not self.model.training:
                entity=self.model(**{k:v for k,v in batch['entity_only'].items() if k!='labels'},return_dict=True).logits.softmax(-1)
                blank=self.model(**{k:v for k,v in batch['blank'].items() if k!='labels'},return_dict=True).logits.softmax(-1)
                self.model.last={'factual_probability':logits.softmax(-1),'entity_only_probability':entity,'blank_probability':blank,'candidate_mask':batch['candidate_mask']}
                # Scores are not normalized probabilities; NLL is reported as
                # unavailable rather than softmax-repairing this inference rule.
                logits=core_scores(logits.softmax(-1),entity,blank,self.coefficients,batch['candidate_mask'])
                # Keep factual CE as a separately scoped diagnostic, not NLL of corrected scores.
                self.loss_scope='factual_branch_cross_entropy_not_corrected_score_NLL'
                self.model.last['corrected_scores']=logits
            return loss,logits
        output=self.model(**batch,return_dict=True)
        loss=focal_loss(output.logits,batch['labels']) if self.model_id=='luke_focal' else output.loss
        return loss,output.logits
    def finalize_vocabulary(self):self.foundation.finalize_vocabulary()
    def trace_fields(self,batch,row):return {'representation':'LUKE_word_and_two_role_entity_states','variant':self.model_id,'dev_coefficients':self.coefficients}
    def parameter_trace(self):return head_gradients(self.model)

def make(model_id,assets,records,*,device='cuda:0',checkpointing=True,config=None):
    if model_id.startswith('luke_'):
        parser=DependencyParser(assets['parser']) if model_id=='luke_cfie' else None
        return LukeVariants(model_id,assets['luke_source'],assets['luke'],records,device=device,checkpointing=checkpointing,parser=parser)
    if model_id=='sequence_routing_h3':return SequenceRouting(assets['roberta'],assets['sequence_source'],assets['routing'],device=device,checkpointing=checkpointing,config=config)
    if model_id=='curriculum':return SyntaxCurriculum(assets['roberta'],DependencyParser(assets['parser']),device=device,checkpointing=checkpointing,config=config)
    if model_id=='metaentail':
        from .metaentail_neural import MetaEntailFoundation
        return MetaEntailFoundation(assets['mnli'],device=device,checkpointing=checkpointing)
    if model_id=='ptr':
        from .ptr import PTR
        return PTR(Path(assets['ptr_source']),Path(assets['roberta']),device=device,checkpointing=checkpointing,metadata=Path(assets['ptr_metadata']))
    from .improved import Improved
    formats={'typed_marker_comparison_2025':'typed_entity_marker_punct','typed_marker_punct':'typed_entity_marker_punct','typed_marker':'typed_entity_marker','entity_marker_punct':'entity_marker_punct','entity_marker':'entity_marker'}
    adapter=Improved(Path(assets['improved_source']),Path(assets['roberta']),formats[model_id],device=device,checkpointing=checkpointing)
    # The author's @autocast() decorates the forward itself. Disabling only an
    # outer context does NOT implement this lane's explicitly declared FP32.
    method=adapter.model.forward
    if not hasattr(method,'__wrapped__'):raise ValueError('pinned marker autocast wrapper changed')
    adapter.model.forward=method.__wrapped__.__get__(adapter.model,type(adapter.model))
    adapter.amp_enabled=False
    return adapter
