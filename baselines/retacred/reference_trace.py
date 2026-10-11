"""Actual train/test numeric evidence; no post-hoc forward is called here."""
from __future__ import annotations
import json,math

def json_numbers(value):
    """JSON display projection only; original tensor values remain unchanged."""
    if isinstance(value,float) and not math.isfinite(value):return "NaN" if math.isnan(value) else "+Infinity" if value>0 else "-Infinity"
    if isinstance(value,list):return [json_numbers(x) for x in value]
    if isinstance(value,dict):return {k:json_numbers(v) for k,v in value.items()}
    return value

from pathlib import Path
from .common import atomic_json,digest
from .improved import SampleCapture,leaves
from .train import save_torch

class Capture(SampleCapture):
    def __init__(self,model,rows,batch_size):
        super().__init__(model,rows[0],batch_size);self.rows=list(rows)
    def select(self,tensor):
        value=tensor.detach()
        if value.ndim and value.shape[0]==self.batch_size:value=value[self.rows]
        return value.cpu().clone()
    def snapshot_mechanism(self):
        for name,module in self.model.named_modules():
            for key,tensor in leaves(getattr(module,'last',{}),'mechanism.'+name):
                self.values[key]=self.select(tensor)

def head_snapshot(adapter):
    values={name:p.detach().cpu().clone() for name,p in adapter.model.named_parameters()
            if any(part in name for part in ('classifier','bilinear','mlp.','extra_token_embeddings.'))}
    if getattr(adapter,'model_id',None)=='ptr':
        import torch
        ids=torch.unique(torch.cat(adapter.processor.prompt_label_idx)).to(adapter.device)
        values['verbalizer_word_rows']=adapter.model.model.embeddings.word_embeddings.weight[ids].detach().cpu().clone()
    return values

def emit(adapter,batch,records,capture,*,phase,step,event,directory,loss,logits,extra=None,parameter_before=None):
    import torch
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    for name,tensor in leaves(batch,'input'):capture.values[name]=capture.select(tensor)
    capture.values['output.logits_or_scores']=capture.select(logits)
    if loss is not None:capture.values['output.physical_batch_loss']=loss.detach().cpu().clone()
    if phase=='train':
        capture.values.update(adapter.parameter_trace())
        for name,before in (parameter_before or {}).items():
            if name=='verbalizer_word_rows':
                ids=torch.unique(torch.cat(adapter.processor.prompt_label_idx)).to(adapter.device)
                parameter=adapter.model.model.embeddings.word_embeddings.weight[ids]
            else:parameter=dict(adapter.model.named_parameters())[name]
            capture.values['parameter_before.'+name]=before
            capture.values['parameter_after.'+name]=parameter.detach().cpu().clone()
    if not capture.values or not any('embedding' in name for name in capture.values):
        raise ValueError('sample embedding values were not observed')
    if not any('softmax' in name or 'attention_scores_to_probabilities' in name for name in capture.values):
        raise ValueError('actual attention probability values were not observed')
    if phase=='train' and not any('gradient' in name for name in capture.values):
        raise ValueError('actual activation/parameter gradients were not observed')
    tensor_path=directory/f'{phase}-update{step}-event{event}.pt';save_torch(tensor_path,capture.values)
    # Reopen every tensor, including full numeric values, not metadata only.
    restored=torch.load(tensor_path,map_location='cpu',weights_only=True,mmap=True)
    representations=[]
    for name,tensor in capture.values.items():
        if name not in restored or not torch.equal(tensor,restored[name]):raise ValueError('numeric trace roundtrip differs')
        representations.append({'id':name,'title':name.replace('.',' / '),'dtype':str(tensor.dtype),'shape':list(tensor.shape),
            'numel':tensor.numel(),'artifact_key':name,'value_status':'complete_values',
            'meaning':'actual '+phase+' event; key identifies module, input/output or gradient; no values reconstructed'})
    del restored
    payload={'schema':'sample-trace.v2','status':'complete','phase':phase,'capture_origin':'actual_'+phase+'_event',
        'optimizer_update':step,'event':event,'raw_inputs':records,'selected_physical_rows':capture.rows,
        'tensor_artifact':tensor_path.name,'sha256':digest(tensor_path),'size_bytes':tensor_path.stat().st_size,
        'representations':representations,'nodes':capture.nodes,'target_access':'training_loss_or_post_prediction_evaluation',
        'outputs':json_numbers(logits.detach().cpu()[capture.rows].tolist()), 'extra':extra or {},
        'actual_values_required':True,'truncation':False,'model_variant':getattr(adapter,'model_id',None)}
    if hasattr(adapter,'tokenizer'):
        payload['tokens']=[adapter.tokenizer.convert_ids_to_tokens(batch['input_ids'][i].detach().cpu().tolist()) for i in capture.rows]
        payload['token_ids']=[batch['input_ids'][i].detach().cpu().tolist() for i in capture.rows]
    artifact={'path':tensor_path.name,'sha256':payload['sha256'],'bytes':payload['size_bytes'],'visibility':'private'}
    for representation in representations:
        key=representation['id']
        if 'embedding' in key:meaning='词、位置或实体标记由整数ID映射成的向量；该键标明具体嵌入模块和输入/输出。'
        elif 'softmax' in key or 'attention_scores_to_probabilities' in key:meaning='模型当次计算的归一化权重；attention表示各位置对其他位置的关注程度，routing表示路由分配。'
        elif 'gradient' in key:meaning='当次反向传播的梯度；激活梯度跟随所选物理行，参数梯度汇总整个逻辑batch。'
        elif 'parameter_before' in key:meaning='本次优化器更新之前的分类参数，和同名parameter_after构成实际更新对。'
        elif 'parameter_after' in key:meaning='本次优化器更新之后的分类参数，非额外重放或历史补算。'
        elif 'adjacency' in key:meaning='真实依存分析构建的图连接矩阵；对称边和自环是本重建的明确选择。'
        elif 'counterfactual' in key:meaning='执行已声明反事实干预后得到的实际分支张量；该键标明分支和处理模块。'
        elif 'routing' in key or 'head_3' in key:meaning='H3顺序路由的实际中间表示；位置与编码层深度参与三段路由。'
        elif 'logits' in key or 'scores' in key:meaning='当前分支或最终关系的原始分数；CoRE修正分数不被伪称为概率。'
        elif key.startswith('input'):meaning='送入当前模型分支的真实张量；labels只用于训练损失或预测后的评价，不参与句子编码。'
        else:meaning='当次'+key.replace('.',' / ')+'模块的真实输入或输出；可按shape逐轴读取完整数值。'
        shape=representation['shape'];axes=['axis_'+str(i) for i in range(len(shape))]
        if len(shape)==4 and 'depth_stack' in key:axes=['physical_row','token_position','encoder_depth','hidden_feature']
        elif len(shape)==4 and ('attention' in key or 'softmax' in key):axes=['physical_row_or_candidate','attention_head','query_position','key_position']
        elif len(shape)==3:axes=['physical_row_or_candidate','position','feature']
        elif len(shape)==2:axes=['row_or_position','feature_or_class']
        representation.update(status='observed',meaning=meaning,axis_semantics=axes,
            artifact_refs=[artifact],portal_preview={'tensor_key':key,'privacy':'private_coordinate_inspector_not_public_json'},
            value_capture={'status':'complete','shape':shape,'logical_element_count':representation['numel'],
                'stored_element_count':representation['numel'],'artifact_refs':[artifact],
                'reader_projection':{'status':'available','kind':'coordinate_inspector','reader_ref':'python:baselines.retacred.reference_trace.ValueReader.read'}},
            display={'label':representation['title'],'meaning':meaning,'show_default':True})
    raw={k:v for k,v in records[0].items() if k!='relation'};gold=records[0].get('relation')
    predicted=int(logits[capture.rows[0]].argmax())
    labels=getattr(adapter,'labels',None)
    predicted_relation=labels[predicted] if labels is not None and predicted<len(labels) else None
    stages=[{'stage_id':'data','canonical_stage':'data','input_refs':['source:sample'], 'output_refs':['sample'],
            'purpose':'给出原句和指定的两个实体。','outputs':raw},
        {'stage_id':'preprocess','canonical_stage':'preprocess','input_refs':['sample'],
            'output_refs':[k for k in capture.values if k.startswith('input.')], 'purpose':'按此模型的标记、候选或图结构生成实际输入。'}]
    for node in capture.nodes:
        stages.append({'stage_id':node['id'],'canonical_stage':'scoring','input_refs':node['inputs'],'output_refs':node['outputs'],
            'purpose':'实际计算模块：'+node['module']})
    if phase=='train':
        stages.append({'stage_id':'training','canonical_stage':'training','input_refs':['output.physical_batch_loss','source:training_target'],
            'output_refs':[k for k in capture.values if 'gradient' in k or 'parameter_after' in k],
            'purpose':'真实损失、反向传播和参数更新；参数梯度属于整个逻辑batch。'})
    stages.append({'stage_id':'selection','canonical_stage':'selection','input_refs':['output.logits_or_scores'],
        'output_refs':['prediction'],'outputs':{'predicted_relation':predicted_relation,'nli_order':(extra or {}).get('nli_order')},
        'purpose':'关系模型选出关系；NLI模型先输出三类判断，组级关系选择另见grouped-decision.json。'})
    stages.append({'stage_id':'evaluation','canonical_stage':'evaluation','input_refs':['prediction','source:post_prediction_reference'],
        'output_refs':['correctness'],'outputs':{'reference_relation':gold,'predicted_relation':predicted_relation},'purpose':'预测结束后再对照正确关系。'})
    payload.update(schema_version='sample-trace.v2',trace_completeness='complete',trace_id=tensor_path.stem,
        samples=[{'sample_id':raw.get('id'), 'lane':'auxiliary_review' if (extra or {}).get('stage')=='curriculum_review' else 'training' if phase=='train' else 'evaluation','split':(extra or {}).get('source_split',phase),
            'roles':{'source_sample':{'sentence':' '.join(raw.get('token',[])),'tokens':raw.get('token')},
                'sequence_or_features':{'token_ids':payload.get('token_ids')},
                'task_objects':{role:{'span':[raw.get(prefix+'_start'),raw.get(prefix+'_end')],
                    'text':' '.join(raw.get('token',[])[raw.get(prefix+'_start',0):raw.get(prefix+'_end',-1)+1]),'type':raw.get(prefix+'_type')}
                    for role,prefix in [('subject','subj'),('object','obj')]},
                'target_or_gold':gold,'prediction':predicted_relation},'representations':representations,'stages':stages}],
        task_contract={'task_type':'sentence_relation_classification','required_value_representations':[r['id'] for r in representations],
            'required_splits':[phase],'checkpoint_policy':'actual_event_not_post_hoc_reconstructed'})
    path=tensor_path.with_suffix('.json');atomic_json(path,payload)
    return {'phase':phase,'path':path.name,'sha256':digest(path),'tensor_sha256':payload['sha256']}

def validate(run,traces):
    if {t['phase'] for t in traces}!={'train','test'}:raise ValueError('actual train and test events required')
    for t in traces:
        path=Path(run)/'case_study'/t['path']
        if path.name!=t['path'] or digest(path)!=t['sha256']:raise ValueError('unsafe or changed trace manifest')
        value=json.loads(path.read_text());tensor=path.parent/value['tensor_artifact']
        if tensor.parent!=path.parent or tensor.is_symlink():raise ValueError('changed tensor evidence')
        from .reference_archive import verify_available
        verify_available(tensor,value['sha256'],value['size_bytes'])
        if value.get('status')!='complete' or any(r['value_status']!='complete_values' for r in value['representations']):
            raise ValueError('metadata-only or partial evidence')

class ValueReader:
    """Private multidimensional slicer; full tensors remain on the exchange."""
    def __init__(self,path):
        import torch
        self.path=Path(path);self.trace=json.loads(self.path.read_text());name=self.trace['tensor_artifact']
        if Path(name).name!=name:raise ValueError('unsafe artifact name')
        artifact=self.path.parent/name
        from .reference_archive import verify_available, materialize
        verify_available(artifact,self.trace['sha256'],self.trace['size_bytes'])
        with materialize(artifact) as local:
            self.values=torch.load(local,map_location='cpu',weights_only=True,mmap=True)
    def read(self,key,*,prefix=(),window=None):
        import math
        tensor=self.values[key]
        if len(prefix)>tensor.ndim or any(type(x)is not int or not 0<=x<tensor.shape[i] for i,x in enumerate(prefix)):raise ValueError('invalid prefix coordinate')
        remaining=tensor.shape[len(prefix):]
        if len(remaining)>2:raise ValueError('select leading axes to expose at most a matrix')
        windows=window or [(0,n) for n in remaining]
        if len(windows)!=len(remaining) or any(len(w)!=2 or any(type(x)is not int for x in w) or not 0<=w[0]<w[1]<=n for w,n in zip(windows,remaining)):raise ValueError('invalid matrix window')
        if math.prod(b-a for a,b in windows)>4096:raise ValueError('bounded inspector allows at most4096 values per read')
        return json_numbers(tensor[tuple(prefix)+tuple(slice(a,b) for a,b in windows)].tolist())
