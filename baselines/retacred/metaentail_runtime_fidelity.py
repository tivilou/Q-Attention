"""Bounded MNLI diagnostic: actual group traces and exact two-update recovery.

No formal training, author optimizer equivalence or reported-score certificate.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path

from .assets import reject_symlinks
from .common import ROOT, atomic_json, digest, fingerprint
from .luke import require_runtime_memory
from .luke_runtime_fidelity import tensor_identity, state_identity
from .metaentail import fit_type_filter, grouped_prediction, NLI_LABELS
from .metaentail_checkpoint import REVISION, checkpoint_spec
from .metaentail_neural import MetaEntailFoundation, collate_pairs

MEMORY_MINIMUM = {'tiny':768*1024**2, 'full':24*1024**3}
PENDING = ['general_domain_author_training_recipe', 'canonical_abstention_evaluator_binding',
           'independent_paper_mechanism_review', 'formal_full_weight_CUDA_and_multi_seed_score']


def description(mode, device):
    if mode not in MEMORY_MINIMUM or device not in ('cpu','cuda'):
        raise ValueError('explicit diagnostic mode/device required')
    return {'schema':'metaentail-runtime-gate.v1', 'mode':mode, 'device':device,
            'formal_training':False, 'formal_comparison_admitted':False,
            'minimum_host_headroom_bytes':MEMORY_MINIMUM[mode], 'seed':13,
            'synthetic_train_examples':3, 'selected_train_groups':1, 'selected_test_groups':1,
            'pairs_per_selected_group':3, 'optimizer_updates':2, 'replayed_updates':1,
            'precision':'FP32', 'rtol':0, 'atol':0,
            'weight_scope':'random_tiny_architecture' if mode=='tiny' else 'hash_locked_pretrained_MNLI',
            'tokenization':'synthetic_non_BPE_ids' if mode=='tiny' else 'actual_official_fast_BPE',
            'optimizer':{'type':'torch.optim.AdamW','lr':1e-5,'eps':1e-8,'betas':[0.9,0.999],
                         'weight_decay':0.01,'gradient_clip_norm':1.0,
                         'warmup_steps':0,'schedule_steps':4,'physical_partition':'whole_selected_group'},
            'optimizer_claim':'diagnostic_only_not_author_recipe', 'pending':list(PENDING)}


def synthetic_records():
    rows=[]
    for i,(verb,label) in enumerate(zip(('visited','joined','attended'),
            ('no_relation','per:employee_of','per:schools_attended'))):
        rows.append({'id':'diagnostic-train-'+str(i),'token':['Alice',verb,'Acme','.'],
                     'subj_start':0,'subj_end':0,'obj_start':2,'obj_end':2,
                     'subj_type':'PERSON','obj_type':'ORGANIZATION','relation':label})
    test=copy.deepcopy(rows[1]); test['id']='diagnostic-test-0'
    test['token']=['Bob','joined','Beta','.']
    return rows,test


def make_adapter(checkpoint, mode, device):
    import torch
    from transformers import RobertaConfig, RobertaForSequenceClassification
    from .metaentail import training_group, inference_group
    records,test=synthetic_records()
    feasibility=fit_type_filter(records,split='train')
    if mode=='full':
        adapter=MetaEntailFoundation(checkpoint,device=device)
        encoded=[adapter.encode(row,feasibility,split='train') for row in records[1:]]
        evaluated=adapter.encode(test,feasibility,split='test')
        tokenization='actual_official_fast_BPE'
    else:
        config=RobertaConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
            num_attention_heads=4,max_position_embeddings=64,num_labels=3,
            hidden_dropout_prob=0.1,attention_probs_dropout_prob=0.1,
            id2label=dict(enumerate(name.upper() for name in NLI_LABELS)),
            label2id={name.upper():i for i,name in enumerate(NLI_LABELS)})
        config._attn_implementation='eager'
        adapter=object.__new__(MetaEntailFoundation)
        adapter.torch,adapter.device,adapter.nli_order=torch,torch.device(device),NLI_LABELS
        adapter.model=RobertaForSequenceClassification(config).to(device)
        def encoded_synthetic(raw,split):
            group=training_group(raw,feasibility) if split=='train' else inference_group(raw,feasibility)
            features=[]
            for index,candidate in enumerate(group['candidates']):
                # Explicit synthetic IDs: this path never claims text/BPE alignment.
                ids=[0,10+(raw['id'].endswith('2')),11,12,2,2,20+index,21,2]
                row={'sample_id':raw['id'],'candidate_index':index,'candidate_relation':candidate['relation'],
                     'input_ids':ids,'attention_mask':[1]*len(ids),'filter_identity':group['filter_identity'],
                     'split':split,'tokenization':'synthetic_non_BPE_ids','tokens':None}
                if split=='train': row['labels']=NLI_LABELS.index(candidate['nli_target'])
                features.append(row)
            return {'group':group,'features':features,'split':split,'nli_order':list(NLI_LABELS),
                    'filter_identity':group['filter_identity']}
        encoded=[encoded_synthetic(row,'train') for row in records[1:]]
        evaluated=encoded_synthetic(test,'test'); tokenization='synthetic_non_BPE_ids'
    batches=[adapter.batch(item['features']) for item in encoded]
    evaluation_batch=adapter.batch(evaluated['features'])
    return adapter,batches,evaluation_batch,encoded,evaluated,records,test,feasibility,tokenization


def capture_group(adapter, pair_count):
    """Capture all pair rows of this one original group, never unrelated groups."""
    from .improved import SampleCapture
    from torch.utils._python_dispatch import TorchDispatchMode
    class GroupCapture(SampleCapture):
        def select(self,tensor):
            return tensor.detach().cpu().clone()
        def __enter__(self):
            super().__enter__()
            capture=self
            class ClassifierObserver(TorchDispatchMode):
                def __torch_dispatch__(self,func,types,args=(),kwargs=None):
                    result=func(*args,**(kwargs or {}))
                    kind=None
                    if (str(func)=='aten.select.int' and args[0].ndim==3
                            and args[0].shape[0]==pair_count and args[1]==1 and args[2]==0
                            and result.ndim==2):
                        kind='cls_readout'
                    elif str(func)=='aten.tanh.default' and result.ndim==2 and result.shape[0]==pair_count:
                        kind='classifier_tanh'
                    if kind and 'metaentail_'+kind not in capture.counts:
                        capture.counts['metaentail_'+kind]=1; key='functional.metaentail_'+kind
                        capture.nodes.append({'id':key,'module':'actual_'+kind,
                            'inputs':[capture.save(key+'.input',args[0])],
                            'outputs':[capture.save(key+'.output',result)]})
                    return result
            self.classifier_observer=ClassifierObserver(); self.classifier_observer.__enter__()
            return self
        def __exit__(self,*exception):
            try: self.classifier_observer.__exit__(*exception)
            finally: super().__exit__(*exception)
    return GroupCapture(adapter.model,row=0,batch_size=pair_count)


def gradient_identity(model):
    import torch
    result={}
    for name,parameter in model.named_parameters():
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all(): raise FloatingPointError('nonfinite gradient: '+name)
            result[name]=dict(tensor_identity(parameter.grad),nonzero=bool(torch.count_nonzero(parameter.grad)))
    required=('roberta.embeddings.word_embeddings.weight',
              'roberta.encoder.layer.0.attention.self.query.weight',
              'classifier.dense.weight','classifier.out_proj.weight')
    if any(name not in result or not result[name]['nonzero'] for name in required):
        raise AssertionError('embedding/attention/dense/NLI-head nonzero gradient path absent')
    return result


def validate_trace(metadata, *, phase, pairs, layers):
    """Check evidence structure without interpreting placeholders as tensors."""
    if phase not in ('train','test') or pairs<=0 or layers<=0: raise ValueError('invalid trace scope')
    tensors=metadata.get('tensors',{})
    required=['task_input.input_ids','task_input.attention_mask','task_output.logits','task_output.probabilities',
              'functional.metaentail_cls_readout.input','functional.metaentail_cls_readout.output',
              'functional.metaentail_classifier_tanh.input','functional.metaentail_classifier_tanh.output',
              'classifier.dense.call0.input.0','classifier.dense.call0.output.tensor',
              'classifier.out_proj.call0.input.0','classifier.out_proj.call0.output.tensor']
    required += ['model.hidden_states.layer'+str(i) for i in range(layers+1)]
    required += ['model.attentions.layer'+str(i) for i in range(layers)]
    required += [f'functional.attention_scores_to_probabilities.call{i}.{suffix}'
                 for i in range(layers) for suffix in ('input','output')]
    for key in required:
        row=tensors.get(key,{})
        if (row.get('value_storage')!='complete_numeric_tensor' or row.get('numel',0)<=0
                or not row.get('shape') or row['shape'][0]!=pairs):
            raise AssertionError('complete aligned group tensor absent: '+key)
    for key in ('task_output.logits','task_output.probabilities'):
        if tensors[key]['shape']!=[pairs,3]: raise AssertionError('NLI head tensor shape changed')
    if phase=='train':
        if 'task_input.labels' not in tensors or 'task_output.loss' not in tensors:
            raise AssertionError('training target/loss absent')
        if not any(key.endswith('.gradient') for key in tensors): raise AssertionError('activation gradients absent')
        if not any(key.startswith('classifier.parameter_gradient.') for key in tensors):
            raise AssertionError('parameter gradients absent')
    elif 'task_input.labels' in tensors or 'task_output.loss' in tensors:
        raise AssertionError('test trace contains gold-dependent model target/loss')


def record_capture(directory,capture,batch,output,adapter,encoded,raw,*,phase,tokenization):
    import torch
    from .train import save_torch
    for key,value in batch.items(): capture.save('task_input.'+key,value)
    capture.save('task_output.logits',output.logits)
    probabilities=output.logits.softmax(-1)
    capture.save('task_output.probabilities',probabilities)
    for i,value in enumerate(output.hidden_states): capture.save('model.hidden_states.layer'+str(i),value)
    for i,value in enumerate(output.attentions): capture.save('model.attentions.layer'+str(i),value)
    if phase=='train':
        capture.values['task_output.loss']=output.loss.detach().cpu().clone()
        capture.values.update(adapter.parameter_trace())
    metadata=capture.metadata()
    validate_trace(metadata,phase=phase,pairs=len(encoded['features']),layers=adapter.model.config.num_hidden_layers)
    decision=adapter.decide(encoded,probabilities.detach().cpu().tolist())
    target=directory/(phase+'-group.pt'); save_torch(target,capture.values)
    reloaded=torch.load(target,map_location='cpu',weights_only=True)
    if state_identity(reloaded)!=state_identity(capture.values): raise AssertionError('numeric group trace reload differs')
    manifest={'schema':'metaentail-diagnostic-group-trace.v1','status':'complete_diagnostic_scope',
        'phase':phase,'capture_origin':'actual_training_forward_backward' if phase=='train' else 'actual_test_forward',
        'raw_input':raw,'tokenization':tokenization,'encoded_group':encoded,
        'pair_rows':len(encoded['features']),'all_candidate_pairs_captured':True,
        'target_relation':raw['relation'],'target_access':'training_loss' if phase=='train' else 'post_prediction_bookkeeping_only',
        'decision':decision,'loss':float(output.loss.detach()) if phase=='train' else None,
        'tensor_artifact':target.name,'sha256':digest(target),'size_bytes':target.stat().st_size,
        'numerical_values':'all_captured_tensors_in_private_pt_not_summaries',**metadata}
    atomic_json(target.with_suffix('.json'),manifest)
    if json.loads(target.with_suffix('.json').read_text())!=manifest: raise AssertionError('trace JSON reload differs')
    return {'json':target.with_suffix('.json').name,'pt':target.name,'sha256':manifest['sha256'],
            'pairs':manifest['pair_rows'],'tensors':len(capture.values)}


def resume_contract(adapter,encoded,feasibility,mode,device):
    order=[[row['sample_id'],row['candidate_relation']] for group in encoded for row in group['features']]
    return {'schema':'metaentail-diagnostic-resume.v1','scope':'private_synthetic_diagnostic_only',
        'mode':mode,'device':device,'seed':13,'config':adapter.model.config.to_dict(),
        'checkpoint':adapter.identity() if mode=='full' else 'random_tiny_not_pretrained',
        'filter':feasibility.to_dict(),'encoded_train_group_hashes':[fingerprint(group) for group in encoded],
        'expanded_pair_order':order,'physical_groups':[len(group['features']) for group in encoded],
        'optimizer':description(mode,device)['optimizer'],
        'source_sha256':{name:digest(ROOT/'baselines/retacred'/name) for name in
            ('metaentail_runtime_fidelity.py','metaentail_neural.py','metaentail.py','improved.py')}}


def exercise(adapter,batches,test_batch,encoded,evaluated,records,test,feasibility,tokenization,mode,device,output):
    import torch
    from transformers import get_linear_schedule_with_warmup
    from .train import rng_state,restore_rng,restore_checkpoint,save_torch
    model=adapter.model; batch=batches[0]
    model.eval()
    with torch.no_grad():
        direct=model(**batch,return_dict=True)
        loss,logits,_=adapter.forward(batch)
        torch.testing.assert_close(logits,direct.logits,rtol=0,atol=0)
        expected_ce=torch.nn.functional.cross_entropy(logits,batch['labels'])
        torch.testing.assert_close(loss,expected_ce,rtol=0,atol=0)
        _,changed,_=adapter.forward(dict(batch,labels=(batch['labels']+1)%3))
        torch.testing.assert_close(logits,changed,rtol=0,atol=0)
    del direct,loss,logits,changed,expected_ce
    model.train(); model.zero_grad(set_to_none=True)
    initial_rng=rng_state()
    direct=model(**batch,return_dict=True,output_hidden_states=True,output_attentions=True)
    direct.loss.backward()
    expected_gradients=gradient_identity(model); expected_rng=state_identity(rng_state())
    expected_loss,expected_logits=direct.loss.detach(),direct.logits.detach()
    del direct
    model.zero_grad(set_to_none=True); restore_rng(initial_rng)
    with capture_group(adapter,len(encoded[0]['features'])) as capture:
        _,_,traced=adapter.forward(batch,trace=True); traced.loss.backward()
    torch.testing.assert_close(traced.loss,expected_loss,rtol=0,atol=0)
    torch.testing.assert_close(traced.logits,expected_logits,rtol=0,atol=0)
    if gradient_identity(model)!=expected_gradients or state_identity(rng_state())!=expected_rng:
        raise AssertionError('group tracing changed gradients or dropout RNG')
    train_trace=record_capture(output,capture,batch,traced,adapter,encoded[0],records[1],
                              phase='train',tokenization=tokenization)
    del capture,traced,expected_loss,expected_logits,expected_gradients,initial_rng
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,eps=1e-8,betas=(0.9,0.999),weight_decay=0.01)
    scheduler=get_linear_schedule_with_warmup(optimizer,0,4)
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    before=tensor_identity(model.classifier.out_proj.weight)
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step(); scheduler.step()
    if tensor_identity(model.classifier.out_proj.weight)==before: raise AssertionError('vacuous optimizer update')
    contract=resume_contract(adapter,encoded,feasibility,mode,device)
    cursor={'epoch':0,'optimizer_update':1,'next_group':1,'next_expanded_pair':len(encoded[0]['features'])}
    checkpoint=output/'post-update1.pt'
    save_torch(checkpoint,{'contract':contract,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
        'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'rng':rng_state(),'cursor':cursor})
    def continuation():
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss,logits,_=adapter.forward(batches[1]); loss.backward()
        if not math.isfinite(float(loss.detach())): raise FloatingPointError('nonfinite continuation loss')
        gradients=fingerprint(gradient_identity(model))
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step(); scheduler.step()
        return {'model':fingerprint(state_identity(model.state_dict())),
            'optimizer':fingerprint(state_identity(optimizer.state_dict())),'scheduler':state_identity(scheduler.state_dict()),
            'rng':fingerprint(state_identity(rng_state())),'loss':float(loss.detach()),
            'logits':tensor_identity(logits),'gradients':gradients}
    uninterrupted=continuation()
    rejected=copy.deepcopy(contract); rejected['seed']=29
    try: restore_checkpoint(checkpoint,rejected,model,optimizer,scheduler,scaler)
    except ValueError: pass
    else: raise AssertionError('changed seed contract accepted')
    if (fingerprint(state_identity(model.state_dict()))!=uninterrupted['model']
            or fingerprint(state_identity(rng_state()))!=uninterrupted['rng']):
        raise AssertionError('rejected checkpoint mutated model/RNG')
    restored=restore_checkpoint(checkpoint,contract,model,optimizer,scheduler,scaler)
    if restored['cursor']!=cursor or optimizer.param_groups[0]['lr']!=1e-5*0.75:
        raise AssertionError('expanded-pair cursor/scheduler recovery mismatch')
    del restored
    resumed=continuation()
    if resumed!=uninterrupted: raise AssertionError('resumed update differs from uninterrupted update')
    model.zero_grad(set_to_none=True); model.eval()
    with torch.no_grad():
        direct=model(**test_batch,return_dict=True,output_hidden_states=True,output_attentions=True)
        expected=tensor_identity(direct.logits); del direct
        with capture_group(adapter,len(evaluated['features'])) as capture:
            _,_,traced=adapter.forward(test_batch,trace=True)
    if traced.loss is not None or tensor_identity(traced.logits)!=expected:
        raise AssertionError('test gold isolation or trace transparency failed')
    test_trace=record_capture(output,capture,test_batch,traced,adapter,evaluated,test,
                             phase='test',tokenization=tokenization)
    return {'native_forward_and_pair_CE':'passed','gold_input_isolation':'passed',
        'nonzero_embedding_attention_NLI_gradients':'passed','train_dropout_trace_transparency':'passed',
        'diagnostic_nonzero_update_and_scheduler':'passed','changed_resume_contract_rejected_before_mutation':'passed',
        'exact_post_update_pair_cursor_resume':'passed','test_forward_trace_transparency':'passed',
        'actual_complete_train_test_group_tensor_roundtrip':'passed','resume_state':resumed,
        'traces':[train_trace,test_trace],'checkpoint':{'file':checkpoint.name,'sha256':digest(checkpoint)},
        'author_optimizer_equivalence':'pending_not_tested'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('tiny','full'),default='full')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'.external/assets/mnli'/REVISION/'model')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--describe',action='store_true',help='read-only, no model/weights/network/writes')
    args=parser.parse_args(argv); result=description(args.mode,args.device)
    if args.describe: print(json.dumps(result,indent=2)); return 0
    if args.output_dir is None: parser.error('--output-dir required for bounded diagnostic')
    output=args.output_dir.absolute(); reject_symlinks(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error('diagnostic output must be a new or empty directory')
    try:
        try: result['host_headroom_bytes']=require_runtime_memory(MEMORY_MINIMUM[args.mode])
        except RuntimeError as error:
            result.update(status='pending_memory_gate',reason=str(error),backend_imported=False,weights_read=False)
            atomic_json(output/'runtime-evidence.json',result); print(json.dumps(result,indent=2)); return 2
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TF='0',USE_FLAX='0')
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        import torch
        import random
        import numpy as np
        torch.set_num_threads(1); random.seed(13); np.random.seed(13); torch.manual_seed(13)
        torch.use_deterministic_algorithms(True)
        if args.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('requested CUDA unavailable')
        output.mkdir(parents=True,exist_ok=True)
        items=make_adapter(args.checkpoint,args.mode,args.device)
        result['checks']=exercise(*items,args.mode,args.device,output)
        result.update(status='tiny_runtime_verified' if args.mode=='tiny' else 'full_foundation_runtime_verified',
            backend_imported=True,weights_read=args.mode=='full',
            environment={'torch':torch.__version__,'transformers':__import__('transformers').__version__})
        if args.mode=='tiny': result['pending'].append('full_weight_loading_forward_gradients_resume_CUDA')
        else: result['loaded_identity']=items[0].identity()
        atomic_json(output/'runtime-evidence.json',result)
        print(json.dumps({key:value for key,value in result.items() if key!='checks'},indent=2)); return 0
    except Exception as error:
        result.update(status='failed',error_type=type(error).__name__,reason=str(error))
        atomic_json(output/'runtime-evidence.json',result); print(json.dumps(result,indent=2)); return 1


if __name__=='__main__': raise SystemExit(main())
