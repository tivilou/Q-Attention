"""Paper-derived MNLI neural foundation; formal runner admission stays closed."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

from .assets import reject_symlinks
from .common import fingerprint
from .luke import require_runtime_memory
from .metaentail import FeasibleTypes, grouped_prediction, inference_group, training_group
from .metaentail_checkpoint import checkpoint_spec, config_contract, encode_pair
from .mnli_assets import verify_checkpoint

HEAD_SHAPES = {'classifier.dense.weight':[1024,1024], 'classifier.dense.bias':[1024],
               'classifier.out_proj.weight':[3,1024], 'classifier.out_proj.bias':[3],
               'roberta.embeddings.word_embeddings.weight':[50265,1024],
               'roberta.embeddings.position_embeddings.weight':[514,1024]}


UNUSED_POOLER_SHAPES = {'roberta.pooler.dense.weight': [1024, 1024],
                        'roberta.pooler.dense.bias': [1024]}

def inspect_safetensors(path):
    """Bounded header/selected tensor byte hashes; no torch or numeric inference.

    Caller must verify the complete pinned file before trusting this diagnostic.
    """
    path=Path(path); reject_symlinks(path)
    with path.open('rb') as handle:
        raw=handle.read(8)
        if len(raw)!=8: raise ValueError('safetensors header missing')
        size=struct.unpack('<Q',raw)[0]
        if not 2<=size<=1024*1024 or size+8>=path.stat().st_size:
            raise ValueError('safetensors header size invalid')
        header=json.loads(handle.read(size))
        ranges=[]; tensor_hashes={}; summary={}; total=path.stat().st_size-size-8
        dtype_bytes={'F32':4,'I64':8,'I32':4,'F16':2,'BF16':2,'BOOL':1}
        for name,row in header.items():
            if name=='__metadata__': continue
            shape=row.get('shape'); offsets=row.get('data_offsets'); dtype=row.get('dtype')
            if (not isinstance(shape,list) or any(type(v) is not int or v<0 for v in shape)
                    or not isinstance(offsets,list) or len(offsets)!=2
                    or any(type(v) is not int for v in offsets) or dtype not in dtype_bytes):
                raise ValueError('invalid safetensors tensor header')
            start,end=offsets; count=1
            for dim in shape: count*=dim
            if not 0<=start<=end<=total or end-start!=count*dtype_bytes[dtype]:
                raise ValueError('safetensors tensor extent/shape mismatch')
            ranges.append((start,end)); summary[name]={'shape':shape,'dtype':dtype}
        ordered=sorted(ranges)
        if (not ordered or ordered[0][0]!=0 or ordered[-1][1]!=total
                or any(a[1]!=b[0] for a,b in zip(ordered,ordered[1:]))):
            raise ValueError('safetensors overlaps/holes invalid')
        # The pinned official checkpoint stores a pooler that this classifier
        # never instantiates or calls. Validate its bytes before permitting
        # this exact pair in Transformers' unexpected-key report.
        present = set(UNUSED_POOLER_SHAPES).intersection(summary)
        if present and present != set(UNUSED_POOLER_SHAPES):
            raise ValueError('incomplete official unused pooler pair')
        selected = dict(HEAD_SHAPES)
        if present: selected.update(UNUSED_POOLER_SHAPES)
        for name,expected in selected.items():
            if summary.get(name)!={'shape':expected,'dtype':'F32'}:
                raise ValueError('pretrained MNLI head/embedding shape mismatch: '+name)
            start,end=header[name]['data_offsets']; handle.seek(8+size+start)
            value=hashlib.sha256(); remaining=end-start
            while remaining:
                block=handle.read(min(1024*1024,remaining))
                if not block: raise ValueError('safetensors truncated tensor')
                value.update(block); remaining-=len(block)
            tensor_hashes[name]=value.hexdigest()
    return {'status':'safetensors_structure_verified','header_bytes':size,'tensor_count':len(summary),
            'selected_tensors':{name:dict(summary[name],sha256=tensor_hashes[name]) for name in HEAD_SHAPES},
            'verified_unused_pooler': {name:dict(summary[name],sha256=tensor_hashes[name])
                                      for name in present},
            'neural_runtime_verified':False}


def loading_keys_checked(info, *, verified_unused_pooler=None):
    keys=('missing_keys','unexpected_keys','mismatched_keys','error_msgs')
    if not isinstance(info,dict) or any(key not in info or not isinstance(info[key],list) for key in keys):
        raise ValueError('explicit pretrained loading report required')
    if any(info[key] for key in ('missing_keys','mismatched_keys','error_msgs')):
        raise ValueError('MNLI pretrained loading keys must match exactly')
    unexpected=info['unexpected_keys']
    if unexpected:
        if len(unexpected)!=2 or set(unexpected)!=set(UNUSED_POOLER_SHAPES):
            raise ValueError('MNLI unexpected keys outside verified unused pooler')
        if not isinstance(verified_unused_pooler,dict) or set(verified_unused_pooler)!=set(UNUSED_POOLER_SHAPES):
            raise ValueError('unused pooler header and byte verification required')
        for name,shape in UNUSED_POOLER_SHAPES.items():
            row=verified_unused_pooler[name]
            if (not isinstance(row,dict) or row.get('shape')!=shape or row.get('dtype')!='F32'
                    or not isinstance(row.get('sha256'),str) or len(row['sha256'])!=64
                    or any(c not in '0123456789abcdef' for c in row['sha256'])):
                raise ValueError('unused pooler verified shape/dtype/hash mismatch')
    return {'ignored_unused_pooler': sorted(unexpected), 'active_parameters_exact':True}


def encoded_group(record, feasibility, tokenizer, nli_order, *, split, max_length):
    if split not in ('train','dev','test'): raise ValueError('explicit train/dev/test split required')
    # Verify supplied checkpoint order without assuming a fixed target column.
    from .metaentail import nli_order_checked
    order=nli_order_checked(nli_order)
    group=training_group(record,feasibility) if split=='train' else inference_group(record,feasibility)
    rows=[]
    for index,candidate in enumerate(group['candidates']):
        feature=encode_pair(group['premise'],candidate['hypothesis'],tokenizer,max_length=max_length)
        feature.update(sample_id=group['sample_id'],candidate_relation=candidate['relation'],candidate_index=index,
                       filter_identity=group['filter_identity'],split=split)
        if split=='train': feature['labels']=order.index(candidate['nli_target'])
        rows.append(feature)
    return {'group':group,'features':rows,'split':split,'nli_order':list(order),
            'filter_identity':group['filter_identity']}


def collate_pairs(features):
    """Unpadded input features -> pad1/attention0; labels only for all-train rows."""
    if not features: raise ValueError('empty pair batch')
    train=features[0].get('split')=='train'
    if any((row.get('split')=='train')!=train or ('labels' in row)!=train for row in features):
        raise ValueError('mixed train/evaluation labels in pair batch')
    filters={row['filter_identity'] for row in features}
    if len(filters)!=1: raise ValueError('mixed fitted filters in pair batch')
    width=max(len(row['input_ids']) for row in features)
    if not 4<width<=512: raise ValueError('pair batch exceeds MNLI capacity')
    batch={'input_ids':[],'attention_mask':[]}
    if train: batch['labels']=[]
    for row in features:
        ids=row['input_ids']; mask=row['attention_mask']
        if (len(ids)!=len(mask) or any(value!=1 for value in mask)
                or any(type(value) is not int or not 0<=value<50265 for value in ids)
                or ids[0]!=0 or ids[-1]!=2):
            raise ValueError('malformed unpadded MNLI pair')
        batch['input_ids'].append(ids+[1]*(width-len(ids)))
        batch['attention_mask'].append(mask+[0]*(width-len(ids)))
        if train:
            if type(row['labels']) is not int or row['labels'] not in (0,1,2):
                raise ValueError('NLI training label outside three-class head')
            batch['labels'].append(row['labels'])
    return batch


class MetaEntailFoundation:
    protocol='metaentail_mnli_foundation_not_formally_admitted'
    amp_enabled=False

    def __init__(self, checkpoint, *, device='cpu', checkpointing=False):
        # Reuse the unchanged conservative12GiB gate before backend/weight reads.
        try: require_runtime_memory()
        except RuntimeError as error:
            raise RuntimeError('MNLI neural runtime pending at pre-import memory gate: '+str(error)) from None
        self.checkpoint=Path(checkpoint)
        self.asset_evidence=verify_checkpoint(self.checkpoint)
        self.tensor_evidence=inspect_safetensors(self.checkpoint/'model.safetensors')
        config=json.loads((self.checkpoint/'config.json').read_text())
        tokenizer_config=json.loads((self.checkpoint/'tokenizer_config.json').read_text())
        self.binding=config_contract(config,tokenizer_config)
        self.nli_order=tuple(self.binding['nli_order'])
        import torch
        from safetensors import safe_open
        from transformers import RobertaForSequenceClassification, RobertaTokenizerFast
        self.torch,self.device=torch,torch.device(device)
        self.tokenizer=RobertaTokenizerFast.from_pretrained(self.checkpoint,local_files_only=True,add_prefix_space=False)
        self.model,self.loading_info=RobertaForSequenceClassification.from_pretrained(
            self.checkpoint,local_files_only=True,output_loading_info=True,use_safetensors=True,
            ignore_mismatched_sizes=False,attn_implementation='eager')
        self.loading_adaptation=loading_keys_checked(self.loading_info,
            verified_unused_pooler=self.tensor_evidence['verified_unused_pooler'])
        actual=self.model.config
        if (actual.num_labels!=3 or actual.hidden_size!=1024 or actual.vocab_size!=50265
                or {str(k):v for k,v in actual.id2label.items()}!=config['id2label']
                or actual.label2id!=config['label2id']):
            raise ValueError('loaded MNLI head/config drift')
        self.loaded_tensor_equality={}
        state=self.model.state_dict()
        with safe_open(str(self.checkpoint/'model.safetensors'),framework='pt',device='cpu') as saved:
            for name in HEAD_SHAPES:
                source=saved.get_tensor(name)
                parameter=state[name].detach().cpu()
                if parameter.dtype!=torch.float32 or not torch.equal(parameter,source):
                    raise ValueError('loaded pretrained tensor mismatch: '+name)
                self.loaded_tensor_equality[name]=True
        if checkpointing: self.model.gradient_checkpointing_enable()
        self.model.to(self.device)

    def encode(self, record, feasibility, *, split, max_length=512):
        return encoded_group(record,feasibility,self.tokenizer,self.nli_order,split=split,max_length=max_length)

    def batch(self, features):
        return {key:self.torch.tensor(value,dtype=self.torch.long,device=self.device)
                for key,value in collate_pairs(features).items()}

    def forward(self, batch, *, trace=False):
        output=self.model(**batch,return_dict=True,output_hidden_states=trace,output_attentions=trace)
        return output.loss,output.logits,output

    def capture(self, *, row, batch_size):
        # Observes real module I/O/softmax/activation gradients in the current
        # forward/backward event. It never performs a second replay.
        from .improved import SampleCapture
        return SampleCapture(self.model,row=row,batch_size=batch_size)

    def parameter_trace(self):
        return {'classifier.parameter_gradient.'+name:parameter.grad.detach().cpu().clone()
                for name,parameter in self.model.classifier.named_parameters() if parameter.grad is not None}

    def decide(self, encoded, probabilities):
        if encoded['nli_order']!=list(self.nli_order): raise ValueError('encoded group NLI order changed')
        return grouped_prediction(encoded['group'],probabilities,nli_order=self.nli_order)

    def identity(self):
        return {'checkpoint_sha256':self.asset_evidence['files'],'nli_order':list(self.nli_order),
                'loaded_tensor_equality':self.loaded_tensor_equality,'protocol':self.protocol,
                'loading_adaptation':self.loading_adaptation,
                'formal_admission':False,'identity_sha256':fingerprint(self.asset_evidence['files'])}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--inspect-only',action='store_true')
    args=parser.parse_args(argv)
    if args.inspect_only:
        result={'assets':verify_checkpoint(args.checkpoint),
                'structure':inspect_safetensors(args.checkpoint/'model.safetensors'),'formal_admission':False}
        print(json.dumps(result,indent=2))
    else:
        try: instance=MetaEntailFoundation(args.checkpoint)
        except RuntimeError as error:
            if 'MNLI neural runtime pending at pre-import memory gate:' not in str(error): raise
            print(json.dumps({'status':'pending_memory','detail':str(error),
                              'neural_runtime_verified':False,'formal_admission':False,'model_jobs':0}))
            return 3
        print(json.dumps(instance.identity(),indent=2))


if __name__=='__main__': raise SystemExit(main())
