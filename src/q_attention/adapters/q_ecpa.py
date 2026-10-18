"""Scoped, single-layer toy adapter; leaves the stable baseline untouched."""
from __future__ import annotations

import torch
from torch import nn
from q_attention.models.relation_transformer import AttentionInterventionPassThrough
from q_attention.plugins.q_ecpa import EntityCoherentPath


class _Intervention(AttentionInterventionPassThrough):
    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel
        self.inputs = None
        self.capture = False
        self.state = {}

    def after_scores(self, scores, query, key, value, *, attention_mask=None, layer_index=0):
        if self.inputs is None:
            raise RuntimeError("ECPA requires scoped subject/object input; use ECPAToyModel")
        subject, obj = self.inputs
        adjusted, state = self.kernel(query, scores, attention_mask, subject, obj,
                                      capture=self.capture)
        if self.capture:
            masked = adjusted.masked_fill(~attention_mask[:,None,None,:].bool(),
                                          torch.finfo(adjusted.dtype).min)
            state.update(query=query, key=key, value=value, base_scores=scores,
                         adjusted_scores=adjusted, attention=torch.softmax(masked, -1))
            self.state.update(state)
        return adjusted, value


class ECPAToyModel(nn.Module):
    """One encoder layer, dropout=0; rejects untested parallel configurations.

    Forward hooks only observe tensors. All actual attention and classification
    computations still run through RelationExtractionModel's original forward.
    """
    def __init__(self, base, kernel: EntityCoherentPath):
        super().__init__()
        if base.config.num_layers != 1 or base.config.dropout != 0 or base.model_parallel_enabled:
            raise ValueError("toy adapter requires one layer, dropout=0, no model parallelism")
        attention = base.encoder.layers[0].attn
        if type(attention.attention_intervention) is not AttentionInterventionPassThrough:
            raise ValueError("ECPA cannot silently replace an existing intervention")
        attention.attention_intervention = _Intervention(kernel)
        self.base = base
        self.last_capture = {}

    @property
    def intervention(self):
        return self.base.encoder.layers[0].attn.attention_intervention

    def forward(self, input_ids, attention_mask, subject_mask, object_mask, *, capture=False):
        hook = self.intervention
        if hook.inputs is not None:
            raise RuntimeError("concurrent or recursive use of scoped toy adapter")
        hook.inputs, hook.capture, hook.state = (subject_mask, object_mask), capture, {}
        self.last_capture = {}
        handles = []
        if capture:
            def observe(name):
                def save(_module, _inputs, output):
                    hook.state[name] = output
                return save
            def observe_input(name):
                def save(_module, inputs):
                    hook.state[name] = inputs[0]
                return save
            encoder, layer = self.base.encoder, self.base.encoder.layers[0]
            for module, name in ((encoder.token_embedding,"token_embedding"),
                                 (encoder.position_embedding,"position_embedding"),
                                 (encoder.dropout,"embedding"), (layer.attn,"attention_output"),
                                 (layer.attn_norm,"attention_hidden"), (layer.ffn,"ffn"),
                                 (layer,"hidden"), (self.base.classifier[1],"classifier_hidden")):
                handles.append(module.register_forward_hook(observe(name)))
            handles.append(layer.attn.out_proj.register_forward_pre_hook(observe_input("context")))
            handles.append(self.base.classifier.register_forward_pre_hook(observe_input("pooled")))
        try:
            logits = self.base(input_ids, attention_mask, subject_mask, object_mask)
            if capture:
                hook.state.update(input_ids=input_ids, attention_mask=attention_mask,
                                  subject_mask=subject_mask, object_mask=object_mask,
                                  logits=logits, probabilities=logits.softmax(-1))
                self.last_capture = {name: value.detach().cpu().clone()
                                     for name, value in hook.state.items()}
            return logits
        finally:
            for handle in handles:
                handle.remove()
            hook.inputs, hook.capture, hook.state = None, False, {}
