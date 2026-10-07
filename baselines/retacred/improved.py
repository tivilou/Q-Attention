"""Thin adapter: marker preprocessing and forward are the pinned upstream code."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

from .common import digest


def load_module(source: Path, name: str):
    spec = importlib.util.spec_from_file_location("retacred_upstream_" + name, source / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    # Do not create untracked files in upstream or trust stale .pyc caches.
    exec(compile((source / (name + ".py")).read_bytes(), str(source / (name + ".py")), "exec"), module.__dict__)
    return module


class Improved:
    amp_enabled = True
    protocol = "common_adapted_v1"
    optimizer_contract = "transformers_AdamW_correct_bias_true_weight_decay_0_lr3e-5_eps1e-6"
    precision_contract = "upstream_cuda_autocast_fp16_gradscaler"
    def __init__(self, source: Path, checkpoint: Path, input_format: str, *, device: str, checkpointing: bool = False):
        import torch
        from transformers import AutoConfig, AutoTokenizer
        self.torch = torch
        self.device = torch.device(device)
        self.args = SimpleNamespace(model_name_or_path=str(checkpoint), input_format=input_format,
                                    max_seq_length=512, dropout_prob=0.1, num_class=40)
        self.source = source
        self.config = AutoConfig.from_pretrained(checkpoint, local_files_only=True, num_labels=40)
        # Upstream was written before SDPA; eager attention also exposes actual
        # score/probability intermediates to the bounded trace recorder.
        self.config._attn_implementation = "eager"
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        self.prepro = load_module(source, "prepro")
        self.processor = self.prepro.RETACREDProcessor(self.args, self.tokenizer)
        self.labels = [label for label, _ in sorted(self.processor.LABEL_TO_ID.items(), key=lambda item: item[1])]
        self.upstream = load_module(source, "model")
        self.model = self.upstream.REModel(self.args, self.config)
        if checkpointing:
            self.model.encoder.gradient_checkpointing_enable()
        self.model.to(self.device)
        self.collate = load_module(source, "utils").collate_fn

    def encode(self, records: list[dict]) -> list[dict]:
        features = []
        for item in records:
            tokens = [self.prepro.convert_token(token) for token in item["token"]]
            ids, subject, object_ = self.processor.tokenize(tokens, item["subj_type"], item["obj_type"],
                         item["subj_start"], item["subj_end"], item["obj_start"], item["obj_end"])
            if not 0 <= subject < len(ids) or not 0 <= object_ < len(ids):
                raise ValueError("upstream truncated an entity marker; refusing to silently change representation")
            features.append({"input_ids": ids, "labels": self.processor.LABEL_TO_ID[item["relation"]],
                             "ss": subject, "os": object_})
        return features

    def batch(self, features: list[dict]) -> dict:
        self.finalize_vocabulary()
        ids, mask, labels, subject, object_ = self.collate(features)
        return {"input_ids": ids.to(self.device), "attention_mask": mask.to(self.device),
                "labels": labels.to(self.device), "ss": subject.to(self.device), "os": object_.to(self.device)}

    def finalize_vocabulary(self):
        # Typed special markers are added during tokenization, not constructor
        # setup. Resize after all splits have been encoded, before optimizer.
        if self.model.encoder.get_input_embeddings().num_embeddings != len(self.tokenizer):
            self.model.encoder.resize_token_embeddings(len(self.tokenizer))

    def forward(self, batch: dict):
        # No alternative pooling, analytic rewrite or classifier replacement.
        return self.model(**batch)

    def source_manifest(self) -> dict:
        return {name: digest(self.source / name) for name in ("prepro.py", "model.py", "utils.py", "train_retacred.py")}

    def trace_fields(self, batch, row):
        return {"entity_marker_positions": {"subject": int(batch["ss"][row]), "object": int(batch["os"][row])}}

    def parameter_trace(self):
        return {"classifier.parameter_gradient." + name: parameter.grad.detach().cpu().clone()
                for name, parameter in self.model.classifier.named_parameters() if parameter.grad is not None}


def leaves(value, prefix=""):
    import torch
    if torch.is_tensor(value):
        yield prefix or "tensor", value
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from leaves(item, f"{prefix}.{index}".strip("."))
    elif isinstance(value, dict):
        for name, item in value.items():
            yield from leaves(item, f"{prefix}.{name}".strip("."))


class SampleCapture:
    """Detached values from the actual batch, including leaf I/O and backward gradients.

    Nothing is obtained by a second replay or post-training reconstruction.
    Hook registration must not change RNG, model mode or model outputs.
    """
    def __init__(self, model, row: int, batch_size: int):
        self.model, self.row, self.batch_size = model, row, batch_size
        self.values, self.nodes, self.handles, self.counts = {}, [], [], {}

    def select(self, tensor):
        value = tensor.detach()
        if value.ndim and value.shape[0] == self.batch_size:
            value = value[self.row:self.row+1]
        return value.cpu().clone()

    def save(self, name: str, tensor):
        self.values[name] = self.select(tensor)
        return name

    def __enter__(self):
        from torch.utils._python_dispatch import TorchDispatchMode
        capture = self
        class ActualSoftmaxObserver(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                result = func(*args, **(kwargs or {}))
                verbalizers = getattr(capture.model, "prompt_label_idx", None)
                if (str(func) == "aten.where.self" and verbalizers is not None
                        and args[2].ndim == 3 and "ptr_embedding_selection" not in capture.counts):
                    capture.counts["ptr_embedding_selection"] = 1
                    key = "functional.ptr_embedding_selection"
                    capture.nodes.append({"id": key, "module": "actual_zero_flag_raw_embedding_selection",
                        "inputs": [capture.save(key+".flags", args[0]), capture.save(key+".extra", args[1]),
                                   capture.save(key+".raw", args[2])],
                        "outputs": [capture.save(key+".selected", result)]})
                position = capture.counts.get("ptr_projection", 0)
                if (str(func) == "aten.mm.default" and verbalizers is not None and position < len(verbalizers)
                        and args[0].shape == (capture.batch_size, capture.model.model.config.hidden_size)
                        and args[1].shape == (capture.model.model.config.hidden_size, len(verbalizers[position]))):
                    capture.counts["ptr_projection"] = position + 1
                    key = f"functional.ptr_verbalizer_projection.position{position}"
                    capture.nodes.append({"id": key, "module": "actual_mask_hidden_to_tied_verbalizer_logits",
                        "inputs": [capture.save(key+".hidden", args[0]),
                                   capture.save(key+".decoder", args[1])],
                        "outputs": [capture.save(key+".logits", result)]})
                classifier = getattr(capture.model, "classifier", None)
                if (str(func) == "aten.cat.default" and classifier is not None
                        and len(args[0]) == 2 and result.ndim == 2
                        and all(value.ndim == 2 and value.shape[-1]*2 == classifier[0].in_features for value in args[0])
                        and result.shape[-1] == classifier[0].in_features
                        and "entity_readout" not in capture.counts):
                    # Observe the exact source operands: never recompute entity
                    # pooling from a stored encoder state after the event.
                    capture.counts["entity_readout"] = 1
                    key = "functional.entity_readout"
                    capture.nodes.append({"id":key, "module":"actual_subject_object_concatenation",
                        "inputs":[capture.save(key+".subject",args[0][0]), capture.save(key+".object",args[0][1])],
                        "outputs":[capture.save(key+".concatenated",result)]})
                if str(func) in ("aten._softmax.default", "aten._log_softmax.default"):
                    kind = "attention_scores_to_probabilities" if args[0].ndim == 4 else "classification_log_softmax"
                    count = capture.counts.get(kind, 0)
                    capture.counts[kind] = count + 1
                    key = f"functional.{kind}.call{count}"
                    node = {"id":key, "module":"actual_"+str(func),
                            "inputs":[capture.save(key+".input", args[0])],
                            "outputs":[capture.save(key+".output", result)]}
                    capture.nodes.append(node)
                return result
        self.dispatch = ActualSoftmaxObserver()
        self.dispatch.__enter__()
        def hook(name):
            def record(_module, inputs, output):
                count = self.counts.get(name, 0)
                self.counts[name] = count + 1
                key = f"{name}.call{count}"
                node = {"id": key, "module": name, "inputs": [], "outputs": []}
                for label, tensor in leaves(inputs):
                    node["inputs"].append(self.save(key + ".input." + label, tensor))
                for label, tensor in leaves(output):
                    tensor_key = key + ".output." + label
                    node["outputs"].append(self.save(tensor_key, tensor))
                    if tensor.requires_grad:
                        def gradient_hook(grad, target=tensor_key):
                            self.save(target + ".gradient", grad)
                            return None  # never replace or modify the training gradient
                        tensor.register_hook(gradient_hook)
                self.nodes.append(node)
            return record
        for name, module in self.model.named_modules():
            if not name or not list(module.children()) or name in (
                    "encoder.embeddings", "encoder", "classifier", "model.embeddings", "model", "model.encoder"):
                self.handles.append(module.register_forward_hook(hook(name or "source_model")))
        return self

    def __exit__(self, *exception):
        for handle in self.handles:
            handle.remove()
        self.dispatch.__exit__(*exception)

    def metadata(self) -> dict:
        return {"nodes": self.nodes, "tensors": {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                 "numel": value.numel(), "value_storage": "complete_numeric_tensor"}
                 for key, value in self.values.items()},
                "gradient_semantics": "selected sample activations; parameter gradients, when present, aggregate the logical batch"}
