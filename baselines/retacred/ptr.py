"""Normal PTR: import the pinned forward, preprocessing and optimizer functions.

Only the exact objective block is extracted from the upstream top-level runner.
Its dataset eval(), training, test access and filesystem writes are never run.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import io
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

from .common import ROOT, digest
from .ptr_resources import REVISION, validate_resources
from .updates import Optimizers, Schedulers

SOURCE_FILES = ("src/arguments.py", "src/data_prompt.py", "src/modeling.py",
                "src/optimizing.py", "src/templating.py", "src/run_prompt.py",
                "scripts/run_large_retacred.sh")


@contextmanager
def arguments_binding(module):
    previous = sys.modules.get("arguments")
    sys.modules["arguments"] = module
    try:
        yield
    finally:
        if previous is None:
            sys.modules.pop("arguments", None)
        else:
            sys.modules["arguments"] = previous


def module_from_bytes(path, *, replace=None):
    text = path.read_text()
    if replace:
        before, after = replace
        if text.count(before) != 1:
            raise ValueError("PTR source device-adaptation anchor changed")
        text = text.replace(before, after)
    module = ModuleType("retacred_ptr_" + path.stem)
    exec(compile(text, str(path), "exec"), module.__dict__)
    return module


def source_objective(path):
    """Compile just logits -> five CE + relation CE, not the top-level script."""
    import torch
    tree = ast.parse(path.read_text())
    candidates = [node for node in ast.walk(tree) if isinstance(node, ast.For)
                  and ast.unparse(node.iter) == "enumerate(tqdm(train_dataloader, desc='Iteration'))"]
    if len(candidates) != 1:
        raise ValueError("PTR source objective loop anchor changed")
    body = []
    for node in candidates[0].body:
        body.append(node)
        if (isinstance(node, ast.AugAssign) and ast.unparse(node.target) == "loss"
                and ast.unparse(node.value) == "criterion(final_logits, batch['labels'])"):
            break
    else:
        raise ValueError("PTR relation CE term is absent")
    if (ast.unparse(body[0]) != "logits = model(**batch)"
            or "loss /= len(logits)" not in ast.unparse(ast.Module(body=body, type_ignores=[]))):
        raise ValueError("PTR five-objective block changed")
    function = ast.FunctionDef(name="objective", args=ast.arguments(posonlyargs=[],
        args=[ast.arg(arg=name) for name in ("model", "train_dataset", "batch", "criterion")],
        kwonlyargs=[], kw_defaults=[], defaults=[]), body=body + [ast.parse("return loss, final_logits, logits").body[0]],
        decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {"torch": torch}
    exec(compile(module, str(path) + "#objective-only", "exec"), namespace)
    return namespace["objective"]


class SourceRuntime:
    def __init__(self, source, args):
        self.arguments = module_from_bytes(source/"src/arguments.py")
        self.arguments._GLOBAL_ARGS = args
        # Source templating calls the CLI parser; substitute already frozen args.
        self.arguments.get_args_parser = lambda: args
        from transformers import RobertaModel
        class LocalEagerRoberta:
            @staticmethod
            def from_pretrained(*positional, **kwargs):
                return RobertaModel.from_pretrained(*positional, **kwargs,
                    local_files_only=True, attn_implementation="eager")
        self.arguments._MODEL_CLASSES["roberta"]["model"] = LocalEagerRoberta
        with arguments_binding(self.arguments):
            self.templating = module_from_bytes(source/"src/templating.py")
            self.data = module_from_bytes(source/"src/data_prompt.py", replace=(
                "self.prompt_id_2_label = self.prompt_id_2_label.long().cuda()",
                "self.prompt_id_2_label = self.prompt_id_2_label.long().to(self.device)"))
            self.modeling = module_from_bytes(source/"src/modeling.py")
            self.optimizing = module_from_bytes(source/"src/optimizing.py")
        self.objective = source_objective(source/"src/run_prompt.py")


class PTR:
    amp_enabled = False  # Released source is FP32; no invented AMP protocol.
    protocol = "ptr_source_objective_dual_updates_common_logical_batches_v1"
    optimizer_contract = "source_dual_AdamW_encoder3e-5_extra1e-5_eps1e-6_wd.01_bias_LN_excluded_warmup500"
    precision_contract = "source_fp32_no_autocast"

    def __init__(self, source: Path, checkpoint: Path, *, device: str,
                 checkpointing: bool = False, metadata: Path | None = None, max_seq_length=512):
        import torch
        from transformers import RobertaTokenizer
        self.source, self.device, self.torch = source, torch.device(device), torch
        self.metadata_dir = metadata or ROOT/".external/retacred/ptr_metadata"/REVISION
        resource = validate_resources(self.metadata_dir)
        self.args = SimpleNamespace(model_name_or_path=str(checkpoint), model_type="roberta", cache_dir="",
            data_dir=str(self.metadata_dir), temps="temp.txt", max_seq_length=max_seq_length, new_tokens=5,
            learning_rate=3e-5, learning_rate_for_new_token=1e-5, weight_decay=.01,
            adam_epsilon=1e-6, warmup_steps=500, gradient_accumulation_steps=1, num_train_epochs=5)
        self.runtime = SourceRuntime(source, self.args)
        self.tokenizer = RobertaTokenizer.from_pretrained(checkpoint, local_files_only=True)
        with redirect_stdout(io.StringIO()):
            temps = self.runtime.templating.get_temps(self.tokenizer)
            # Skip the unsafe upstream constructor's eval(line), not its methods.
            self.processor = self.runtime.data.REPromptDataset.__new__(self.runtime.data.REPromptDataset)
            self.processor.rel2id = resource["labels"]
            self.processor.NA_NUM, self.processor.num_class = 0, 40
            self.processor.args, self.processor.temps, self.processor.device = self.args, temps, self.device
            self.processor.get_labels(self.tokenizer)
        structures = {tuple(tuple(ids) for ids in item["mask_ids"]) for item in self.processor.temp_ids.values()}
        if len(structures) != 1 or len(self.processor.prompt_label_idx) != 5:
            raise ValueError("PTR model input depends on target relation or mask count differs")
        self.labels = [name for name, _ in sorted(resource["labels"].items(), key=lambda item: item[1])]
        self.model = self.runtime.modeling.Model(self.args, self.tokenizer,
            [ids.to(self.device) for ids in self.processor.prompt_label_idx]).to(self.device)
        self.config = self.model.model.config
        if checkpointing:
            self.model.model.gradient_checkpointing_enable()
        self.criterion = torch.nn.CrossEntropyLoss()

    def fresh_source_model(self):
        runtime = SourceRuntime(self.source, self.args)
        return runtime.modeling.Model(self.args, self.tokenizer,
            [ids.to(self.device) for ids in self.processor.prompt_label_idx]).to(self.device), runtime

    @staticmethod
    def source_record(item):
        tokens = item["token"]
        return {"token": tokens, "h": {"name": " ".join(tokens[item["subj_start"]:item["subj_end"]+1])},
                "t": {"name": " ".join(tokens[item["obj_start"]:item["obj_end"]+1])}, "relation": item["relation"]}

    def encode(self, records):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            tensors = self.processor.list2tensor([self.source_record(item) for item in records], self.tokenizer)
        if not (tensors["mlm_labels"].ge(0).sum(-1) == 5).all() or tensors["input_flags"].any():
            raise ValueError("PTR requires exactly five mask positions and dormant extra-token flags")
        return [{name: int(value[index]) if name == "labels" else value[index]
                 for name, value in tensors.items()} for index in range(len(records))]

    def batch(self, features):
        return {name: (self.torch.tensor([item[name] for item in features], dtype=self.torch.long)
                      if name == "labels" else self.torch.stack([item[name] for item in features])).to(self.device)
                for name in features[0]}

    def finalize_vocabulary(self):
        if self.model.model.get_input_embeddings().num_embeddings != len(self.tokenizer):
            raise ValueError("PTR tokenizer/embedding vocabulary mismatch; no silent resize")

    def forward(self, batch):
        loss, logits, _ = self.runtime.objective(self.model, self.processor, batch, self.criterion)
        return loss, logits

    def make_updates(self, groups_per_epoch, epochs, *, model=None, runtime=None, warmup=None):
        runtime = runtime or self.runtime
        args = runtime.arguments._GLOBAL_ARGS
        old = (args.num_train_epochs, args.warmup_steps)
        args.num_train_epochs = epochs
        if warmup is not None:
            args.warmup_steps = warmup  # Explicit tiny-gate override, never used by formal trainer.
        try:
            optimizer, scheduler, extra, extra_scheduler = runtime.optimizing.get_optimizer(
                model if model is not None else self.model, range(groups_per_epoch))
        finally:
            args.num_train_epochs, args.warmup_steps = old
        return Optimizers(encoder=optimizer, extra=extra), Schedulers(encoder=scheduler, extra=extra_scheduler)

    def source_manifest(self):
        return {name: digest(self.source/name) for name in SOURCE_FILES}

    def trace_fields(self, batch, row):
        return {"prompt_mask_positions": batch["mlm_labels"][row].ge(0).nonzero().flatten().tolist(),
                "relation_verbalizer_indices": self.processor.prompt_id_2_label.detach().cpu().tolist(),
                "verbalizer_token_ids": [ids.tolist() for ids in self.processor.prompt_label_idx],
                "loss_scope": "actual_physical_batch_mean_five_prompt_CE_plus_relation_CE"}

    def parameter_trace(self):
        # Tied verbalizer rows are the source readout parameters, not a new head.
        result = {}
        weight = self.model.model.embeddings.word_embeddings.weight
        if weight.grad is not None:
            ids = self.torch.unique(self.torch.cat(self.processor.prompt_label_idx)).to(self.device)
            result["verbalizer.parameter_token_ids"] = ids.cpu().clone()
            result["verbalizer.parameter_gradient"] = weight.grad[ids].detach().cpu().clone()
        for name, parameter in self.model.named_parameters():
            if (name.startswith("mlp.") or name.startswith("extra_token_embeddings.")) and parameter.grad is not None:
                result["extra.parameter_gradient." + name] = parameter.grad.detach().cpu().clone()
        return result
