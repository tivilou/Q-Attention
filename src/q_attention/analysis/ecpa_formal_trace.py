"""Private full-value Re-TACRED traces; no raw samples enter public reports."""
from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import torch

from q_attention.analysis.ecpa_trace import BASE, GRAPH, display, read_values
from q_attention.experiments.ecpa_protocol import atomic_json, read, sha

COMMON = ("input_ids","attention_mask","subject_mask","object_mask","token_embedding",
          "position_embedding","embedding","pooled","classifier_hidden","logits","probabilities","loss")
LAYER = ("query","key","value","base_scores","adjusted_scores","attention","context",
         "attention_output","attention_hidden","ffn","hidden")


def inventory(mode, layers, heads):
    result = {name:BASE[name] for name in COMMON}
    result["logits"] = ("generation","Relation scores","Classifier scores in the frozen training label-map order.",["relation"])
    for index in range(layers):
        for name in LAYER:
            stage,title,meaning,axes = BASE[name]
            result[f"layer{index}_{name}"] = (f"layer{index}_{stage}",f"Layer {index+1}: {title}",meaning,axes)
    if mode not in ("baseline","disabled","original_qk"):
        result.update({f"layer0_{name}":("layer0_scoring",title,meaning,axes) for name,(title,meaning,axes) in GRAPH.items()})
    if mode == "original_qk":
        for index in range(layers):
            for name in ("qk_raw","qk_residual"):
                result[f"layer{index}_{name}"] = (f"layer{index}_scoring",name,
                    "Actual original-QK pair observable or context-centered bounded score residual.",["head","query","key"])
            for head in range(heads):
                for name,axes in (("angles",["query-key pair","query/key angle"]),("states",["query-key pair","circuit step","basis state"])):
                    result[f"layer{index}_qk_head{head}_{name}"] = (f"layer{index}_scoring",f"Layer {index+1}, head {head+1}: QK {name}",
                        "Actual angles or complex state after preparation, controlled phase, and local rotations; pair order is query-major.",axes)
    result["parameter_gradient"] = ("training","Trainable parameter gradients",
        "Full gradient vector in named-parameter order. Real training: whole logical minibatch; replay: this sample only; empty when no parameter is trainable.",["parameter"])
    return result


def state_identity(model):
    import hashlib
    digest = hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def gradients(model):
    values = [p.grad.detach().reshape(-1).cpu() if p.grad is not None else torch.zeros(p.numel())
              for p in model.parameters() if p.requires_grad]
    return torch.cat(values) if values else torch.empty(0)


class FormalTrace:
    def __init__(self, root, mode, config, contract, labels):
        self.root,self.mode,self.config,self.labels = Path(root),mode,config,labels
        self.fields = inventory(mode,config["model"]["num_layers"],config["model"]["num_heads"])
        path = self.root/"sample_trace.json"
        experiment = dict(experiment_id=config["name"],run_id=contract["run_id"],model_identity=mode,
            code_revision=contract["git_revision"],config_sha256=contract["fingerprint"],
            dataset="Re-TACRED" if config["profile"]=="formal_seed13" else "public synthetic canary",dataset_identity=contract["data"],seed=config["seed"])
        if path.exists():
            self.trace = read(path)
            if self.trace["experiment"] != experiment:
                raise ValueError("trace resume identity mismatch")
        else:
            self.trace = dict(schema_version="sample-trace.v2",trace_id=f"ecpa-{mode}",trace_completeness="partial",
                experiment=experiment,sample_selection=dict(rule="First three record indexes in every split, frozen before results; first train record at actual update 0 when trainable.",
                population_scope=config["profile"],selected_count=0),coverage=dict(data="observed",preprocess="observed",scoring="observed",context="observed",generation="observed",
                evaluation="observed",training="observed" if mode not in ("disabled","random_phase") else "not_applicable",retrieval="not_applicable",selection="not_applicable",diagnosis="not_applicable"),
                task_contract=dict(task_type="relation classification",required_roles=["sentence","tokens","subject","object","correct_relation","predicted_relation","checkpoint_identity"],
                    required_representations=list(self.fields),required_value_representations=list(self.fields),required_splits=["train","valid","test"],
                    required_checkpoints=["initial","best","final"],required_lanes=["replay"],
                    checkpoint_policy="Three frozen cases per split at initial/best/final; actual update-0 train lane only for optimized jobs. No target reaches intervention."),samples=[])

    def add(self, row, split, index, checkpoint, lane, values, update, model_sha):
        if set(values) != set(self.fields):
            raise ValueError(f"numeric capture differs from declared inventory: {set(values)^set(self.fields)}")
        sample_id = f"{split}-{index:04d}:{checkpoint}:{lane}"
        reps, outputs = [], {}
        for name,(stage,title,meaning,axes) in self.fields.items():
            array = values[name].detach().cpu().numpy()
            if array.dtype.hasobject or not np.isfinite(array).all():
                raise ValueError(f"nonfinite/object case tensor: {name}")
            relative = f"values/{sample_id.replace(':','_')}/{name}.npy"
            path = self.root/relative
            path.parent.mkdir(parents=True,exist_ok=True)
            temporary = path.with_suffix(".npy.tmp")
            with temporary.open("wb") as stream:
                np.save(stream,array,allow_pickle=False)
            temporary.replace(path)
            ref = dict(path=relative,sha256=sha(path),bytes=path.stat().st_size,visibility="exchange_only")
            reps.append(dict(id=name,stage_id=stage,status="observed",meaning=meaning,shape=list(array.shape),dtype=str(array.dtype),axis_semantics=axes,
                artifact_refs=[ref],display=display(title,meaning,name),value_capture=dict(status="complete",shape=list(array.shape),logical_element_count=array.size,
                stored_element_count=array.size,artifact_refs=[ref],reader_projection=dict(status="available",kind="coordinate_inspector",coordinate_access="arbitrary",
                reader_ref=f"q_attention.analysis.ecpa_trace:read_values#{sample_id}/{name}",source_sha256=[ref["sha256"]]))))
            outputs.setdefault(stage,[]).append(name)
        id_to_label = {v:k for k,v in self.labels.items()}
        def entity(span):
            return dict(text=" ".join(row.tokens[span[0]:span[1]]),span=list(span))
        roles = dict(sentence=" ".join(row.tokens),tokens=list(row.tokens),subject=entity(row.subject),object=entity(row.object),
            correct_relation=row.label,predicted_relation=id_to_label[int(values["logits"].argmax())],
            checkpoint_identity=dict(optimizer_updates_completed=update,model_state_sha256=model_sha,capture="actual update before optimizer.step" if lane=="training" else "checkpoint replay; no update"))
        lineage = {"data":["source:sentence","source:entity_spans"],"preprocess":["tokens","entity_spans","external:model_checkpoint"],
            "generation":[f"layer{self.config['model']['num_layers']-1}_hidden","subject_mask","object_mask","attention_mask","external:model_checkpoint"],
            "evaluation":["logits","source:correct_relation"],"training":["loss","external:model_checkpoint"]}
        for layer in range(self.config["model"]["num_layers"]):
            prior = "embedding" if layer==0 else f"layer{layer-1}_hidden"
            lineage[f"layer{layer}_scoring"] = [prior,"attention_mask","subject_mask","object_mask","external:model_checkpoint"]
            lineage[f"layer{layer}_context"] = [prior,f"layer{layer}_adjusted_scores",f"layer{layer}_value","attention_mask","external:model_checkpoint"]
        stages = []
        names = ["data","preprocess"]+[name for layer in range(self.config["model"]["num_layers"]) for name in (f"layer{layer}_scoring",f"layer{layer}_context")]+["generation","evaluation","training"]
        for stage in names:
            canonical = stage.split("_",1)[1] if stage.startswith("layer") else stage
            purpose = f"{stage}: consume the referenced actual inputs and emit observed values for this sample."
            stages.append(dict(stage_id=stage,stage=canonical,canonical_stage=canonical,lane=lane,status="observed",purpose=purpose,
                input_refs=lineage[stage],output_refs=["sentence","tokens","entity_spans"] if stage=="data" else outputs.get(stage,[]),
                checkpoint=model_sha,target_access=("training_only" if stage=="training" else "outside_generation_input" if stage=="evaluation" else "post_generation_only" if stage=="generation" else "not_available"),display=display(stage,purpose)))
        sample = dict(sample_id=sample_id,lane=lane,split=split,checkpoint=checkpoint,roles=roles,representations=reps,stages=stages)
        self.trace["samples"] = [s for s in self.trace["samples"] if s["sample_id"] != sample_id]+[sample]
        self.save()

    def save(self):
        self.trace["sample_selection"].update(selected_count=len(self.trace["samples"]),selected_sample_ids=[s["sample_id"] for s in self.trace["samples"]])
        atomic_json(self.root/"sample_trace.json",self.trace)

    def complete(self):
        self.trace["trace_completeness"] = "complete"
        self.save()
        report = validate(self.root/"sample_trace.json",self.mode,self.config)
        atomic_json(self.root/"trace_validation.json",report)
        return report


def validate(path, mode, config):
    path = Path(path)
    trace = read(path)
    from q_attention.analysis.ecpa_contract_validator import validate as validate_contract
    errors,warnings = validate_contract(trace,require_complete=True)
    if errors or warnings:
        raise ValueError(f"sample-trace.v2 structural gate failed: {(errors+warnings)[:12]}")
    expected = set(inventory(mode,config["model"]["num_layers"],config["model"]["num_heads"]))
    if trace["schema_version"] != "sample-trace.v2" or set(trace["task_contract"]["required_value_representations"]) != expected:
        raise ValueError("incomplete formal trace contract")
    seen = set()
    for sample in trace["samples"]:
        seen.add(sample["sample_id"])
        reps = {r["id"]:r for r in sample["representations"]}
        if set(reps) != expected or len(reps) != len(sample["representations"]):
            raise ValueError("missing or duplicate complete value")
        if any(s["checkpoint"] != sample["roles"]["checkpoint_identity"]["model_state_sha256"] for s in sample["stages"]):
            raise ValueError("checkpoint lineage mismatch")
        arrays = {}
        for name,rep in reps.items():
            capture = rep["value_capture"]
            ref = capture["artifact_refs"][0]
            source = (path.parent/ref["path"]).resolve()
            if not source.is_relative_to(path.parent.resolve()) or sha(source) != ref["sha256"]:
                raise ValueError("unsafe or corrupted numeric artifact")
            array = np.load(source,allow_pickle=False)
            if list(array.shape) != rep["shape"] or str(array.dtype) != rep["dtype"] or not np.isfinite(array).all():
                raise ValueError("bad numeric shape/dtype/value")
            if (array.size != capture["stored_element_count"] or array.size != capture["logical_element_count"]
                    or capture["shape"] != rep["shape"] or ref["bytes"] != source.stat().st_size
                    or rep["artifact_refs"] != capture["artifact_refs"] or capture["reader_projection"]["source_sha256"] != [ref["sha256"]]):
                raise ValueError("incomplete coordinate/projection coverage")
            if array.size:
                coordinate = tuple(n-1 for n in array.shape)
                for component,fn in (("real",np.real),("imag",np.imag),("magnitude",np.abs)):
                    value = read_values(path,sample["sample_id"],name,coordinate,component=component)
                    if not np.array_equal(value,fn(array[coordinate])):
                        raise ValueError("reader projection differs from stored values")
            arrays[name] = array
        if not np.allclose(arrays["embedding"],arrays["token_embedding"]+arrays["position_embedding"],atol=1e-6):
            raise ValueError("embedding lineage mismatch")
        for index in range(config["model"]["num_layers"]):
            prefix = f"layer{index}_"
            context = arrays[prefix+"attention"]@arrays[prefix+"value"]
            context = context.transpose(1,0,2).reshape(arrays[prefix+"context"].shape)
            if not np.allclose(context,arrays[prefix+"context"],atol=2e-5,rtol=2e-5):
                raise ValueError("attention/context lineage mismatch")
        roles = sample["roles"]
        if arrays["parameter_gradient"].size != sum(v["elements"] for v in trace["parameter_layout"]):
            raise ValueError("gradient vector does not match named parameter layout")
        for name in ("subject","object"):
            start,end = roles[name]["span"]
            mask = arrays[name+"_mask"]
            if not np.all(mask[start:end]) or int(mask.sum()) != end-start:
                raise ValueError("entity span does not match model input")
    for split in config["case_study"]["splits"]:
        for index in config["case_study"]["indices"]:
            for checkpoint in config["case_study"]["checkpoints"]:
                if f"{split}-{index:04d}:{checkpoint}:replay" not in seen:
                    raise ValueError("missing frozen split/checkpoint/sample")
    if mode not in ("disabled","random_phase") and "train-0000:update-0:training" not in seen:
        raise ValueError("missing actual training case")
    return dict(status="passed",samples=len(seen),full_numeric_artifacts=len(seen)*len(expected),
        private_values=True,reader="arbitrary hash-checked tensor coordinates",semantic_lineage="validated")
