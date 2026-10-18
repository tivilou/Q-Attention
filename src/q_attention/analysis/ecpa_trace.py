"""Full-value, sample-trace.v2 producer and hash-checked coordinate reader.

Only public synthetic toy inputs are supported. Never reuse this public-safe
projection policy for private Re-TACRED records without an explicit adapter.
"""
from __future__ import annotations
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
import numpy as np

from q_attention.tasks.ecpa_toy_data import LABELS


BASE = {
    "input_ids": ("preprocess", "Token IDs", "Vocabulary index for each token; zero marks padding.", ["token"]),
    "attention_mask": ("preprocess", "Valid tokens", "True positions belong to the observed sentence.", ["token"]),
    "subject_mask": ("preprocess", "Subject span", "Positions of the first entity whose relation is requested.", ["token"]),
    "object_mask": ("preprocess", "Object span", "Positions of the second entity whose relation is requested.", ["token"]),
    "token_embedding": ("preprocess", "Token embeddings", "Learned numeric features looked up from token IDs.", ["token","feature"]),
    "position_embedding": ("preprocess", "Position embeddings", "Learned features identifying each token's sentence position.", ["token","feature"]),
    "embedding": ("preprocess", "Encoder input", "Sum of token and position embeddings supplied to attention.", ["token","feature"]),
    "query": ("scoring", "Queries", "Per-head features describing what each token seeks.", ["head","query token","feature"]),
    "key": ("scoring", "Keys", "Per-head features used to match each candidate token.", ["head","key token","feature"]),
    "value": ("scoring", "Values", "Per-head content combined by the attention weights.", ["head","key token","feature"]),
    "base_scores": ("scoring", "Original attention scores", "Query-key dot products before the path intervention.", ["head","query token","key token"]),
    "adjusted_scores": ("scoring", "Adjusted attention scores", "Original dot products plus the bounded path residual, before padding mask.", ["head","query token","key token"]),
    "attention": ("context", "Attention weights", "Masked softmax of adjusted scores; valid key weights sum to one.", ["head","query token","key token"]),
    "context": ("context", "Combined values", "Attention-weighted values, concatenated across heads before output projection.", ["token","feature"]),
    "attention_output": ("context", "Attention projection", "Combined values after the attention output projection.", ["token","feature"]),
    "attention_hidden": ("context", "Attention residual state", "Input plus attention output, normalized before the feedforward block.", ["token","feature"]),
    "ffn": ("context", "Feedforward output", "Token-wise nonlinear transformation before the second residual addition.", ["token","feature"]),
    "hidden": ("context", "Encoder output", "Final token features after both attention and feedforward residuals.", ["token","feature"]),
    "pooled": ("generation", "Relation features", "Concatenated subject, object and sentence mean features.", ["feature"]),
    "classifier_hidden": ("generation", "Classifier hidden state", "Nonlinear relation features supplied to the last classifier layer.", ["feature"]),
    "logits": ("generation", "Relation scores", "Unnormalized scores for employee_of, resident_of and founder_of.", ["relation"]),
    "probabilities": ("generation", "Relation probabilities", "Softmax of relation scores; argmax produces the predicted relation.", ["relation"]),
    "loss": ("evaluation", "Sample loss", "Cross-entropy computed only after prediction, using the correct relation.", []),
    "classifier_gradient": ("training", "Classifier gradient", "Full gradient of the last classifier's weights for this sample loss.", ["relation","feature"]),
    "plugin_gradient": ("training", "Intervention gradients", "Flattened condition weight then bias gradients; empty for disabled/random frozen operator.", ["parameter"]),
    "diagnostics": ("diagnosis", "Mechanism checks", "Maximum coherent/dephased probability difference, first-triangle absolute cycle flux and intervention gradient norm; graph entries zero for disabled.", ["diagnostic"]),
}
GRAPH = {
    "entity_pair": ("Ordered entity features", "Subject and object query-feature means concatenated in that order.", ["head","ordered pair feature"]),
    "skew": ("Entity-conditioned operator", "Antisymmetric learned operator converting token features into directed phases.", ["head","feature","feature"]),
    "edges": ("Token graph", "Valid adjacent and skip-one edges; these form triangles and exclude padding.", ["head","destination","source"]),
    "degree": ("Node degree", "Number of valid neighbors used for symmetric graph normalization.", ["head","token"]),
    "phase": ("Edge phases", "Antisymmetric phase angles in radians; directed cycle sums can be nonzero.", ["head","destination","source"]),
    "hamiltonian": ("Complex propagation operator", "Hermitian normalized graph with complex phase on each edge.", ["head","destination","source"]),
    "unitary": ("One-step unitary", "Complex amplitude transition exp(-i delta H), before measuring probabilities.", ["head","destination","source"]),
    "amplitudes": ("Coherent path states", "All U^t amplitude matrices, including each propagation step.", ["head","step","destination","source"]),
    "single": ("One-step probabilities", "Squared magnitudes of the one-step unitary; shared Q/dephased transition.", ["head","destination","source"]),
    "dephased_steps": ("Dephased path states", "All B^t probability matrices; coherence is removed after each step.", ["head","step","destination","source"]),
    "pq": ("Coherent attention prior", "Squared final coherent amplitudes transposed to query-by-key order.", ["head","query token","key token"]),
    "pd": ("Dephased attention prior", "Final dephased probability transition transposed to query-by-key order.", ["head","query token","key token"]),
    "transition": ("Classical one-step attention", "Real softmax edge transition from the same learned phase features, with self edges.", ["head","query token","key token"]),
    "classical_steps": ("Classical path states", "Every repeated real transition; final state is used by the classical control.", ["head","step","query token","key token"]),
    "probability": ("Selected path prior", "Prior selected by the explicit condition: coherent, dephased or classical.", ["head","query token","key token"]),
    "residual": ("Attention score residual", "Alpha times clipped centered log prior, zero on invalid query/key positions.", ["head","query token","key token"]),
}
ALL_STAGES = ("data","preprocess","scoring","context","generation","evaluation","training","diagnosis")


def inventory(mode):
    values = dict(BASE)
    if mode != "disabled":
        values.update({name:("scoring", *info) for name,info in GRAPH.items()})
    return values


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def display(label, help_text, ref=None):
    return dict(summary=help_text, default_view="visual" if ref else "human_summary",
                visible_fields=[dict(id="meaning",label=label,help=help_text,value=label)],
                visualizations=([dict(id="values",kind="axis_aware_tensor",title=label,
                                      help=help_text,data_ref=ref)] if ref else []))


class TraceWriter:
    def __init__(self, root, mode, experiment, resume=False):
        self.root, self.mode = Path(root), mode
        path = self.root / "sample_trace.json"
        if resume and path.exists():
            self.trace = json.loads(path.read_text())
            if self.trace["experiment"] != experiment:
                raise ValueError("resumed trace source/config/model identity mismatch")
        else:
            required = list(inventory(mode))
            coverage = {stage:"observed" for stage in ALL_STAGES}
            coverage.update(retrieval="not_applicable", selection="not_applicable")
            self.trace = dict(schema_version="sample-trace.v2",trace_id=f"ecpa-{mode}",
                              trace_completeness="partial", experiment=experiment,
                              sample_selection=dict(rule="First sample ID in each split, frozen before training; plus the same train case in a real update.",
                                                    population_scope="Public synthetic ECPA toy, not Re-TACRED",selected_count=0),
                              coverage=coverage, task_contract=dict(task_type="synthetic relation classification",
                              required_roles=["sentence","tokens","subject","object","correct_relation","predicted_relation","checkpoint_identity"],
                              required_representations=required,required_value_representations=required,
                              required_splits=["train","valid","test"],required_checkpoints=["initial","best","final"],
                              required_lanes=["training","replay"],
                              checkpoint_policy="Initial/best/final each cross all three splits; actual update trace is training, snapshots are replay."),samples=[])

    def add(self, row, split, checkpoint, lane, tensors, *, update, model_sha256):
        sample_id = f"{row['id']}:{checkpoint}:{lane}"
        if set(tensors) != set(inventory(self.mode)):
            raise ValueError(f"trace inventory mismatch: {set(inventory(self.mode)) ^ set(tensors)}")
        representations, outputs = [], {stage:[] for stage in ALL_STAGES}
        for name,(stage,label,meaning,axes) in inventory(self.mode).items():
            tensor = tensors[name]
            array = tensor.detach().cpu().numpy()
            if array.dtype.hasobject or not np.isfinite(array).all():
                raise ValueError(f"invalid numeric evidence {name}")
            relative = f"values/{sample_id.replace(':','_')}/{name}.npy"
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".npy.tmp")
            with temporary.open("wb") as stream:
                np.save(stream, array, allow_pickle=False)
            temporary.replace(path)
            artifact = dict(path=relative,sha256=digest(path),bytes=path.stat().st_size,visibility="portal")
            projection = dict(status="available",kind="coordinate_inspector",coordinate_access="arbitrary",
                              reader_ref=f"q_attention.analysis.ecpa_trace:read_values#{sample_id}/{name}",
                              source_sha256=[artifact["sha256"]])
            representations.append(dict(id=name,stage_id=stage,status="observed",meaning=meaning,
                                        shape=list(array.shape),dtype=str(array.dtype),axis_semantics=axes,
                                        artifact_refs=[artifact],display=display(label,meaning,name),
                                        value_capture=dict(status="complete",shape=list(array.shape),
                                        logical_element_count=array.size,stored_element_count=array.size,
                                        artifact_refs=[artifact],reader_projection=projection)))
            outputs[stage].append(name)
        roles = dict(sentence=" ".join(row["tokens"]),tokens=row["tokens"],
                     subject=dict(text=row["tokens"][row["subject"]],span=[row["subject"],row["subject"]+1]),
                     object=dict(text=row["tokens"][row["object"]],span=[row["object"],row["object"]+1]),
                     correct_relation=LABELS[row["gold"]],
                     predicted_relation=LABELS[int(tensors["logits"].argmax())],
                     checkpoint_identity=dict(optimizer_updates_completed=update,model_state_sha256=model_sha256,
                                              capture="pre-update" if lane == "training" else "checkpoint-replay"))
        stages = []
        consumed = {
            "data": ["source:sentence","source:entity_spans"],
            "preprocess": ["tokens","entity_spans","external:model_checkpoint"],
            "scoring": ["embedding","attention_mask","subject_mask","object_mask","external:model_checkpoint"],
            "context": ["embedding","adjusted_scores","value","attention_mask","external:model_checkpoint"],
            "generation": ["hidden","subject_mask","object_mask","attention_mask","external:model_checkpoint"],
            "evaluation": ["logits","source:correct_relation"],
            "training": ["loss","external:model_checkpoint"],
            "diagnosis": ["plugin_gradient"] + (["pq","pd","phase"] if self.mode != "disabled" else []),
        }
        for stage in ALL_STAGES:
            emitted = ["sentence","tokens","entity_spans"] if stage == "data" else outputs[stage]
            inputs = consumed[stage]
            purpose = {"data":"Read the sentence and the two queried entities.",
                       "preprocess":"Map tokens and positions to embeddings, keeping entity masks.",
                       "scoring":"Construct path transitions and adjust query-key scores.",
                       "context":"Combine values by attention, then apply residual and feedforward blocks.",
                       "generation":"Pool entity/sentence features and predict their relation without the target.",
                       "evaluation":"Compare the relation scores with the correct relation.",
                       "training":("Backward of this real optimizer minibatch." if lane == "training" else
                                   "Diagnostic loss backward during checkpoint replay; no optimizer update."),
                       "diagnosis":"Inspect gradients and the difference between coherent and dephased transitions."}[stage]
            stages.append(dict(stage_id=stage,stage=stage,canonical_stage=stage,lane=lane,status="observed",
                               purpose=purpose,input_refs=inputs,output_refs=emitted,
                               checkpoint=model_sha256,
                               target_access="post_generation_only" if stage == "generation" else "not_available",
                               display=display(stage.title(),purpose)))
        sample = dict(sample_id=sample_id,lane=lane,split=split,checkpoint=checkpoint,
                      roles=roles,representations=representations,stages=stages)
        self.trace["samples"] = [x for x in self.trace["samples"] if x["sample_id"] != sample_id] + [sample]
        self.save()

    def save(self):
        samples = self.trace["samples"]
        self.trace["sample_selection"].update(selected_count=len(samples),selected_sample_ids=[x["sample_id"] for x in samples])
        atomic_json(self.root/"sample_trace.json",self.trace)

    def complete(self):
        validate_trace(self.root/"sample_trace.json")
        self.trace["trace_completeness"] = "complete"
        self.save()


@lru_cache(maxsize=8)
def _reader_document(path, modified_ns, size):
    # Cache metadata only. Actual value files are ALWAYS rehashed below. Atomic
    # JSON replacement/mtime changes invalidate this bounded parsed-document cache.
    return json.loads(Path(path).read_text())


def read_values(trace_path, sample_id, name, prefix=(), *, component="real", limit=4096):
    """Resolve an arbitrary prefix to a bounded matrix/vector/coordinate slice.

    Immutable hash verification precedes every read. Prefix axes can select head
    and step independently; callers need not load all values into a web page.
    """
    trace_path = Path(trace_path)
    stat = trace_path.stat()
    payload = _reader_document(str(trace_path.resolve()),stat.st_mtime_ns,stat.st_size)
    sample = next(x for x in payload["samples"] if x["sample_id"] == sample_id)
    rep = next(x for x in sample["representations"] if x["id"] == name)
    artifact = rep["value_capture"]["artifact_refs"][0]
    root = trace_path.parent.resolve()
    path = (root / artifact["path"]).resolve()
    if not path.is_relative_to(root) or digest(path) != artifact["sha256"]:
        raise ValueError("trace artifact escapes root or fails hash verification")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if tuple(rep["shape"]) != array.shape or str(array.dtype) != rep["dtype"]:
        raise ValueError("trace artifact shape/dtype mismatch")
    if len(prefix) > array.ndim or any(type(i) is not int or i<0 or i>=array.shape[k] for k,i in enumerate(prefix)):
        raise ValueError("invalid tensor coordinates")
    result = np.asarray(array[tuple(prefix)])
    if result.size > limit:
        raise ValueError("choose more prefix axes for a bounded slice")
    if component not in ("real","imag","magnitude"):
        raise ValueError("unknown complex component")
    return {"real":np.real,"imag":np.imag,"magnitude":np.abs}[component](result).copy()


def validate_trace(trace_path):
    """Independent content gate: coverage, complete arrays and reader equality."""
    trace_path = Path(trace_path)
    payload = json.loads(trace_path.read_text())
    mode = payload["experiment"]["model_identity"]
    expected = set(inventory(mode))
    if set(payload["task_contract"]["required_value_representations"]) != expected:
        raise ValueError("trace contract omitted implemented numeric states")
    observed = set()
    for sample in payload["samples"]:
        identity = sample["roles"]["checkpoint_identity"]
        sha = identity["model_state_sha256"]
        if type(identity["optimizer_updates_completed"]) is not int or identity["optimizer_updates_completed"] < 0:
            raise ValueError("missing actual optimizer update identity")
        if len(sha) != 64 or any(x not in "0123456789abcdef" for x in sha):
            raise ValueError("invalid actual model state hash")
        if any(stage.get("checkpoint") != sha for stage in sample["stages"]):
            raise ValueError("stage checkpoint differs from captured model state")
        observed.add((sample["split"],sample["checkpoint"],sample["lane"]))
        by_id = {x["id"]:x for x in sample["representations"]}
        if set(by_id) != expected or len(by_id) != len(sample["representations"]):
            raise ValueError("sample omitted/duplicated a numeric state")
        for name, rep in by_id.items():
            artifact = rep["value_capture"]["artifact_refs"][0]
            source = (trace_path.parent/artifact["path"]).resolve()
            if not source.is_relative_to(trace_path.parent.resolve()):
                raise ValueError("unsafe artifact path")
            if digest(source) != artifact["sha256"]:
                raise ValueError("full-value artifact hash mismatch")
            full = np.load(source,allow_pickle=False)
            capture = rep["value_capture"]
            if (capture["logical_element_count"] != full.size or capture["stored_element_count"] != full.size
                    or capture["shape"] != list(full.shape) or artifact["bytes"] != source.stat().st_size
                    or rep["artifact_refs"] != capture["artifact_refs"]
                    or capture["reader_projection"]["source_sha256"] != [artifact["sha256"]]):
                raise ValueError("full-value artifact coverage/projection mismatch")
            if not np.isfinite(full).all() or full.size != math.prod(rep["shape"]):
                raise ValueError("invalid/incomplete numeric tensor")
            for coordinates in (tuple(0 for _ in full.shape),tuple(n-1 for n in full.shape)):
                if not full.size:
                    read_values(trace_path,sample["sample_id"],name)
                    break
                for component,fn in (("real",np.real),("imag",np.imag),("magnitude",np.abs)):
                    value = read_values(trace_path,sample["sample_id"],name,coordinates,component=component)
                    if not np.array_equal(value,fn(full[coordinates])):
                        raise ValueError("reader value differs from actual source")
        # Verify a key dataflow edge numerically, not just matching metadata.
        values = lambda name: np.load(trace_path.parent/by_id[name]["artifact_refs"][0]["path"],allow_pickle=False)
        if not np.allclose(values("embedding"),values("token_embedding")+values("position_embedding"),atol=1e-7):
            raise ValueError("embedding lineage mismatch")
        context = values("attention") @ values("value")
        context = context.transpose(1,0,2).reshape(values("context").shape)
        if not np.allclose(context,values("context"),atol=1e-6):
            raise ValueError("attention-to-context lineage mismatch")
    for split in ("train","valid","test"):
        for checkpoint in ("initial","best","final"):
            if (split,checkpoint,"replay") not in observed:
                raise ValueError(f"missing selected sample coverage: {split}/{checkpoint}")
    if ("train","update-0","training") not in observed:
        raise ValueError("missing actual training-lane sample")
    return dict(status="passed",samples=len(payload["samples"]),representations=len(expected),
                full_value_artifacts=len(expected)*len(payload["samples"]))
