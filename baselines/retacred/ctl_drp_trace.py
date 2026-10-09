"""Producer-owned CTL sample-trace.v2, complete detached values and local reader.

All artifacts are private. No web route, raw-data publication or reconstructed
training forward is implied by the coordinate reader implemented here.
"""
from __future__ import annotations
from contextlib import contextmanager
import copy
import json
import math
from pathlib import Path

from .common import atomic_json, digest
from .improved import SampleCapture

STAGES = ("data", "preprocess", "training", "retrieval", "scoring", "selection", "context", "generation", "evaluation", "diagnosis")
ROLES = ["source_sample", "sequence_or_features", "task_objects", "target_or_gold", "prediction", "diagnosis"]


def inventory(layers, phase="test"):
    """Required IDs derive from the implemented encoder, not captured-key availability."""
    rows = [
        ("token_ids", "task.input_ids", "preprocess", "候选句对的 token ID", "候选关系对应的句子和描述联合编码；ID 是词表中的位置，不是语义向量。"),
        ("mask", "task.attention_mask", "preprocess", "有效 token 掩码", "1 标明真实 token，0 标明同一物理批次中用于补齐长度的 padding。"),
        ("word_embedding", "encoder.embeddings.word_embeddings.call0.output.tensor", "embedding", "词向量", "每个候选句对的 token ID 经可训练词表查询得到的词向量。"),
        ("position_embedding", "encoder.embeddings.position_embeddings.call0.output.tensor", "embedding", "位置向量", "模型为每个 token 的序列位置查出的向量，补充词序信息。"),
        ("type_embedding", "encoder.embeddings.token_type_embeddings.call0.output.tensor", "embedding", "类型向量", "RoBERTa 的 token-type 向量；不是原始实体的 PERSON 等类别。"),
        ("embedding_sum", "encoder.embeddings.call0.output.tensor", "embedding", "进入 Transformer 的表示", "词、位置、token-type 向量相加后经过归一化和 dropout 的实际结果。"),
    ]
    for i in range(layers):
        base=f"encoder.encoder.layer.{i}"
        for name, key, label, meaning in (
            ("input", ".attention.self.query.call0.input.0", "层输入", "上一层输出的上下文 token 表示；首层来自 embedding 阶段。"),
            ("Q", ".attention.self.query.call0.output.tensor", "查询向量 Q", "token 想从其他 token 获取何种信息；尚未拆分成注意力头。"),
            ("K", ".attention.self.key.call0.output.tensor", "键向量 K", "token 可供其他 token 匹配的信息；尚未拆分成注意力头。"),
            ("V", ".attention.self.value.call0.output.tensor", "值向量 V", "实际被注意力权重汇总的信息；尚未拆分成注意力头。"),
            ("context", ".attention.output.dense.call0.input.0", "注意力汇总", "按注意力概率加权 V 并合并所有注意力头后的实际上下文向量。"),
            ("attention_output", ".attention.output.LayerNorm.call0.output.tensor", "注意力块输出", "注意力投影、dropout、残差相加和归一化后的实际表示。"),
            ("feedforward", ".intermediate.intermediate_act_fn.call0.output.tensor", "前馈激活", "每个 token 独立经过前馈扩展和 GELU 得到的非线性中间表示。"),
            ("output", ".output.LayerNorm.call0.output.tensor", "层输出", "前馈投影、dropout、残差相加和归一化后的表示，供下一层使用。"),
        ):
            rows.append((f"layer{i}.{name}",base+key,f"layer{i}",f"第{i+1}层 · {label}",f"第{i+1}层：{meaning}"))
        rows.extend([
            (f"layer{i}.attention_scores",f"functional.attention_scores_to_probabilities.call{i}.input",f"layer{i}",f"第{i+1}层 · 注意力分数", "按头拆分的 Q/K 匹配分数，含 padding 掩码；这是 softmax 的真实输入。"),
            (f"layer{i}.attention_probabilities",f"functional.attention_scores_to_probabilities.call{i}.output",f"layer{i}",f"第{i+1}层 · 注意力概率", "softmax 的真实输出；最后两轴是查询 token 与被关注 token，发生在注意力 dropout 之前。"),
        ])
    rows.extend([
        ("first_token","task.first_token","readout","每个候选的首 token 表示","最后一层首 token 的原始表示；没有 pooler 或 L2 归一化。"),
        ("candidate_scores","task.scores","readout","候选关系分数","同一个标量仿射头对每个句子—关系描述联合表示打出的分数。"),
        ("candidate_probabilities","task.probabilities","selection","候选关系概率","只在这个样本自己的候选集合内归一化，最高概率候选成为预测关系。"),
    ])
    if phase=="train":
        rows.extend([
            ("head_before","pre_update.head.weight","readout","更新前的打分头权重","产生这次预测的共享打分头权重；记录发生在训练 forward 之前。"),
            ("selected_loss","task.selected_sentence_loss","training","所选样本损失","正确关系在本样本候选集合中的负对数概率，未乘物理批次累积权重。"),
            ("head_gradient","logical_batch.head_gradient.weight","training","打分头梯度","整个逻辑 batch 加权累积后的参数梯度；不是只归因于所选样本的梯度。"),
            ("head_after","post_update.head.weight","training","更新后的打分头权重","AdamW 完成这个逻辑 batch 的一次更新后实际读出的参数。"),
        ])
    return rows


def observability(adapter):
    selected = {s:min(range(len(adapter.data.records[s])), key=lambda i:adapter.data.records[s][i]["id"]) for s in ("train","test")}
    return {"schema":"ctl-drp-observability.v1", "selection_rule":"lexicographically_first_sample_id_per_train_and_test_split",
            "selected":{s:{"index":i,"sample_id":adapter.data.records[s][i]["id"]} for s,i in selected.items()},
            "required_representations":{s:[r[0] for r in inventory(adapter.model.encoder.config.num_hidden_layers,s)] for s in ("train","test")},
            "checkpoints":{"train":"first_selected_training_event_pre_and_post_logical_update","test":"final_post_update_evaluation"},
            "values":"complete_detached_selected_sample_all_candidate_rows",
            "target_boundary":"training_loss_only_or_post_prediction_evaluation",
            "visibility":"private_local_coordinate_reader; exchange_and_portal_binding_separate",
            "formal_comparison_admitted":False}


class CandidateCapture(SampleCapture):
    def __init__(self, model, groups, selected):
        self.begin = sum(len(g["candidates"]) for g in groups[:selected])
        self.end = self.begin+len(groups[selected]["candidates"])
        self.pairs = sum(len(g["candidates"]) for g in groups)
        super().__init__(model, row=0, batch_size=-1)

    def select(self, tensor):
        value=tensor.detach()
        if value.ndim >= 2 and value.shape[0] == self.pairs:
            value=value[self.begin:self.end]
        return value.cpu().clone()


def display(label, meaning, ref=None):
    return {"summary":meaning,"default_view":"human_summary",
            "visible_fields":[{"id":"value","label":label,"help":meaning,**({"value_ref":ref} if ref else {})}],
            "visualizations":[]}


class TraceObserver:
    def __init__(self, adapter, directory, *, phase, index, update, experiment):
        self.adapter,self.directory,self.phase,self.index,self.update=adapter,Path(directory),phase,index,update
        self.experiment=copy.deepcopy(experiment)
        self.sample_id=adapter.data.records[phase][index]["id"]
        self.capture=self.pending=None
        self.path=None

    @contextmanager
    def context(self, phase, groups, batch, scale):
        matches=[i for i,g in enumerate(groups) if g["sample_id"]==self.sample_id]
        if phase!=self.phase or self.path is not None or not matches:
            yield
            return
        if self.pending is not None or len(matches)!=1:
            raise ValueError("selected trace duplicated within logical update")
        self.group=copy.deepcopy(groups[matches[0]])
        self.group.pop("gold",None)
        self.local=matches[0];self.scale=scale
        self.capture=CandidateCapture(self.adapter.model,groups,self.local)
        with self.capture:
            self.capture.save("task.input_ids",batch["input_ids"])
            self.capture.save("task.attention_mask",batch["attention_mask"])
            if phase=="train":self.capture.values["pre_update.head.weight"]=self.adapter.model.head.weight.detach().cpu().clone()
            yield
        self.pending=self.capture
        self.capture=None

    def forward(self,result):
        if self.capture is None:return
        self.capture.save("task.first_token",result["first_token"])
        values=self.capture.values
        values["task.scores"]=result["scores"][self.capture.begin:self.capture.end].detach().cpu().clone()
        values["task.probabilities"]=result["probabilities"][self.local].detach().cpu().clone()
        self.prediction=result["predictions"][self.local]
        if result["per_sentence_loss"] is not None:
            values["task.selected_sentence_loss"]=result["per_sentence_loss"][self.local].detach().cpu().clone()

    def post_update(self):
        if self.pending is None:return
        for name,param in self.adapter.model.head.named_parameters():
            if param.grad is not None:self.pending.values["logical_batch.head_gradient."+name]=param.grad.detach().cpu().clone()
            self.pending.values["post_update.head."+name]=param.detach().cpu().clone()
        self._write()

    def post_evaluation(self):
        if self.pending is not None:self._write()

    def _write(self):
        import torch
        self.directory.mkdir(parents=True,exist_ok=True)
        event=f"{self.phase}-index{self.index}-update{self.update}"
        path=self.directory/(event+".pt")
        if path.exists():raise ValueError("trace event path already exists; preserve evidence instead of overwrite")
        values={k:v.contiguous() for k,v in self.pending.values.items()}
        required=inventory(self.adapter.model.encoder.config.num_hidden_layers,self.phase)
        missing=[key for _,key,*_ in required if key not in values]
        if missing:raise ValueError("trace_incomplete: required actual tensors absent: "+repr(missing))
        if path.exists():raise ValueError('fresh actual trace required')
        from .train import save_torch
        save_torch(path,values)
        artifact={"path":path.name,"sha256":digest(path),"bytes":path.stat().st_size,"visibility":"private"}
        representations=[]
        for rep,key,stage,label,meaning in required:
            tensor=values[key];shape=list(tensor.shape)
            axes=(["candidate_relation","attention_head","query_token","key_token"] if "attention_" in rep and tensor.ndim==4
                  else ["candidate_relation","token","feature"] if tensor.ndim==3
                  else ["candidate_relation","feature"] if tensor.ndim==2
                  else ["candidate_relation"] if tensor.ndim==1 else [])
            if rep in ("head_before","head_gradient","head_after"):axes=["scalar_output","encoder_feature"]
            representations.append({"id":rep,"stage_id":stage,"status":"observed","meaning":meaning,
                "shape":shape,"dtype":str(tensor.dtype),"axis_semantics":axes,
                "portal_preview":{"tensor_key":key,"privacy":"private_coordinate_inspector_not_public_json"},
                "artifact_refs":[artifact],"display":display(label,meaning,rep),
                "value_capture":{"status":"complete","shape":shape,"logical_element_count":tensor.numel(),"stored_element_count":tensor.numel(),
                    "artifact_refs":[artifact],"reader_projection":{"status":"available","kind":"coordinate_inspector","coordinate_access":"bounded_tiles",
                    "reader_ref":"python:baselines.retacred.ctl_drp_trace.ValueReader.read","source_sha256":[artifact["sha256"]]}}})
        # Producer-owned stage references are actual transformation boundaries.
        stages=[]
        def stage(id_,canonical,inputs,outputs,purpose,content=None):
            stages.append({"stage_id":id_,"stage":id_,"canonical_stage":canonical,"lane":"training" if self.phase=="train" else "evaluation",
                           "status":"observed","step":len(stages),"purpose":purpose,"input_refs":inputs,"output_refs":outputs,
                           "display":display(id_,purpose),"outputs":content or {},"target_access":"training_only" if self.phase=="train" else "post_generation_only"})
        record=copy.deepcopy(self.adapter.data.records[self.phase][self.index]);gold=record.pop("relation")
        stage("data","data",["source:physical_sample"],["sample"],"输入一句话，并指定需要判断关系的主语和宾语。",record)
        stage("preprocess","preprocess",["sample"],["token_ids","mask"],"插入实体标记，并为每个训练集允许的候选关系构造句子—描述对。",self.group)
        stage("embedding","scoring",["token_ids","mask"],[r[0] for r in required if r[2]=="embedding"],"把词表 ID 转换成包含位置和类型信息的向量。")
        previous="embedding_sum"
        for i in range(self.adapter.model.encoder.config.num_hidden_layers):
            outputs=[r[0] for r in required if r[2]==f"layer{i}"]
            stage(f"layer{i}","scoring",[previous,"mask"],outputs,f"第{i+1}层先汇总相关 token，再经过前馈变换，得到新的上下文表示。")
            previous=f"layer{i}.output"
        stage("readout","scoring",[previous,"source:current_head_parameters"],[r[0] for r in required if r[2]=="readout"],"取每个候选的首 token 表示，用共享标量头打分。")
        stage("selection","selection",["candidate_scores"],["candidate_probabilities","prediction"],"仅在当前句子的候选关系中归一化，并取分数最大的关系。",{"predicted_relation":self.prediction})
        if self.phase=="train":
            stage("training","training",["candidate_probabilities","head_before","source:training_target"],["optimizer_event"]+[r[0] for r in required if r[2]=="training"],
                  "用正确关系计算损失，反向传播并完成整个逻辑 batch 的参数更新。",
                  {"selected_loss":float(values["task.selected_sentence_loss"]),"physical_loss_weight":self.scale,
                   "parameter_gradient_scope":"aggregate_weighted_logical_batch_not_selected_sample_attribution","post_update":self.update})
        stage("evaluation","evaluation",["prediction","source:post_prediction_reference"],["correctness"],"预测完成后，与真实关系比较。",{"reference_relation":gold,"correct":self.prediction==gold})
        stage("diagnosis","diagnosis",["correctness"],["diagnosis_result"],"记录候选覆盖和事件身份；不改写预测或正确标签。",
              {"gold_in_candidate_set":gold in self.group["candidates"],"samples_dropped":0})
        lane="training" if self.phase=="train" else "evaluation"
        coverage={s:("observed" if s in {"data","preprocess","scoring","selection","evaluation","diagnosis"} or s=="training" and self.phase=="train" else "not_applicable") for s in STAGES}
        ids=[r[0] for r in required]
        sample={"sample_id":self.sample_id,"lane":lane,"split":self.phase,"checkpoint":"selected_train_update" if self.phase=="train" else "final_post_update",
                "roles":{"source_sample":{"sentence":" ".join(record["token"]),"tokens":record["token"]},
                         "sequence_or_features":{"candidate_sentences":self.group["pairs"]},"task_objects":self.group["marked"]["entities"],
                         "target_or_gold":gold,"prediction":self.prediction,"diagnosis":{"gold_in_H":gold in self.group["candidates"]}},
                "representations":representations,"stages":stages}
        payload={"schema_version":"sample-trace.v2","trace_id":event,"trace_completeness":"complete","experiment":self.experiment,
                 "sample_selection":{"rule":"lexicographically_first_sample_id_in_split","population_scope":self.phase,"selected_count":1,"selected_sample_ids":[self.sample_id]},
                 "coverage":coverage,"samples":[sample],"task_contract":{"task_type":"sentence_relation_classification",
                 "required_roles":ROLES,"required_representations":ids,"required_value_representations":ids,"required_splits":[self.phase],
                 "required_checkpoints":[sample["checkpoint"]],"required_lanes":[lane],"checkpoint_policy":"actual_event_not_post_hoc_reconstructed"}}
        self.path=self.directory/(event+".json");atomic_json(self.path,payload)
        validate(self.path,layers=self.adapter.model.encoder.config.num_hidden_layers)
        self.pending=None


class ValueReader:
    """Private safe source-linked reader; each request at most4096 numeric elements."""
    def __init__(self, trace):
        import torch
        self.trace=Path(trace).absolute()
        if self.trace.resolve()!=self.trace:raise ValueError("regular trace required")
        self.payload=json.loads(self.trace.read_text())
        self.representations={r["id"]:r for r in self.payload["samples"][0]["representations"]}
        artifacts={r["artifact_refs"][0]["path"]:r["artifact_refs"][0] for r in self.representations.values()}
        self.values={}
        for name, spec in artifacts.items():
            if Path(name).name!=name or name in (".",".."):raise ValueError("unsafe numeric artifact name")
            path=self.trace.parent/name
            from .reference_archive import verify_available, materialize
            verify_available(path,spec['sha256'],spec['bytes'])
            with materialize(path) as local:
                self.values[name]=torch.load(local,map_location="cpu",weights_only=True,mmap=True)

    def tensor(self, rep):
        import torch
        r=self.representations[rep];key=r["portal_preview"]["tensor_key"]
        tensor=self.values[r["artifact_refs"][0]["path"]][key]
        if not torch.is_tensor(tensor) or list(tensor.shape)!=r["shape"] or str(tensor.dtype)!=r["dtype"]:
            raise ValueError("numeric representation shape/dtype drift")
        return tensor

    def read(self, rep, *, prefix=(), window=None):
        tensor=self.tensor(rep)
        if len(prefix)>tensor.ndim or any(type(v) is not int or not 0<=v<tensor.shape[i] for i,v in enumerate(prefix)):
            raise ValueError("invalid prefix-axis coordinate")
        remaining=list(tensor.shape[len(prefix):])
        if len(remaining)>2:raise ValueError("choose prefix axes to expose a matrix/vector/scalar")
        if window is None:window=[(0,n) for n in remaining]
        if len(window)!=len(remaining) or any(len(w)!=2 or any(type(v) is not int for v in w) or not 0<=w[0]<w[1]<=n for w,n in zip(window,remaining)):
            raise ValueError("invalid bounded slice")
        if math.prod(w[1]-w[0] for w in window)>4096:raise ValueError("reader slice exceeds4096 element bound")
        index=tuple(prefix)+tuple(slice(*w) for w in window)
        value=tensor[index]
        return {"values":value.tolist(),"shape":list(value.shape),"dtype":str(value.dtype),
                "source_sha256":self.representations[rep]["artifact_refs"][0]["sha256"]}


def validate(path, *, layers):
    from .reference_archive import ctl_archived_validation
    if ctl_archived_validation(path,layers):return
    reader=ValueReader(path);payload=reader.payload
    sample=payload["samples"][0];required={r[0] for r in inventory(layers,sample["split"])}
    if (payload.get("schema_version")!="sample-trace.v2" or set(payload["task_contract"]["required_value_representations"])!=required
            or set(reader.representations)!=required or any(not sample["roles"].get(r) for r in ROLES)):
        raise ValueError("trace_incomplete: implemented inventory/roles differ")
    produced=set();all_outputs=set()
    for stage in sample["stages"]:
        if any(ref not in produced and not ref.startswith(("source:","external:")) for ref in stage["input_refs"]):
            raise ValueError("producer lineage gap")
        produced.update(stage["output_refs"]);all_outputs.update(stage["output_refs"])
    if not required<=all_outputs:raise ValueError("required states are not producer-owned stage outputs")
    for name,r in reader.representations.items():
        t=reader.tensor(name);c=r["value_capture"]
        if c["logical_element_count"]!=t.numel() or c["stored_element_count"]!=t.numel() or c["shape"]!=list(t.shape):
            raise ValueError("full coordinate coverage differs")
        for coordinate in (tuple(0 for _ in t.shape),tuple(n-1 for n in t.shape)):
            if reader.read(name,prefix=coordinate)["values"]!=t[coordinate].item():raise ValueError("reader value differs")
    # Independently verify actual transformation continuity, not names alone.
    import torch
    previous=reader.tensor("embedding_sum")
    for i in range(layers):
        if not torch.equal(previous,reader.tensor(f"layer{i}.input")):raise ValueError("actual layer lineage mismatch")
        previous=reader.tensor(f"layer{i}.output")
    if not torch.equal(previous[:,0,:],reader.tensor("first_token")):raise ValueError("raw CLS readout mismatch")
    p=reader.tensor("candidate_probabilities")
    if not torch.allclose(p,reader.tensor("candidate_scores").softmax(0),atol=1e-7,rtol=1e-6):raise ValueError("candidate denominator mismatch")
    return {"status":"complete_local_values_verified","representations":len(required),"coordinate_reader":"actual_first_last_values_all_required_states"}
