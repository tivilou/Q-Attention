"""Read-only loopback reader for audited Q-ECPA reports and private case values.

The static portal receives bounded slices, never a raw archive or filesystem
path. This program does not import torch, train, export, upload or publish.
"""
from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np

MODES = ("baseline", "disabled", "quantum", "dephased", "classical", "random_phase", "entity_permuted", "original_qk")
ASSETS = Path(__file__).resolve().parents[1] / "research-web/apps/ecpa-case-reader"
STAGES = {
    "data": ("原始样本", "读取句子和指定的主、客体；正确关系在模型预测之后才用于损失。"),
    "preprocess": ("分词与嵌入", "按已保存的词表索引查找词嵌入，加上位置嵌入，生成第一层编码器的输入。"),
    "generation": ("关系预测", "汇聚最后一层的主实体、客实体和句子表示，经分类器得到关系分数和概率。"),
    "evaluation": ("损失与核对", "在预测完成后，使用正确关系计算交叉熵；标签不进入注意力干预。"),
    "training": ("反向传播", "实际训练记录保存本次逻辑批次的参数梯度；重放记录只做诊断反向，不更新参数。"),
}
LABELS = {
    "input_ids": "词表索引", "attention_mask": "有效词位置", "subject_mask": "主实体位置", "object_mask": "客实体位置",
    "token_embedding": "词嵌入", "position_embedding": "位置嵌入", "embedding": "相加后的输入嵌入",
    "query": "查询向量 Q", "key": "匹配向量 K", "value": "内容向量 V", "base_scores": "原始注意力分数",
    "adjusted_scores": "干预后的注意力分数", "attention": "注意力权重", "context": "按权重汇合的内容",
    "attention_output": "注意力输出投影", "attention_hidden": "注意力残差与归一化", "ffn": "前馈网络输出", "hidden": "本层最终词表示",
    "entity_pair": "有序实体对表示", "skew": "实体条件反对称算子", "edges": "词之间的图连接", "degree": "图节点的度",
    "phase": "边上的相位", "hamiltonian": "复数传播算子 H", "unitary": "单步酉演化 U", "amplitudes": "各传播步的复振幅",
    "single": "单步转移概率", "dephased_steps": "各步去相干概率", "pq": "相干路径先验", "pd": "去相干路径先验",
    "transition": "经典单步转移", "classical_steps": "各步经典路径状态", "probability": "本条件选用的路径先验", "residual": "注意力分数残差",
    "pooled": "实体与句子的汇聚表示", "classifier_hidden": "分类器中间表示", "logits": "各关系的预测分数",
    "probabilities": "各关系的预测概率", "loss": "该样本交叉熵", "parameter_gradient": "可训练参数的完整梯度",
    "qk_raw": "原 QK 可观测量", "qk_residual": "原 QK 分数残差",
}
AXES = {"token":"词位置", "head":"注意力头", "feature":"特征维度", "query token":"查询词", "key token":"候选词",
        "destination":"目标词", "source":"来源词", "step":"传播步", "relation":"关系类别", "parameter":"参数位置",
        "ordered pair feature":"有序实体对特征", "query":"查询词", "key":"候选词", "basis state":"基态",
        "circuit step":"电路步骤", "query-key pair":"查询词—候选词对", "query/key angle":"查询/候选旋转角"}
MEANINGS = {
    "input_ids":"每个词在训练词表中的整数编号，用于查找词嵌入；编号不是词本身的连续特征。",
    "attention_mask":"1 表示真实词，0 表示补齐位置；注意力不使用补齐词。",
    "subject_mask":"为 1 的位置属于指定主实体；其表示会按实体范围汇聚。", "object_mask":"为 1 的位置属于指定客实体；主、客体顺序有意义。",
    "token_embedding":"按每个词的词表编号查到的特征向量，尚未包含位置信息。", "position_embedding":"按句子中的词位置查到的位置特征。",
    "embedding":"词嵌入与位置嵌入相加后的实际值，直接进入第一层编码器。",
    "query":"各注意力头为每个词计算的查询特征，表示它需要匹配什么信息。", "key":"各注意力头为候选词计算的匹配特征，与 Q 点积形成原始分数。",
    "value":"候选词所携带的内容向量，后续按注意力权重相加。", "base_scores":"本层 Q 与 K 的缩放点积，尚未叠加本方法的干预。",
    "adjusted_scores":"原始分数加上当前条件的路径残差；随后才进行掩码与 softmax。", "attention":"掩码和 softmax 后的权重，对每个有效查询的候选词权重之和为 1。",
    "context":"注意力矩阵乘以 V，各头连接起来，得到每个词收集到的内容。", "attention_output":"汇合后的内容通过注意力输出线性层的结果。",
    "attention_hidden":"把注意力输出加回本层输入，再归一化得到的词状态。", "ffn":"归一化词状态经前馈网络计算后的结果，尚未进行第二次残差相加。",
    "hidden":"注意力与前馈网络都完成后，本层最终的词表示；下一层继续使用它。",
    "entity_pair":"按主实体、客实体的顺序拼接查询特征的均值，作为相位条件。", "skew":"从实体对生成的反对称特征算子，用于计算有向边的相位。",
    "edges":"有效词之间相邻及隔一词的连接；图中可形成三角环，补齐词被排除。", "degree":"每个词节点的邻居数量，用来归一化图上的传播强度。",
    "phase":"边上的有向相位角，单位弧度；反向边角度相反，环路的相位和可以非零。", "hamiltonian":"将归一化图连接与复数边相位结合得到的厄米传播算子。",
    "unitary":"根据 H 计算 exp(-i·δ·H)，描述尚未测量概率的单步复振幅转移。", "amplitudes":"每一步 U 的连续传播结果，保留复数路径振幅，可查看实部、虚部或模。",
    "single":"单步酉演化振幅的模平方，相干版与去相干版使用同一个单步概率起点。", "dephased_steps":"每步去掉相干后反复相乘的概率矩阵，用于相同 H 的去相干对照。",
    "pq":"最终相干振幅取模平方并调整为查询词—候选词顺序，形成相干先验。", "pd":"逐步去相干传播后的最终概率，调整为查询词—候选词顺序。",
    "transition":"匹配经典对照从相同相位特征计算的实数 softmax 边转移。", "classical_steps":"经典转移矩阵逐步相乘的状态；最后一步用于经典注意力先验。",
    "probability":"当前实验条件实际选择使用的路径先验，可以来自相干、去相干或经典传播。", "residual":"路径先验取对数、按查询中心化、裁剪并缩放后的注意力分数残差。",
    "pooled":"按主实体、客实体和句子均值拼接的关系特征。", "classifier_hidden":"汇聚后的关系特征经分类器非线性变换得到的中间表示。",
    "logits":"分类器对各关系的未归一化预测分数。", "probabilities":"关系分数经过 softmax 得到的概率，最大概率对应模型判断。",
    "loss":"在预测之后按正确关系计算的交叉熵，不是注意力输入。", "parameter_gradient":"按已保存的参数顺序拼接的完整梯度。实际训练为整个逻辑批次的梯度；重放为单案例诊断梯度。",
    "qk_raw":"原始 QK 电路计算的 query-key 成对可观测量。", "qk_residual":"原始 QK 可观测量经过上下文中心化、限幅和缩放得到的打分残差。",
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def inside(root, relative):
    path = Path(relative)
    if path.is_absolute() or any(part in ("", ".", "..") or part.startswith(".") for part in path.parts):
        raise ValueError("unsafe artifact reference")
    target = root.joinpath(path)
    if any(parent.is_symlink() for parent in (target, *target.parents) if parent != root.parent):
        raise ValueError("symlink artifact reference")
    target = target.resolve()
    if not target.is_relative_to(root):
        raise ValueError("artifact outside evidence root")
    return target


def stage_info(stage):
    if stage.startswith("layer"):
        layer, operation = stage.split("_", 1)
        number = int(layer[5:]) + 1
        if operation == "scoring":
            return f"第 {number} 层：注意力打分", "从本层输入生成 Q/K/V；若此层有干预，则加入已保存的路径先验残差。"
        return f"第 {number} 层：内容汇聚", "注意力加权 V，接着做投影、残差、归一化和前馈变换；最终词表示交给下一层。"
    return STAGES[stage]


def rep_label(rep):
    name = rep["id"]
    if name.startswith("layer"):
        layer, name = name.split("_", 1)
        prefix = f"第 {int(layer[5:])+1} 层 · "
    else:
        prefix = ""
    return prefix + LABELS.get(name, rep["display"]["visible_fields"][0]["label"])


class CaseReader:
    def __init__(self, report_dir, case_dir):
        self.report_dir, self.case_dir = Path(report_dir).resolve(), Path(case_dir).resolve()
        marker = read(self.report_dir / "REPORT_COMPLETE.json")
        if marker["schema_version"] != "ecpa-report.v1":
            raise ValueError("not a complete ECPA report")
        if not {"case_study.json", "run_manifest.json", "run_config.json"} <= set(marker["files"]):
            raise ValueError("report marker omits required reader inputs")
        for name, digest in marker["files"].items():
            if sha(inside(self.report_dir, name)) != digest:
                raise ValueError("report integrity failure")
        index = read(self.report_dir / "case_study.json")
        if index["trace_completeness"] != "complete" or set(index["jobs"]) != set(MODES):
            raise ValueError("incomplete case inventory")
        self.manifest = read(self.report_dir / "run_manifest.json")
        if marker["fingerprint"] != self.manifest["fingerprint"]:
            raise ValueError("report fingerprint mismatch")
        self.config = read(self.report_dir / "run_config.json")
        self.documents, self.trace_files, self.trace_hashes = {}, {}, {}
        for job in MODES:
            path = inside(self.case_dir, index["jobs"][job]["evidence_path"])
            digest = index["jobs"][job]["trace_sha256"]
            if sha(path) != digest:
                raise ValueError("trace integrity failure")
            document = read(path)
            if (document["schema_version"] != "sample-trace.v2" or document["trace_completeness"] != "complete"
                    or document["experiment"]["config_sha256"] != marker["fingerprint"]
                    or document["experiment"]["model_identity"] != job):
                raise ValueError("trace source identity mismatch")
            self.documents[job], self.trace_files[job], self.trace_hashes[job] = document, path, digest

    def sample(self, job, sample_id):
        if job not in self.documents:
            raise ValueError("unknown condition")
        if sha(self.trace_files[job]) != self.trace_hashes[job]:
            raise ValueError("trace changed after opening")
        return next(s for s in self.documents[job]["samples"] if s["sample_id"] == sample_id)

    def index(self):
        return dict(profile=self.config["profile"], run_id=self.manifest["run_id"],
                    boundary="合成 canary 验收，不是正式效果结果" if self.config["profile"] == "synthetic_canary" else "冻结 seed 13 的私有选定案例；案例不替代全量指标",
                    selection=self.documents["quantum"]["sample_selection"]["rule"],
                    jobs={job:[{key:s[key] for key in ("sample_id","split","checkpoint","lane")} for s in doc["samples"]] for job,doc in self.documents.items()})

    def project(self, job, sample_id):
        sample = self.sample(job, sample_id)
        reps = []
        for rep in sample["representations"]:
            short=rep["id"].split("_",1)[1] if rep["id"].startswith("layer") else rep["id"]
            meaning=MEANINGS.get(short,rep["meaning"])
            reps.append(dict(id=rep["id"],owner_stage=rep["stage_id"],label=rep_label(rep),meaning=meaning,
                direction="output",unit="radian" if short=="phase" else "unitless",visibility="reader",
                logical_transition_group=rep["stage_id"],missing_state="observed",
                shape=rep["shape"],dtype=rep["dtype"],axes=[AXES.get(a,a) for a in rep["axis_semantics"]],
                empty=rep["value_capture"]["stored_element_count"] == 0))
        stages = []
        for stage in sample["stages"]:
            label, meaning = stage_info(stage["stage_id"])
            stages.append(dict(id=stage["stage_id"],label=label,meaning=meaning,status=stage["status"],
                               input_refs=stage["input_refs"],output_refs=stage["output_refs"]))
        return dict(sample_id=sample_id, roles=sample["roles"], stages=stages, representations=reps,
                    lane=sample["lane"], split=sample["split"], checkpoint=sample["checkpoint"],
                    label_order=[label for label,_ in sorted(self.manifest["data"]["label_to_id"].items(),key=lambda item:item[1])] if "label_to_id" in self.manifest["data"] else [])

    def values(self, job, sample_id, name, prefix=(), *, view="matrix", component="real", row=0, column=0):
        sample = self.sample(job, sample_id)
        rep = next(r for r in sample["representations"] if r["id"] == name)
        if component not in ("real","imag","magnitude") or view not in ("matrix","vector"):
            raise ValueError("invalid tensor view")
        ref = rep["value_capture"]["artifact_refs"][0]
        path = inside(self.trace_files[job].parent,ref["path"])
        if path.stat().st_size != ref["bytes"] or sha(path) != ref["sha256"]:
            raise ValueError("numeric integrity failure")
        array = np.load(path,allow_pickle=False,mmap_mode="r")
        if list(array.shape) != rep["shape"] or str(array.dtype) != rep["dtype"] or array.dtype.hasobject:
            raise ValueError("numeric shape/dtype mismatch")
        dimensions = min(array.ndim, 2 if view == "matrix" else 1)
        if (not isinstance(prefix,(tuple,list)) or len(prefix) != array.ndim-dimensions
                or any(type(i) is not int or i<0 or i>=array.shape[k] for k,i in enumerate(prefix))):
            raise ValueError("invalid tensor prefix")
        if type(row) is not int or type(column) is not int or row<0 or column<0:
            raise ValueError("invalid tile offset")
        tail = array[tuple(prefix)]
        if dimensions == 2:
            if tail.size and (row >= tail.shape[0] or column >= tail.shape[1]):
                raise ValueError("tile outside matrix")
            tile = tail[row:row+32,column:column+32]
        elif dimensions == 1:
            if tail.size and column >= tail.shape[0]:
                raise ValueError("tile outside vector")
            tile = tail[column:column+128]
        else:
            tile = tail
        value = {"real":np.real,"imag":np.imag,"magnitude":np.abs}[component](tile)
        if not np.isfinite(value).all():
            raise ValueError("nonfinite numeric evidence")
        return dict(values=value.tolist(),shape=list(value.shape),full_shape=rep["shape"],prefix=list(prefix),
                    row=row,column=column,component=component,view=view)


def handler(reader, asset_root=ASSETS):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # Sample identifiers and request parameters are private.

        def send(self, status, data, content_type="application/json; charset=utf-8"):
            body = data if isinstance(data,bytes) else json.dumps(data,ensure_ascii=False,allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type",content_type)
            self.send_header("Content-Length",str(len(body)))
            self.send_header("Cache-Control","no-store")
            self.send_header("X-Content-Type-Options","nosniff")
            self.send_header("Content-Security-Policy","default-src 'self'; frame-ancestors 'none'; object-src 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            port = self.server.server_port
            allowed = {f"127.0.0.1:{port}",f"localhost:{port}"}
            origin = self.headers.get("Origin")
            if self.headers.get("Host") not in allowed or (origin and origin not in {"http://"+h for h in allowed}):
                return self.send(403,{"error":"Only this loopback reader origin is permitted"})
            parsed = urlsplit(self.path)
            if parsed.path == "/favicon.ico":
                return self.send(204,b"","image/x-icon")
            assets = {"/":"index.html","/index.html":"index.html","/reader.js":"reader.js","/reader.css":"reader.css", "/research-editorial.css":"research-editorial.css", "/theme.css":"theme.css"}
            if parsed.path in assets:
                path = asset_root/assets[parsed.path]
                mime={".html":"text/html; charset=utf-8",".js":"text/javascript; charset=utf-8",".css":"text/css; charset=utf-8"}[path.suffix]
                return self.send(200,path.read_bytes(),mime)
            try:
                q = parse_qs(parsed.query,strict_parsing=True,max_num_fields=16) if parsed.query else {}
                def arg(key, default=None):
                    if key in q and len(q[key]) != 1:
                        raise ValueError("duplicate parameter")
                    if key not in q and default is None:
                        raise ValueError("missing parameter")
                    return q.get(key,[default])[0]
                if parsed.path == "/api/index":
                    data=reader.index()
                elif parsed.path == "/api/sample":
                    data=reader.project(arg("job"),arg("sample"))
                elif parsed.path == "/api/values":
                    data=reader.values(arg("job"),arg("sample"),arg("representation"),json.loads(arg("prefix","[]")),
                        view=arg("view","matrix"),component=arg("component","real"),row=int(arg("row","0")),column=int(arg("column","0")))
                else:
                    return self.send(404,{"error":"No raw file or archive route"})
                self.send(200,data)
            except (ValueError,KeyError,StopIteration,TypeError,IndexError):
                self.send(400,{"error":"Invalid request or changed evidence; cannot display this value"})
            except OSError:
                self.send(409,{"error":"Evidence unavailable; no value inferred"})
    return Handler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir",type=Path,required=True)
    parser.add_argument("--case-dir",type=Path,required=True)
    parser.add_argument("--port",type=int,default=4191)
    args=parser.parse_args()
    reader=CaseReader(args.report_dir,args.case_dir)
    server=ThreadingHTTPServer(("127.0.0.1",args.port),handler(reader))
    print(f"Q-ECPA private case reader: http://127.0.0.1:{server.server_port}/",flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
