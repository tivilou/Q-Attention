"""Paper-derived MetaEntail primitives, not an admitted neural model adapter."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from string import Formatter
from types import MappingProxyType

from .common import PACKAGE, atomic_json, fingerprint

NLI_LABELS = ("contradiction", "neutral", "entailment")


def paper_spec():
    value = json.loads((PACKAGE/"metaentail_paper.lock.json").read_text(encoding="utf-8"))
    if (value["schema"] != "metaentail-paper-primitives.v1" or value["model_implemented"] is not False
            or value["strict_comparison_eligible"] is not False or value["provenance_mode"] != "paper"):
        raise ValueError("MetaEntail preparation/admission contract drift")
    rows = value["templates"]
    if (len(rows) != 40 or len({r["id"] for r in rows}) != 40
            or any(type(r["canonical_id"]) is not int for r in rows)
            or sorted(r["canonical_id"] for r in rows) != list(range(40))
            or next(r["id"] for r in rows if r["canonical_id"] == 0) != "no_relation"):
        raise ValueError("canonical40 template mapping invalid")
    for row in rows:
        parts = list(Formatter().parse(row["pattern"]))
        fields = [field for _, field, _, _ in parts if field is not None]
        if ("subject" not in fields or set(fields)-{"subject", "object"}
                or any(spec or conversion for _, _, spec, conversion in parts)):
            raise ValueError("invalid hypothesis format field")
    labels = {row["id"] for row in rows}
    for group in value["exclusive_groups"]:
        if len(set(group)) != len(group) or len(group)<2 or set(group)-labels or "no_relation" in group:
            raise ValueError("invalid exclusive group")
    return value


_SPEC = paper_spec()
RELATIONS = tuple(row["id"] for row in sorted(_SPEC["templates"], key=lambda row: row["canonical_id"]))
TEMPLATES = MappingProxyType({row["id"]: row["pattern"] for row in _SPEC["templates"]})
EXCLUSIVE = tuple(frozenset(group) for group in _SPEC["exclusive_groups"])
del _SPEC


def nli_target(gold_relation, hypothesis_relation):
    if gold_relation not in TEMPLATES or hypothesis_relation not in TEMPLATES:
        raise ValueError("unknown canonical relation")
    if gold_relation == hypothesis_relation:
        return "entailment"
    if ("no_relation" in (gold_relation, hypothesis_relation)
            or any(gold_relation in group and hypothesis_relation in group for group in EXCLUSIVE)):
        return "contradiction"
    return "neutral"


def nli_order_checked(order):
    order = tuple(order)
    if len(order) != 3 or set(order) != set(NLI_LABELS):
        raise ValueError("explicit checkpoint NLI label order required")
    return order


def record_input(record):
    """Only named input fields are read: inference never touches record.relation."""
    identity, tokens = record["id"], record["token"]
    if not isinstance(identity, str) or not identity or not isinstance(tokens, list) or not tokens:
        raise ValueError("nonempty string sample id and raw token list required")
    if any(not isinstance(token, str) or not token.strip() for token in tokens):
        raise ValueError("invalid raw token")
    result = {"sample_id": identity, "premise": " ".join(tokens)}
    for role, prefix in (("subject", "subj"), ("object", "obj")):
        start, end, kind = record[prefix+"_start"], record[prefix+"_end"], record[prefix+"_type"]
        if (type(start) is not int or type(end) is not int or not 0 <= start <= end < len(tokens)
                or not isinstance(kind, str) or not kind.strip()):
            raise ValueError("invalid inclusive entity span or type")
        result[role] = {"text": " ".join(tokens[start:end+1]), "span": [start,end], "type": kind}
    if max(result["subject"]["span"][0], result["object"]["span"][0]) <= min(
            result["subject"]["span"][1], result["object"]["span"][1]):
        raise ValueError("overlapping entity spans need a declared adaptation")
    return result


@dataclass(frozen=True)
class FeasibleTypes:
    pairs: tuple[frozenset[tuple[str, str]], ...]
    training_projection_sha256: str
    training_count: int

    def __post_init__(self):
        if (type(self.training_count) is not int or self.training_count<=0
                or len(self.training_projection_sha256)!=64
                or any(c not in "0123456789abcdef" for c in self.training_projection_sha256)
                or not isinstance(self.pairs, tuple) or len(self.pairs)!=40):
            raise ValueError("invalid training-filter identity")
        for pairs in self.pairs:
            if not isinstance(pairs, frozenset) or any(
                    not isinstance(pair,tuple) or len(pair)!=2
                    or any(not isinstance(t,str) or not t.strip() for t in pair) for pair in pairs):
                raise ValueError("invalid ordered entity type pairs")
        if not any(self.pairs):
            raise ValueError("training filter is empty")

    def candidates(self, subject_type, object_type):
        return tuple(relation for relation, pairs in zip(RELATIONS,self.pairs)
                     if (subject_type,object_type) in pairs)

    def to_dict(self):
        value = {"schema":"metaentail-train-type-filter.v1", "fit_split":"train",
                 "training_projection_sha256":self.training_projection_sha256, "training_count":self.training_count,
                 "projection_fields":["id","subj_type","obj_type","relation"],
                 "pairs":{label:[list(pair) for pair in sorted(pairs)] for label,pairs in zip(RELATIONS,self.pairs)}}
        value["artifact_sha256"] = fingerprint(value)
        return value

    @classmethod
    def from_dict(cls, value):
        payload = {key:item for key,item in value.items() if key!="artifact_sha256"}
        if (fingerprint(payload)!=value.get("artifact_sha256") or value.get("fit_split")!="train"
                or value.get("schema")!="metaentail-train-type-filter.v1"
                or value.get("projection_fields")!=["id","subj_type","obj_type","relation"]
                or set(value["pairs"])!=set(RELATIONS)):
            raise ValueError("invalid or changed train-only filter artifact")
        packed = []
        for label in RELATIONS:
            rows = value["pairs"][label]
            if not isinstance(rows,list) or any(not isinstance(row,list) or len(row)!=2
                    or any(not isinstance(t,str) or not t.strip() for t in row) for row in rows):
                raise ValueError("invalid serialized type pair")
            pairs = frozenset(tuple(row) for row in rows)
            if len(pairs)!=len(rows):
                raise ValueError("duplicate serialized type pair")
            packed.append(pairs)
        return cls(tuple(packed),value["training_projection_sha256"],value["training_count"])


def fit_type_filter(records, *, split):
    if split!="train":
        raise ValueError("type feasibility must be fit on train only")
    pairs = {label:set() for label in RELATIONS}
    identities, projection_hash = set(), hashlib.sha256()
    for record in records:
        fields = record_input(record)
        identity, gold = fields["sample_id"], record["relation"]
        if identity in identities or gold not in TEMPLATES:
            raise ValueError("duplicate training sample identity or unknown relation")
        identities.add(identity)
        pair = fields["subject"]["type"],fields["object"]["type"]
        pairs[gold].add(pair)
        row = [identity,*pair,gold]
        projection_hash.update((json.dumps(row,ensure_ascii=True,separators=(",",":"))+"\n").encode())
    return FeasibleTypes(tuple(frozenset(pairs[label]) for label in RELATIONS),projection_hash.hexdigest(),len(identities))


def inference_group(record, feasibility):
    group = record_input(record)
    subject, object_ = group["subject"],group["object"]
    labels = feasibility.candidates(subject["type"],object_["type"])
    group["candidates"] = [{"relation":label,"hypothesis":TEMPLATES[label].format(
                           subject=subject["text"],object=object_["text"])} for label in labels]
    group["filter_identity"] = fingerprint(feasibility.to_dict())
    group["target_access"] = "none"
    return group


def training_group(record, feasibility):
    group = inference_group(record,feasibility)
    gold = record["relation"]
    if gold not in TEMPLATES:
        raise ValueError("unknown training relation")
    if gold not in {item["relation"] for item in group["candidates"]}:
        raise ValueError("training gold excluded by filter; wrong filter or training split")
    group["target_relation"],group["target_access"] = gold,"training_NLI_target_only"
    for item in group["candidates"]:
        item["nli_target"] = nli_target(gold,item["relation"])
    return group


def grouped_prediction(group, probabilities, *, nli_order):
    """Choose only among argmax-entail rows; preserve an honest abstention."""
    order = nli_order_checked(nli_order)
    labels = [item["relation"] for item in group["candidates"]]
    if len(set(labels))!=len(labels) or any(label not in TEMPLATES for label in labels):
        raise ValueError("invalid hypothesis group relations")
    if labels!=sorted(labels,key=RELATIONS.index):
        raise ValueError("hypothesis group must preserve canonical relation order")
    if len(probabilities)!=len(labels):
        raise ValueError("one probability row required for every feasible hypothesis")
    entail_index = order.index("entailment")
    rows, eligible = [], []
    for label, values in zip(labels,probabilities):
        if (len(values)!=3 or any(isinstance(p,bool) or not isinstance(p,Real)
                or not math.isfinite(p) or not 0<=p<=1 for p in values)
                or not math.isclose(sum(values),1,rel_tol=1e-6,abs_tol=1e-7)):
            raise ValueError("finite normalized three-way NLI probabilities required")
        predicted = order[max(range(3),key=lambda i:values[i])]
        rows.append({"relation":label,"nli_predicted":predicted,
                     "probabilities":[float(p) for p in values],"entailment_probability":float(values[entail_index])})
        if predicted=="entailment":
            eligible.append(rows[-1])
    best = max(eligible,key=lambda item:item["entailment_probability"]) if eligible else None
    return {"sample_id":group["sample_id"],"status":"selected" if best else "abstained",
            "relation":best["relation"] if best else None,
            "confidence":best["entailment_probability"] if best else None,
            "reason":"highest_confidence_entail" if best else "no_entail_prediction" if labels else "no_feasible_hypotheses",
            "nli_label_order":list(order),"candidate_predictions":rows,
            "tie_policy":"NLI_first_checkpoint_index_then_first_canonical_relation",
            "fallback_relation_added":False}


def audit_decisions(gold, predictions):
    """All-example diagnostic counts; not the unrecovered paper entail-only scorer."""
    if not gold or len(gold)!=len(predictions) or any(label not in TEMPLATES for label in gold):
        raise ValueError("aligned nonempty canonical gold and prediction lists required")
    if any(label is not None and label not in TEMPLATES for label in predictions):
        raise ValueError("unknown predicted relation")
    tp = sum(g==p and g!="no_relation" for g,p in zip(gold,predictions))
    fp = sum(p not in (None,"no_relation") and p!=g for g,p in zip(gold,predictions))
    fn = sum(g!="no_relation" and p!=g for g,p in zip(gold,predictions))
    precision,recall = tp/(tp+fp) if tp+fp else 0.,tp/(tp+fn) if tp+fn else 0.
    return {"scope":"all_example_diagnostic_not_paper_evaluator", "examples":len(gold),
            "abstained":predictions.count(None),"coverage":1-predictions.count(None)/len(gold),
            "accuracy":sum(g==p for g,p in zip(gold,predictions))/len(gold),
            "tp":tp,"fp":fp,"fn":fn,"micro_precision":precision,"micro_recall":recall,
            "micro_f1":2*precision*recall/(precision+recall) if precision+recall else 0.}


def toy_report():
    records = [{"id":"toy-"+str(i),"token":["Alice",("visited","joined","attended")[i],"Acme","."],
                "subj_start":0,"subj_end":0,"obj_start":2,"obj_end":2,"subj_type":"PERSON","obj_type":"ORGANIZATION",
                "relation":label} for i,label in enumerate(("no_relation","per:employee_of","per:schools_attended"))]
    feasibility = fit_type_filter(records,split="train")
    rebuilt = FeasibleTypes.from_dict(json.loads(json.dumps(feasibility.to_dict())))
    if rebuilt != feasibility:
        raise AssertionError("type filter serialization mismatch")
    raw = dict(records[1],token=["Alice","joined","Acme","."])
    group = inference_group(raw,rebuilt)
    decision = grouped_prediction(group,[[.1,.2,.7],[.05,.1,.85],[.6,.3,.1]],nli_order=NLI_LABELS)
    if decision["relation"]!="per:employee_of":
        raise AssertionError("grouped entail selection fixture failed")
    counts = audit_decisions(["per:employee_of","per:schools_attended"],[decision["relation"],None])
    if counts["fn"]!=1 or counts["examples"]!=2:
        raise AssertionError("abstention disappeared from diagnostic denominator")
    return {"schema":"metaentail-primitive-fixture.v1","status":"paper_primitives_verified",
            "model_implemented":False,"strict_comparison_eligible":False,"model_jobs_launched":0,
            "templates":40,"meta_class_cells":1600,"filter":feasibility.to_dict(),
            "training_group":training_group(records[1],rebuilt),"inference_group":group,
            "decision":decision,"abstention_counts":counts,"pending":paper_spec()["unresolved"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toy",action="store_true",help="synthetic primitive fixtures only; no model or task data")
    parser.add_argument("--output",type=Path)
    args = parser.parse_args(argv)
    if not args.toy:
        parser.error("--toy required; no neural runner is admitted")
    if args.output and (args.output.exists() or args.output.absolute()!=args.output.resolve()):
        parser.error("output must be a new non-symlinked path")
    result = toy_report()
    if args.output:
        atomic_json(args.output,result)
    print(json.dumps(result,indent=2,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
