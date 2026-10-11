"""Paper translation and isolation checks; never a neural/runtime certificate."""
import ast
from collections import Counter
import copy
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from baselines.retacred import metaentail as m
from baselines.retacred.common import cohort, fingerprint, select_models
from baselines.retacred.metaentail_fidelity import paper_tables, verify_paper

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT/".external/retacred/metaentail-paper-recovery-20261008A"


def record(label="per:employee_of", identity="s", reverse=False, types=("PERSON","ORGANIZATION")):
    return {"id":identity,"token":["Alice","joined","Acme","."],"subj_start":2 if reverse else 0,
            "subj_end":2 if reverse else 0,"obj_start":0 if reverse else 2,"obj_end":0 if reverse else 2,
            "subj_type":types[0],"obj_type":types[1],"relation":label}


@pytest.fixture
def fitted():
    return m.fit_type_filter([record(label,identity=str(i)) for i,label in enumerate(
                ("no_relation","per:employee_of","per:schools_attended"))],split="train")


def test_actual_cached_pdf_all_templates_and_1600_cells():
    result = verify_paper(CACHE)
    assert result["templates_checked"]==40 and result["meta_class_cells_checked"]==1600
    assert result["paper_cell_codes"]=={"contradiction":110,"neutral":1450,"entailment":40}
    assert result["model_runtime_verified"] is False and result["network_requests"]==0


def test_all_canonical_ids_match_exact_pinned_upstream_mapping():
    path = ROOT/".external/retacred/improved/prepro.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=="RETACREDProcessor")
    assignment = next(node for node in ast.walk(cls) if isinstance(node,ast.Assign)
                      and isinstance(node.targets[0],ast.Attribute) and node.targets[0].attr=="LABEL_TO_ID")
    source_mapping = ast.literal_eval(assignment.value)
    assert dict(zip(m.RELATIONS,range(40)))==source_mapping


def test_matrix_symmetry_counts_and_nonexclusive_positive_neutral():
    cells = [m.nli_target(a,b) for a,b in itertools.product(m.RELATIONS,repeat=2)]
    assert Counter(cells)=={"contradiction":110,"neutral":1450,"entailment":40}
    for a,b in itertools.product(m.RELATIONS,repeat=2):
        assert m.nli_target(a,b)==m.nli_target(b,a)
    assert m.nli_target("per:employee_of","per:schools_attended")=="neutral"
    assert m.nli_target("per:parents","per:children")=="contradiction"
    assert m.nli_target("org:members","org:member_of")=="contradiction"


@pytest.mark.parametrize("split",["valid","dev","test","TRAIN",None])
def test_reject_non_train_before_iterating(split):
    def forbidden():
        raise AssertionError("non-train data was touched")
        yield
    with pytest.raises(ValueError,match="train only"):
        m.fit_type_filter(forbidden(),split=split)


def test_train_filter_only_seen_ordered_pairs_and_integrity_roundtrip(fitted):
    assert fitted.candidates("PERSON","ORGANIZATION")==("no_relation","per:employee_of","per:schools_attended")
    assert fitted.candidates("ORGANIZATION","PERSON")==()
    assert fitted.candidates("PERSON","UNSEEN")==()
    payload = json.loads(json.dumps(fitted.to_dict()))
    assert m.FeasibleTypes.from_dict(payload)==fitted
    payload["pairs"]["per:spouse"].append(["PERSON","ORGANIZATION"])
    with pytest.raises(ValueError,match="changed train-only"):
        m.FeasibleTypes.from_dict(payload)


@pytest.mark.parametrize("kind",["duplicate","empty","unknown"])
def test_invalid_training_not_silently_dropped(kind):
    rows = [record(),record()] if kind=="duplicate" else [] if kind=="empty" else [record("missing")]
    with pytest.raises(ValueError):
        m.fit_type_filter(rows,split="train")


def test_inference_never_reads_gold_and_preserves_surfaces(fitted):
    class NoGold(dict):
        def __getitem__(self,key):
            if key=="relation":
                raise AssertionError("inference accessed target")
            return super().__getitem__(key)
        def get(self,key,*args):
            if key=="relation":
                raise AssertionError("inference accessed target")
            return super().get(key,*args)
    raw = NoGold(record())
    before = copy.deepcopy(dict(raw))
    first = m.inference_group(raw,fitted)
    second = m.inference_group(dict(raw,relation="per:spouse"),fitted)
    assert first==second and dict(raw)==before
    assert first["premise"]=="Alice joined Acme ."
    assert first["subject"]["text"]=="Alice" and first["object"]["text"]=="Acme"
    assert "relation" not in first and "target_relation" not in first
    assert first["candidates"][1]["hypothesis"]=="Alice is an employee of Acme."


def test_reverse_roles_multiword_and_literal_template_braces():
    raw = record(reverse=True,types=("ORGANIZATION","PERSON"))
    fields = m.record_input(raw)
    assert fields["subject"]["text"]=="Acme" and fields["object"]["text"]=="Alice"
    raw = record(); raw.update(token=["Alice","Jones","joined","Acme","Corp","."],subj_end=1,obj_start=3,obj_end=4)
    fitted = m.fit_type_filter([raw],split="train")
    group = m.inference_group(raw,fitted)
    assert group["candidates"][0]["hypothesis"]=="Alice Jones is an employee of Acme Corp."
    raw["token"][0]="{object}"
    assert m.inference_group(raw,fitted)["candidates"][0]["hypothesis"].startswith("{object} Jones")


def test_paper_dissolved_omits_object_not_repaired():
    assert m.TEMPLATES["org:dissolved"].format(subject="Acme",object="2001")=="Acme has been dissolved."


@pytest.mark.parametrize("change",[dict(subj_start=-1),dict(subj_start=True),dict(obj_start=0,obj_end=0),
                                  dict(subj_type=""),dict(token=[""]),dict(id="")])
def test_invalid_input_rejected(change):
    with pytest.raises(ValueError):
        m.record_input(dict(record(),**change))


def test_training_target_separate_and_wrong_filter_rejected(fitted):
    train = m.training_group(record(),fitted)
    assert [row["nli_target"] for row in train["candidates"]]==["contradiction","entailment","neutral"]
    with pytest.raises(ValueError,match="training gold excluded"):
        m.training_group(record("per:spouse"),fitted)


@pytest.mark.parametrize("order",list(itertools.permutations(m.NLI_LABELS)))
def test_grouping_uses_explicit_checkpoint_nli_mapping(fitted,order):
    group = m.inference_group(record(),fitted)
    base = [[.1,.2,.7],[.05,.1,.85],[.6,.3,.1]]
    probabilities = [[row[m.NLI_LABELS.index(name)] for name in order] for row in base]
    result = m.grouped_prediction(group,probabilities,nli_order=order)
    assert result["relation"]=="per:employee_of" and result["confidence"]==.85


def test_not_global_max_entail_probability_when_argmax_is_not_entail(fitted):
    group = m.inference_group(record(),fitted)
    result = m.grouped_prediction(group,[[.41,.19,.40],[.34,.31,.35],[.8,.1,.1]],nli_order=m.NLI_LABELS)
    assert result["relation"]=="per:employee_of" and result["confidence"]==.35


def test_abstention_empty_groups_and_no_forced_null(fitted):
    group = m.inference_group(record(),fitted)
    decision = m.grouped_prediction(group,[[.7,.2,.1]]*3,nli_order=m.NLI_LABELS)
    assert decision["relation"] is None and decision["reason"]=="no_entail_prediction"
    assert decision["fallback_relation_added"] is False
    empty = m.inference_group(record(types=("NEW","TYPE")),fitted)
    assert not empty["candidates"]
    decision = m.grouped_prediction(empty,[],nli_order=m.NLI_LABELS)
    assert decision["relation"] is None and decision["reason"]=="no_feasible_hypotheses"


def test_first_checkpoint_index_and_first_canonical_relation_ties(fitted):
    group = m.inference_group(record(),fitted)
    tied_nli = m.grouped_prediction(group,[[.5,0,.5]]*3,nli_order=m.NLI_LABELS)
    assert tied_nli["relation"] is None
    tied_relations = m.grouped_prediction(group,[[.1,.1,.8]]*3,nli_order=m.NLI_LABELS)
    assert tied_relations["relation"]=="no_relation"


@pytest.mark.parametrize("bad",[[math.nan,.1,.9],[math.inf,0,0],[-.1,.1,1],[.2,.2,.2],[True,0,0],[.1,.9]])
def test_invalid_probabilities_not_repaired(fitted,bad):
    with pytest.raises(ValueError):
        m.grouped_prediction(m.inference_group(record(),fitted),[bad]*3,nli_order=m.NLI_LABELS)


def test_misalignment_unknown_nli_and_noncanonical_order_rejected(fitted):
    group = m.inference_group(record(),fitted)
    with pytest.raises(ValueError):
        m.grouped_prediction(group,[],nli_order=m.NLI_LABELS)
    with pytest.raises(ValueError):
        m.grouped_prediction(group,[[.1,.1,.8]]*3,nli_order=(0,1,2))
    group["candidates"].reverse()
    with pytest.raises(ValueError,match="canonical relation order"):
        m.grouped_prediction(group,[[.1,.1,.8]]*3,nli_order=m.NLI_LABELS)


def test_all_examples_denominator_abstentions_and_negative_predictions():
    result = m.audit_decisions(["per:employee_of","per:schools_attended"],["per:employee_of",None])
    assert result["examples"]==2 and result["fn"]==1 and result["coverage"]==.5
    assert result["accuracy"]==.5 and result["micro_f1"]==pytest.approx(2/3)
    result = m.audit_decisions(["no_relation","no_relation","per:employee_of"],[None,"per:employee_of","per:schools_attended"])
    assert result["fp"]==2 and result["fn"]==1 and result["examples"]==3


def test_no_abstention_counts_match_existing_canonical_metric():
    from baselines.retacred.common import metrics
    gold,predicted = [0,7,30,7,30],[0,7,7,0,30]
    existing = metrics(gold,predicted,list(m.RELATIONS),0.)
    actual = m.audit_decisions([m.RELATIONS[i] for i in gold],[m.RELATIONS[i] for i in predicted])
    assert actual["accuracy"]==existing["accuracy"]
    assert actual["micro_f1"]==existing["micro_f1"]


def test_toy_cli_no_backend_weight_network_or_task_data(tmp_path):
    code = '''
import runpy,sys
def audit(event,args):
    if event=='import' and args[0].split('.')[0] in ('torch','transformers','numpy'):
        raise AssertionError('backend import in primitive toy')
    if event.startswith('socket.'):
        raise AssertionError('network in primitive toy')
    if event=='open' and (str(args[0]).endswith(('.bin','.safetensors','.jsonl')) or '/data/relation/' in str(args[0])):
        raise AssertionError('weights or task data in primitive toy')
sys.addaudithook(audit)
runpy.run_module('baselines.retacred.metaentail',run_name='__main__')
'''
    output = tmp_path/"toy.json"
    done = subprocess.run([sys.executable,"-c",code,"--toy","--output",str(output)],cwd=ROOT,
                          env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1"),capture_output=True,text=True,timeout=30)
    assert done.returncode==0,done.stdout+done.stderr
    result = json.loads(output.read_text())
    assert result==json.loads(done.stdout) and result["status"]=="paper_primitives_verified"
    assert not result["model_implemented"] and not result["strict_comparison_eligible"]
    assert result["model_jobs_launched"]==0


def test_output_not_overwritten_and_admission_remains_closed(tmp_path):
    path = tmp_path/"preserved.json"
    path.write_text("preserved")
    with pytest.raises(SystemExit):
        m.main(["--toy","--output",str(path)])
    assert path.read_text()=="preserved"
    registry = cohort()
    assert len(registry["default_models"])==5 and "metaentail" not in registry["default_models"]
    row = next(row for row in registry["models"] if row["id"]=="metaentail")
    assert row["status"]=="blocked" and row["implementation_stage"]=="paper_primitives_prepared"
    with pytest.raises(ValueError):
        select_models("metaentail")
