"""Paper/scalar gates only; no pretrained encoder or formal training."""
import copy
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import pytest
from baselines.retacred import common, ctl_drp_paper as paper

CACHE = Path(os.environ.get("CTL_DRP_EVIDENCE_CACHE",str(common.ROOT/".external/retacred/ctl-drp-paper-20261008I")))
H = {"no_relation":[1.0,-.5],"a":[-.2,.8],"b":[.4,.1],"excluded":[999.0,999.0]}
W = [.3,-.7]
ALLOWED = ["no_relation","a","b"]

def tex():
    lock = paper.load_lock()
    row = next(a for a in lock["artifacts"] if "/src/" in a["url"])
    with tarfile.open(CACHE/row["path"],"r:gz") as archive:
        return archive.extractfile(lock["native_tex"]["member"]).read().decode()

def test_actual_originals_equation_recipe_and_canonical_coverage():
    result = paper.verify_cache(CACHE,paper.load_lock())
    assert result["originals"] == 12 and result["history_trees"] == 5
    assert result["coverage"]["paper_descriptions"] == 46
    assert result["coverage"]["canonical_covered"] == 40
    assert len(result["coverage"]["extra_canonicalized_names"]) == 6
    assert not result["coverage"]["type_predicate_recovered"]
    assert not result["source_executed"] and not result["archive_extracted"] and not result["checkpoint_loaded"]

def test_sup_unsup_is_initialization_not_label_free_training():
    result = paper.audit(paper.load_lock(),common.cohort(),CACHE)
    assert result["sup_unsup_difference"] == "pretrained_initialization_not_downstream_label_supervision"
    assert result["recipe"] == {"batch_size":512,"learning_rate":1e-5,"epochs":10,"batch_unit":"unbound"}
    assert result["variant_scores"]["sup"]["f1"] == 91.6
    assert result["variant_scores"]["unsup"]["f1"] == 91.2
    assert not result["neural_runtime_verified"] and not result["formal_comparison_admitted"]

def test_closed_form_equation2_against_direct_finite_scores():
    result = paper.candidate_reference(H,W,1.4,ALLOWED,"a")
    scores = [sum(a*b for a,b in zip(H[k],W))+1.4 for k in ALLOWED]
    expected = [math.exp(z)/sum(map(math.exp,scores)) for z in scores]
    assert list(result["probabilities"].values()) == pytest.approx(expected,abs=1e-14)
    assert result["loss"] == pytest.approx(-math.log(expected[1]),abs=1e-14)
    assert "excluded" not in result["probabilities"]
    assert result["denominator_scope"] == "one_sentence_allowed_candidates_plus_na"

def test_analytic_weight_and_embedding_gradients_against_finite_difference():
    result = paper.candidate_reference(H,W,.2,ALLOWED,"a")
    epsilon = 1e-5
    for j in range(len(W)):
        plus,minus = W[:],W[:]
        plus[j] += epsilon
        minus[j] -= epsilon
        numeric = (paper.candidate_reference(H,plus,.2,ALLOWED,"a")["loss"]
                   -paper.candidate_reference(H,minus,.2,ALLOWED,"a")["loss"])/(2*epsilon)
        assert result["weight_gradient"][j] == pytest.approx(numeric,abs=1e-9)
    for key in ALLOWED:
        for j in range(len(W)):
            plus,minus = copy.deepcopy(H),copy.deepcopy(H)
            plus[key][j] += epsilon
            minus[key][j] -= epsilon
            numeric = (paper.candidate_reference(plus,W,.2,ALLOWED,"a")["loss"]
                       -paper.candidate_reference(minus,W,.2,ALLOWED,"a")["loss"])/(2*epsilon)
            assert result["embedding_gradients"][key][j] == pytest.approx(numeric,abs=1e-9)

def test_shared_bias_cancels_and_extreme_logits_remain_stable():
    zero = paper.candidate_reference(H,W,0,ALLOWED,"a")
    large = paper.candidate_reference(H,W,1e100,ALLOWED,"a")
    assert zero["probabilities"] == large["probabilities"] and zero["loss"] == large["loss"]
    assert zero["bias_gradient"] == large["bias_gradient"] == 0
    result = paper.candidate_reference({"no_relation":[1000.0],"a":[-1000.0]},[1.0],.3,["no_relation","a"],"a")
    assert result["loss"] == 2000 and sum(result["probabilities"].values()) == 1

def test_na_only_and_ties_do_not_invent_a_tie_policy():
    only = paper.candidate_reference(H,W,2,["no_relation"],"no_relation")
    assert only["loss"] == 0 and only["probabilities"] == {"no_relation":1}
    assert only["weight_gradient"] == [0,0]
    tied = paper.candidate_reference({"no_relation":[1],"a":[1]},[2],.3,["a","no_relation"])
    assert tied["maximizers"] == ["a","no_relation"]
    assert "loss" not in tied

def test_candidate_scope_is_not_gold_conditioned_or_cross_sentence():
    left = paper.candidate_reference(H,W,.2,ALLOWED,"a")
    right = paper.candidate_reference(H,W,.2,ALLOWED,"b")
    assert left["probabilities"] == right["probabilities"]
    with pytest.raises(ValueError,match="no automatic reinsertion"):
        paper.candidate_reference(H,W,0,["no_relation"],"a")
    independently = paper.candidate_reference(H,W,0,["no_relation","a"],"a")
    assert independently["probabilities"]["a"] != left["probabilities"]["a"]

@pytest.mark.parametrize("bad", [float("nan"),float("inf"),True,"1.0"])
def test_invalid_numeric_inputs_rejected(bad):
    with pytest.raises(ValueError,match="finite numeric"):
        paper.candidate_reference(H,[bad,.2],0,ALLOWED,"a")

@pytest.mark.parametrize("allowed", [[],["a"],["no_relation","a","a"],["no_relation","unknown"]])
def test_candidate_contract_errors(allowed):
    with pytest.raises(ValueError):
        paper.candidate_reference(H,W,0,allowed,"a")

def test_paper_scope_negative_and_alias_coverage_errors():
    text = tex()
    facts = paper.paper_facts(text)
    with pytest.raises(ValueError,match="equation2"):
        paper.paper_facts(text.replace(r"\label{eq2}",r"\label{other}"))
    labels = paper.load_lock()["canonical_labels"]
    descriptions = facts["descriptions"].copy()
    del descriptions["no-relation"]
    with pytest.raises(ValueError,match="missing"):
        paper.description_coverage(descriptions,labels)
    descriptions = facts["descriptions"].copy()
    descriptions["no_relation"] = "second"
    with pytest.raises(ValueError,match="ambiguous"):
        paper.description_coverage(descriptions,labels)

def test_lock_does_not_fill_unbound_fields_or_promote_neural(tmp_path):
    for field in ("formal_comparison_admitted","reported_score_rank_eligible","neural_runtime_verified",
                  "type_predicate_recovered","checkpoint_identity_bound"):
        lock = paper.load_lock()
        lock[field] = True
        path = tmp_path/"lock.json"
        path.write_text(json.dumps(lock))
        with pytest.raises(ValueError,match="boundary"):
            paper.load_lock(path)
    lock = paper.load_lock()
    lock["unresolved"] = []
    path.write_text(json.dumps(lock))
    with pytest.raises(ValueError):
        paper.load_lock(path)

def test_original_hash_and_canonical_source_tampering(tmp_path):
    lock = paper.load_lock()
    for choice in ("pinned-prepro.py",lock["artifacts"][1]["path"]):
        directory = tmp_path/choice.replace(".","_")
        directory.mkdir()
        for name in ["collection.json","pinned-prepro.py"]+[a["path"] for a in lock["artifacts"]]:
            shutil.copyfile(CACHE/name,directory/name)
        with (directory/choice).open("ab") as stream:
            stream.write(b" ")
        with pytest.raises(ValueError,match="hash"):
            paper.verify_cache(directory,lock)

def test_cohort_preserved_and_both_ctl_variants_still_blocked():
    registry = common.cohort()
    assert len(registry["models"]) == 16 and len(common.select_models("ready")) == 5
    for model in paper.MODELS:
        with pytest.raises(ValueError,match="not runnable"):
            common.select_models(model)
    result = paper.audit(paper.load_lock(),registry)
    assert result["cache_evidence"]["status"] == "originals_not_verified_this_invocation"

def test_actual_cli_backend_and_network_tripwire():
    code = """import builtins,runpy,socket,sys
original=builtins.__import__
def guard(name,*args,**kwargs):
    if name.split('.')[0] in {'torch','transformers','requests'} or name in {'urllib.request','http.client'}:
        raise AssertionError('backend/network import: '+name)
    return original(name,*args,**kwargs)
builtins.__import__=guard
def deny(*args,**kwargs):
    raise AssertionError('network used')
socket.socket=deny
sys.argv=['audit','--verify-cache',sys.argv[1]]
runpy.run_module('baselines.retacred.ctl_drp_paper',run_name='__main__')
"""
    result = subprocess.run([sys.executable,"-c",code,str(CACHE)],cwd=common.ROOT,
                            text=True,capture_output=True,timeout=30)
    assert result.returncode == 2, result.stderr
    value = json.loads(result.stdout)
    assert value["status"] == "recipe_neural_unresolved"
    assert value["cache_evidence"]["originals"] == 12 and value["model_jobs"] == 0
    assert not value["backend_imported"] and value["network_requests"] == 0
