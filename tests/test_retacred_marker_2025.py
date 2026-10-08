"""Bounded offline evidence/negative gates; never instantiate a model."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import pytest

from baselines.retacred import common, marker_2025_audit as audit

CACHE = Path(os.environ.get("MARKER_2025_EVIDENCE_CACHE",
    str(common.ROOT / ".external/retacred/marker-2025-paper-20261008H")))

def tex():
    lock = audit.load_lock()
    path = next(x for x in lock["artifacts"] if "/src/" in x["url"])["path"]
    with tarfile.open(CACHE/path, "r:gz") as archive:
        return archive.extractfile(lock["source_tex"]["member"]).read().decode("utf-8")

def test_actual_originals_and_native_tables():
    lock = audit.load_lock()
    actual = audit.verify_cache(CACHE,lock)
    assert actual["originals"] == 6
    assert actual["native_tex_facts"]["percentage_scores"]["Re-TACRED"] == 91.53
    assert actual["native_tex_facts"]["retacred_counts"] == {"train":58465,"dev":19584,"test":13418,"classes":40}
    assert actual["repository_query_hits"] == [0,0]
    assert not actual["archive_extracted"] and not actual["source_executed"]

def test_input_family_is_not_full_architecture_or_recipe():
    lock = audit.load_lock()
    result = audit.audit(lock,common.cohort(),CACHE)
    assert result["architecture_equivalence"] == "input_family_only_not_certified"
    assert result["published"]["input_family"] == "entity_preserving_typed_punctuation"
    assert all(result["published"][key] is False for key in (
        "checkpoint_size_revision_bound","grid_or_selected_numeric_recipe_bound",
        "no_relation_denominator_bound","seed_aggregation_bound"))
    assert not result["source_search"]["global_source_absence_inferred"]
    assert not result["formal_comparison_admitted"] and result["model_jobs"] == 0

def test_no_cache_does_not_claim_originals_verified():
    result = audit.audit(audit.load_lock(), common.cohort())
    assert result["cache_evidence"]["status"] == "originals_not_verified_this_invocation"

@pytest.mark.parametrize("field", ["formal_comparison_admitted", "reported_score_rank_eligible",
                                  "scores_are_project_measurements"])
def test_lock_cannot_promote_missing_evidence(tmp_path,field):
    value = audit.load_lock()
    value[field] = True
    target = tmp_path/"lock.json"
    target.write_text(json.dumps(value))
    with pytest.raises(ValueError,match="boundary"):
        audit.load_lock(target)

def test_lock_preserves_missing_recipe_and_bounded_scope(tmp_path):
    original = audit.load_lock()
    for mutation in (lambda x:x["unresolved"].clear(),
                     lambda x:x["artifacts"][0].update(url="https://example.invalid/other"),
                     lambda x:x["source_search"].update(global_source_absence_inferred=True),
                     lambda x:x["published"].update(checkpoint_size_revision_bound=True)):
        value = copy.deepcopy(original)
        mutation(value)
        target = tmp_path/"lock.json"
        target.write_text(json.dumps(value))
        with pytest.raises(ValueError):
            audit.load_lock(target)

def test_actual_table_scope_and_missing_marker_statement():
    text = tex()
    facts = audit.parse_tex(text)
    assert list(facts["percentage_scores"].values()) == [74.84,83.09,91.53]
    with pytest.raises(ValueError,match="scope"):
        audit.parse_tex(text.replace("subj-type","unknown-type"))
    with pytest.raises(ValueError,match="table"):
        audit.parse_tex(text.replace("RoBERTa &", "OtherModel &"))
    with pytest.raises(ValueError,match="source link"):
        audit.parse_tex(text + "\nhttps://github.com/example/new-evidence\n")

def test_original_hash_and_receipt_tampering(tmp_path):
    lock = audit.load_lock()
    for choice in (lock["artifacts"][1]["path"], "collection.json"):
        destination = tmp_path/choice.replace(".","_")
        destination.mkdir()
        for name in ["collection.json"] + [a["path"] for a in lock["artifacts"]]:
            shutil.copyfile(CACHE/name,destination/name)
        with (destination/choice).open("ab") as out:
            out.write(b" ")
        with pytest.raises(ValueError,match="hash"):
            audit.verify_cache(destination,lock)

def test_symlink_cache_is_rejected(tmp_path):
    target = tmp_path/"alias"
    try:
        target.symlink_to(CACHE, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation unsupported on this platform")
    with pytest.raises(ValueError,match="real directory"):
        audit.verify_cache(target,audit.load_lock())

def test_cohort_admission_stays_closed_and_ready_five_unchanged():
    registry = common.cohort()
    assert len(registry["models"]) == 16 and len([r for r in registry["models"] if r["rank"] is not None]) == 15
    assert registry["default_models"] == ["typed_marker_punct","typed_marker","entity_marker_punct","entity_marker","ptr"]
    assert len(common.select_models("ready")) == 5
    with pytest.raises(ValueError,match="not runnable"):
        common.select_models(audit.MODEL_ID)
    with pytest.raises(ValueError,match="not runnable"):
        common.select_models("all")
    altered = copy.deepcopy(registry)
    row = next(r for r in altered["models"] if r["id"] == audit.MODEL_ID)
    row["architecture_equivalence"] = "exact_duplicate"
    with pytest.raises(ValueError,match="boundary"):
        audit.audit(audit.load_lock(),altered)

def test_actual_package_cli_no_backend_or_network():
    code = """import builtins, runpy, socket, sys
original = builtins.__import__
def guard(name,*args,**kwargs):
    if name.split('.')[0] in {'torch','transformers','requests'} or name in {'urllib.request','http.client'}:
        raise AssertionError('unexpected backend/network import: '+name)
    return original(name,*args,**kwargs)
builtins.__import__ = guard
def deny(*args,**kwargs):
    raise AssertionError('unexpected network')
socket.socket = deny
sys.argv = ['audit','--verify-cache',sys.argv[1]]
runpy.run_module('baselines.retacred.marker_2025_audit',run_name='__main__')
"""
    result = subprocess.run([sys.executable,"-c",code,str(CACHE)],cwd=common.ROOT,
        text=True,capture_output=True,timeout=30)
    assert result.returncode == 2, result.stderr
    output = json.loads(result.stdout)
    assert output["status"] == "paper_recipe_unverified"
    assert output["cache_evidence"]["status"] == "originals_and_native_tex_verified"
    assert output["network_requests"] == 0 and not output["backend_imported"]
