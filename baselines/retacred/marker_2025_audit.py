"""Offline paper identity/input audit. No model, source execution or admission."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile

PACKAGE = Path(__file__).resolve().parent
MODEL_ID = "typed_marker_comparison_2025"
URLS = (
    "https://arxiv.org/abs/2509.11374", "https://arxiv.org/pdf/2509.11374",
    "https://arxiv.org/html/2509.11374v1", "https://arxiv.org/src/2509.11374v1",
    "https://api.github.com/search/repositories?q=2509.11374&per_page=20",
    "https://api.github.com/search/repositories?q=%22Transformer%20Enhanced%20Relation%20Classification%22&per_page=20",
)
UNRESOLVED = ("checkpoint_and_tokenizer_identity", "entity_type_normalization",
    "representation_pooling_classifier_loss", "optimizer_learning_rate_scheduler_warmup_dropout",
    "batch_accumulation_epoch_budget", "seed_set_and_aggregation", "grid_and_checkpoint_selection",
    "maximum_length_overflow", "label_order_and_no_relation_scorer_denominator")

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def regular(path: Path) -> Path:
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError("evidence must be a regular non-symlink file: " + str(path))
    return path

def load_lock(path: Path = PACKAGE / "marker_2025_evidence.lock.json") -> dict:
    value = json.loads(regular(path).read_text(encoding="utf-8"))
    if (value["schema"] != "retacred-marker-2025-paper-evidence.v1" or value["paper_id"] != "2509.11374v1"
            or value["provenance_mode"] != "paper" or value["status"] != "paper_recipe_unverified"
            or value["formal_comparison_admitted"] is not False
            or value["reported_score_rank_eligible"] is not False
            or value["scores_are_project_measurements"] is not False
            or type(value["model_jobs"]) is not int or value["model_jobs"] != 0
            or value["architecture_equivalence"] != "input_family_only_not_certified"
            or value["unresolved"] != list(UNRESOLVED)):
        raise ValueError("evidence identity/admission boundary differs")
    published = value["published"]
    if (published["table"] != 1 or published["model"] != "RoBERTa" or published["metric_name"] != "micro_f1"
            or published["input_family"] != "entity_preserving_typed_punctuation"
            or published["percentage_scores"] != {"TACRED":74.84,"TACREV":83.09,"Re-TACRED":91.53}
            or published["retacred_counts"] != {"train":58465,"dev":19584,"test":13418,"classes":40}
            or not published["retacred_training_and_evaluation"] or not published["grid_search_mentioned"]
            or any(published[k] is not False for k in ("grid_or_selected_numeric_recipe_bound",
                "no_relation_denominator_bound", "checkpoint_size_revision_bound", "seed_aggregation_bound"))):
        raise ValueError("paper fact/recipe boundary differs")
    if (tuple(a["url"] for a in value["artifacts"]) != URLS
            or len({a["path"] for a in value["artifacts"]}) != 6
            or any(Path(a["path"]).name != a["path"] or a["status"] != 200 for a in value["artifacts"])
            or value["source_search"]["repository_queries"] != list(URLS[-2:])
            or value["source_search"]["author_repository_identified"] is not False
            or value["source_search"]["global_source_absence_inferred"] is not False
            or value["source_search"]["total_counts"] != [0,0]
            or value["source_search"]["incomplete_results"] != [False,False]):
        raise ValueError("bounded evidence collection differs")
    return value

def parse_tex(text: str) -> dict:
    normalized = re.sub(r"\\textbf\{([^{}]+)\}", r"\1", text)
    score = re.search(r"^\s*RoBERTa\s*&\s*(\d+\.\d+)\s*&\s*(\d+\.\d+)\s*&\s*(\d+\.\d+)", normalized, re.M)
    counts = re.search(r"^\s*Re-TACRED\s*&\s*(\d+)\s*&\s*(\d+)\s*&\s*(\d+)\s*&\s*(\d+)", normalized, re.M)
    if not score or not counts:
        raise ValueError("native TeX table binding missing")
    if ("grid search" not in text or "subj-type" not in text or "obj-type" not in text
            or "without adding any new special tokens" not in text
            or "trained with Re-TACRED and also evaluate with it" not in text):
        raise ValueError("native TeX input/task/grid scope differs")
    if re.search(r"https?://github\.com/", text):
        raise ValueError("native TeX contains a source link requiring audit")
    return {"percentage_scores": dict(zip(("TACRED","TACREV","Re-TACRED"), map(float,score.groups()))),
            "retacred_counts": dict(zip(("train","dev","test","classes"), map(int,counts.groups())))}

def verify_cache(cache: Path, lock: dict) -> dict:
    cache = cache.absolute()
    if any(p.is_symlink() for p in (cache, *cache.parents)) or not cache.is_dir():
        raise ValueError("cache root must be a real directory")
    receipt_path = regular(cache / "collection.json")
    if digest(receipt_path) != lock["collection_sha256"]:
        raise ValueError("collection receipt hash mismatch")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if len(receipt) != len(lock["artifacts"]):
        raise ValueError("collection scope mismatch")
    queries = []
    source = None
    for expected, actual in zip(lock["artifacts"], receipt):
        if (any(actual.get(k) != v for k,v in expected.items()) or actual.get("method") != "GET"
                or actual.get("access") != "public_https" or actual.get("retry_count") != 0):
            raise ValueError("collection provenance mismatch")
        path = regular(cache / expected["path"])
        if path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise ValueError("original evidence hash/size mismatch: " + expected["path"])
        if "/search/repositories?" in expected["url"]:
            result = json.loads(path.read_text(encoding="utf-8"))
            if result["total_count"] != 0 or result["incomplete_results"] is not False or result["items"] != []:
                raise ValueError("exact public query result differs")
            queries.append(expected["url"])
        elif "/src/2509.11374v1" in expected["url"]:
            source = path
    if queries != lock["source_search"]["repository_queries"] or source is None:
        raise ValueError("query/source identity mismatch")
    with tarfile.open(source, "r:gz") as archive:
        members = [m for m in archive.getmembers() if m.name == lock["source_tex"]["member"]]
        if len(members) != 1 or not members[0].isfile() or members[0].size != lock["source_tex"]["bytes"]:
            raise ValueError("TeX member identity mismatch")
        # Read a single bounded member; never extract or execute the archive.
        content = archive.extractfile(members[0]).read(lock["source_tex"]["bytes"] + 1)
    if hashlib.sha256(content).hexdigest() != lock["source_tex"]["sha256"]:
        raise ValueError("TeX original hash mismatch")
    facts = parse_tex(content.decode("utf-8"))
    if any(facts[k] != lock["published"][k] for k in facts):
        raise ValueError("actual TeX and declared paper facts differ")
    return {"status":"originals_and_native_tex_verified", "originals":6, "repository_query_hits":[0,0],
            "native_tex_facts": facts, "archive_extracted":False, "source_executed":False}

def audit(lock: dict, registry: dict, cache: Path | None = None) -> dict:
    rows = [r for r in registry["models"] if r["id"] == MODEL_ID]
    if len(rows) != 1:
        raise ValueError("cohort identity differs")
    row = rows[0]
    if (row["status"] != "paper_recipe_unverified" or row["reported_f1"] != 0.9153
            or row["strict_comparison_eligible"] is not False or row["reported_score_rank_eligible"] is not False
            or row["architecture_equivalence"] != "unverified_beyond_typed_punctuation_input_family"
            or MODEL_ID in registry["default_models"]):
        raise ValueError("cohort admission/equivalence boundary differs")
    actual = verify_cache(cache,lock) if cache else {"status":"originals_not_verified_this_invocation"}
    return {"status":"paper_recipe_unverified", "model_id":MODEL_ID, "provenance_mode":"paper",
            "fidelity_status":"not_verifiable", "published":lock["published"], "cache_evidence":actual,
            "source_search":lock["source_search"], "unresolved":lock["unresolved"],
            "architecture_equivalence":lock["architecture_equivalence"], "formal_comparison_admitted":False,
            "reported_score_rank_eligible":False, "scores_are_project_measurements":False,
            "model_jobs":0, "backend_imported":False, "network_requests":0}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-cache", type=Path)
    args = parser.parse_args()
    try:
        lock = load_lock()
        registry = json.loads((PACKAGE / "cohort.json").read_text(encoding="utf-8"))
        output = audit(lock,registry,args.verify_cache)
    except (KeyError, ValueError, OSError, tarfile.TarError) as error:
        print(json.dumps({"status":"invalid_evidence", "error":str(error), "model_jobs":0,
                          "formal_comparison_admitted":False}, indent=2))
        return 1
    print(json.dumps(output, indent=2))
    return 2

if __name__ == "__main__":
    raise SystemExit(main())
