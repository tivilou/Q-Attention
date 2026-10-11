"""CTL-DRP paper primitives: explicit scalar equation, not a neural adapter."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import tarfile

PACKAGE = Path(__file__).resolve().parent
MODELS = ("ctl_drp_supervised", "ctl_drp_unsupervised")
ALIASES = {"org:top-members":"org:top_members/employees", "no-relation":"no_relation"}
UNRESOLVED = ["exact_sup_unsup_checkpoint_revision","complete_H_type_predicate",
    "batch_unit_accumulation_optimizer_scheduler_warmup","max_length_overflow_and_type_text_normalization",
    "special_token_initialization_and_first_token_pooler_projection","dropout_and_dev_checkpoint_selection",
    "exact_seed_list_and_report_aggregation","null_class_evaluator_denominator","neural_gradient_update_resume_and_numeric_cases"]

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def regular(path: Path) -> Path:
    if not path.is_file() or any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError("regular non-symlink evidence required: " + str(path))
    return path

def canonical_labels(content: bytes) -> dict:
    tree = ast.parse(content)
    classes = [n for n in tree.body if isinstance(n,ast.ClassDef) and n.name == "RETACREDProcessor"]
    if len(classes) != 1:
        raise ValueError("canonical label source identity differs")
    mappings = []
    for node in ast.walk(classes[0]):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr == "LABEL_TO_ID" for t in node.targets):
            mappings.append(ast.literal_eval(node.value))
    if (len(mappings) != 1 or len(mappings[0]) != 40
            or set(mappings[0].values()) != set(range(40)) or "no_relation" not in mappings[0]):
        raise ValueError("canonical40 mapping differs")
    return mappings[0]

def paper_facts(tex: str) -> dict:
    equations = re.findall(r"\\begin\{equation\}(.*?)\\end\{equation\}",tex,re.S)
    eq2 = [x for x in equations if r"\label{eq2}" in x]
    if len(eq2) != 1:
        raise ValueError("native equation2 binding missing")
    batch = re.search(r"batch size to (\d+) and learning rate to ([\de.-]+)",tex)
    epoch = re.search(r"trained in (\d+) epochs",tex)
    appendix = re.findall(r"\\multicolumn\{1\}\{\|l\|\}\{([^{}]+)\}\s*&\s*(.*?)\s*\\\\",tex)
    if (not batch or not epoch or len(appendix) != 46 or len({k for k,_ in appendix}) != 46
            or "sup-smi-roberta-large" not in tex or "unsup-smi-roberta-large" not in tex
            or "first subtoken from the last layer" not in tex
            or "several random seeds" not in tex
            or "probability of the positive relation" not in tex):
        raise ValueError("paper mechanism/recipe/description scope differs")
    tables = re.findall(r"\\begin\{table\}.*?\\end\{table\}",tex,re.S)
    target = [t for t in tables if r"\label{tab2}" in t]
    if len(target) != 1:
        raise ValueError("Re-TACRED score table binding missing")
    score_table = re.sub(r"\\textbf\{([^{}]+)\}",r"\1",target[0]).replace(r"\mbox{-}","-")
    scores = {}
    for variant in ("sup","unsup"):
        match = re.search(r"CTL-DRP_\{"+variant+r"\}\}\$\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)",score_table)
        if not match:
            raise ValueError("variant score binding missing: " + variant)
        scores[variant] = dict(zip(("precision","recall","f1"),map(float,match.groups())))
    return {"equation2":re.sub(r"\s+","",eq2[0]),
            "batch_size":int(batch[1]), "learning_rate":float(batch[2]), "epochs":int(epoch[1]),
            "initialization_literals":["sup-smi-roberta-large","unsup-smi-roberta-large"],
            "seed_aggregation":"mean_several_unspecified_random_seeds", "variant_scores":scores,
            "descriptions":dict(appendix)}

def description_coverage(descriptions: dict, labels: dict) -> dict:
    mapped = {}
    for key,text in descriptions.items():
        canonical = ALIASES.get(key,key.replace("-","_"))
        if canonical in mapped:
            raise ValueError("ambiguous paper alias: " + key)
        mapped[canonical] = text
    missing = sorted(set(labels)-set(mapped))
    if missing:
        raise ValueError("missing canonical descriptions: " + str(missing))
    return {"paper_descriptions":len(descriptions), "canonical_covered":len(labels),
            "extra_canonicalized_names":sorted(set(mapped)-set(labels)),
            "aliases":ALIASES, "normalization":"other_hyphens_to_underscores",
            "type_predicate_recovered":False}

def number(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError("finite numeric input required")
    return float(value)

def candidate_reference(embeddings: dict, weight: list, bias: float, allowed: list,
                        gold: str | None = None) -> dict:
    """One sentence's supplied H-candidates, not cross-sentence in-batch negatives.

    The shared bias cancels analytically; no unknown candidate predicate or tie
    policy is invented. This function is a scalar equation fixture only.
    """
    if (not isinstance(allowed,list) or not allowed or any(not isinstance(k,str) for k in allowed)
            or len(set(allowed)) != len(allowed) or "no_relation" not in allowed
            or any(k not in embeddings for k in allowed) or not weight):
        raise ValueError("explicit unique candidates including NA and embeddings required")
    if gold is not None and gold not in allowed:
        raise ValueError("gold excluded by supplied candidates; no automatic reinsertion")
    w = [number(x) for x in weight]
    number(bias)
    h = []
    for key in allowed:
        vector = embeddings[key]
        if not isinstance(vector,(list,tuple)) or len(vector) != len(w):
            raise ValueError("candidate embedding dimensions differ")
        h.append([number(x) for x in vector])
    scores = [sum(a*b for a,b in zip(vector,w)) for vector in h]
    if not all(math.isfinite(z) for z in scores):
        raise ValueError("nonfinite scalar score")
    maximum = max(scores)
    centered = [z-maximum for z in scores]
    if not all(math.isfinite(z) for z in centered):
        raise ValueError("score dynamic range exceeds scalar fixture")
    normalizer = sum(math.exp(z) for z in centered)
    p = [math.exp(z)/normalizer for z in centered]
    result = {"candidates":list(allowed), "probabilities":dict(zip(allowed,p)),
        "maximizers":[k for k,z in zip(allowed,scores) if z == maximum],
        "bias_cancels":True, "denominator_scope":"one_sentence_allowed_candidates_plus_na",
        "neural_model_executed":False}
    if gold is not None:
        target = allowed.index(gold)
        dz = [probability-(i == target) for i,probability in enumerate(p)]
        result.update(loss=math.log(normalizer)-centered[target],
            weight_gradient=[sum(dz[i]*h[i][j] for i in range(len(h))) for j in range(len(w))],
            bias_gradient=0.0,
            embedding_gradients={key:[delta*a for a in w] for key,delta in zip(allowed,dz)})
    return result

def load_lock(path: Path = PACKAGE/"ctl_drp_paper.lock.json") -> dict:
    lock = json.loads(regular(path).read_text(encoding="utf-8"))
    if (lock["schema"] != "retacred-ctl-drp-paper.v1" or lock["paper_id"] != "2304.04935v1"
            or lock["provenance_mode"] != "paper" or lock["formal_comparison_admitted"] is not False
            or lock["reported_score_rank_eligible"] is not False or lock["neural_runtime_verified"] is not False
            or lock["type_predicate_recovered"] is not False or lock["checkpoint_identity_bound"] is not False
            or lock["sup_unsup_difference"] != "pretrained_initialization_not_downstream_label_supervision"
            or lock["unresolved"] != UNRESOLVED
            or lock["canonical_label_source"]["commit"] != "fe00781116ab4c34cbcd9a2ae5b02513afae87ee"
            or lock["variant_scores"] != {"sup":{"precision":91.3,"recall":91.9,"f1":91.6},
                                          "unsup":{"precision":91.2,"recall":91.2,"f1":91.2}}
            or len(lock["artifacts"]) != 12 or len({a["path"] for a in lock["artifacts"]}) != 12
            or any(Path(a["path"]).name != a["path"] for a in lock["artifacts"])
            or lock["recipe"] != {"batch_size":512,"learning_rate":1e-5,"epochs":10,"batch_unit":"unbound"}
            or lock["coverage"]["canonical_covered"] != 40 or lock["coverage"]["paper_descriptions"] != 46):
        raise ValueError("paper mechanism/admission boundary differs")
    return lock

def verify_cache(cache: Path,lock: dict) -> dict:
    cache = cache.absolute()
    if any(p.is_symlink() for p in (cache,*cache.parents)) or not cache.is_dir():
        raise ValueError("real evidence cache required")
    receipt = regular(cache/"collection.json")
    if sha(receipt) != lock["collection_sha256"]:
        raise ValueError("receipt hash mismatch")
    observed = json.loads(receipt.read_text(encoding="utf-8"))
    if len(observed) != 12:
        raise ValueError("collection scope differs")
    source = None
    tree_count = 0
    for actual,expected in zip(observed,lock["artifacts"]):
        if (any(actual.get(k) != value for k,value in expected.items()) or actual["status"] != 200
                or actual["retry_count"] != 0 or actual["method"] != "GET" or actual["access"] != "public_https"):
            raise ValueError("collection provenance differs")
        path = regular(cache/expected["path"])
        if path.stat().st_size != expected["bytes"] or sha(path) != expected["sha256"]:
            raise ValueError("original evidence hash/size mismatch")
        if "/src/2304.04935v1" in expected["url"]:
            source = path
        if "/git/trees/" in expected["url"]:
            data = json.loads(path.read_text())
            paths = sorted(x["path"] for x in data["tree"])
            if data["truncated"] is not False or paths not in (["README.md"],["README.md","dataset"]):
                raise ValueError("bounded source tree differs")
            tree_count += 1
    if source is None or tree_count != 5:
        raise ValueError("native source/history scope differs")
    with tarfile.open(source,"r:gz") as archive:
        member = archive.getmember(lock["native_tex"]["member"])
        if not member.isfile() or member.size != lock["native_tex"]["bytes"]:
            raise ValueError("native TeX member identity differs")
        tex = archive.extractfile(member).read(member.size+1)
    if hashlib.sha256(tex).hexdigest() != lock["native_tex"]["sha256"]:
        raise ValueError("native TeX hash mismatch")
    facts = paper_facts(tex.decode("utf-8"))
    if (facts["equation2"] != lock["equation2"] or facts["variant_scores"] != lock["variant_scores"]
            or any(facts[k] != lock["recipe"][k] for k in ("batch_size","learning_rate","epochs"))):
        raise ValueError("paper equation/recipe/score binding differs")
    label_path = regular(cache/"pinned-prepro.py")
    if sha(label_path) != lock["canonical_label_source"]["sha256"]:
        raise ValueError("canonical source hash mismatch")
    labels = canonical_labels(label_path.read_bytes())
    if labels != lock["canonical_labels"]:
        raise ValueError("canonical label identity differs")
    coverage = description_coverage(facts["descriptions"],labels)
    if coverage != lock["coverage"]:
        raise ValueError("description coverage differs")
    return {"status":"originals_native_equation_recipe_and_coverage_verified", "originals":12,
            "history_trees":tree_count,"coverage":coverage,"source_executed":False,
            "archive_extracted":False,"checkpoint_loaded":False}

def audit(lock: dict,registry: dict,cache: Path | None = None) -> dict:
    for model in MODELS:
        rows = [x for x in registry["models"] if x["id"] == model]
        if (len(rows) != 1 or rows[0]["status"] != "blocked" or model in registry["default_models"]
                or rows[0]["strict_comparison_eligible"] is not False
                or rows[0]["reported_score_rank_eligible"] is not False
                or rows[0]["reported_f1"] != {MODELS[0]:.916, MODELS[1]:.912}[model]):
            raise ValueError("cohort variant/admission differs")
    evidence = verify_cache(cache,lock) if cache else {"status":"originals_not_verified_this_invocation"}
    return {"status":"recipe_neural_unresolved", "provenance_mode":"paper", "fidelity_status":"not_verifiable",
            "models":list(MODELS),"variant_scores":lock["variant_scores"],"recipe":lock["recipe"],
            "sup_unsup_difference":lock["sup_unsup_difference"],"cache_evidence":evidence,
            "unresolved":lock["unresolved"],"formal_comparison_admitted":False,
            "reported_score_rank_eligible":False,"neural_runtime_verified":False,
            "model_jobs":0,"backend_imported":False,"network_requests":0}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-cache",type=Path)
    args = parser.parse_args()
    try:
        lock = load_lock()
        registry = json.loads((PACKAGE/"cohort.json").read_text(encoding="utf-8"))
        result = audit(lock,registry,args.verify_cache)
    except (ValueError,KeyError,OSError,tarfile.TarError,SyntaxError) as error:
        print(json.dumps({"status":"invalid_evidence","error":str(error),"model_jobs":0,
                          "formal_comparison_admitted":False},indent=2))
        return 1
    print(json.dumps(result,indent=2))
    return 2

if __name__ == "__main__":
    raise SystemExit(main())
