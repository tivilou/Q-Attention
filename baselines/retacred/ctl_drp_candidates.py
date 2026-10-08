"""RECENT-derived train-only candidate primitive; no CTL neural certification."""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import tarfile

from .ctl_drp_paper import load_lock as paper_lock, regular, sha

PACKAGE = Path(__file__).resolve().parent
COMMIT = "4742c9e80517acbd1466a6fb3d5bcb4678e56c47"
RAW = "https://raw.githubusercontent.com/Saintfe/RECENT/" + COMMIT + "/"
PREPARES = ("SpanBERT/prepare_data.py", "gcn-multiple/prepare_data.py")
STATUS = "recent_aggregation_bound_ctl_exact_H_unconfirmed"

def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def aggregation_nodes(content: bytes) -> list:
    """Select only upstream r/s/o/t plus relation-set aggregation, never I/O/gold shortcuts."""
    tree = ast.parse(content)
    split_loops = [n for n in tree.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name)
                   and n.target.id == "data_type"]
    if len(split_loops) != 1 or ast.literal_eval(split_loops[0].iter) != ["train", "dev", "test"]:
        raise ValueError("source split loop differs")
    loops = [n for n in split_loops[0].body if isinstance(n, ast.For)
             and ast.unparse(n.iter) == "enumerate(data)"]
    if len(loops) != 1:
        raise ValueError("source aggregation loop differs")
    nodes = loops[0].body[1:6]
    expected = ["r = d['relation']", "s = d['subj_type']", "o = d['obj_type']", "t = (s, o)"]
    if len(nodes) != 5 or [ast.unparse(n) for n in nodes[:4]] != expected:
        raise ValueError("aggregation source window differs")
    if ast.unparse(nodes[4]) != "if t not in so2reltion:\n    so2reltion[t] = set((r,))\nelse:\n    so2reltion[t].add(r)":
        raise ValueError("aggregation source algorithm differs")
    return nodes

def source_fragment_reference(content: bytes, rows: list) -> dict:
    nodes = aggregation_nodes(content)
    # The selected AST shape is checked above. No import, open, eval, output path,
    # test-label branch, NA exclusion, direct prediction or model code is included.
    loop = ast.For(target=ast.Tuple(elts=[ast.Name(id="i", ctx=ast.Store()), ast.Name(id="d", ctx=ast.Store())], ctx=ast.Store()),
                   iter=ast.Call(func=ast.Name(id="enumerate", ctx=ast.Load()), args=[ast.Name(id="data", ctx=ast.Load())], keywords=[]),
                   body=copy.deepcopy(nodes), orelse=[])
    module = ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[]))
    scope = {"__builtins__": {"enumerate": enumerate, "set": set}, "data": copy.deepcopy(rows), "so2reltion": {}}
    exec(compile(module, "<pinned-recent-aggregation-only>", "exec"), scope)
    return scope["so2reltion"]

def derive_source_facts(contents: dict[str, bytes]) -> dict:
    repo = json.loads(contents["https://api.github.com/repos/Saintfe/RECENT"])
    commit = json.loads(contents["https://api.github.com/repos/Saintfe/RECENT/commits/HEAD"])
    tree = json.loads(contents["https://api.github.com/repos/Saintfe/RECENT/git/trees/HEAD?recursive=1"])
    if repo["full_name"] != "Saintfe/RECENT" or commit["sha"] != COMMIT or tree["truncated"] is not False:
        raise ValueError("RECENT source identity differs")
    entries = {x["path"]: x for x in tree["tree"] if x["type"] == "blob"}
    for url, content in contents.items():
        if url.startswith(RAW):
            name = url[len(RAW):]
            blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
            if entries[name]["sha"] != blob or entries[name]["size"] != len(content):
                raise ValueError("selected source blob differs")
    prep = [contents[RAW + p] for p in PREPARES]
    # ast.dump includes version-dependent optional/empty fields. The exact
    # validated statement text above is shared by supported Python versions.
    ast_hashes = [hashlib.sha256("\n".join(ast.unparse(n) for n in aggregation_nodes(p)).encode()).hexdigest() for p in prep]
    if ast_hashes[0] != ast_hashes[1]:
        raise ValueError("SpanBERT/GCN aggregation algorithm differs")
    for p in prep:
        text = ast.unparse(ast.parse(p))
        if ("if len(so2reltion[t]) == 2:" not in text or "r.remove('no_relation')" not in text
                or "r = r.pop()" not in text or "if data_type == 'test':" not in text
                or "out_f.write('%d %s\\n' % (s['order_id'], r))" not in text
                or "if r == 'no_relation':\n            continue" not in text):
            raise ValueError("heldout gold-derived shortcut witness differs")
    constants = {}
    for n in ast.parse(contents[RAW + "gcn-multiple/utils/constant.py"]).body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ("ID_TO_TYPE_PAIR", "LABEL_TO_ID"):
            constants[n.targets[0].id] = ast.literal_eval(n.value)
    positive = set.union(*(set(v) for v in constants["LABEL_TO_ID"].values()))
    canonical = set(paper_lock()["canonical_labels"])
    return {"source_commit": COMMIT, "source_tree": commit["commit"]["tree"]["sha"],
            "aggregation_normalized_source_sha256": ast_hashes[0], "source_task": "TACRED_not_Re_TACRED",
            "constant_multiclass_groups": len(constants["ID_TO_TYPE_PAIR"]),
            "constant_positive_labels": len(positive),
            "constant_canonical_missing": sorted(canonical - {"no_relation"} - positive),
            "constant_extra_labels": sorted(positive - canonical),
            "heldout_gold_aggregation_and_direct_semantic_shortcut_observed": True,
            "negative_heldout_rows_excluded_from_semantic_groups": True,
            "ctl_exact_H_author_mapping_recovered": False}

def load_lock(path: Path = PACKAGE / "ctl_drp_candidates.lock.json") -> dict:
    lock = json.loads(regular(path).read_text(encoding="utf-8"))
    if (lock["schema"] != "retacred-ctl-drp-candidate-primitive.v1" or lock["source_commit"] != COMMIT
            or lock["formal_comparison_admitted"] is not False or lock["ctl_exact_H_author_mapping_recovered"] is not False
            or lock["source_task"] != "TACRED_not_Re_TACRED" or lock["status"] != STATUS
            or len(lock["artifacts"]) != 12 or len({a["url"] for a in lock["artifacts"]}) != 12
            or len({a["path"] for a in lock["artifacts"]}) != 12
            or any(Path(a["path"]).name != a["path"] for a in lock["artifacts"])):
        raise ValueError("candidate provenance/admission boundary differs")
    return lock

def verify_cache(cache: Path, lock: dict) -> dict:
    cache = cache.absolute()
    if not cache.is_dir() or any(p.is_symlink() for p in (cache, *cache.parents)):
        raise ValueError("real source cache required")
    receipt = regular(cache / "collection.json")
    if sha(receipt) != lock["collection_sha256"]:
        raise ValueError("receipt hash differs")
    records = json.loads(receipt.read_text())
    if len(records) != 12:
        raise ValueError("collection scope differs")
    contents = {}
    archive_path = None
    for actual, expected in zip(records, lock["artifacts"]):
        if (any(actual.get(k) != v for k, v in expected.items()) or actual["status"] != 200
                or actual["method"] != "GET" or actual["retry_count"] != 0 or actual["access"] != "public_https"):
            raise ValueError("source collection provenance differs")
        p = regular(cache / expected["path"])
        if p.stat().st_size != expected["bytes"] or sha(p) != expected["sha256"]:
            raise ValueError("source evidence hash/size differs")
        contents[expected["url"]] = p.read_bytes()
        if expected["url"] == "https://arxiv.org/src/2105.08393v1":
            archive_path = p
    with tarfile.open(archive_path, "r:gz") as a:
        member = a.getmember(lock["native_tex"]["member"])
        if not member.isfile() or member.size != lock["native_tex"]["bytes"]:
            raise ValueError("native source member differs")
        native = a.extractfile(member).read(member.size + 1)
    if hashlib.sha256(native).hexdigest() != lock["native_tex"]["sha256"]:
        raise ValueError("native source hash differs")
    if ("aggregating the relations in the group" not in native.decode()
            or "S(r)" not in native.decode() or "Saintfe/RECENT" not in native.decode()):
        raise ValueError("paper aggregation/code citation differs")
    facts = derive_source_facts(contents)
    if facts != lock["facts"]:
        raise ValueError("source facts differ")
    rows = [{"subj_type": "PERSON", "obj_type": "ORGANIZATION", "relation": r}
            for r in ("no_relation", "per:employee_of", "per:schools_attended")]
    reference = source_fragment_reference(contents[RAW + PREPARES[0]], rows)
    table = fit_train_pairs(rows, source_split="train")
    if set(candidate_labels(table, "PERSON", "ORGANIZATION")) != reference[("PERSON", "ORGANIZATION")]:
        raise ValueError("actual aggregation fragment fixture differs")
    return {"originals_verified": 12, "source_aggregation_fragment_executed": True,
            "whole_upstream_script_executed": False, "neural_model_executed": False}

def fit_train_pairs(rows: list, *, source_split: str) -> dict:
    """Declared train-only observed-pair construction, not the unknown author H.

    Split identity still requires a qualified dataset manifest at runner binding.
    No lookup uses gold labels, and no per-relation Cartesian closure is applied.
    """
    if source_split != "train" or not isinstance(rows, list) or not rows:
        raise ValueError("nonempty declared training rows required; heldout fitting rejected")
    labels = paper_lock()["canonical_labels"]
    supports = {}
    semantic = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("training row must be a mapping")
        s, o, r = (row.get(k) for k in ("subj_type", "obj_type", "relation"))
        if any(not isinstance(v, str) or not v or v.strip() != v for v in (s, o, r)) or r not in labels:
            raise ValueError("exact types and canonical relation required; no alias repair")
        supports.setdefault((s, o), set()).add(r)
        semantic.append([s, o, r])
    pairs = [{"subject_type": s, "object_type": o,
              "relations": sorted(values | {"no_relation"}, key=labels.__getitem__)}
             for (s, o), values in sorted(supports.items())]
    result = {"schema": "ctl-observed-train-pairs.v1", "source_split": "train",
            "construction": "exact_observed_type_pair_support_plus_unconditional_NA",
            "train_semantic_sha256": fingerprint(sorted(semantic)), "training_rows": len(rows), "pairs": pairs,
            "physical_split_identity_verified": False, "ctl_exact_H_author_mapping_recovered": False,
            "formal_comparison_admitted": False,
            "unobserved_canonical_relations": sorted(set(labels) - set.union(*supports.values()) - {"no_relation"})}
    result["table_sha256"] = fingerprint(result)
    return result

def candidate_labels(table: dict, subject_type: str, object_type: str) -> list:
    """Only a frozen table and entity types enter lookup; missing pairs stay diagnostic."""
    if (table.get("table_sha256") != fingerprint({k: v for k, v in table.items() if k != "table_sha256"})
            or table.get("schema") != "ctl-observed-train-pairs.v1" or table.get("source_split") != "train"
            or table.get("construction") != "exact_observed_type_pair_support_plus_unconditional_NA"
            or table.get("formal_comparison_admitted") is not False
            or table.get("ctl_exact_H_author_mapping_recovered") is not False):
        raise ValueError("candidate table contract differs")
    if any(not isinstance(t, str) or not t or t.strip() != t for t in (subject_type, object_type)):
        raise ValueError("exact nonempty type strings required")
    entries = [p for p in table["pairs"] if (p["subject_type"], p["object_type"]) == (subject_type, object_type)]
    if not entries:
        raise LookupError("unseen training type pair; explicit H policy required")
    if len(entries) != 1:
        raise ValueError("duplicate type pair")
    labels = paper_lock()["canonical_labels"]
    relations = entries[0]["relations"]
    if (not isinstance(relations, list) or "no_relation" not in relations
            or any(not isinstance(r, str) or r not in labels for r in relations)
            or len(set(relations)) != len(relations) or sorted(relations, key=labels.__getitem__) != relations):
        raise ValueError("canonical candidate/NA/order boundary differs")
    return list(relations)

def coverage_diagnostic(table: dict, rows: list) -> dict:
    """Separate label-using diagnostic; never changes candidate tables or predictions."""
    labels = paper_lock()["canonical_labels"]
    unknown, excluded = [], []
    for i, row in enumerate(rows):
        if row.get("relation") not in labels:
            raise ValueError("canonical diagnostic gold required")
        try:
            candidates = candidate_labels(table, row["subj_type"], row["obj_type"])
        except LookupError:
            unknown.append(i)
            continue
        if row["relation"] not in candidates:
            excluded.append(i)
    return {"samples": len(rows), "unknown_pair_indices": unknown, "excluded_gold_indices": excluded,
            "candidates_modified": False, "gold_reinserted": False}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-cache", type=Path)
    args = parser.parse_args()
    try:
        lock = load_lock()
        evidence = verify_cache(args.verify_cache, lock) if args.verify_cache else {"originals_verified": 0}
        result = {"status": STATUS, "facts": lock["facts"], "cache_evidence": evidence,
                  "formal_comparison_admitted": False, "ctl_exact_H_author_mapping_recovered": False,
                  "model_jobs": 0, "network_requests": 0, "weights_loaded": False}
    except (ValueError, KeyError, OSError, SyntaxError, tarfile.TarError) as error:
        print(json.dumps({"status": "invalid_evidence", "error": str(error), "formal_comparison_admitted": False}))
        return 1
    print(json.dumps(result, indent=2))
    return 2

if __name__ == "__main__":
    raise SystemExit(main())
