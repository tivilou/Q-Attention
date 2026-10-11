"""Read-only, hash-pinned Sequence Routing recipe audit; blocked candidates exit 2.

Static inspection only. Upstream imports, model construction and training are
never executed. Recovered historical files do not change source admission.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from .assets import reject_symlinks
from .common import digest, git, pin_source
from .sequence_resources import resource_directory, resource_lock, validate_resources


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _literal(node: ast.AST, symbols: dict):
    if isinstance(node, ast.Dict):
        values = {}
        for key, value in zip(node.keys, node.values):
            try:
                values[_literal(key, symbols)] = _literal(value, symbols)
            except (ValueError, TypeError, KeyError):
                continue  # Keep only defined literal facts; no factory execution.
        return values
    if isinstance(node, ast.Name):
        return symbols[node.id]
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "list" and len(node.args) == 1 and not node.keywords):
        return list(_literal(node.args[0], symbols))
    return ast.literal_eval(node)


def _assignments(statements: list[ast.stmt]) -> dict:
    symbols, args = {}, {}
    for statement in statements:
        if not isinstance(statement, ast.Assign):
            continue
        try:
            value = _literal(statement.value, symbols)
        except (ValueError, TypeError, KeyError):
            continue  # Do not execute factories or infer unresolved values.
        for target in statement.targets:
            if isinstance(target, ast.Name):
                symbols[target.id] = value
            elif isinstance(target, ast.Attribute) and ast.unparse(target.value) == "args":
                args[target.attr] = value
    return {"symbols": symbols, "args": args}


def source_facts(source: Path) -> dict:
    trees = {name: ast.parse((source / name).read_text(), filename=name)
             for name in resource_lock()["active_file_sha256"]}
    settings = _function(trees["Experiments.py"], "get_settings")
    defaults = _assignments(settings.body)["symbols"]["Train_Args"]
    classes = next(node for node in ast.walk(settings)
                   if isinstance(node, ast.If) and ast.unparse(node.test) == "args.data_type == 'retacred'")
    branch = next(node for node in ast.walk(_function(trees["Experiments.py"], "experiment_run"))
                  if isinstance(node, ast.If) and ast.unparse(node.test) == "str(experiment) == 'four'")
    four = _assignments(branch.body)["args"]
    batch_branch = next(node for node in branch.body if isinstance(node, ast.If)
                        and "head_conditional" in ast.unparse(node.test))
    four.update(_assignments(batch_branch.orelse)["args"])
    eval_function = _function(trees["Train_Eval.py"], "rc_eval")
    micro = [node for node in ast.walk(eval_function) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "f1_score"
             and any(keyword.arg == "average" and isinstance(keyword.value, ast.Constant)
                     and keyword.value.value == "micro" for keyword in node.keywords)]
    if not micro or any(any(keyword.arg == "labels" for keyword in call.keywords) for call in micro):
        raise ValueError("unrecognized Sequence Routing evaluator")
    imports = [node.module for node in ast.walk(trees["Experiments.py"])
               if isinstance(node, ast.ImportFrom)]
    return {
        "missing_abstraction_import": "Abstraction" in imports and not (source / "Abstraction.py").exists(),
        "settings_class_dimensions": _assignments(classes.body)["args"],
        "active_h3_mix_recipe": {key: four[key] for key in
                                  ("heads", "h3_embed_size", "abstract", "batch_size", "epcoh_n")},
        "learning_rate": defaults["lr"], "scheduler_step": defaults["scheduler_step"],
        "released_evaluator": {"average": "micro", "labels_argument": None,
                                "includes_no_relation": True,
                                "identity_on_single_label_classification": "accuracy"},
        "counter_files_generated_by_preprocessing": True,
        "clean_checkout_counter_bootstrap": "factory_reads_counter_before_preprocessing_writes_it"}


def audit(source: Path) -> dict:
    lock = resource_lock()
    for name, expected in lock["active_file_sha256"].items():
        path = source / name
        reject_symlinks(path)
        if digest(path) != expected:
            raise ValueError("Sequence Routing pinned file drift: " + name)
    if (git("rev-parse", "HEAD", cwd=source) != lock["active_commit"]
            or git("status", "--porcelain", "--untracked-files=no", cwd=source)):
        raise ValueError("Sequence Routing source commit/cleanliness mismatch")
    if git("remote", "get-url", "origin", cwd=source).removesuffix(".git") != lock["repository"]:
        raise ValueError("Sequence Routing source origin mismatch")
    historical = resource_directory()
    reject_symlinks(historical)
    recovered = (validate_resources(historical) if (historical / "Abstraction.py").exists()
                 else {"status": "not_recovered"})
    facts = source_facts(source)
    return {"schema": "sequence-routing-readiness.v1", "status": "not_verifiable",
            "source_commit": lock["active_commit"], "active_file_sha256": lock["active_file_sha256"],
            "historical_abstraction": recovered, "facts": facts,
            "unresolved_metadata": [name for name in lock["unresolved_resources"] if not (source / name).exists()],
            "paper_recipe": {"batch_size": 64, "epochs": 6, "learning_rate": 1e-5,
                             "evidence": "https://aclanthology.org/2025.icnlsp-1.31.pdf#page=11"},
            "blockers": ["canonical40_vs_source41_output_policy_unresolved",
                         "paper64x6_vs_released24x5_recipe_unresolved",
                         "counter_bootstrap_and_exact_label_token_order_unverified",
                         "unversioned_routing_dependency_and_missing_project_license",
                         "released_all_label_metric_vs_benchmark_non_null_metric"],
            "published_score_evaluator_binding": "unverified_do_not_rank_as_same_metric",
            "strict_comparison_eligible": False, "model_jobs_launched": 0,
            "source_modified": False, "historical_source_executed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    result = audit(pin_source("sequence_routing", fetch=False))
    print(json.dumps(result, indent=2))
    raise SystemExit(2)  # This audit is not an adapter or a runtime fidelity certificate.


if __name__ == "__main__":
    main()
