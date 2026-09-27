from __future__ import annotations

"""Summarize and rate a completed Q-EPVG three-seed replication."""

import argparse
import hashlib
import json
from math import sqrt
from pathlib import Path
from statistics import mean, stdev
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CONFIG_SCHEMA = "q-attention.q-epvg-formal-single-seed.v1"
EXPECTED_CASE_SCHEMA = "q-attention.Q-EPVG-case-study.v2"
EXPECTED_TRACE_CONTRACT = "q-attention.case-study-trace-contract.v3"
EXPECTED_SEEDS = (13, 29, 53)
SUPPORTED_MANIFEST_SCHEMAS = {
    "q-attention.q-epvg.formal-task-graph.v2",
    "q-attention.q-epvg.formal-multiseed-manifest.v1",
}
TRACE_VALIDATOR = ROOT / "scripts" / "validate_sample_trace.py"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _trace_errors(path: Path, expected_config_sha: str) -> list[str]:
    import importlib.util

    spec = importlib.util.spec_from_file_location("sample_trace_validator", TRACE_VALIDATOR)
    if spec is None or spec.loader is None:
        return [f"cannot load trace validator: {TRACE_VALIDATOR}"]
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        payload = load_json(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return [str(exc)]
    errors = list(module.validate(payload))
    experiment = payload.get("experiment")
    if not isinstance(experiment, dict) or experiment.get("config_sha256") != expected_config_sha:
        errors.append("experiment.config_sha256 does not match run_config.json")
    return errors


def _validate_group_completion(group_dir: Path) -> None:
    """Allow the scheduler's pre-marker summary pass without weakening export gates."""
    marker = group_dir / "MULTI_SEED_COMPLETE"
    if marker.is_file():
        return
    state_path = group_dir / "multi_seed_run_summary.json"
    if not state_path.is_file():
        raise ValueError(f"missing MULTI_SEED_COMPLETE: {group_dir}")
    state = load_json(state_path)
    if state.get("success") is not True:
        raise ValueError(f"missing MULTI_SEED_COMPLETE: {group_dir}")
    assignments = state.get("tasks", state.get("assignments"))
    if not isinstance(assignments, list) or not assignments:
        raise ValueError(f"missing MULTI_SEED_COMPLETE: {group_dir}")
    if any(not isinstance(item, dict) or item.get("status") != "complete" for item in assignments):
        raise ValueError(f"missing MULTI_SEED_COMPLETE: {group_dir}")


def _t_critical_95(df: int) -> float:
    values = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    }
    return values.get(df, 1.96)


def describe(values: list[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("cannot summarize an empty value list")
    average = mean(values)
    deviation = stdev(values) if len(values) > 1 else 0.0
    if len(values) > 1:
        margin = _t_critical_95(len(values) - 1) * deviation / sqrt(len(values))
        ci = [average - margin, average + margin]
    else:
        ci = [None, None]
    return {
        "values": values,
        "n": len(values),
        "mean": average,
        "std": deviation,
        "ci95": ci,
    }


def _validate_case_study(
    selector_dir: Path,
    *,
    selector: str,
    config: dict[str, Any],
    config_sha: str,
    allow_missing_tensor_files: bool,
) -> dict[str, Any]:
    case = load_json(selector_dir / "case_study.json")
    trace = load_json(selector_dir / "sample_trace.json")
    if case.get("schema_version") != EXPECTED_CASE_SCHEMA:
        raise ValueError(f"{selector}: expected Case Study schema {EXPECTED_CASE_SCHEMA}")
    if trace.get("schema_version") != "sample-trace.v1":
        raise ValueError(f"{selector}: expected sample-trace.v1")
    contract = trace.get("semantic_contract")
    if not isinstance(contract, dict) or contract.get("version") != EXPECTED_TRACE_CONTRACT:
        raise ValueError(f"{selector}: expected trace contract {EXPECTED_TRACE_CONTRACT}")
    experiment = trace.get("experiment")
    if not isinstance(experiment, dict) or experiment.get("config_sha256") != config_sha:
        raise ValueError(f"{selector}: sample-trace config hash mismatch")
    try:
        from q_epvg_report_export import (
            _validate_case_study_payload,
            _validate_source_tensor_files,
        )
        _validate_case_study_payload(case, trace, selector=selector)
        if not allow_missing_tensor_files:
            _validate_source_tensor_files(selector_dir.parent.parent, [selector])
    except Exception as exc:
        raise ValueError(f"{selector}: producer-owned Case Study validation failed: {exc}") from exc
    checkpoints = config.get("case_study", {}).get("checkpoints", [])
    records = config.get("case_study", {}).get("records", {})
    expected_count = sum(len(records.get(split, [])) for split in ("train", "valid", "test")) * len(checkpoints)
    cases = case.get("cases")
    if not isinstance(cases, list) or len(cases) != expected_count:
        raise ValueError(f"{selector}: expected {expected_count} Case Study cases")
    return {"case": case, "trace": trace, "count": expected_count}


def collect(group_dir: Path) -> dict[str, Any]:
    _validate_group_completion(group_dir)
    manifest = load_json(group_dir / "multi_seed_manifest.json")
    protocol_name = str(manifest.get("protocol", "q_epvg"))
    selectors = [str(item) for item in manifest.get("selectors", [])]
    seeds = [int(seed) for seed in manifest.get("seeds", [])]
    if seeds != list(EXPECTED_SEEDS):
        raise ValueError(f"manifest seeds must be {list(EXPECTED_SEEDS)}, found {seeds}")
    if manifest.get("schema_version") not in SUPPORTED_MANIFEST_SCHEMAS:
        raise ValueError("unsupported multi-seed manifest schema")
    commit_values: set[str] = set()
    protocol_values: set[str] = set()
    if not selectors or selectors[0] != "disabled" or len(set(selectors)) != len(selectors):
        raise ValueError("manifest selectors are invalid")
    rows: dict[str, list[dict[str, float]]] = {selector: [] for selector in selectors}
    seed_records: list[dict[str, Any]] = []
    for seed in seeds:
        seed_dir = group_dir / f"seed_{seed}"
        if not (seed_dir / "RUN_COMPLETE").is_file():
            raise ValueError(f"missing RUN_COMPLETE for seed {seed}")
        if (seed_dir / "RUN_FAILED").exists():
            raise ValueError(f"seed {seed} contains RUN_FAILED")
        summary = load_json(seed_dir / "run_summary.json")
        if summary.get("formal_experiment") is not True or summary.get("stage") != "formal_single_seed":
            raise ValueError(f"seed {seed} is not a formal single-seed run")
        if int(summary.get("seed", -1)) != seed:
            raise ValueError(f"seed summary mismatch for {seed}")
        if summary.get("selectors") != list(selectors):
            raise ValueError(f"seed {seed} selector set differs from frozen protocol")
        if summary.get("test_used_for_training_or_selection") is not False:
            raise ValueError(f"seed {seed} violates held-out test contract")
        provenance = summary.get("provenance")
        if not isinstance(provenance, dict) or provenance.get("git_dirty") is not False:
            raise ValueError(f"seed {seed} lacks clean provenance")
        commit = provenance.get("git_revision")
        if not isinstance(commit, str):
            raise ValueError(f"seed {seed} lacks git_revision")
        commit_values.add(commit)
        config_path = seed_dir / "run_config.json"
        config = load_json(config_path)
        if config.get("schema_version") != EXPECTED_CONFIG_SCHEMA:
            raise ValueError(f"seed {seed} has an unsupported Q-EPVG config schema")
        protocol_fingerprint = hashlib.sha256(
            json.dumps(
                {**{key: value for key, value in config.items() if key not in {"seed", "replication"}}, "seed": 0},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        protocol_values.add(protocol_fingerprint)
        trace_status: dict[str, str] = {}
        config_sha = sha256(config_path)
        for selector in tuple(item for item in selectors if item != "disabled"):
            result = _validate_case_study(
                seed_dir / "selectors" / selector,
                selector=selector,
                config=config,
                config_sha=config_sha,
                allow_missing_tensor_files=(seed_dir / "imported_report.json").is_file(),
            )
            trace_status[selector] = f"complete:{result['count']}"
        by_selector = {row.get("selector"): row for row in summary.get("rows", [])}
        for selector in selectors:
            row = by_selector.get(selector)
            if not isinstance(row, dict):
                raise ValueError(f"seed {seed} missing summary row {selector}")
            try:
                valid = float(row["valid"]["metrics"]["macro_f1"])
                test = float(row["test"]["metrics"]["macro_f1"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"seed {seed} has malformed {selector} metrics") from exc
            rows[selector].append({"valid_macro_f1": valid, "test_macro_f1": test})
        imported_path = seed_dir / "imported_report.json"
        imported = imported_path.is_file()
        imported_metadata = load_json(imported_path) if imported else None
        if imported:
            if not isinstance(imported_metadata, dict) or imported_metadata.get("validated_against_git_commit") != manifest.get("git_commit"):
                raise ValueError(f"seed {seed} imported report was not validated against this scheduler commit")
        elif commit != str(manifest.get("git_commit")):
            raise ValueError(f"seed {seed} commit differs from manifest commit")
        seed_records.append({"seed": seed, "git_revision": commit, "config_sha256": sha256(config_path), "trace_status": trace_status, "imported_report": imported, "import_metadata": imported_metadata})
    if any(
        not bool(record.get("imported_report")) and record["git_revision"] != str(manifest.get("git_commit"))
        for record in seed_records
    ):
        raise ValueError(f"fresh seed commits do not match manifest commit: {sorted(commit_values)}")
    if len(protocol_values) != 1:
        raise ValueError("seed configs differ beyond the declared seed field")

    baseline = rows["disabled"]
    candidate_selector = str(config.get("candidate"))
    classical_selector = str(config.get("matched_control"))
    structural_selector = str(config.get("structural_control"))
    for selector in (candidate_selector, classical_selector, structural_selector):
        if selector not in rows:
            raise ValueError(f"configured selector is absent from the replication: {selector}")
    q = rows[candidate_selector]
    classical = rows[classical_selector]
    q_deltas = [q_item["test_macro_f1"] - base["test_macro_f1"] for q_item, base in zip(q, baseline, strict=True)]
    classical_deltas = [c_item["test_macro_f1"] - base["test_macro_f1"] for c_item, base in zip(classical, baseline, strict=True)]
    classical_relative = [delta / abs(base["test_macro_f1"]) if base["test_macro_f1"] else None for delta, base in zip(classical_deltas, baseline, strict=True)]
    if any(value is None for value in classical_relative):
        raise ValueError("cannot compute classical relative gain with zero baseline")
    q_relative = [delta / abs(base["test_macro_f1"]) if base["test_macro_f1"] else None for delta, base in zip(q_deltas, baseline, strict=True)]
    aggregate = {
        selector: {
            metric: describe([item[metric] for item in values])
            for metric in ("valid_macro_f1", "test_macro_f1")
        }
        for selector, values in rows.items()
    }
    aggregate[candidate_selector]["delta_test_macro_f1_vs_disabled"] = describe(q_deltas)
    aggregate[candidate_selector]["relative_test_gain_vs_disabled"] = describe([float(value) for value in q_relative])
    aggregate[classical_selector]["delta_test_macro_f1_vs_disabled"] = describe(classical_deltas)
    aggregate[classical_selector]["relative_test_gain_vs_disabled"] = describe([float(value) for value in classical_relative])
    structural_deltas = [
        rows[structural_selector][index]["test_macro_f1"] - base["test_macro_f1"]
        for index, base in enumerate(baseline)
    ]
    aggregate[structural_selector]["delta_test_macro_f1_vs_disabled"] = describe(structural_deltas)
    q_delta = aggregate[candidate_selector]["delta_test_macro_f1_vs_disabled"]
    classical_relative_stats = aggregate[classical_selector]["relative_test_gain_vs_disabled"]
    l2_gate = q_delta["n"] >= 3 and q_delta["mean"] > 0 and q_delta["ci95"][0] is not None and q_delta["ci95"][0] > 0
    qi_gate = classical_relative_stats["mean"] > 0.01
    return {
        "schema_version": "q-attention.q-epvg.formal-multiseed-summary.v1",
        "stage": "replication",
        "group_dir": str(group_dir),
        "seeds": seeds,
        "git_commit": str(manifest.get("git_commit")),
        "protocol_fingerprint": next(iter(protocol_values)),
        "seed_records": seed_records,
        "protocol": protocol_name,
        "source_git_revisions": sorted(commit_values),
        "selectors": list(selectors),
        "candidate": candidate_selector,
        "matched_control": classical_selector,
        "structural_control": structural_selector,
        "aggregate": aggregate,
        "gates": {
            "l2_reproducible_utility_gate": l2_gate,
            "quantum_inspired_mean_relative_gain_gate": qi_gate,
            "quantum_inspired_threshold": 0.01,
            "test_used_for_training_or_selection": False,
        },
        "claim_ceiling": "L2_reproducible_utility" if l2_gate else "L1_utility_candidate",
        "interpretation": "L2 requires the paired 95% CI of Q-EPVG minus disabled test macro-F1 to remain strictly above zero; otherwise retain L1.",
    }


def write_markdown(payload: dict[str, Any], output: Path) -> None:
    candidate_selector = str(payload["candidate"])
    classical_selector = str(payload["matched_control"])
    lines = [
        "# Q-EPVG Re-TACRED Formal Multi-Seed Summary",
        "",
        f"Seeds: `{', '.join(map(str, payload['seeds']))}`",
        f"Commit: `{payload['git_commit']}`",
        f"Claim ceiling: `{payload['claim_ceiling']}`",
        "",
        "| selector | test macro-F1 mean +/- std | delta vs disabled mean +/- std | paired 95% CI |",
        "| --- | ---: | ---: | ---: |",
    ]
    for selector, label in (
        ("disabled", "disabled"),
        (candidate_selector, "Q-EPVG candidate"),
        (classical_selector, "matched classical control"),
        (str(payload["structural_control"]), "random parity control"),
    ):
        item = payload["aggregate"][selector]
        delta = item.get("delta_test_macro_f1_vs_disabled")
        delta_text = f"{delta['mean']:.6f} +/- {delta['std']:.6f}" if delta else "n/a"
        ci_text = f"[{delta['ci95'][0]:.6f}, {delta['ci95'][1]:.6f}]" if delta and delta["ci95"][0] is not None else "n/a"
        lines.append(f"| {label} | {item['test_macro_f1']['mean']:.6f} +/- {item['test_macro_f1']['std']:.6f} | {delta_text} | {ci_text} |")
    lines.extend(
        [
            "",
            f"L2 reproducible-utility gate: `{str(payload['gates']['l2_reproducible_utility_gate']).lower()}`",
            f"Classical quantum-inspired mean relative-gain gate (>1%): `{str(payload['gates']['quantum_inspired_mean_relative_gain_gate']).lower()}`",
            "",
            "The intervals are paired seed-level descriptive 95% t intervals. They do not establish quantum advantage or wall-clock speedup.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args(argv)
    payload = collect(args.group_dir.resolve())
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(payload, args.output_md)
    print(f"Summary written: {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
