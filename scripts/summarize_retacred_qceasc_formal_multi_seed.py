from __future__ import annotations

"""Summarize and rate a completed Q-CEASC three-seed replication."""

import argparse
import hashlib
import json
from math import sqrt
from pathlib import Path
from statistics import mean, stdev
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SELECTORS = ("disabled", "q_ceasc", "classical_ceasc")
EXPECTED_SEEDS = (13, 29, 53)
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


def collect(group_dir: Path) -> dict[str, Any]:
    marker = group_dir / "MULTI_SEED_COMPLETE"
    if not marker.is_file():
        raise ValueError(f"missing MULTI_SEED_COMPLETE: {group_dir}")
    manifest = load_json(group_dir / "multi_seed_manifest.json")
    seeds = [int(seed) for seed in manifest.get("seeds", [])]
    if seeds != list(EXPECTED_SEEDS):
        raise ValueError(f"manifest seeds must be {list(EXPECTED_SEEDS)}, found {seeds}")
    if manifest.get("schema_version") != "q-attention.q-ceasc.formal-multiseed-manifest.v1":
        raise ValueError("unsupported multi-seed manifest schema")
    commit_values: set[str] = set()
    protocol_values: set[str] = set()
    rows: dict[str, list[dict[str, float]]] = {selector: [] for selector in SELECTORS}
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
        if summary.get("selectors") != list(SELECTORS):
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
        protocol = hashlib.sha256(
            json.dumps(
                {**{key: value for key, value in config.items() if key not in {"seed", "replication"}}, "seed": 0},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        protocol_values.add(protocol)
        trace_status: dict[str, str] = {}
        for selector in ("q_ceasc", "classical_ceasc"):
            trace_path = seed_dir / "selectors" / selector / "sample_trace.json"
            case_path = seed_dir / "selectors" / selector / "case_study.json"
            if not case_path.is_file():
                raise ValueError(f"missing case study for seed {seed}, selector {selector}")
            if not trace_path.is_file():
                raise ValueError(f"missing sample trace for seed {seed}, selector {selector}")
            errors = _trace_errors(trace_path, sha256(config_path))
            if errors:
                raise ValueError(f"invalid sample trace for seed {seed}, selector {selector}: {'; '.join(errors)}")
            trace_status[selector] = "complete"
        by_selector = {row.get("selector"): row for row in summary.get("rows", [])}
        for selector in SELECTORS:
            row = by_selector.get(selector)
            if not isinstance(row, dict):
                raise ValueError(f"seed {seed} missing summary row {selector}")
            try:
                valid = float(row["valid"]["metrics"]["macro_f1"])
                test = float(row["test"]["metrics"]["macro_f1"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"seed {seed} has malformed {selector} metrics") from exc
            rows[selector].append({"valid_macro_f1": valid, "test_macro_f1": test})
        seed_records.append({"seed": seed, "git_revision": commit, "config_sha256": sha256(config_path), "trace_status": trace_status})
    if commit_values != {str(manifest.get("git_commit"))}:
        raise ValueError(f"seed commits do not match manifest commit: {sorted(commit_values)}")
    if len(protocol_values) != 1:
        raise ValueError("seed configs differ beyond the declared seed field")

    baseline = rows["disabled"]
    q = rows["q_ceasc"]
    classical = rows["classical_ceasc"]
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
    aggregate["q_ceasc"]["delta_test_macro_f1_vs_disabled"] = describe(q_deltas)
    aggregate["q_ceasc"]["relative_test_gain_vs_disabled"] = describe([float(value) for value in q_relative])
    aggregate["classical_ceasc"]["delta_test_macro_f1_vs_disabled"] = describe(classical_deltas)
    aggregate["classical_ceasc"]["relative_test_gain_vs_disabled"] = describe([float(value) for value in classical_relative])
    q_delta = aggregate["q_ceasc"]["delta_test_macro_f1_vs_disabled"]
    classical_relative_stats = aggregate["classical_ceasc"]["relative_test_gain_vs_disabled"]
    l2_gate = q_delta["n"] >= 3 and q_delta["mean"] > 0 and q_delta["ci95"][0] is not None and q_delta["ci95"][0] > 0
    qi_gate = classical_relative_stats["mean"] > 0.01
    return {
        "schema_version": "q-attention.q-ceasc.formal-multiseed-summary.v1",
        "stage": "replication",
        "group_dir": str(group_dir),
        "seeds": seeds,
        "git_commit": next(iter(commit_values)),
        "protocol_fingerprint": next(iter(protocol_values)),
        "seed_records": seed_records,
        "selectors": list(SELECTORS),
        "aggregate": aggregate,
        "gates": {
            "l2_reproducible_utility_gate": l2_gate,
            "quantum_inspired_mean_relative_gain_gate": qi_gate,
            "quantum_inspired_threshold": 0.01,
            "test_used_for_training_or_selection": False,
        },
        "claim_ceiling": "L2_reproducible_utility" if l2_gate else "L1_utility_candidate",
        "interpretation": "L2 requires the paired 95% CI of Q-CEASC minus disabled test macro-F1 to remain strictly above zero; otherwise retain L1.",
    }


def write_markdown(payload: dict[str, Any], output: Path) -> None:
    q = payload["aggregate"]["q_ceasc"]
    c = payload["aggregate"]["classical_ceasc"]
    lines = [
        "# Q-CEASC Re-TACRED Formal Multi-Seed Summary",
        "",
        f"Seeds: `{', '.join(map(str, payload['seeds']))}`",
        f"Commit: `{payload['git_commit']}`",
        f"Claim ceiling: `{payload['claim_ceiling']}`",
        "",
        "| selector | test macro-F1 mean +/- std | delta vs disabled mean +/- std | paired 95% CI |",
        "| --- | ---: | ---: | ---: |",
    ]
    for selector, label in (("disabled", "disabled"), ("q_ceasc", "Q-CEASC"), ("classical_ceasc", "classical CEASC")):
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
