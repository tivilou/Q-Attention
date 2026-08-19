#!/usr/bin/env python3
"""Aggregate a completed frozen consensus quantum-estimator multi-seed run."""

from __future__ import annotations

import argparse
from itertools import product
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FROZEN_SEEDS = (7, 11, 13, 17, 23)
SELECTORS = (
    "disabled",
    "q_consensus_quantum",
    "classical_consensus_control",
    "q_consensus_shuffled_query",
    "q_consensus_magnitude",
)
SPLITS = ("valid", "test")
T_CRITICAL_95_DF4 = 2.7764451051977987


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def stats(values: list[float]) -> dict[str, Any]:
    if len(values) != len(FROZEN_SEEDS):
        raise ValueError("frozen summary statistics require exactly five values")
    average = mean(values)
    sample_std = stdev(values)
    half_width = T_CRITICAL_95_DF4 * sample_std / math.sqrt(len(values))
    return {
        "values": values,
        "n": len(values),
        "mean": average,
        "std": sample_std,
        "sem": sample_std / math.sqrt(len(values)),
        "ci95": {
            "lower": average - half_width,
            "upper": average + half_width,
            "method": "two-sided Student-t interval with df=4",
        },
    }


def paired_sign_flip(differences: list[float]) -> dict[str, Any]:
    """Exact directional and two-sided paired randomization tests."""
    if len(differences) != len(FROZEN_SEEDS):
        raise ValueError("paired test requires exactly five differences")
    observed = mean(differences)
    permuted = [
        mean([sign * value for sign, value in zip(signs, differences, strict=True)])
        for signs in product((-1.0, 1.0), repeat=len(differences))
    ]
    tolerance = 1e-12
    greater = sum(value >= observed - tolerance for value in permuted) / len(permuted)
    two_sided = sum(abs(value) >= abs(observed) - tolerance for value in permuted) / len(permuted)
    return {
        "differences": differences,
        "mean_difference": observed,
        "greater_p": greater,
        "two_sided_p": two_sided,
        "permutations": len(permuted),
        "method": "exact paired sign-flip randomization test",
        "direction": "quantum greater than control",
    }


def _selector_metrics(summary: dict[str, Any], split: str, selector: str) -> dict[str, Any]:
    try:
        metrics = summary["selectors"][split][selector]
    except KeyError as exc:
        raise ValueError(f"missing {split}/{selector} metrics for seed {summary.get('seed')}") from exc
    if metrics.get("selector") != selector:
        raise ValueError(f"selector identity mismatch for seed {summary.get('seed')}: {selector}")
    return metrics


def seed_invariants(summary: dict[str, Any]) -> dict[str, bool]:
    baseline_replay = bool(summary.get("gate", {}).get("baseline_replay"))
    selector_complete = all(
        selector in summary.get("selectors", {}).get(split, {})
        for split in SPLITS
        for selector in SELECTORS
    )
    witness_consistent = selector_complete
    witness_nontrivial = selector_complete
    action_bounded_zero_sum = selector_complete
    baseline_metrics_consistent = selector_complete
    if selector_complete:
        active_selectors = SELECTORS[1:]
        for split in SPLITS:
            active_rates = [
                float(_selector_metrics(summary, split, selector)["active_rate"])
                for selector in active_selectors
            ]
            witness_consistent = witness_consistent and max(active_rates) - min(active_rates) <= 1e-12
            witness_nontrivial = witness_nontrivial and 0.0 < active_rates[0] < 1.0
            expected_baseline = float(summary["baseline"][split]["accuracy"])
            for selector in SELECTORS:
                metrics = _selector_metrics(summary, split, selector)
                baseline_metrics_consistent = baseline_metrics_consistent and abs(
                    float(metrics["baseline_accuracy"]) - expected_baseline
                ) <= 1e-12
                action_bounded_zero_sum = action_bounded_zero_sum and bool(
                    metrics["residual_finite"]
                ) and float(metrics["residual_zero_sum_error"]) <= 1e-5
    return {
        "selector_complete": selector_complete,
        "baseline_replay": baseline_replay,
        "baseline_metrics_consistent": baseline_metrics_consistent,
        "witness_consistent_across_active_selectors": witness_consistent,
        "witness_nontrivial_on_heldout_splits": witness_nontrivial,
        "actions_finite_bounded_zero_sum": action_bounded_zero_sum
        and bool(summary.get("gate", {}).get("residual_invariants")),
    }


def collect(group_dir: Path) -> dict[str, Any]:
    if not (group_dir / "MULTI_SEED_COMPLETE").is_file():
        raise ValueError(f"multi-seed execution is incomplete: {group_dir}")
    manifest = load_json(group_dir / "multi_seed_manifest.json")
    execution = load_json(group_dir / "multi_seed_execution_summary.json")
    if execution.get("execution_success") is not True:
        raise ValueError("execution summary is not successful")
    seeds = tuple(int(seed) for seed in manifest.get("seeds", []))
    if seeds != FROZEN_SEEDS:
        raise ValueError(f"unexpected seed set or order: {seeds}")
    if manifest.get("git_dirty") is not False:
        raise ValueError("formal run manifest recorded a dirty worktree")
    config_path = ROOT / str(manifest["config_path"])
    config = load_json(config_path)
    if config.get("seeds") != list(FROZEN_SEEDS):
        raise ValueError("master config seed set is not frozen")

    summaries: list[dict[str, Any]] = []
    invariants: dict[str, dict[str, bool]] = {}
    for seed in FROZEN_SEEDS:
        seed_dir = group_dir / f"seed_{seed}"
        if not (seed_dir / "SEED_COMPLETE").is_file():
            raise ValueError(f"seed output is incomplete: {seed_dir}")
        summary = load_json(seed_dir / "run_summary.json")
        if summary.get("formal_experiment") is not True:
            raise ValueError(f"seed {seed} is not marked formal")
        if summary.get("run_type") != "frozen_multiseed_synthetic_validation":
            raise ValueError(f"seed {seed} run type mismatch")
        if int(summary.get("seed", -1)) != seed:
            raise ValueError(f"seed directory and summary mismatch: {seed}")
        provenance = summary.get("provenance", {})
        if provenance.get("git_commit") != manifest.get("git_commit"):
            raise ValueError(f"seed {seed} used a different commit")
        if provenance.get("git_dirty") is not False:
            raise ValueError(f"seed {seed} recorded a dirty worktree")
        if provenance.get("master_config_sha256") != manifest.get("config_sha256"):
            raise ValueError(f"seed {seed} used a different master config")
        if provenance.get("source_sha256") != manifest.get("source_sha256"):
            raise ValueError(f"seed {seed} source hashes differ from the manifest")
        summaries.append(summary)
        invariants[str(seed)] = seed_invariants(summary)

    aggregate: dict[str, Any] = {}
    for selector in SELECTORS:
        aggregate[selector] = {}
        for split in SPLITS:
            rows = [_selector_metrics(summary, split, selector) for summary in summaries]
            aggregate[selector][split] = {
                metric: stats([float(row[metric]) for row in rows])
                for metric in (
                    "accuracy",
                    "accuracy_delta",
                    "wrong_correction_rate",
                    "harm_rate",
                    "active_candidate_accuracy",
                )
            }
            aggregate[selector][split]["counts"] = {
                "baseline_wrong_queries": [int(row["baseline_wrong_queries"]) for row in rows],
                "corrected_queries": [int(row["corrected_queries"]) for row in rows],
                "harmed_correct_queries": [int(row["harmed_correct_queries"]) for row in rows],
            }

    quantum = "q_consensus_quantum"
    controls = {
        "product": "classical_consensus_control",
        "shuffled": "q_consensus_shuffled_query",
        "magnitude": "q_consensus_magnitude",
    }
    comparisons: dict[str, Any] = {}
    for split in SPLITS:
        q_values = aggregate[quantum][split]["accuracy_delta"]["values"]
        comparisons[split] = {}
        for label, selector in controls.items():
            control_values = aggregate[selector][split]["accuracy_delta"]["values"]
            differences = [
                q_value - control_value
                for q_value, control_value in zip(q_values, control_values, strict=True)
            ]
            comparisons[split][f"quantum_minus_{label}"] = {
                "stats": stats(differences),
                "paired_test": paired_sign_flip(differences),
            }

    pooled_harm: dict[str, Any] = {}
    for split in SPLITS:
        harmed = sum(
            int(_selector_metrics(summary, split, quantum)["harmed_correct_queries"])
            for summary in summaries
        )
        correct = sum(
            int(summary["baseline"][split]["queries"])
            - int(_selector_metrics(summary, split, quantum)["baseline_wrong_queries"])
            for summary in summaries
        )
        pooled_harm[split] = {
            "harmed_correct_queries": harmed,
            "baseline_correct_queries": correct,
            "rate": harmed / max(correct, 1),
        }

    seed_gate_passes = sum(summary.get("gate", {}).get("status") == "pass" for summary in summaries)
    all_invariants = all(all(values.values()) for values in invariants.values())
    gate_config = config["aggregate_gate"]
    conditions = {
        "required_seed_count": len(summaries) == int(gate_config["required_seed_count"]),
        "minimum_seed_gate_passes": seed_gate_passes
        >= int(gate_config["minimum_seed_gate_passes"]),
        "all_protocol_invariants": all_invariants
        if bool(gate_config["require_all_invariants"])
        else True,
        "quantum_gain_ci95_positive": all(
            aggregate[quantum][split]["accuracy_delta"]["ci95"]["lower"]
            > float(gate_config["minimum_quantum_gain_ci95_lower"])
            for split in SPLITS
        ),
        "quantum_beats_product": all(
            comparisons[split]["quantum_minus_product"]["stats"]["mean"]
            > float(gate_config["minimum_quantum_product_mean_margin"])
            and comparisons[split]["quantum_minus_product"]["paired_test"]["greater_p"]
            <= float(gate_config["maximum_quantum_product_sign_flip_p"])
            for split in SPLITS
        ),
        "quantum_beats_shuffled": all(
            comparisons[split]["quantum_minus_shuffled"]["stats"]["mean"]
            >= float(gate_config["minimum_quantum_shuffled_mean_margin"])
            for split in SPLITS
        ),
        "quantum_beats_magnitude": all(
            comparisons[split]["quantum_minus_magnitude"]["stats"]["mean"]
            >= float(gate_config["minimum_quantum_magnitude_mean_margin"])
            for split in SPLITS
        ),
        "quantum_harm_bounded": all(
            pooled_harm[split]["rate"]
            <= float(gate_config["maximum_pooled_harm_rate"])
            for split in SPLITS
        ),
    }
    return {
        "schema_version": "q-attention.q-consensus-quantum-estimator-frozen-multiseed-summary.v1",
        "status": "complete",
        "group_dir": str(group_dir),
        "git_commit": manifest["git_commit"],
        "config_path": manifest["config_path"],
        "config_sha256": manifest["config_sha256"],
        "seeds": list(FROZEN_SEEDS),
        "seed_gate": {
            "passed": seed_gate_passes,
            "failed": len(summaries) - seed_gate_passes,
            "statuses": {str(summary["seed"]): summary["gate"]["status"] for summary in summaries},
        },
        "invariants": invariants,
        "aggregate": aggregate,
        "comparisons": comparisons,
        "pooled_quantum_harm": pooled_harm,
        "gate": {
            **conditions,
            "status": "pass" if all(conditions.values()) else "fail",
            "next_real_data_authorized": False,
            "finite_shot_authorized": False,
            "hardware_claim": False,
            "quantum_advantage_claim": False,
        },
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Frozen Consensus Quantum-Estimator Multi-Seed Summary",
        "",
        f"Seeds: `{', '.join(str(seed) for seed in payload['seeds'])}`",
        f"Commit: `{payload['git_commit']}`",
        f"Scientific gate: **{payload['gate']['status']}**",
        f"Single-seed gates: `{payload['seed_gate']['passed']}/{len(payload['seeds'])}` passed",
        "",
        "| selector | valid delta mean +/- std (95% CI) | test delta mean +/- std (95% CI) |",
        "| --- | ---: | ---: |",
    ]
    for selector in SELECTORS:
        valid = payload["aggregate"][selector]["valid"]["accuracy_delta"]
        test = payload["aggregate"][selector]["test"]["accuracy_delta"]
        lines.append(
            f"| {selector} | {valid['mean']:.6f} +/- {valid['std']:.6f} "
            f"([{valid['ci95']['lower']:.6f}, {valid['ci95']['upper']:.6f}]) | "
            f"{test['mean']:.6f} +/- {test['std']:.6f} "
            f"([{test['ci95']['lower']:.6f}, {test['ci95']['upper']:.6f}]) |"
        )
    lines.extend(
        [
            "",
            "| split | quantum - product | exact one-sided p | quantum - shuffled | quantum - magnitude | pooled harm |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for split in SPLITS:
        comparisons = payload["comparisons"][split]
        product_comparison = comparisons["quantum_minus_product"]
        lines.append(
            f"| {split} | {product_comparison['stats']['mean']:.6f} | "
            f"{product_comparison['paired_test']['greater_p']:.6f} | "
            f"{comparisons['quantum_minus_shuffled']['stats']['mean']:.6f} | "
            f"{comparisons['quantum_minus_magnitude']['stats']['mean']:.6f} | "
            f"{payload['pooled_quantum_harm'][split]['rate']:.6f} |"
        )
    lines.extend(["", "## Gate", ""])
    for name, passed in payload["gate"].items():
        if isinstance(passed, bool):
            lines.append(f"- `{name}`: `{'pass' if passed else 'fail'}`")
    lines.extend(
        [
            "",
            "This frozen synthetic validation does not authorize real-data, finite-shot, hardware-speedup, or quantum-advantage claims.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-dir", required=True, type=Path)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()
    group_dir = args.group_dir.resolve()
    payload = collect(group_dir)
    output_json = args.output_json or group_dir / "aggregate_summary.json"
    output_md = args.output_md or group_dir / "aggregate_summary.md"
    output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_markdown(payload, output_md)
    print(json.dumps({"output": str(output_json), "gate": payload["gate"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
