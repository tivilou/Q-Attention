#!/usr/bin/env python3
"""Qualify candidate diagnostics before a new quantum mechanism is trained."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


FIXED_SEEDS = (7, 11, 13, 17, 23)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", required=True)
    parser.add_argument("--routing-v2", required=True)
    parser.add_argument("--rescue-bank", required=True)
    parser.add_argument("--balanced", required=True)
    parser.add_argument(
        "--output-root", default="runs/q_headroom_action_certificate"
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def current_certificate(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload["results"]
    seeds = tuple(int(row["seed"]) for row in rows)
    headroom = [int(row["oracle_headroom_vs_best_restricted"]) for row in rows]
    mean_headroom = sum(headroom) / len(headroom)
    conditions = {
        "fixed_seed_set": seeds == FIXED_SEEDS,
        "baseline_validity": True,
        "minimum_mean_headroom_two": mean_headroom >= 2.0,
        "minimum_four_eligible_seeds": sum(value >= 2 for value in headroom) >= 4,
    }
    return {
        "benchmark": "current_query_indexed",
        "conditions": conditions,
        "headroom_by_seed": dict(zip(map(str, seeds), headroom)),
        "mean_headroom": mean_headroom,
        "status": "pass" if all(conditions.values()) else "fail",
        "failure_class": None if all(conditions.values()) else "insufficient_headroom",
    }


def routing_certificate(payload: dict[str, Any]) -> dict[str, Any]:
    gate = payload["gate"]
    conditions = {
        "exact_marginals": bool(gate["marginals_exact"]),
        "baseline_validity": bool(gate["baseline_accuracy_parity"]),
        "oracle_safety": bool(gate["oracle_correct_retention"]),
        "oracle_residual_invariants": bool(gate["oracle_residual_invariants"]),
    }
    return {
        "benchmark": "counterbalanced_routing_v2",
        "conditions": conditions,
        "baseline_accuracy_gap": float(gate["baseline_accuracy_gap"]),
        "status": "pass" if all(conditions.values()) else "fail",
        "failure_class": None if all(conditions.values()) else "invalid_baseline",
    }


def rescue_bank_certificate(payload: dict[str, Any]) -> dict[str, Any]:
    gate = payload["gate"]
    row = payload["results"][0]
    headroom = int(row["oracle_headroom_vs_best_restricted"])
    conditions = {
        "split_invariants": bool(gate["split_invariants"]),
        "baseline_validity": bool(gate["baseline_accuracy_parity"]),
        "minimum_headroom_two": headroom >= 2,
        "oracle_safety": bool(gate["oracle_correct_retention"]),
        "oracle_residual_invariants": bool(gate["oracle_residual_invariants"]),
    }
    return {
        "benchmark": "fixed_query_rescue_bank",
        "conditions": conditions,
        "oracle_headroom": headroom,
        "status": "pass" if all(conditions.values()) else "fail",
        "failure_class": None if all(conditions.values()) else "insufficient_headroom",
    }


def balanced_geometry_certificate(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload["results"]
    by_key = {(row["selector"], int(row["seed"])): row for row in rows}
    expected = {
        (selector, seed)
        for selector in ("q_causal_transport", "classical_causal_transport")
        for seed in FIXED_SEEDS
    }
    complete = expected <= set(by_key)
    per_seed = []
    for seed in FIXED_SEEDS:
        quantum = by_key.get(("q_causal_transport", seed))
        classical = by_key.get(("classical_causal_transport", seed))
        if quantum is None or classical is None:
            continue
        per_seed.append(
            {
                "seed": seed,
                "quantum_mass_gain": float(quantum["context_target_mass_gain"]),
                "classical_mass_gain": float(classical["context_target_mass_gain"]),
                "mass_slack": float(quantum["context_target_mass_gain"])
                - float(classical["context_target_mass_gain"]),
                "quantum_influence_gain": float(
                    quantum["target_counterfactual_influence_gain"]
                ),
                "classical_influence_gain": float(
                    classical["target_counterfactual_influence_gain"]
                ),
                "influence_slack": float(
                    quantum["target_counterfactual_influence_gain"]
                )
                - float(classical["target_counterfactual_influence_gain"]),
            }
        )
    conditions = {
        "fixed_seed_set": tuple(int(seed) for seed in payload["seeds"])
        == FIXED_SEEDS,
        "matched_rows_complete": complete,
        "positive_quantum_mass_gain_all_seeds": bool(per_seed)
        and all(row["quantum_mass_gain"] > 0.0 for row in per_seed),
        "positive_quantum_influence_gain_all_seeds": bool(per_seed)
        and all(row["quantum_influence_gain"] > 0.0 for row in per_seed),
        "classical_slack_mass_all_seeds": bool(per_seed)
        and all(row["mass_slack"] >= 0.005 for row in per_seed),
        "classical_slack_influence_all_seeds": bool(per_seed)
        and all(row["influence_slack"] >= 0.005 for row in per_seed),
    }
    return {
        "benchmark": "balanced_causal_value_geometry",
        "conditions": conditions,
        "per_seed": per_seed,
        "status": "pass" if all(conditions.values()) else "fail",
        "failure_class": None
        if all(conditions.values())
        else "matched_control_slack_failure",
    }


def build_certificate(
    current: dict[str, Any],
    routing: dict[str, Any],
    rescue: dict[str, Any],
    balanced: dict[str, Any],
) -> dict[str, Any]:
    benchmarks = [
        current_certificate(current),
        routing_certificate(routing),
        rescue_bank_certificate(rescue),
        balanced_geometry_certificate(balanced),
    ]
    qualified = [row["benchmark"] for row in benchmarks if row["status"] == "pass"]
    return {
        "schema_version": "q-attention.headroom-action-certificate.v1",
        "status": "pass" if qualified else "fail",
        "revision": git_revision(),
        "benchmarks": benchmarks,
        "qualified_benchmarks": qualified,
        "gate": {
            "new_mechanism_canary_authorized": bool(qualified),
            "five_seed_mechanism_authorized": False,
            "real_data_authorized": False,
            "status": "pass" if qualified else "fail",
        },
        "interpretation": {
            "certificate_is_not_a_quantum_mechanism": True,
            "invalid_baseline_is_distinct_from_low_headroom": True,
            "positive_average_geometry_is_insufficient_without_fixed_seed_classical_slack": True,
        },
    }


def main() -> None:
    args = parse_args()
    paths = {
        "current": Path(args.current).resolve(),
        "routing_v2": Path(args.routing_v2).resolve(),
        "rescue_bank": Path(args.rescue_bank).resolve(),
        "balanced": Path(args.balanced).resolve(),
    }
    summary = build_certificate(
        load_json(paths["current"]),
        load_json(paths["routing_v2"]),
        load_json(paths["rescue_bank"]),
        load_json(paths["balanced"]),
    )
    summary["sources"] = {
        name: {"path": str(path), "sha256": sha256(path)}
        for name, path in paths.items()
    }
    output = Path(args.output_root) / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "gate": summary["gate"]}, sort_keys=True))


if __name__ == "__main__":
    main()
