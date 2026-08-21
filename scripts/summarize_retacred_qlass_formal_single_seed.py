#!/usr/bin/env python3
"""Create a compact, test-set-auditable summary for one Q-LASS run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--config", default="configs/retacred_qlass_formal_single_seed.json", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    config = read_json(args.config)
    baseline = read_json(run_dir / "baseline" / "metrics.json")
    quantum = read_json(run_dir / "quantum_global_context" / "test" / "metrics.json")
    classical = read_json(run_dir / "classical_global_context" / "test" / "metrics.json")
    quantum_valid = read_json(run_dir / "quantum_global_context" / "valid" / "metrics.json")
    classical_valid = read_json(run_dir / "classical_global_context" / "valid" / "metrics.json")

    payload = {
        "run_dir": str(run_dir),
        "config": config,
        "seed": config["seed"],
        "selection_metric": config["selection_metric"],
        "data_split_contract": {"train_used_for_fit": True, "valid_used_for_selection": True, "test_used_for_training_or_selection": False},
        "baseline_best_valid": baseline["best_valid"],
        "valid": {
            "quantum": quantum_valid["steered"],
            "classical": classical_valid["steered"],
            "quantum_minus_baseline": quantum_valid["delta_vs_baseline"],
            "classical_minus_baseline": classical_valid["delta_vs_baseline"],
        },
        "test": {
            "quantum": quantum["steered"],
            "classical": classical["steered"],
            "quantum_minus_baseline": quantum["delta_vs_baseline"],
            "classical_minus_baseline": classical["delta_vs_baseline"],
        },
        "quantum_minus_classical": {
            key: quantum["steered"][key] - classical["steered"][key]
            for key in ("accuracy", "macro_f1", "loss", "correct_label_margin")
        },
        "action_contract": quantum["checkpoint_metadata"].get("action_contract"),
    }
    (run_dir / "run_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    test_q = payload["test"]["quantum"]
    test_c = payload["test"]["classical"]
    delta = payload["quantum_minus_classical"]
    lines = [
        "# Q-LASS Re-TACRED 单 seed 汇总",
        "",
        f"- seed: `{payload['seed']}`",
        f"- selection: `{payload['selection_metric']}`（仅 valid）",
        "- test 未用于训练、checkpoint 选择或调参。",
        "",
        "| split | Q-LASS macro-F1 | classical macro-F1 | Q-LASS - classical |",
        "| --- | ---: | ---: | ---: |",
        f"| valid | {payload['valid']['quantum']['macro_f1']:.6f} | {payload['valid']['classical']['macro_f1']:.6f} | {(payload['valid']['quantum']['macro_f1'] - payload['valid']['classical']['macro_f1']):+.6f} |",
        f"| test | {test_q['macro_f1']:.6f} | {test_c['macro_f1']:.6f} | {delta['macro_f1']:+.6f} |",
        "",
        "test Q-LASS - classical："
        f" accuracy {delta['accuracy']:+.6f}，loss {delta['loss']:+.6f}，"
        f"correct-label margin {delta['correct_label_margin']:+.6f}。",
    ]
    (run_dir / "run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
