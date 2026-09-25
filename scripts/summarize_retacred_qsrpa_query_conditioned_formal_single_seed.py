#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

METHODS = [
    "quantum_global_context",
    "classical_global_context",
    "quantum_soft_role_pair",
    "classical_soft_role_pair",
    "quantum_query_conditioned_soft_role_pair",
    "classical_query_conditioned_soft_role_pair",
]
METRICS = ("accuracy", "macro_f1", "loss", "correct_label_margin")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    root = parser.parse_args().run_dir.resolve()
    payload = {
        "run_dir": str(root),
        "seed": 13,
        "test_used_for_training_or_selection": False,
        "methods": {},
    }
    for method in METHODS:
        payload["methods"][method] = {
            split: read(root / method / split / "metrics.json")["steered"]
            for split in ("valid", "test")
        }
    candidate = "quantum_query_conditioned_soft_role_pair"
    matched = "classical_query_conditioned_soft_role_pair"
    disabled = "quantum_global_context"
    payload["candidate_minus_matched"] = {
        split: {
            key: payload["methods"][candidate][split][key]
            - payload["methods"][matched][split][key]
            for key in METRICS
        }
        for split in ("valid", "test")
    }
    payload["candidate_minus_disabled"] = {
        split: {
            key: payload["methods"][candidate][split][key]
            - payload["methods"][disabled][split][key]
            for key in METRICS
        }
        for split in ("valid", "test")
    }
    (root / "run_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Query-conditioned Q-SRPA Re-TACRED 单 seed 汇总",
        "",
        "- seed: `13`",
        "- checkpoint selection: valid only",
        "- test 未用于训练、选择或调参。",
        "",
        "| split | candidate | matched classical | candidate - matched | candidate - disabled |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for split in ("valid", "test"):
        candidate_value = payload["methods"][candidate][split]["macro_f1"]
        matched_value = payload["methods"][matched][split]["macro_f1"]
        lines.append(
            f"| {split} | {candidate_value:.6f} | {matched_value:.6f} | "
            f"{payload['candidate_minus_matched'][split]['macro_f1']:+.6f} | "
            f"{payload['candidate_minus_disabled'][split]['macro_f1']:+.6f} |"
        )
    (root / "run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
