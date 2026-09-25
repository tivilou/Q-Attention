#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    root = parser.parse_args().run_dir.resolve()
    methods = ["quantum_global_context", "classical_global_context", "quantum_srpa", "classical_srpa"]
    payload = {"run_dir": str(root), "seed": 13, "test_used_for_training_or_selection": False, "methods": {}}
    for method in methods:
        payload["methods"][method] = {split: read(root / method / split / "metrics.json")["steered"] for split in ("valid", "test")}
    payload["qsrpa_minus_classical"] = {split: {key: payload["methods"]["quantum_srpa"][split][key] - payload["methods"]["classical_srpa"][split][key] for key in ("accuracy", "macro_f1", "loss", "correct_label_margin")} for split in ("valid", "test")}
    (root / "run_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    lines = ["# Q-SRPA Re-TACRED 单 seed 汇总", "", "- seed: `13`", "- checkpoint selection: valid only", "- test 未用于训练、选择或调参。", "", "| split | Q-SRPA | classical SRPA | Q-SRPA - classical |", "| --- | ---: | ---: | ---: |"]
    for split in ("valid", "test"):
        q = payload["methods"]["quantum_srpa"][split]["macro_f1"]
        c = payload["methods"]["classical_srpa"][split]["macro_f1"]
        lines.append(f"| {split} | {q:.6f} | {c:.6f} | {q-c:+.6f} |")
    (root / "run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
