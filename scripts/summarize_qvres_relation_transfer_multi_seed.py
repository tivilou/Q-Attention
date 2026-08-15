from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any


SELECTORS = (
    "disabled",
    "q_causal_transport",
    "classical_causal_transport",
    "q_causal_key_only",
)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def stats(values: list[float]) -> dict[str, Any]:
    return {
        "values": values,
        "n": len(values),
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
    }


def collect(group_dir: Path) -> dict[str, Any]:
    marker = group_dir / "MULTI_SEED_COMPLETE"
    if not marker.is_file():
        raise ValueError(f"multi-seed run is incomplete: {group_dir}")
    seed_dirs = sorted(group_dir.glob("seed_*"), key=lambda path: int(path.name.split("_")[-1]))
    if not seed_dirs:
        raise ValueError(f"no seed directories found: {group_dir}")

    rows: dict[str, list[dict[str, float]]] = {selector: [] for selector in SELECTORS}
    seeds: list[int] = []
    commit_values: set[str] = set()
    for seed_dir in seed_dirs:
        summary_path = seed_dir / "run_summary.json"
        if not (seed_dir / "RUN_COMPLETE").is_file() or not summary_path.is_file():
            raise ValueError(f"incomplete seed output: {seed_dir}")
        summary = load_json(summary_path)
        if summary.get("formal_experiment") is not True or summary.get("status") != "pass":
            raise ValueError(f"seed is not a completed formal run: {seed_dir}")
        provenance = summary.get("provenance", {})
        if provenance.get("git_dirty") is not False:
            raise ValueError(f"seed recorded a dirty worktree: {seed_dir}")
        commit = provenance.get("git_commit")
        if not isinstance(commit, str):
            raise ValueError(f"missing git commit provenance: {seed_dir}")
        commit_values.add(commit)
        seeds.append(int(seed_dir.name.split("_")[-1]))
        result_map = {row["selector"]: row for row in summary.get("results", [])}
        for selector in SELECTORS:
            if selector not in result_map:
                raise ValueError(f"missing selector {selector} in {summary_path}")
            row = result_map[selector]
            valid = float(row["valid"]["metrics"]["macro_f1"])
            test = float(row["test"]["metrics"]["macro_f1"])
            delta = float(row["test"]["delta_vs_baseline"]["delta_macro_f1"])
            rows[selector].append({"valid_macro_f1": valid, "test_macro_f1": test, "delta_test_macro_f1": delta})

    if len(commit_values) != 1:
        raise ValueError(f"seed runs used different commits: {sorted(commit_values)}")
    aggregate: dict[str, Any] = {}
    for selector, values in rows.items():
        aggregate[selector] = {
            metric: stats([float(item[metric]) for item in values])
            for metric in ("valid_macro_f1", "test_macro_f1", "delta_test_macro_f1")
        }

    q_values = [item["test_macro_f1"] for item in rows["q_causal_transport"]]
    classical_values = [item["test_macro_f1"] for item in rows["classical_causal_transport"]]
    paired = [q - c for q, c in zip(q_values, classical_values, strict=True)]
    return {
        "schema_version": "q-attention.q-vres.formal-multiseed-summary.v1",
        "group_dir": str(group_dir),
        "seeds": seeds,
        "git_commit": next(iter(commit_values)),
        "selectors": SELECTORS,
        "aggregate": aggregate,
        "comparisons": {
            "q_causal_transport_minus_classical_causal_transport_test_macro_f1": stats(paired),
        },
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Q-VRES Re-TACRED Formal Multi-Seed Summary",
        "",
        f"Seeds: `{', '.join(str(seed) for seed in payload['seeds'])}`",
        f"Commit: `{payload['git_commit']}`",
        "",
        "| selector | valid macro-F1 mean +/- std | test macro-F1 mean +/- std | delta vs baseline mean +/- std |",
        "| --- | ---: | ---: | ---: |",
    ]
    for selector in payload["selectors"]:
        aggregate = payload["aggregate"][selector]
        lines.append(
            f"| {selector} | {aggregate['valid_macro_f1']['mean']:.6f} +/- {aggregate['valid_macro_f1']['std']:.6f} | "
            f"{aggregate['test_macro_f1']['mean']:.6f} +/- {aggregate['test_macro_f1']['std']:.6f} | "
            f"{aggregate['delta_test_macro_f1']['mean']:.6f} +/- {aggregate['delta_test_macro_f1']['std']:.6f} |"
        )
    comparison = payload["comparisons"]["q_causal_transport_minus_classical_causal_transport_test_macro_f1"]
    lines.extend(
        [
            "",
            "Paired Q-VRES minus classical causal transport test macro-F1:",
            f"`{comparison['mean']:.6f} +/- {comparison['std']:.6f}`",
            "",
            "This summary reports paired descriptive statistics; it does not by itself establish statistical significance or a quantum speedup.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize completed Q-VRES formal multi-seed runs.")
    parser.add_argument("--group-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()
    payload = collect(args.group_dir.resolve())
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(payload, args.output_md)
    print(f"Summary written: {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
