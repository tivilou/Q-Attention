from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from summarize_relation_run import METRIC_KEYS, summarize_run  # noqa: E402


def metric_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "std": None, "ci95_low": None, "ci95_high": None}
    average = statistics.mean(values)
    if len(values) < 2:
        return {"n": 1, "mean": average, "std": None, "ci95_low": None, "ci95_high": None}
    std = statistics.stdev(values)
    margin = 1.96 * std / math.sqrt(len(values))
    return {
        "n": len(values),
        "mean": average,
        "std": std,
        "ci95_low": average - margin,
        "ci95_high": average + margin,
    }


def aggregate_run_summaries(summaries: list[Mapping[str, Any]]) -> dict[str, Any]:
    rows_by_variant: dict[str, list[Mapping[str, Any]]] = {}
    runs: list[dict[str, Any]] = []
    for summary in summaries:
        pipeline = summary.get("pipeline", {})
        reproducibility = pipeline.get("reproducibility", {}) if isinstance(pipeline, Mapping) else {}
        runs.append(
            {
                "run_dir": summary.get("run_dir"),
                "seed": reproducibility.get("global_seed") if isinstance(reproducibility, Mapping) else None,
            }
        )
        for row in summary.get("rows", []):
            if not isinstance(row, Mapping) or row.get("variant") is None:
                continue
            rows_by_variant.setdefault(str(row["variant"]), []).append(row)

    variants: dict[str, Any] = {}
    for variant, rows in sorted(rows_by_variant.items()):
        metrics: dict[str, Any] = {}
        for key in METRIC_KEYS:
            values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
            metrics[key] = metric_stats(values)
        for key in METRIC_KEYS:
            delta_key = f"delta_{key}"
            values = [float(row[delta_key]) for row in rows if isinstance(row.get(delta_key), (int, float))]
            metrics[delta_key] = metric_stats(values)
        deltas = [float(row["delta_macro_f1"]) for row in rows if isinstance(row.get("delta_macro_f1"), (int, float))]
        variants[variant] = {
            "num_runs": len(rows),
            "metrics": metrics,
            "positive_delta_macro_f1": sum(value > 0.0 for value in deltas),
            "nonnegative_delta_macro_f1": sum(value >= 0.0 for value in deltas),
        }

    return {"num_runs": len(summaries), "runs": runs, "variants": variants}


def fmt(value: Any) -> str:
    return "" if value is None else f"{float(value):.6f}"


def markdown_table(summary: Mapping[str, Any]) -> str:
    header = ["variant", "n", "macro_f1_mean", "macro_f1_std", "delta_mean", "delta_95%_CI", "positive_seeds"]
    lines = [
        "# Relation Multi-Seed Summary",
        "",
        f"Runs: {summary.get('num_runs', 0)}",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    variants = summary.get("variants", {})
    for name, payload in variants.items() if isinstance(variants, Mapping) else []:
        metrics = payload.get("metrics", {})
        macro = metrics.get("macro_f1", {})
        delta = metrics.get("delta_macro_f1", {})
        ci = ""
        if delta.get("ci95_low") is not None:
            ci = f"[{fmt(delta.get('ci95_low'))}, {fmt(delta.get('ci95_high'))}]"
        positive = payload.get("positive_delta_macro_f1", 0)
        delta_n = delta.get("n", 0)
        values = [
            str(name),
            str(payload.get("num_runs", 0)),
            fmt(macro.get("mean")),
            fmt(macro.get("std")),
            fmt(delta.get("mean")),
            ci,
            "" if not delta_n else f"{positive}/{delta_n}",
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def resolve_run_dirs(run_dirs: str | None, run_root: str | None) -> list[Path]:
    paths = [Path(item.strip()) for item in (run_dirs or "").split(",") if item.strip()]
    if run_root:
        paths.extend(sorted(path for path in Path(run_root).glob("seed_*") if path.is_dir()))
    unique = list(dict.fromkeys(path.resolve() for path in paths))
    if not unique:
        raise ValueError("provide --run_dirs or --run_root containing seed_* directories")
    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Q-Attention relation runs across random seeds.")
    parser.add_argument("--run_dirs", default=None, help="Comma-separated pipeline run directories")
    parser.add_argument("--run_root", default=None, help="Directory containing seed_* run directories")
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_markdown", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = resolve_run_dirs(args.run_dirs, args.run_root)
    summaries = [summarize_run(path) for path in run_dirs]
    aggregate = aggregate_run_summaries(summaries)
    output_root = Path(args.run_root) if args.run_root else run_dirs[0].parent
    output_json = Path(args.output_json) if args.output_json else output_root / "seed_summary.json"
    output_markdown = Path(args.output_markdown) if args.output_markdown else output_root / "seed_summary.md"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8")
    output_markdown.write_text(markdown_table(aggregate), encoding="utf-8")
    print(json.dumps({"output_json": str(output_json), "output_markdown": str(output_markdown), "num_runs": len(run_dirs)}, sort_keys=True))


if __name__ == "__main__":
    main()
