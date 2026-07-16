from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

METRIC_KEYS = ("macro_f1", "accuracy", "macro_precision", "macro_recall", "loss")


def read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_float(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return str(value)


def metric_row(name: str, metrics: Mapping[str, Any] | None, *, delta: Mapping[str, Any] | None = None, note: str = "") -> dict[str, Any]:
    row: dict[str, Any] = {"variant": name, "note": note}
    for key in METRIC_KEYS:
        row[key] = None if metrics is None else metrics.get(key)
        row[f"delta_{key}"] = None if delta is None else delta.get(key)
    return row


def filter_note(row: Mapping[str, Any]) -> str:
    family = row.get("family")
    filt = row.get("filter", {})
    if not isinstance(filt, Mapping):
        return str(family or "")
    parts = [str(family)] if family else []
    mode = filt.get("mode")
    if mode:
        parts.append(str(mode))
    for key in ("rank", "threshold", "sharpness"):
        value = filt.get(key)
        if value is not None:
            parts.append(f"{key}={value}")
    gain = row.get("gain")
    if gain is not None:
        parts.append(f"gain={gain}")
    return ", ".join(parts)


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    rows: list[dict[str, Any]] = []

    pipeline = read_json_if_exists(run_path / "pipeline_summary.json")
    baseline = read_json_if_exists(run_path / "baseline" / "metrics.json")
    classical = read_json_if_exists(run_path / "classical_steering_eval" / "metrics.json")
    quantum = read_json_if_exists(run_path / "quantum_steering_eval" / "metrics.json")
    spectral = read_json_if_exists(run_path / "spectral_filter_sweep" / "summary.json")
    routing = read_json_if_exists(run_path / "relation_routing_eval" / "metrics.json")

    protocol = pipeline.get("evaluation_protocol", {}) if isinstance(pipeline, Mapping) else {}
    final_split = protocol.get("final_split", "valid") if isinstance(protocol, Mapping) else "valid"
    final_baseline = None
    baseline_split = "valid"
    for candidate in (
        None if classical is None else classical.get("baseline"),
        None if quantum is None else quantum.get("baseline"),
        None if routing is None else routing.get("baseline"),
        None if spectral is None else spectral.get("test_baseline"),
    ):
        if isinstance(candidate, Mapping):
            final_baseline = candidate
            baseline_split = str(final_split)
            break
    if final_baseline is None and baseline is not None and isinstance(baseline.get("best_valid"), Mapping):
        final_baseline = baseline["best_valid"]
    if final_baseline is not None:
        note = "held-out test split" if baseline_split == "test" else "best validation checkpoint"
        rows.append(metric_row("baseline", final_baseline, note=note))

    if classical is not None:
        rows.append(metric_row("classical_steering", classical.get("steered"), delta=classical.get("delta_vs_baseline")))

    if quantum is not None:
        rows.append(metric_row("quantum_steering", quantum.get("steered"), delta=quantum.get("delta_vs_baseline")))

    spectral_selection = None if spectral is None else spectral.get("best_by_macro_f1")
    spectral_test = None if spectral is None else spectral.get("best_on_test")
    spectral_best = spectral_test if isinstance(spectral_test, Mapping) else spectral_selection
    if isinstance(spectral_best, Mapping):
        note = filter_note(spectral_best)
        if isinstance(spectral_test, Mapping):
            note = f"{note}, selected on validation"
        rows.append(
            metric_row(
                "spectral_sweep_best",
                spectral_best.get("metrics"),
                delta=spectral_best.get("delta_vs_baseline"),
                note=note,
            )
        )

    if routing is not None:
        summary = routing.get("routing_summary", {})
        note = ""
        if isinstance(summary, Mapping):
            entropy = summary.get("mean_entropy")
            note = "" if entropy is None else f"mean_entropy={float(entropy):.6f}"
        rows.append(metric_row("adaptive_routing", routing.get("routed"), delta=routing.get("delta_vs_baseline"), note=note))

    return {
        "run_dir": str(run_path),
        "pipeline": pipeline,
        "evaluation_protocol": protocol,
        "rows": rows,
        "spectral_best": spectral_best,
        "spectral_selection": spectral_selection,
        "routing_summary": None if routing is None else routing.get("routing_summary"),
    }


def markdown_table(summary: Mapping[str, Any]) -> str:
    rows = summary.get("rows", [])
    protocol = summary.get("evaluation_protocol", {})
    final_split = protocol.get("final_split", "valid") if isinstance(protocol, Mapping) else "valid"
    header = ["variant", "macro_f1", "accuracy", "macro_precision", "macro_recall", "loss", "delta_macro_f1", "note"]
    lines = [
        "# Relation Run Summary",
        "",
        f"Run directory: `{summary.get('run_dir')}`",
        "",
        f"Reported split: `{final_split}`",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, Mapping):
            continue
        values = [
            str(row.get("variant", "")),
            fmt_float(row.get("macro_f1")),
            fmt_float(row.get("accuracy")),
            fmt_float(row.get("macro_precision")),
            fmt_float(row.get("macro_recall")),
            fmt_float(row.get("loss")),
            fmt_float(row.get("delta_macro_f1")),
            str(row.get("note", "")),
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a Q-Attention relation pipeline run.")
    parser.add_argument("--run_dir", required=True, help="Run directory produced by run_relation_smoke_pipeline.py")
    parser.add_argument("--output_json", default=None, help="Output JSON path; defaults to <run_dir>/run_summary.json")
    parser.add_argument("--output_markdown", default=None, help="Output Markdown path; defaults to <run_dir>/run_summary.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    summary = summarize_run(run_dir)
    output_json = Path(args.output_json) if args.output_json else run_dir / "run_summary.json"
    output_markdown = Path(args.output_markdown) if args.output_markdown else run_dir / "run_summary.md"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    output_markdown.write_text(markdown_table(summary), encoding="utf-8")
    print(json.dumps({"output_json": str(output_json), "output_markdown": str(output_markdown), "rows": len(summary["rows"])}, sort_keys=True))


if __name__ == "__main__":
    main()
