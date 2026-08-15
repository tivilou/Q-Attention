from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any, Mapping


SELECTORS = (
    "disabled",
    "q_causal_transport",
    "classical_causal_transport",
    "q_causal_key_only",
)
TRAINABLE_SELECTORS = SELECTORS[1:]


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _stage_identity(summary: Mapping[str, Any]) -> tuple[Any, ...]:
    provenance = summary["provenance"]
    result_seeds = tuple(sorted(int(row["seed"]) for row in summary["results"]))
    return (
        summary["run_type"],
        summary["formal_experiment"],
        summary["partial_selector_run"],
        result_seeds,
        provenance["git_commit"],
        provenance["git_branch"],
        provenance["git_dirty"],
        provenance["config_sha256"],
        provenance["train"]["source_sha256"],
        provenance["valid"]["source_sha256"],
        provenance["test"]["source_sha256"],
        provenance["train"]["records"],
        provenance["valid"]["records"],
        provenance["test"]["records"],
    )


def _metric_signature(summary: Mapping[str, Any]) -> tuple[float, ...]:
    baseline = summary["baseline"]
    return tuple(
        float(baseline[split]["metrics"][metric])
        for split in ("valid", "test")
        for metric in ("accuracy", "macro_f1", "loss")
    )


def _write_markdown(summary: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Q-VRES Formal Single-Seed Selector-Parallel Run",
        "",
        f"Seed: `{summary['seed']}`",
        f"Pilot validation gate: `{summary['pilot_validation_gate']['status']}`",
        "",
        "| selector | valid macro-F1 | test macro-F1 | delta test macro-F1 | parameters |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["results"]:
        lines.append(
            f"| {row['selector']} | {row['valid']['metrics']['macro_f1']:.6f} | "
            f"{row['test']['metrics']['macro_f1']:.6f} | "
            f"{row['test']['delta_vs_baseline']['delta_macro_f1']:.6f} | "
            f"{row['trainable_parameters']} |"
        )
    gate = summary["pilot_validation_gate"]
    lines.extend(
        [
            "",
            "The pilot gate uses validation macro-F1 only:",
            f"- Q-VRES minus baseline: `{gate['q_minus_baseline_valid_macro_f1']:.6f}`",
            f"- Q-VRES minus classical transport: `{gate['q_minus_classical_valid_macro_f1']:.6f}`",
            "",
            "Test metrics are reported but must not be used to tune the method or decide hyperparameters.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def aggregate_selector_parallel_run(
    run_dir: Path,
    baseline_stage_dir: Path,
    selector_stage_dirs: Mapping[str, Path],
    gpu_assignments: Mapping[str, int],
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    baseline_stage_dir = baseline_stage_dir.resolve()
    if set(selector_stage_dirs) != set(TRAINABLE_SELECTORS):
        raise ValueError(f"selector stages must be exactly {TRAINABLE_SELECTORS}")
    expected_assignment_stages = {"baseline", *TRAINABLE_SELECTORS}
    if set(gpu_assignments) != expected_assignment_stages:
        raise ValueError(f"GPU assignments must be exactly {sorted(expected_assignment_stages)}")

    manifest = load_json(run_dir / "selector_parallel_manifest.json")

    stage_summaries: dict[str, dict[str, Any]] = {
        "disabled": load_json(baseline_stage_dir / "run_summary.json")
    }
    for selector, stage_dir in selector_stage_dirs.items():
        stage_summaries[selector] = load_json(stage_dir.resolve() / "run_summary.json")

    for selector, summary in stage_summaries.items():
        if summary.get("status") != "pass":
            raise ValueError(f"stage {selector} did not pass")
        if summary.get("formal_experiment") is not True:
            raise ValueError(f"stage {selector} is not marked formal")
        if summary.get("partial_selector_run") is not True:
            raise ValueError(f"stage {selector} is not a partial selector worker")
        rows = summary.get("results")
        if not isinstance(rows, list) or len(rows) != 1 or rows[0].get("selector") != selector:
            raise ValueError(f"stage {selector} contains unexpected selector rows")

    identities = {_stage_identity(summary) for summary in stage_summaries.values()}
    if len(identities) != 1:
        raise ValueError("selector workers used different code, data, or formal settings")
    baseline_signatures = {_metric_signature(summary) for summary in stage_summaries.values()}
    if len(baseline_signatures) != 1:
        raise ValueError("selector workers did not evaluate the same frozen baseline")

    baseline_model_dir = (baseline_stage_dir / "baseline").resolve()
    for selector in TRAINABLE_SELECTORS:
        worker_model_dir = Path(stage_summaries[selector]["baseline"]["model_dir"]).resolve()
        if worker_model_dir != baseline_model_dir:
            raise ValueError(f"stage {selector} used a different baseline model")

    results = [stage_summaries[selector]["results"][0] for selector in SELECTORS]
    if not all(bool(row["finite"]) for row in results):
        raise ValueError("one or more selector rows contain non-finite metrics")
    parameter_counts = {
        int(row["trainable_parameters"])
        for row in results
        if row["selector"] != "disabled"
    }
    if len(parameter_counts) != 1:
        raise ValueError(f"trainable selector parameter counts differ: {sorted(parameter_counts)}")

    baseline_summary = stage_summaries["disabled"]
    seed = int(results[0]["seed"])
    if any(int(row["seed"]) != seed for row in results):
        raise ValueError("selector workers used different seeds")
    provenance = baseline_summary["provenance"]
    if manifest.get("seed") != seed:
        raise ValueError("selector scheduler manifest seed does not match worker outputs")
    if manifest.get("git_commit") != provenance.get("git_commit"):
        raise ValueError("selector scheduler and workers used different commits")
    if manifest.get("config_sha256") != provenance.get("config_sha256"):
        raise ValueError("selector scheduler and workers used different configs")
    available_gpu_ids = {int(item["id"]) for item in manifest.get("gpus", [])}
    assigned_gpu_ids = {int(gpu_id) for gpu_id in gpu_assignments.values()}
    if not assigned_gpu_ids.issubset(available_gpu_ids):
        raise ValueError("GPU assignments contain a device absent from the scheduler manifest")
    q_valid = float(stage_summaries["q_causal_transport"]["results"][0]["valid"]["metrics"]["macro_f1"])
    baseline_valid = float(results[0]["valid"]["metrics"]["macro_f1"])
    classical_valid = float(
        stage_summaries["classical_causal_transport"]["results"][0]["valid"]["metrics"]["macro_f1"]
    )
    if q_valid > baseline_valid and q_valid > classical_valid:
        pilot_status = "pass"
    elif q_valid >= baseline_valid:
        pilot_status = "review"
    else:
        pilot_status = "fail"
    pilot_gate = {
        "status": pilot_status,
        "selection_split": "valid",
        "q_valid_macro_f1": q_valid,
        "baseline_valid_macro_f1": baseline_valid,
        "classical_valid_macro_f1": classical_valid,
        "q_minus_baseline_valid_macro_f1": q_valid - baseline_valid,
        "q_minus_classical_valid_macro_f1": q_valid - classical_valid,
        "test_metrics_must_not_drive_method_changes": True,
    }

    baseline_destination = run_dir / "baseline"
    baseline_destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(baseline_stage_dir / "baseline" / "metrics.json", baseline_destination / "metrics.json")
    baseline_train_log = baseline_stage_dir / "baseline_train.log"
    if baseline_train_log.is_file():
        shutil.copy2(baseline_train_log, run_dir / "baseline_train.log")
    for selector in SELECTORS:
        source_stage = baseline_stage_dir if selector == "disabled" else selector_stage_dirs[selector]
        destination = run_dir / "selectors" / selector
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_stage / "selectors" / selector / "metrics.json", destination / "metrics.json")

    provenance = dict(provenance)
    provenance["cuda_devices"] = {
        stage: stage_summaries[stage if stage in stage_summaries else "disabled"]["provenance"].get("cuda_device")
        for stage in gpu_assignments
    }
    summary = {
        "schema_version": "q-attention.q-vres.formal-selector-parallel.v1",
        "run_type": "formal_full_relation_transfer",
        "formal_experiment": True,
        "partial_selector_run": False,
        "parallel_mode": "selectors",
        "status": "pass",
        "seed": seed,
        "run_dir": str(run_dir),
        "device": "cuda",
        "selectors": list(SELECTORS),
        "baseline": baseline_summary["baseline"],
        "results": results,
        "screen_gate": {
            "status": "pass",
            "finite_metrics": True,
            "matched_parameter_counts": sorted(parameter_counts),
            "matched_parameters": True,
            "controls_present": {selector: True for selector in SELECTORS},
            "all_controls_required": True,
            "task_gain_is_not_automatically_accepted": True,
        },
        "pilot_validation_gate": pilot_gate,
        "parallel_execution": {
            "gpu_assignments": dict(gpu_assignments),
            "baseline_stage_dir": str(baseline_stage_dir),
            "selector_stage_dirs": {
                selector: str(path.resolve()) for selector, path in selector_stage_dirs.items()
            },
        },
        "provenance": provenance,
    }
    run_config = {
        "schema_version": "q-attention.q-vres.selector-parallel-config.v1",
        "seed": seed,
        "formal_experiment": True,
        "parallel_mode": "selectors",
        "run_dir": str(run_dir),
        "baseline_stage_dir": str(baseline_stage_dir),
        "selector_stage_dirs": {
            selector: str(path.resolve()) for selector, path in selector_stage_dirs.items()
        },
        "gpu_assignments": dict(gpu_assignments),
        "stage_run_configs": {
            selector: load_json(
                (baseline_stage_dir if selector == "disabled" else selector_stage_dirs[selector])
                / "run_config.json"
            )
            for selector in SELECTORS
        },
    }
    write_json(run_dir / "run_config.json", run_config)
    write_json(run_dir / "run_summary.json", summary)
    _write_markdown(summary, run_dir / "run_summary.md")
    return summary


def _parse_mapping(values: list[str], *, label: str, value_type: type) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{label} must use name=value syntax")
        name, raw = value.split("=", 1)
        if not name or name in result:
            raise ValueError(f"invalid or duplicate {label} name: {name!r}")
        result[name] = value_type(raw)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate one Q-VRES selector-parallel seed run.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--baseline-stage", required=True, type=Path)
    parser.add_argument("--selector-stage", action="append", default=[])
    parser.add_argument("--gpu-assignment", action="append", default=[])
    args = parser.parse_args()
    selector_dirs = _parse_mapping(args.selector_stage, label="selector stage", value_type=Path)
    gpu_assignments = _parse_mapping(args.gpu_assignment, label="GPU assignment", value_type=int)
    aggregate_selector_parallel_run(
        args.run_dir,
        args.baseline_stage,
        selector_dirs,
        gpu_assignments,
    )
    print(f"Aggregated selector-parallel run: {args.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
