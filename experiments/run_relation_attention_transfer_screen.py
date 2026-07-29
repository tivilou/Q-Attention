from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q_attention.experiments import (  # noqa: E402
    choose_device,
    evaluate_relation_attention_score_kernel,
    load_relation_run,
    make_relation_loader,
    summarize_transfer_screen,
)
from q_attention.tasks.relation import (  # noqa: E402
    load_relation_jsonl,
    sample_relation_records,
    sample_relation_records_proportional,
    write_relation_jsonl,
)


FAMILIES = ("quantum", "classical")
STAGES = ("core", "evidence", "routing")
SPLITS = ("valid", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a private Re-TACRED validation/test transfer screen for the "
            "quantum score core, evidence selector, and expert router."
        )
    )
    parser.add_argument(
        "--train_path", default="data/relation/retacred/train.jsonl"
    )
    parser.add_argument(
        "--valid_path", default="data/relation/retacred/valid.jsonl"
    )
    parser.add_argument(
        "--test_path", default="data/relation/retacred/test.jsonl"
    )
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--baseline_train_limit", type=int, default=4096)
    parser.add_argument("--train_limit", type=int, default=512)
    parser.add_argument("--valid_limit", type=int, default=512)
    parser.add_argument("--test_limit", type=int, default=512)
    parser.add_argument("--baseline_epochs", type=int, default=8)
    parser.add_argument("--core_epochs", type=int, default=6)
    parser.add_argument("--evidence_epochs", type=int, default=6)
    parser.add_argument("--routing_epochs", type=int, default=10)
    parser.add_argument("--baseline_batch_size", type=int, default=64)
    parser.add_argument("--plugin_batch_size", type=int, default=32)
    parser.add_argument("--random_repeats", type=int, default=2)
    parser.add_argument("--min_baseline_macro_f1", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _default_output_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "runs" / f"retacred_attention_transfer_screen_{timestamp}"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _run_stage(name: str, command: list[str], logs_dir: Path) -> None:
    log_path = logs_dir / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"event": "stage_started", "stage": name, "log": str(log_path)}))
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = environment.get("PYTHONHASHSEED", "0")
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"command": command}) + "\n")
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        tail = log_path.read_text(encoding="utf-8").splitlines()[-30:]
        raise RuntimeError(
            f"stage {name} failed with exit code {completed.returncode}:\n"
            + "\n".join(tail)
        )
    print(json.dumps({"event": "stage_completed", "stage": name}))


def _subset_manifest(records: list[Any], source_path: Path, sampler: str) -> dict[str, Any]:
    return {
        "source_path": str(source_path),
        "source_size_bytes": source_path.stat().st_size,
        "selected_records": len(records),
        "selected_label_counts": dict(sorted(Counter(record.label for record in records).items())),
        "sampler": sampler,
    }


def _prepare_subsets(args: argparse.Namespace, data_dir: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    source_paths = {
        "train": _resolve_path(args.train_path),
        "valid": _resolve_path(args.valid_path),
        "test": _resolve_path(args.test_path),
    }
    full = {split: load_relation_jsonl(path) for split, path in source_paths.items()}
    selected = {
        "baseline_train": sample_relation_records(
            full["train"], args.baseline_train_limit, seed=args.seed, stratified=True
        ),
        "train": sample_relation_records(
            full["train"], args.train_limit, seed=args.seed, stratified=True
        ),
        "valid": sample_relation_records_proportional(
            full["valid"], args.valid_limit, seed=args.seed + 101
        ),
        "test": sample_relation_records_proportional(
            full["test"], args.test_limit, seed=args.seed + 211
        ),
    }
    train_labels = {record.label for record in selected["train"]}
    evaluation_labels = {
        record.label for split in ("valid", "test") for record in selected[split]
    }
    missing = evaluation_labels - train_labels
    if missing:
        raise ValueError(f"training subset does not cover evaluation labels: {sorted(missing)}")

    subset_paths: dict[str, Path] = {}
    manifest: dict[str, Any] = {"seed": args.seed, "splits": {}}
    source_split = {
        "baseline_train": "train",
        "train": "train",
        "valid": "valid",
        "test": "test",
    }
    for split in ("baseline_train", "train", "valid", "test"):
        path = data_dir / f"{split}.jsonl"
        write_relation_jsonl(selected[split], path)
        subset_paths[split] = path
        sampler = (
            "balanced_round_robin"
            if split in {"baseline_train", "train"}
            else "proportional_largest_remainder"
        )
        manifest["splits"][split] = _subset_manifest(
            selected[split], source_paths[source_split[split]], sampler
        )
    _write_json(data_dir / "subset_manifest.json", manifest)
    return subset_paths, manifest


def _baseline_command(
    args: argparse.Namespace,
    paths: dict[str, Path],
    model_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "experiments" / "train_relation_baseline.py"),
        "--train_path",
        str(paths["baseline_train"]),
        "--valid_path",
        str(paths["valid"]),
        "--output_dir",
        str(model_dir),
        "--epochs",
        str(args.baseline_epochs),
        "--batch_size",
        str(args.baseline_batch_size),
        "--lr",
        "0.001",
        "--dim",
        "64",
        "--num_layers",
        "2",
        "--num_heads",
        "4",
        "--ff_dim",
        "128",
        "--dropout",
        "0.1",
        "--max_length",
        "128",
        "--selection_metric",
        "valid_loss",
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]


def _core_command(
    args: argparse.Namespace,
    family: str,
    paths: dict[str, Path],
    model_dir: Path,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "experiments" / "train_relation_attention_score_kernel.py"),
        "--model_dir",
        str(model_dir),
        "--train_path",
        str(paths["train"]),
        "--valid_path",
        str(paths["valid"]),
        "--output_dir",
        str(output_dir),
        "--kernel_type",
        family,
        "--num_qubits",
        "4",
        "--depth",
        "2",
        "--score_readout",
        "observable",
        "--input_encoding",
        "factorized_shared",
        "--query_scope",
        "all",
        "--epochs",
        str(args.core_epochs),
        "--batch_size",
        str(args.plugin_batch_size),
        "--lr",
        "0.001",
        "--selection_metric",
        "valid_loss",
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]


def _evidence_command(
    args: argparse.Namespace,
    family: str,
    paths: dict[str, Path],
    model_dir: Path,
    core_checkpoint: Path,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "experiments" / "train_relation_counterfactual_evidence.py"),
        "--model_dir",
        str(model_dir),
        "--core_checkpoint",
        str(core_checkpoint),
        "--train_path",
        str(paths["train"]),
        "--valid_path",
        str(paths["valid"]),
        "--output_dir",
        str(output_dir),
        "--evidence_type",
        family,
        "--evidence_readout",
        "factorized_observable",
        "--epochs",
        str(args.evidence_epochs),
        "--batch_size",
        str(args.plugin_batch_size),
        "--lr",
        "0.01",
        "--random_repeats",
        str(args.random_repeats),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]


def _routing_command(
    args: argparse.Namespace,
    family: str,
    paths: dict[str, Path],
    model_dir: Path,
    evidence_checkpoint: Path,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "experiments" / "train_relation_expert_routing.py"),
        "--model_dir",
        str(model_dir),
        "--core_checkpoint",
        str(evidence_checkpoint),
        "--train_path",
        str(paths["train"]),
        "--valid_path",
        str(paths["valid"]),
        "--output_dir",
        str(output_dir),
        "--router_type",
        family,
        "--epochs",
        str(args.routing_epochs),
        "--batch_size",
        str(args.plugin_batch_size),
        "--lr",
        "0.01",
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]


def _evaluation_command(
    args: argparse.Namespace,
    model_dir: Path,
    checkpoint: Path,
    data_path: Path,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "experiments" / "eval_relation_attention_score_kernel.py"),
        "--model_dir",
        str(model_dir),
        "--checkpoint",
        str(checkpoint),
        "--data_path",
        str(data_path),
        "--output_dir",
        str(output_dir),
        "--batch_size",
        str(args.plugin_batch_size),
        "--random_repeats",
        str(args.random_repeats),
        "--random_seed",
        str(args.seed + 9001),
        "--device",
        args.device,
    ]


def _evaluate_baseline(
    model_dir: Path,
    data_path: Path,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    artifacts = load_relation_run(model_dir, device)
    loader = make_relation_loader(
        load_relation_jsonl(data_path),
        artifacts.vocab,
        artifacts.label_to_id,
        batch_size=batch_size,
        shuffle=False,
    )
    metrics = evaluate_relation_attention_score_kernel(
        artifacts.model,
        loader,
        device,
        len(artifacts.label_to_id),
        adapter=None,
    )
    del artifacts
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_shared_baseline(
    expected: dict[str, float],
    observed: dict[str, float],
    *,
    stage: str,
    tolerance: float = 1e-8,
) -> None:
    for name in ("loss", "correct_label_margin", "macro_f1"):
        if abs(float(expected[name]) - float(observed[name])) > tolerance:
            raise RuntimeError(f"{stage} did not evaluate the shared frozen baseline")


def _render_summary(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    rows = [
        "# Re-TACRED Attention Transfer Screen",
        "",
        f"- Screen pass: `{str(decision['screen_pass']).lower()}`",
        f"- Seed: `{payload['config']['seed']}`",
        "- Selection: validation only; test evaluated after all checkpoints were fixed.",
        "",
        "| Stage | Split | Q loss gain | Q margin gain | Q-vs-C loss gain | Q-vs-C margin gain | Pass |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for stage in STAGES:
        for split in SPLITS:
            result = decision["stages"][stage]["splits"][split]
            increment = result["quantum_increment"]
            comparison = result["quantum_over_classical"]
            passed = result["quantum_increment_pass"] and result["quantum_over_classical_pass"]
            rows.append(
                f"| {stage} | {split} | {increment['loss_reduction']:.6f} | "
                f"{increment['correct_label_margin_gain']:.6f} | "
                f"{comparison['loss_reduction']:.6f} | "
                f"{comparison['correct_label_margin_gain']:.6f} | "
                f"{str(passed).lower()} |"
            )
    rows.extend(
        [
            "",
            "## Controls",
            "",
            f"- Parameter match: `{json.dumps(decision['parameter_match'], sort_keys=True)}`",
            f"- Uniform routing identity: `{json.dumps(decision['routing_uniform_identity'], sort_keys=True)}`",
            "",
        ]
    )
    return "\n".join(rows)


def main() -> None:
    args = parse_args()
    positive = (
        args.baseline_train_limit,
        args.train_limit,
        args.valid_limit,
        args.test_limit,
        args.baseline_epochs,
        args.core_epochs,
        args.evidence_epochs,
        args.routing_epochs,
        args.baseline_batch_size,
        args.plugin_batch_size,
        args.random_repeats,
    )
    if min(positive) <= 0:
        raise ValueError("limits, epochs, batch sizes, and random_repeats must be positive")
    if args.min_baseline_macro_f1 < 0.0:
        raise ValueError("min_baseline_macro_f1 must be non-negative")

    output_dir = _resolve_path(args.output_dir) if args.output_dir else _default_output_dir()
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse transfer-screen directory: {output_dir}")
    output_dir.mkdir(parents=True)
    logs_dir = output_dir / "logs"
    paths, subset_manifest = _prepare_subsets(args, output_dir / "private_subsets")
    config = {**vars(args), "output_dir": str(output_dir)}
    _write_json(output_dir / "run_config.json", config)

    model_dir = output_dir / "baseline"
    _run_stage("baseline_train", _baseline_command(args, paths, model_dir), logs_dir)
    baseline_training = _read_json(model_dir / "metrics.json")
    baseline_macro_f1 = float(baseline_training["best_valid"]["macro_f1"])
    if baseline_macro_f1 < args.min_baseline_macro_f1:
        raise RuntimeError(
            "shared baseline is too weak for a transfer decision: "
            f"macro_f1={baseline_macro_f1:.6f} < {args.min_baseline_macro_f1:.6f}"
        )

    stage_dirs = {
        stage: {family: output_dir / stage / family for family in FAMILIES}
        for stage in STAGES
    }
    checkpoints: dict[str, dict[str, Path]] = {stage: {} for stage in STAGES}
    for family in FAMILIES:
        _run_stage(
            f"core_{family}_train",
            _core_command(args, family, paths, model_dir, stage_dirs["core"][family]),
            logs_dir,
        )
        checkpoints["core"][family] = (
            stage_dirs["core"][family] / "attention_score_kernel.pt"
        )
    for family in FAMILIES:
        _run_stage(
            f"evidence_{family}_train",
            _evidence_command(
                args,
                family,
                paths,
                model_dir,
                checkpoints["core"][family],
                stage_dirs["evidence"][family],
            ),
            logs_dir,
        )
        checkpoints["evidence"][family] = (
            stage_dirs["evidence"][family] / "counterfactual_evidence.pt"
        )
    for family in FAMILIES:
        _run_stage(
            f"routing_{family}_train",
            _routing_command(
                args,
                family,
                paths,
                model_dir,
                checkpoints["evidence"][family],
                stage_dirs["routing"][family],
            ),
            logs_dir,
        )
        checkpoints["routing"][family] = (
            stage_dirs["routing"][family] / "expert_router.pt"
        )

    device = choose_device(args.device)
    baseline: dict[str, dict[str, float]] = {}
    evaluations: dict[str, dict[str, dict[str, Any]]] = {
        stage: {family: {} for family in FAMILIES} for stage in STAGES
    }
    evaluation_dir = output_dir / "evaluation"
    for split in SPLITS:
        if split == "test":
            _write_json(
                output_dir / "test_evaluation_started.json",
                {"after_validation": True, "checkpoints": {
                    stage: {family: str(path) for family, path in family_paths.items()}
                    for stage, family_paths in checkpoints.items()
                }},
            )
        baseline[split] = _evaluate_baseline(
            model_dir,
            paths[split],
            batch_size=args.plugin_batch_size,
            device=device,
        )
        _write_json(evaluation_dir / "baseline" / f"{split}.json", baseline[split])
        for stage in STAGES:
            for family in FAMILIES:
                target = evaluation_dir / stage / family / split
                _run_stage(
                    f"evaluate_{stage}_{family}_{split}",
                    _evaluation_command(
                        args,
                        model_dir,
                        checkpoints[stage][family],
                        paths[split],
                        target,
                    ),
                    logs_dir,
                )
                payload = _read_json(target / "metrics.json")
                _assert_shared_baseline(
                    baseline[split], payload["baseline"], stage=f"{stage}/{family}/{split}"
                )
                evaluations[stage][family][split] = payload

    stage_metrics = {
        stage: {
            family: {
                split: evaluations[stage][family][split]["steered"]
                for split in SPLITS
            }
            for family in FAMILIES
        }
        for stage in STAGES
    }
    routing_uniform = {
        family: {
            split: evaluations["routing"][family][split]["uniform_routing"]
            for split in SPLITS
        }
        for family in FAMILIES
    }
    metric_files = {
        stage: {
            family: _read_json(stage_dirs[stage][family] / "metrics.json")
            for family in FAMILIES
        }
        for stage in STAGES
    }
    parameter_counts = {
        "core": {
            family: metric_files["core"][family]["trainable_parameters"]
            for family in FAMILIES
        },
        "evidence": {
            family: metric_files["evidence"][family]["selector_trainable_parameters"]
            for family in FAMILIES
        },
        "routing": {
            family: metric_files["routing"][family]["router_trainable_parameters"]
            for family in FAMILIES
        },
    }
    decision = summarize_transfer_screen(
        baseline=baseline,
        stages=stage_metrics,
        routing_uniform=routing_uniform,
        parameter_counts=parameter_counts,
    )
    summary = {
        "config": config,
        "subset_manifest": subset_manifest,
        "decision": decision,
    }
    _write_json(output_dir / "transfer_screen_summary.json", summary)
    (output_dir / "transfer_screen_summary.md").write_text(
        _render_summary(summary), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "event": "screen_completed",
                "output_dir": str(output_dir),
                "screen_pass": decision["screen_pass"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
