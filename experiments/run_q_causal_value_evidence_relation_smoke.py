from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q_attention.adapters import AttentionScoreHookConfig, AttentionScoreKernelAdapter  # noqa: E402
from q_attention.experiments.progress import tracked_batches  # noqa: E402
from q_attention.experiments.relation_steering import (  # noqa: E402
    choose_device,
    load_relation_run,
    make_relation_loader,
    move_batch,
)
from q_attention.metrics import classification_metrics  # noqa: E402
from q_attention.plugins.q_causal_value_evidence import (  # noqa: E402
    CausalValueTransportConfig,
    build_causal_value_transport_kernel,
)
from q_attention.tasks.relation import (  # noqa: E402
    load_relation_jsonl,
    sample_relation_records,
    sample_relation_records_proportional,
    write_relation_jsonl,
)


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", default="configs/q_vres_relation_smoke.json")
    bootstrap_args, _ = bootstrap.parse_known_args()
    config_payload: dict[str, Any] = {}
    if bootstrap_args.config is not None:
        config_payload = json.loads(
            resolve_path(bootstrap_args.config).read_text(encoding="utf-8")
        )
    configurable = {
        "train_path",
        "valid_path",
        "output_root",
        "output_dir",
        "model_dir",
        "device",
        "max_train_records",
        "max_valid_records",
        "epochs",
        "batch_size",
        "dim",
        "num_layers",
        "num_heads",
        "ff_dim",
        "dropout",
        "lr",
        "log_every_batches",
        "seed",
        "register_qubits",
        "depth",
        "angle_scale",
        "max_transport",
        "initial_transport",
        "evidence_floor",
    }
    parser = argparse.ArgumentParser(
        description="Run a short real-data Q-VRES integration smoke test."
    )
    parser.add_argument("--config", default=bootstrap_args.config)
    parser.add_argument(
        "--train_path",
        default="data/relation/retacred/train.jsonl",
    )
    parser.add_argument(
        "--valid_path",
        default="data/relation/retacred/valid.jsonl",
    )
    parser.add_argument(
        "--output_root",
        default="runs/q_causal_value_evidence_relation_smoke",
    )
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max_train_records", type=int, default=32)
    parser.add_argument("--max_valid_records", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ff_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log_every_batches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--register_qubits", type=int, default=2)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--angle_scale", type=float, default=1.0)
    parser.add_argument("--max_transport", type=float, default=0.25)
    parser.add_argument("--initial_transport", type=float, default=0.05)
    parser.add_argument("--evidence_floor", type=float, default=1e-6)
    parser.set_defaults(
        **{key: value for key, value in config_payload.items() if key in configurable}
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_run_dir(output_root: str | Path, explicit_output: str | None) -> Path:
    if explicit_output is not None:
        output_dir = resolve_path(explicit_output)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    root = resolve_path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for duplicate_index in range(1000):
        suffix = "" if duplicate_index == 0 else f"-{duplicate_index:02d}"
        output_dir = root / f"{timestamp}{suffix}"
        try:
            output_dir.mkdir()
        except FileExistsError:
            continue
        return output_dir
    raise RuntimeError(f"could not create a unique run directory under {root}")


def materialize_subset(
    source_path: Path,
    output_path: Path,
    limit: int,
    *,
    seed: int,
    split: str,
    required_labels: set[str] | None = None,
) -> dict[str, Any]:
    records = load_relation_jsonl(source_path)
    if limit <= 0 or len(records) <= limit:
        selected = records
        sampling = "source"
    elif split == "train":
        required = sorted(required_labels or ())
        if len(required) > limit:
            raise ValueError(
                "max_train_records is smaller than the number of labels required "
                "by the validation subset"
            )
        by_label: dict[str, list[int]] = {}
        for index, record in enumerate(records):
            by_label.setdefault(record.label, []).append(index)
        required_indices: list[int] = []
        for label in required:
            if label not in by_label:
                raise ValueError(f"validation label {label!r} is absent from the training source")
            required_indices.append(by_label[label][0])
        remaining_indices = [
            index for index in range(len(records)) if index not in set(required_indices)
        ]
        remaining_records = [records[index] for index in remaining_indices]
        selected = [records[index] for index in required_indices]
        selected.extend(
            sample_relation_records(
                remaining_records,
                limit - len(selected),
                seed=seed,
            )
        )
        sampling = "balanced_stratified_with_validation_label_coverage"
    else:
        selected = sample_relation_records_proportional(records, limit, seed=seed)
        sampling = "proportional_stratified"
    write_relation_jsonl(selected, output_path)
    return {
        "source_path": str(source_path),
        "source_sha256": file_sha256(source_path),
        "path": str(output_path),
        "records": len(selected),
        "source_records": len(records),
        "sampling": sampling,
        "seed": seed,
    }


def run_logged_command(command: list[str], log_path: Path) -> dict[str, Any]:
    print(json.dumps({"event": "command_start", "command": command}, sort_keys=True), flush=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_handle.write(line)
        return_code = process.wait()
    result = {
        "command": command,
        "returncode": return_code,
        "duration_seconds": round(time.perf_counter() - started, 3),
        "log_path": str(log_path),
    }
    print(json.dumps({"event": "command_complete", **result}, sort_keys=True), flush=True)
    if return_code != 0:
        raise RuntimeError(f"command failed with return code {return_code}: {command}")
    return result


def metric_delta(current: Mapping[str, float], baseline: Mapping[str, float]) -> dict[str, float]:
    keys = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "loss")
    return {f"delta_{key}": float(current[key]) - float(baseline[key]) for key in keys}


def evaluate_model(
    model: torch.nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    num_labels: int,
    *,
    adapter: AttentionScoreKernelAdapter | None = None,
    stage: str,
) -> dict[str, Any]:
    model.eval()
    predictions: list[int] = []
    labels: list[int] = []
    total_loss = 0.0
    total_items = 0
    total_batches = len(loader)  # type: ignore[arg-type]
    batches = tracked_batches(
        loader,
        total_batches=total_batches,
        stage=stage,
        phase="validation",
        log_every_batches=1,
    )
    with torch.no_grad():
        for batch in batches:
            batch = move_batch(batch, device)
            if adapter is None:
                logits = model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["subject_mask"],
                    batch["object_mask"],
                )
            else:
                hook_config = AttentionScoreHookConfig(
                    attention_mask=batch["attention_mask"],
                    subject_mask=batch["subject_mask"],
                    object_mask=batch["object_mask"],
                )
                with adapter.steering(hook_config):
                    logits = model(
                        batch["input_ids"],
                        batch["attention_mask"],
                        batch["subject_mask"],
                        batch["object_mask"],
                    )
            if not torch.isfinite(logits).all():
                raise FloatingPointError(f"non-finite logits during {stage}")
            loss = F.cross_entropy(logits, batch["labels"])
            total_loss += float(loss.item()) * int(batch["labels"].shape[0])
            total_items += int(batch["labels"].shape[0])
            predictions.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())
    metrics = classification_metrics(predictions, labels, num_labels)
    metrics["loss"] = total_loss / max(total_items, 1)
    return {"metrics": metrics, "items": total_items, "batches": total_batches}


def probe_qvres_inputs(
    model: torch.nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    kernel: torch.nn.Module,
    adapter: AttentionScoreKernelAdapter,
    num_labels: int,
) -> dict[str, Any]:
    calls = {path: 0 for path in model.score_module_paths}  # type: ignore[attr-defined]
    value_calls = {path: 0 for path in model.score_module_paths}  # type: ignore[attr-defined]
    input_shapes: dict[str, list[list[int]]] = {path: [] for path in calls}
    errors: list[str] = []
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_probe(path: str):
        def probe(_module: torch.nn.Module, inputs: tuple[object, ...]) -> None:
            calls[path] += 1
            if len(inputs) != 4:
                errors.append(f"{path}: expected 4 score-hook inputs, got {len(inputs)}")
                return
            if not all(isinstance(item, torch.Tensor) for item in inputs):
                errors.append(f"{path}: score-hook inputs are not all tensors")
                return
            if inputs[3] is not None:
                value_calls[path] += 1
            input_shapes[path].append([list(item.shape) for item in inputs])  # type: ignore[union-attr]

        return probe

    for path in calls:
        module = dict(model.named_modules())[path]  # type: ignore[attr-defined]
        handles.append(module.register_forward_pre_hook(make_probe(path)))
    try:
        result = evaluate_model(
            model,
            loader,
            device,
            num_labels,
            adapter=adapter,
            stage="q_vres_probe",
        )
    finally:
        for handle in handles:
            handle.remove()
    expected_calls = result["batches"]
    expected_per_layer = expected_calls
    return {
        "evaluation": result,
        "calls": calls,
        "value_calls": value_calls,
        "input_shapes": input_shapes,
        "errors": errors,
        "expected_calls_per_layer": expected_per_layer,
        "all_layers_called": all(value == expected_per_layer for value in calls.values()),
        "all_calls_received_value": all(value == calls[path] for path, value in value_calls.items()),
        "kernel_parameters": sum(parameter.numel() for parameter in kernel.parameters()),
    }


def gradient_probe(
    model: torch.nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    kernel: torch.nn.Module,
    adapter: AttentionScoreKernelAdapter,
) -> dict[str, Any]:
    batch = next(iter(loader))
    batch = move_batch(batch, device)
    model.zero_grad(set_to_none=True)
    kernel.zero_grad(set_to_none=True)
    hook_config = AttentionScoreHookConfig(
        attention_mask=batch["attention_mask"],
        subject_mask=batch["subject_mask"],
        object_mask=batch["object_mask"],
    )
    adapter.attach(hook_config)
    try:
        logits = model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["subject_mask"],
            batch["object_mask"],
        )
        loss = F.cross_entropy(logits, batch["labels"])
        loss.backward()
    finally:
        adapter.remove()
    gradients = [parameter.grad for parameter in kernel.parameters()]
    finite = all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    nonzero = sum(
        int(gradient is not None and bool(torch.any(gradient.abs() > 0).item()))
        for gradient in gradients
    )
    return {
        "loss": float(loss.item()),
        "parameter_count": len(gradients),
        "finite_gradients": finite,
        "nonzero_gradient_tensors": nonzero,
    }


def markdown_summary(summary: Mapping[str, Any]) -> str:
    baseline = summary["baseline"]["metrics"]
    qvres = summary["q_vres"]["metrics"]
    checks = summary["integration_checks"]
    lines = [
        "# Q-VRES Real-Data Integration Smoke",
        "",
        f"Run directory: `{summary['run_dir']}`",
        "",
        "This is an integration smoke test, not a task-level result claim.",
        "",
        "| variant | accuracy | macro_f1 | loss |",
        "| --- | ---: | ---: | ---: |",
        f"| baseline | {baseline['accuracy']:.6f} | {baseline['macro_f1']:.6f} | {baseline['loss']:.6f} |",
        f"| q_vres | {qvres['accuracy']:.6f} | {qvres['macro_f1']:.6f} | {qvres['loss']:.6f} |",
        "",
        "## Integration checks",
        "",
        f"- Status: `{summary['status']}`",
        f"- Hook calls per layer: `{checks['expected_calls_per_layer']}`",
        f"- All layers called: `{checks['all_layers_called']}`",
        f"- Every call received value: `{checks['all_calls_received_value']}`",
        f"- Finite gradients: `{checks['gradient_probe']['finite_gradients']}`",
        f"- Nonzero Q-VRES gradient tensors: `{checks['gradient_probe']['nonzero_gradient_tensors']}`",
        f"- Probe errors: `{checks['probe_errors']}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.max_train_records <= 0 or args.max_valid_records <= 0:
        raise ValueError("max_train_records and max_valid_records must be positive")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if args.dim % args.num_heads != 0:
        raise ValueError("dim must be divisible by num_heads")
    set_seed(args.seed)
    device = choose_device(args.device)
    run_dir = create_run_dir(args.output_root, args.output_dir)
    data_dir = run_dir / "smoke_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_source = resolve_path(args.train_path)
    valid_source = resolve_path(args.valid_path)
    train_path = data_dir / "train.jsonl"
    valid_path = data_dir / "valid.jsonl"
    valid_info = materialize_subset(
        valid_source,
        valid_path,
        args.max_valid_records,
        seed=args.seed,
        split="valid",
    )
    valid_records_for_coverage = load_relation_jsonl(valid_path)
    train_info = materialize_subset(
        train_source,
        train_path,
        args.max_train_records,
        seed=args.seed,
        split="train",
        required_labels={record.label for record in valid_records_for_coverage},
    )

    config_payload = vars(args).copy()
    config_payload.update(
        {
            "run_dir": str(run_dir),
            "device_resolved": str(device),
            "train": train_info,
            "valid": valid_info,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    (run_dir / "run_config.json").write_text(
        json.dumps(config_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"event": "run_started", "run_dir": str(run_dir), "device": str(device)}, sort_keys=True), flush=True)

    baseline_dir = resolve_path(args.model_dir) if args.model_dir else run_dir / "baseline"
    command_result = None
    if args.model_dir is None:
        command = [
            sys.executable,
            str(ROOT / "experiments" / "train_relation_baseline.py"),
            "--train_path",
            str(train_path),
            "--valid_path",
            str(valid_path),
            "--output_dir",
            str(baseline_dir),
            "--device",
            args.device,
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--dim",
            str(args.dim),
            "--num_layers",
            str(args.num_layers),
            "--num_heads",
            str(args.num_heads),
            "--ff_dim",
            str(args.ff_dim),
            "--dropout",
            str(args.dropout),
            "--lr",
            str(args.lr),
            "--log_every_batches",
            str(args.log_every_batches),
            "--seed",
            str(args.seed),
        ]
        command_result = run_logged_command(command, run_dir / "baseline_train.log")

    artifacts = load_relation_run(baseline_dir, device)
    valid_records = load_relation_jsonl(valid_path)
    valid_loader = make_relation_loader(
        valid_records,
        artifacts.vocab,
        artifacts.label_to_id,
        batch_size=args.batch_size,
    )
    baseline_eval = evaluate_model(
        artifacts.model,
        valid_loader,
        device,
        len(artifacts.label_to_id),
        stage="baseline_eval",
    )
    kernel_config = CausalValueTransportConfig(
        num_layers=artifacts.model.config.num_layers,
        num_heads=artifacts.model.config.num_heads,
        head_dim=artifacts.model.config.dim // artifacts.model.config.num_heads,
        register_qubits=args.register_qubits,
        depth=args.depth,
        angle_scale=args.angle_scale,
        max_transport=args.max_transport,
        initial_transport=args.initial_transport,
        evidence_floor=args.evidence_floor,
        seed=args.seed + 307,
    )
    kernel = build_causal_value_transport_kernel("quantum", kernel_config).to(device)
    adapter = AttentionScoreKernelAdapter(artifacts.model, artifacts.model.score_module_paths, kernel)
    probe = probe_qvres_inputs(
        artifacts.model,
        valid_loader,
        device,
        kernel,
        adapter,
        len(artifacts.label_to_id),
    )
    qvres_eval = probe["evaluation"]
    gradient = gradient_probe(artifacts.model, valid_loader, device, kernel, adapter)
    checks = {
        "all_layers_called": probe["all_layers_called"],
        "all_calls_received_value": probe["all_calls_received_value"],
        "finite_qvres_metrics": all(torch.isfinite(torch.tensor(value)) for value in qvres_eval["metrics"].values()),
        "gradient_probe": gradient,
        "probe_errors": probe["errors"],
        "hook_call_counts": probe["calls"],
        "value_call_counts": probe["value_calls"],
        "expected_calls_per_layer": probe["expected_calls_per_layer"],
    }
    checks["passed"] = bool(
        checks["all_layers_called"]
        and checks["all_calls_received_value"]
        and checks["finite_qvres_metrics"]
        and gradient["finite_gradients"]
        and not checks["probe_errors"]
    )
    summary = {
        "schema_version": 1,
        "run_type": "q_vres_real_data_integration_smoke",
        "status": "pass" if checks["passed"] else "fail",
        "run_dir": str(run_dir),
        "device": str(device),
        "baseline": {
            "model_dir": str(baseline_dir),
            "metrics": baseline_eval["metrics"],
            "items": baseline_eval["items"],
            "command": command_result,
        },
        "q_vres": {
            "metrics": qvres_eval["metrics"],
            "delta_vs_baseline": metric_delta(qvres_eval["metrics"], baseline_eval["metrics"]),
            "items": qvres_eval["items"],
            "metadata": kernel.metadata(),
            "trainable_parameters": sum(parameter.numel() for parameter in kernel.parameters()),
            "probe_shapes": probe["input_shapes"],
        },
        "integration_checks": checks,
        "provenance": {
            "train": train_info,
            "valid": valid_info,
            "git_commit": _git_output("rev-parse", "HEAD"),
            "git_branch": _git_output("branch", "--show-current"),
            "git_dirty": bool(_git_output("status", "--porcelain")),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (run_dir / "run_summary.md").write_text(markdown_summary(summary), encoding="utf-8")
    print(json.dumps({"event": "run_complete", "status": summary["status"], "run_dir": str(run_dir)}, sort_keys=True), flush=True)
    if summary["status"] != "pass":
        raise RuntimeError(f"Q-VRES integration checks failed; see {run_dir / 'run_summary.json'}")


def _git_output(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()


if __name__ == "__main__":
    main()
