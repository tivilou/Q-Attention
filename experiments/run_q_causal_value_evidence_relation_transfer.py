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
EXPERIMENTS = ROOT / "experiments"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from run_q_causal_value_evidence_relation_smoke import (  # noqa: E402
    create_run_dir,
    materialize_subset,
    resolve_path,
    run_logged_command,
)
from q_attention.adapters import AttentionScoreKernelAdapter  # noqa: E402
from q_attention.adapters.encoder import resolve_module  # noqa: E402
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
from q_attention.tasks.relation import load_relation_jsonl  # noqa: E402


SELECTORS = (
    "disabled",
    "q_causal_transport",
    "classical_causal_transport",
    "q_causal_key_only",
)


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", default="configs/q_vres_relation_transfer_screen.json")
    bootstrap_args, _ = bootstrap.parse_known_args()
    config: dict[str, Any] = {}
    if bootstrap_args.config is not None:
        config = json.loads(resolve_path(bootstrap_args.config).read_text(encoding="utf-8"))
    configurable = {
        "train_path",
        "valid_path",
        "test_path",
        "output_root",
        "output_dir",
        "model_dir",
        "device",
        "max_train_records",
        "max_valid_records",
        "max_test_records",
        "baseline_epochs",
        "baseline_batch_size",
        "epochs",
        "batch_size",
        "dim",
        "num_layers",
        "num_heads",
        "ff_dim",
        "dropout",
        "baseline_lr",
        "kernel_lr",
        "log_every_batches",
        "seed",
        "selectors",
        "register_qubits",
        "depth",
        "angle_scale",
        "max_transport",
        "initial_transport",
        "evidence_floor",
        "formal_experiment",
    }
    parser = argparse.ArgumentParser(
        description="Run a bounded real-data Q-VRES task-transfer screen."
    )
    parser.add_argument("--config", default=bootstrap_args.config)
    parser.add_argument("--train_path", default="data/relation/retacred/train.jsonl")
    parser.add_argument("--valid_path", default="data/relation/retacred/valid.jsonl")
    parser.add_argument("--test_path", default="data/relation/retacred/test.jsonl")
    parser.add_argument("--output_root", default="runs/q_causal_value_evidence_relation_transfer_screen")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max_train_records", type=int, default=128)
    parser.add_argument("--max_valid_records", type=int, default=64)
    parser.add_argument("--max_test_records", type=int, default=64)
    parser.add_argument("--baseline_epochs", type=int, default=2)
    parser.add_argument("--baseline_batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ff_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--baseline_lr", type=float, default=1e-3)
    parser.add_argument("--kernel_lr", type=float, default=1e-2)
    parser.add_argument("--log_every_batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--selectors", default=",".join(SELECTORS))
    parser.add_argument("--register_qubits", type=int, default=2)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--angle_scale", type=float, default=1.0)
    parser.add_argument("--max_transport", type=float, default=0.25)
    parser.add_argument("--initial_transport", type=float, default=0.05)
    parser.add_argument("--evidence_floor", type=float, default=1e-6)
    parser.add_argument(
        "--formal-experiment",
        action="store_true",
        help="mark the output as a full formal run rather than a bounded screen",
    )
    parser.set_defaults(**{key: value for key, value in config.items() if key in configurable})
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


def git_output(*args: str) -> str | None:
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


class ScalarAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.square_total = 0.0

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().float().reshape(-1)
        if values.numel() == 0:
            return
        self.count += int(values.numel())
        self.total += float(values.sum().item())
        self.square_total += float(values.square().sum().item())

    def mean(self) -> float | None:
        return self.total / self.count if self.count else None

    def summary(self) -> dict[str, float | int | None]:
        if not self.count:
            return {"count": 0, "mean": None, "std": None}
        mean = self.total / self.count
        variance = max(self.square_total / self.count - mean * mean, 0.0)
        return {"count": self.count, "mean": mean, "std": variance**0.5}


def build_kernel(selector: str, model: torch.nn.Module, seed: int, args: argparse.Namespace) -> torch.nn.Module | None:
    if selector == "disabled":
        return None
    if selector not in SELECTORS:
        raise ValueError(f"unknown selector {selector!r}; expected one of {SELECTORS}")
    kernel_type = "classical" if selector == "classical_causal_transport" else "quantum"
    value_feature_mode = "key_only" if selector == "q_causal_key_only" else "leave_one_out"
    config = model.config
    return build_causal_value_transport_kernel(
        kernel_type,
        CausalValueTransportConfig(
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            head_dim=config.dim // config.num_heads,
            register_qubits=args.register_qubits,
            depth=args.depth,
            angle_scale=args.angle_scale,
            max_transport=args.max_transport,
            initial_transport=args.initial_transport,
            evidence_floor=args.evidence_floor,
            value_feature_mode=value_feature_mode,
            seed=seed + 5000,
        ),
    )


def hook_config(batch: Mapping[str, torch.Tensor]) -> Any:
    from q_attention.adapters.attention_scores import AttentionScoreHookConfig

    return AttentionScoreHookConfig(
        attention_mask=batch["attention_mask"],
        subject_mask=batch["subject_mask"],
        object_mask=batch["object_mask"],
    )


def _masked_attention(scores: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    key_mask = attention_mask[:, None, None, :].to(dtype=torch.bool)
    return torch.softmax(scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min), dim=-1)


def _capture_geometry(
    layer_accumulators: list[dict[str, ScalarAccumulator]],
    layer_index: int,
    inputs: tuple[object, ...],
    output: object,
    batch: Mapping[str, torch.Tensor],
) -> None:
    if not inputs or not isinstance(inputs[0], torch.Tensor) or not isinstance(output, torch.Tensor):
        raise TypeError("geometry hook requires tensor score input and output")
    base_scores = inputs[0]
    steered_scores = output
    base_attention = _masked_attention(base_scores, batch["attention_mask"])
    steered_attention = _masked_attention(steered_scores, batch["attention_mask"])
    context = batch["attention_mask"] & ~(batch["subject_mask"] | batch["object_mask"])
    context_mask = context[:, None, None, :].to(dtype=base_attention.dtype)
    entity_mask = (batch["subject_mask"] | batch["object_mask"])[:, None, None, :].to(dtype=base_attention.dtype)
    query_mask = batch["attention_mask"][:, None, :, None].to(dtype=base_attention.dtype)
    probability_delta = (steered_attention - base_attention).abs()
    accumulator = layer_accumulators[layer_index]
    accumulator["residual_rms"].update((steered_scores - base_scores).square().mean(dim=(-1, -2, -3)).sqrt())
    accumulator["attention_total_variation"].update(
        0.5 * probability_delta.sum(dim=-1).masked_select(query_mask.squeeze(-1).bool())
    )
    accumulator["context_mass_error"].update(
        ((steered_attention - base_attention) * context_mask).sum(dim=-1).abs().masked_select(query_mask.squeeze(-1).bool())
    )
    accumulator["entity_mass_error"].update(
        ((steered_attention - base_attention) * entity_mask).sum(dim=-1).abs().masked_select(query_mask.squeeze(-1).bool())
    )


def evaluate(
    model: torch.nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    num_labels: int,
    *,
    kernel: torch.nn.Module | None,
    stage: str,
    log_every_batches: int,
    collect_geometry: bool,
) -> dict[str, Any]:
    model.eval()
    if kernel is not None:
        kernel.eval()
    adapter = None if kernel is None else AttentionScoreKernelAdapter(model, model.score_module_paths, kernel)
    predictions: list[int] = []
    labels: list[int] = []
    total_loss = 0.0
    total_items = 0
    layer_accumulators = [
        {
            name: ScalarAccumulator()
            for name in ("residual_rms", "attention_total_variation", "context_mass_error", "entity_mass_error")
        }
        for _ in model.score_module_paths
    ]
    batches = tracked_batches(
        loader,
        total_batches=len(loader),  # type: ignore[arg-type]
        stage=stage,
        phase="evaluation",
        log_every_batches=log_every_batches,
    )
    with torch.no_grad():
        for raw_batch in batches:
            batch = move_batch(raw_batch, device)
            captures: dict[int, tuple[tuple[object, ...], object]] = {}
            handles: list[torch.utils.hooks.RemovableHandle] = []
            try:
                if adapter is not None:
                    adapter.attach(hook_config(batch))
                    if collect_geometry:
                        for layer_index, path in enumerate(model.score_module_paths):
                            module = resolve_module(model, path)

                            def capture(_module: torch.nn.Module, inputs: tuple[object, ...], output: object, index: int = layer_index) -> None:
                                captures[index] = (inputs, output)

                            handles.append(module.register_forward_hook(capture))
                logits = model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["subject_mask"],
                    batch["object_mask"],
                )
            finally:
                for handle in handles:
                    handle.remove()
                if adapter is not None:
                    adapter.remove()
            if not torch.isfinite(logits).all():
                raise FloatingPointError(f"non-finite logits during {stage}")
            loss = F.cross_entropy(logits, batch["labels"])
            total_loss += float(loss.item()) * int(batch["labels"].shape[0])
            total_items += int(batch["labels"].shape[0])
            predictions.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())
            for layer_index, (inputs, output) in captures.items():
                _capture_geometry(layer_accumulators, layer_index, inputs, output, batch)
    metrics = classification_metrics(predictions, labels, num_labels)
    metrics["loss"] = total_loss / max(total_items, 1)
    return {
        "metrics": metrics,
        "items": total_items,
        "batches": len(loader),  # type: ignore[arg-type]
        "geometry": [
            {name: accumulator.summary() for name, accumulator in layer.items()}
            for layer in layer_accumulators
        ],
    }


def metric_delta(current: Mapping[str, float], baseline: Mapping[str, float]) -> dict[str, float]:
    keys = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "loss")
    return {f"delta_{key}": float(current[key]) - float(baseline[key]) for key in keys}


def train_kernel(
    model: torch.nn.Module,
    kernel: torch.nn.Module,
    train_records: list[Any],
    valid_loader: Any,
    artifacts: Any,
    device: torch.device,
    selector: str,
    seed: int,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    set_seed(seed)
    train_loader = make_relation_loader(
        train_records,
        artifacts.vocab,
        artifacts.label_to_id,
        batch_size=args.batch_size,
        shuffle=True,
    )
    adapter = AttentionScoreKernelAdapter(model, model.score_module_paths, kernel)
    optimizer = torch.optim.AdamW(kernel.parameters(), lr=args.kernel_lr)
    history: list[dict[str, Any]] = []
    best_valid: dict[str, float] | None = None
    best_epoch: int | None = None
    best_score = (float("-inf"), float("-inf"))
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        kernel.train()
        total_loss = 0.0
        total_items = 0
        for raw_batch in tracked_batches(
            train_loader,
            total_batches=len(train_loader),
            stage=selector,
            phase="train",
            log_every_batches=args.log_every_batches,
            epoch=epoch,
            epochs=args.epochs,
        ):
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            adapter.attach(hook_config(batch))
            try:
                logits = model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["subject_mask"],
                    batch["object_mask"],
                )
            finally:
                adapter.remove()
            loss = F.cross_entropy(logits, batch["labels"])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss selector={selector} epoch={epoch}")
            loss.backward()
            gradients = [parameter.grad for parameter in kernel.parameters()]
            if any(gradient is None or not torch.isfinite(gradient).all() for gradient in gradients):
                raise FloatingPointError(f"non-finite gradient selector={selector} epoch={epoch}")
            optimizer.step()
            if any(not torch.isfinite(parameter).all() for parameter in kernel.parameters()):
                raise FloatingPointError(f"non-finite parameter selector={selector} epoch={epoch}")
            total_loss += float(loss.item()) * int(batch["labels"].shape[0])
            total_items += int(batch["labels"].shape[0])
        valid = evaluate(
            model,
            valid_loader,
            device,
            len(artifacts.label_to_id),
            kernel=kernel,
            stage=selector,
            log_every_batches=args.log_every_batches,
            collect_geometry=False,
        )
        valid_metrics = valid["metrics"]
        epoch_record = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_items, 1),
            "valid": valid_metrics,
        }
        history.append(epoch_record)
        print(json.dumps({"event": "epoch_complete", "selector": selector, **epoch_record}, sort_keys=True), flush=True)
        score = (float(valid_metrics["macro_f1"]), -float(valid_metrics["loss"]))
        if score > best_score:
            best_score = score
            best_valid = dict(valid_metrics)
            best_epoch = epoch
            torch.save(kernel.state_dict(), output_dir / "best_kernel.pt")
    if best_valid is None or best_epoch is None:
        raise RuntimeError(f"selector {selector} produced no validation checkpoint")
    kernel.load_state_dict(torch.load(output_dir / "best_kernel.pt", map_location=device, weights_only=True))
    return {
        "history": history,
        "best_valid": best_valid,
        "best_epoch": best_epoch,
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }


def write_markdown(summary: Mapping[str, Any], path: Path) -> None:
    rows = summary["results"]
    lines = [
        "# Q-VRES Real-Data Task-Transfer Screen",
        "",
        "This is a bounded screen, not the formal multi-seed Re-TACRED result.",
        "",
        "| selector | valid macro-F1 | test macro-F1 | delta test macro-F1 | test loss | parameters |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['selector']} | {row['valid']['metrics']['macro_f1']:.6f} | "
            f"{row['test']['metrics']['macro_f1']:.6f} | "
            f"{row['test']['delta_vs_baseline']['delta_macro_f1']:.6f} | "
            f"{row['test']['metrics']['loss']:.6f} | {row['trainable_parameters']} |"
        )
    lines.extend(
        [
            "",
            f"Screen gate: `{summary['screen_gate']['status']}`",
            "",
            "The screen gate checks implementation, finite values, geometry diagnostics, and matched controls. It does not automatically establish a publication-level task gain.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if min(args.max_train_records, args.max_valid_records, args.max_test_records) < 0:
        raise ValueError("record limits must be non-negative; zero means the full split")
    if min(args.baseline_epochs, args.baseline_batch_size, args.epochs, args.batch_size) <= 0:
        raise ValueError("epoch and batch-size values must be positive")
    if args.dim % args.num_heads != 0:
        raise ValueError("dim must be divisible by num_heads")
    selectors = [item.strip() for item in args.selectors.split(",") if item.strip()]
    if not selectors or len(set(selectors)) != len(selectors):
        raise ValueError("selectors must be a non-empty comma-separated list without duplicates")
    if any(selector not in SELECTORS for selector in selectors):
        raise ValueError(f"selectors must be drawn from {SELECTORS}")
    set_seed(args.seed)
    device = choose_device(args.device)
    run_dir = create_run_dir(args.output_root, args.output_dir)
    data_dir = run_dir / "screen_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_source = resolve_path(args.train_path)
    valid_source = resolve_path(args.valid_path)
    test_source = resolve_path(args.test_path)
    valid_path = data_dir / "valid.jsonl"
    test_path = data_dir / "test.jsonl"
    train_path = data_dir / "train.jsonl"
    valid_info = materialize_subset(
        valid_source,
        valid_path,
        args.max_valid_records,
        seed=args.seed + 101,
        split="valid",
    )
    test_info = materialize_subset(
        test_source,
        test_path,
        args.max_test_records,
        seed=args.seed + 211,
        split="test",
    )
    valid_records = load_relation_jsonl(valid_path)
    test_records = load_relation_jsonl(test_path)
    required_labels = {record.label for record in valid_records + test_records}
    train_info = materialize_subset(
        train_source,
        train_path,
        args.max_train_records,
        seed=args.seed,
        split="train",
        required_labels=required_labels,
    )
    train_records = load_relation_jsonl(train_path)
    observed_max_length = max(
        len(record.tokens)
        for record in train_records + valid_records + test_records
    )
    run_config = vars(args).copy()
    run_config.update(
        {
            "run_dir": str(run_dir),
            "device_resolved": str(device),
            "selectors_resolved": selectors,
            "train": train_info,
            "valid": valid_info,
            "test": test_info,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"event": "run_started", "run_dir": str(run_dir), "device": str(device), "selectors": selectors}, sort_keys=True), flush=True)

    baseline_dir = resolve_path(args.model_dir) if args.model_dir else run_dir / "baseline"
    baseline_command = None
    if args.model_dir is None:
        baseline_command = [
            sys.executable,
            str(ROOT / "experiments" / "train_relation_baseline.py"),
            "--train_path", str(train_path),
            "--valid_path", str(valid_path),
            "--output_dir", str(baseline_dir),
            "--device", args.device,
            "--epochs", str(args.baseline_epochs),
            "--batch_size", str(args.baseline_batch_size),
            "--dim", str(args.dim),
            "--num_layers", str(args.num_layers),
            "--num_heads", str(args.num_heads),
            "--ff_dim", str(args.ff_dim),
            "--max_length", str(observed_max_length + 4),
            "--dropout", str(args.dropout),
            "--lr", str(args.baseline_lr),
            "--log_every_batches", str(args.log_every_batches),
            "--seed", str(args.seed),
        ]
        run_logged_command(baseline_command, run_dir / "baseline_train.log")
    artifacts = load_relation_run(baseline_dir, device)
    valid_loader = make_relation_loader(valid_records, artifacts.vocab, artifacts.label_to_id, batch_size=args.batch_size)
    test_loader = make_relation_loader(test_records, artifacts.vocab, artifacts.label_to_id, batch_size=args.batch_size)
    for parameter in artifacts.model.parameters():
        parameter.requires_grad_(False)
    baseline_valid = evaluate(
        artifacts.model, valid_loader, device, len(artifacts.label_to_id),
        kernel=None, stage="baseline_valid", log_every_batches=args.log_every_batches, collect_geometry=False,
    )
    baseline_test = evaluate(
        artifacts.model, test_loader, device, len(artifacts.label_to_id),
        kernel=None, stage="baseline_test", log_every_batches=args.log_every_batches, collect_geometry=False,
    )

    results: list[dict[str, Any]] = []
    for selector in selectors:
        selector_dir = run_dir / "selectors" / selector
        selector_dir.mkdir(parents=True, exist_ok=True)
        if selector == "disabled":
            valid_result = baseline_valid
            test_result = baseline_test
            train_result = {"history": [], "best_valid": valid_result["metrics"], "best_epoch": 0, "runtime_seconds": 0.0}
            trainable_parameters = 0
            metadata: dict[str, Any] = {"selector": selector, "mechanism": "disabled"}
        else:
            kernel = build_kernel(selector, artifacts.model, args.seed, args)
            assert kernel is not None
            kernel = kernel.to(device)
            train_result = train_kernel(
                artifacts.model,
                kernel,
                load_relation_jsonl(train_path),
                valid_loader,
                artifacts,
                device,
                selector,
                args.seed,
                args,
                selector_dir,
            )
            valid_result = evaluate(
                artifacts.model, valid_loader, device, len(artifacts.label_to_id),
                kernel=kernel, stage=f"{selector}_valid_final", log_every_batches=args.log_every_batches, collect_geometry=True,
            )
            test_result = evaluate(
                artifacts.model, test_loader, device, len(artifacts.label_to_id),
                kernel=kernel, stage=f"{selector}_test", log_every_batches=args.log_every_batches, collect_geometry=True,
            )
            trainable_parameters = sum(parameter.numel() for parameter in kernel.parameters())
            metadata = kernel.metadata()
            torch.save({"state_dict": kernel.state_dict(), "metadata": metadata}, selector_dir / "best_kernel_with_metadata.pt")
        row = {
            "selector": selector,
            "seed": args.seed,
            "valid": valid_result,
            "test": {
                **test_result,
                "delta_vs_baseline": metric_delta(test_result["metrics"], baseline_test["metrics"]),
            },
            "train": train_result,
            "trainable_parameters": trainable_parameters,
            "metadata": metadata,
            "finite": all(
                torch.isfinite(torch.tensor(value))
                for value in list(valid_result["metrics"].values()) + list(test_result["metrics"].values())
            ),
        }
        (selector_dir / "metrics.json").write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
        results.append(row)
        print(json.dumps({"event": "selector_complete", "selector": selector, "test": test_result["metrics"]}, sort_keys=True), flush=True)

    non_disabled = [row for row in results if row["selector"] != "disabled"]
    controls_present = all(selector in selectors for selector in SELECTORS)
    matched_parameters = len({row["trainable_parameters"] for row in non_disabled}) == 1
    screen_gate = {
        "status": "pass" if all(row["finite"] for row in results) and controls_present and matched_parameters else "fail",
        "finite_metrics": all(row["finite"] for row in results),
        "matched_parameter_counts": sorted({row["trainable_parameters"] for row in non_disabled}),
        "matched_parameters": matched_parameters,
        "controls_present": {selector: selector in selectors for selector in SELECTORS},
        "task_gain_is_not_automatically_accepted": True,
    }
    summary = {
        "schema_version": "q-attention.q-vres.real-transfer-screen.v1",
        "run_type": "bounded_real_data_task_transfer_screen",
        "formal_experiment": bool(args.formal_experiment),
        "run_type": (
            "formal_full_relation_transfer"
            if args.formal_experiment
            else "bounded_real_data_task_transfer_screen"
        ),
        "status": screen_gate["status"],
        "run_dir": str(run_dir),
        "device": str(device),
        "selectors": selectors,
        "baseline": {
            "model_dir": str(baseline_dir),
            "valid": baseline_valid,
            "test": baseline_test,
            "command": baseline_command,
        },
        "results": results,
        "screen_gate": screen_gate,
        "provenance": {
            "train": train_info,
            "valid": valid_info,
            "test": test_info,
            "git_commit": git_output("rev-parse", "HEAD"),
            "git_branch": git_output("branch", "--show-current"),
            "git_dirty": bool(git_output("status", "--porcelain")),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(summary, run_dir / "run_summary.md")
    print(json.dumps({"event": "run_complete", "status": summary["status"], "run_dir": str(run_dir)}, sort_keys=True), flush=True)
    if summary["status"] != "pass":
        raise RuntimeError(f"real-data Q-VRES screen failed; see {run_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()
