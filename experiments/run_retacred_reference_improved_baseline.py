#!/usr/bin/env python3
"""Run the pinned Improved Baseline through a declared project adapter.

The upstream encoder, marker construction, entity-position readout, loss, and
optimizer semantics remain active. This adapter adds deterministic data paths,
audited metrics, checkpoint/provenance artifacts, and bounded case-study
traces without changing the upstream model computation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q_attention.metrics import classification_report  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(*args: str, cwd: Path = ROOT) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=False, timeout=15
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-dir", required=True)
    parser.add_argument("--data-dir", required=True, help="prepared train.json/dev.json/test.json directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name-or-path", default="roberta-large")
    parser.add_argument("--input-format", default="typed_entity_marker_punct")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--test-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--adam-epsilon", type=float, default=1e-6)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--num-train-epochs", type=float, default=5.0)
    parser.add_argument("--evaluation-steps", type=int, default=500)
    parser.add_argument("--dropout-prob", type=float, default=0.1)
    parser.add_argument("--num-class", type=int, default=40)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--case-study-per-split", type=int, default=2)
    return parser.parse_args()


def set_seed(seed: int, torch: ModuleType, numpy: ModuleType) -> None:
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def import_upstream(upstream_dir: Path) -> tuple[ModuleType, ModuleType, ModuleType]:
    if str(upstream_dir) not in sys.path:
        sys.path.insert(0, str(upstream_dir))
    model_module = importlib.import_module("model")
    prepro_module = importlib.import_module("prepro")
    utils_module = importlib.import_module("utils")
    return model_module, prepro_module, utils_module


def as_namespace(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        **vars(args),
        config_name="",
        tokenizer_name="",
        project_name="q_attention_reference",
        run_name="retacred-improved-baseline",
        n_gpu=1,
    )


def load_split(preprocessor: Any, path: Path, args: SimpleNamespace) -> list[dict[str, Any]]:
    return preprocessor.read(str(path))


def batch_to_device(batch: tuple[Any, ...], device: Any) -> dict[str, Any]:
    return {
        "input_ids": batch[0].to(device),
        "attention_mask": batch[1].to(device),
        "labels": batch[2].to(device),
        "ss": batch[3].to(device),
        "os": batch[4].to(device),
    }


def evaluate(
    model: Any,
    features: list[dict[str, Any]],
    collate_fn: Any,
    torch: ModuleType,
    numpy: ModuleType,
    source_get_f1: Any,
    label_names: list[str],
    device: Any,
    batch_size: int,
) -> dict[str, Any]:
    from torch.utils.data import DataLoader
    import torch.nn.functional as F

    loader = DataLoader(features, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    model.eval()
    predictions: list[int] = []
    labels: list[int] = []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            inputs = batch_to_device(batch, device)
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                ss=inputs["ss"],
                os=inputs["os"],
            )
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            loss = F.cross_entropy(logits.float(), inputs["labels"])
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise FloatingPointError("reference baseline produced non-finite logits or loss")
            total_loss += float(loss.item()) * int(inputs["labels"].shape[0])
            predictions.extend(logits.argmax(dim=-1).detach().cpu().tolist())
            labels.extend(inputs["labels"].detach().cpu().tolist())
    report = classification_report(predictions, labels, len(label_names), label_names=label_names)
    source_precision, source_recall, source_f1 = source_get_f1(
        numpy.asarray(labels, dtype=numpy.int64), numpy.asarray(predictions, dtype=numpy.int64)
    )
    return {
        "items": len(labels),
        "batches": len(loader),
        "metrics": {
            **report["metrics"],
            "loss": total_loss / max(len(labels), 1),
            "source_non_null_micro_precision": source_precision,
            "source_non_null_micro_recall": source_recall,
            "source_non_null_micro_f1": source_f1,
        },
        "per_class": report["per_class"],
        "confusion_matrix": report["confusion_matrix"],
        "predictions": predictions,
        "labels": labels,
    }


def selection_key(result: dict[str, Any]) -> tuple[float, float]:
    metrics = result["metrics"]
    return float(metrics["macro_f1"]), -float(metrics["loss"])


def tensor_stats(tensor: Any) -> dict[str, Any]:
    detached = tensor.detach().float().cpu()
    return {
        "shape": list(detached.shape),
        "dtype": str(tensor.dtype),
        "min": float(detached.min().item()) if detached.numel() else 0.0,
        "max": float(detached.max().item()) if detached.numel() else 0.0,
        "mean": float(detached.mean().item()) if detached.numel() else 0.0,
        "std": float(detached.std(unbiased=False).item()) if detached.numel() else 0.0,
    }


def sample_trace(
    model: Any,
    feature: dict[str, Any],
    record: dict[str, Any],
    tokenizer: Any,
    label_names: list[str],
    torch: ModuleType,
    device: Any,
    split: str,
    index: int,
    tensor_store: dict[str, Any],
) -> dict[str, Any]:
    import torch.nn.functional as F

    input_ids = torch.tensor([feature["input_ids"]], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.float, device=device)
    ss = torch.tensor([feature["ss"]], dtype=torch.long, device=device)
    os = torch.tensor([feature["os"]], dtype=torch.long, device=device)
    with torch.no_grad():
        encoder_outputs = model.encoder(input_ids, attention_mask=attention_mask)
        hidden = encoder_outputs[0]
        embeddings = model.encoder.embeddings(input_ids) if hasattr(model.encoder, "embeddings") else None
        subject_hidden = hidden[0, ss[0]]
        object_hidden = hidden[0, os[0]]
        entity_representation = torch.cat((subject_hidden, object_hidden), dim=-1)
        logits = model.classifier(entity_representation)
        probabilities = F.softmax(logits.float(), dim=-1)
    key = f"{split}_{index}"
    tensor_payload = {
        "input_embeddings": embeddings.detach().cpu() if embeddings is not None else None,
        "encoder_last_hidden_state": hidden.detach().cpu(),
        "subject_marker_hidden_state": subject_hidden.detach().cpu(),
        "object_marker_hidden_state": object_hidden.detach().cpu(),
        "entity_pair_representation": entity_representation.detach().cpu(),
        "logits": logits.detach().cpu(),
        "probabilities": probabilities.detach().cpu(),
    }
    tensor_store[key] = tensor_payload
    tensor_ref = f"case_study_tensors.pt::{key}"
    token_strings = tokenizer.convert_ids_to_tokens(feature["input_ids"])
    prediction = int(logits.argmax(dim=-1).item())
    gold = int(feature["labels"])
    subject_start, subject_end = record["subj_start"], record["subj_end"] + 1
    object_start, object_end = record["obj_start"], record["obj_end"] + 1
    return {
        "split": split,
        "index": index,
        "input": {
            "sentence": " ".join(record["token"]),
            "tokens": record["token"],
            "subject": {"text": " ".join(record["token"][subject_start:subject_end]), "span": [subject_start, subject_end]},
            "object": {"text": " ".join(record["token"][object_start:object_end]), "span": [object_start, object_end]},
            "gold_relation": label_names[gold],
        },
        "stages": [
            {
                "name": "upstream_marker_tokenization",
                "input": "raw tokens, inclusive entity spans, and entity types",
                "output": {
                    "token_ids": feature["input_ids"],
                    "tokens": token_strings,
                    "subject_marker_position": int(feature["ss"]),
                    "object_marker_position": int(feature["os"]),
                },
            },
            {
                "name": "embedding",
                "input": "marker-aware token IDs",
                "output": {"tensor_ref": tensor_ref, "tensor_key": "input_embeddings", "summary": tensor_stats(embeddings) if embeddings is not None else None},
            },
            {
                "name": "roberta_encoder",
                "input": "embedding sequence and attention mask",
                "output": {"tensor_ref": tensor_ref, "tensor_key": "encoder_last_hidden_state", "summary": tensor_stats(hidden)},
            },
            {
                "name": "entity_position_readout",
                "input": "subject/object marker hidden states",
                "output": {
                    "tensor_ref": tensor_ref,
                    "tensor_keys": ["subject_marker_hidden_state", "object_marker_hidden_state", "entity_pair_representation"],
                    "subject_summary": tensor_stats(subject_hidden),
                    "object_summary": tensor_stats(object_hidden),
                    "pair_summary": tensor_stats(entity_representation),
                },
            },
            {
                "name": "classifier",
                "input": "concatenated subject/object representation",
                "output": {
                    "tensor_ref": tensor_ref,
                    "logits": logits.detach().cpu().flatten().tolist(),
                    "probabilities": probabilities.detach().cpu().flatten().tolist(),
                    "predicted_relation": label_names[prediction],
                },
            },
        ],
    }


def choose_case_indices(result: dict[str, Any], count: int) -> list[int]:
    if count <= 0:
        return []
    selected: list[int] = []
    for index, (gold, prediction) in enumerate(zip(result["labels"], result["predictions"])):
        if gold != prediction:
            selected.append(index)
            break
    for index, gold in enumerate(result["labels"]):
        if gold != 0 and index not in selected:
            selected.append(index)
            break
    for index in range(len(result["labels"])):
        if index not in selected:
            selected.append(index)
        if len(selected) >= count:
            break
    return selected[:count]


def write_markdown(summary: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Re-TACRED Improved Baseline reference",
        "",
        f"- seed: `{summary['seed']}`",
        f"- upstream commit: `{summary['upstream']['commit']}`",
        f"- model: `{summary['configuration']['model_name_or_path']}`",
        f"- input format: `{summary['configuration']['input_format']}`",
        "- test used for training or selection: `false`",
        "",
        "| split | micro-P | micro-R | micro-F1 | accuracy | macro-P | macro-R | macro-F1 | loss | source non-null F1 | items |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, result in summary["splits"].items():
        metrics = result["metrics"]
        lines.append(
            f"| {split} | {metrics['micro_precision']:.6f} | {metrics['micro_recall']:.6f} | "
            f"{metrics['micro_f1']:.6f} | {metrics['accuracy']:.6f} | "
            f"{metrics['macro_precision']:.6f} | {metrics['macro_recall']:.6f} | "
            f"{metrics['macro_f1']:.6f} | {metrics['loss']:.6f} | "
            f"{metrics['source_non_null_micro_f1']:.6f} | {result['items']} |"
        )
    lines.extend([
        "",
        "## Evidence status",
        "",
        "This is a source-faithful adapted run. The upstream forward path is preserved; the project adapter adds complete metrics, checkpoint, provenance, and bounded case-study traces.",
        "",
        "## Case study",
        "",
        "Train and test samples are traced from raw tokens through marker tokenization, embeddings, encoder states, entity-position readout, and classifier logits. Full tensors are kept in the private raw run and are referenced by `case_study.json`.",
    ])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.seed < 0 or args.train_batch_size <= 0 or args.test_batch_size <= 0:
        raise ValueError("seed must be non-negative and batch sizes must be positive")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    upstream_dir = Path(args.upstream_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    manifest_path = data_dir / "data_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    try:
        import numpy as np
        import torch
        import torch.nn.functional as F  # noqa: F401
        from torch.cuda.amp import GradScaler
        from torch.utils.data import DataLoader
        from transformers import AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup
    except ImportError as exc:
        raise RuntimeError("Improved Baseline requires torch, transformers, numpy, tqdm, and ujson") from exc

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available")
    device = torch.device(args.device)
    set_seed(args.seed, torch, np)
    model_module, prepro_module, utils_module = import_upstream(upstream_dir)
    runtime_args = as_namespace(args)
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=args.num_class)
    config.gradient_checkpointing = True
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    processor = prepro_module.RETACREDProcessor(runtime_args, tokenizer)
    split_paths = {"train": data_dir / "train.json", "valid": data_dir / "dev.json", "test": data_dir / "test.json"}
    raw_records = {split: json.loads(path.read_text(encoding="utf-8")) for split, path in split_paths.items()}
    features = {split: load_split(processor, path, runtime_args) for split, path in split_paths.items()}
    label_to_id = dict(processor.LABEL_TO_ID)
    label_names = [label for label, _ in sorted(label_to_id.items(), key=lambda item: item[1])]
    if len(label_names) != args.num_class:
        raise ValueError(f"upstream label map has {len(label_names)} classes, expected {args.num_class}")

    model = model_module.REModel(runtime_args, config)
    if len(processor.new_tokens) > 0:
        model.encoder.resize_token_embeddings(len(tokenizer))
    model.to(device)

    train_loader = DataLoader(
        features["train"], batch_size=args.train_batch_size, shuffle=True,
        collate_fn=utils_module.collate_fn, drop_last=True, num_workers=0,
    )
    total_steps = int(len(train_loader) * args.num_train_epochs // args.gradient_accumulation_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)
    try:
        from transformers.optimization import AdamW
    except ImportError:
        AdamW = torch.optim.AdamW
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=max(total_steps, 1)
    )
    scaler = GradScaler(enabled=device.type == "cuda")
    train_history: list[dict[str, Any]] = []
    num_steps = 0
    evaluation_points: list[dict[str, Any]] = []
    model.zero_grad()
    for epoch in range(int(args.num_train_epochs)):
        model.train()
        for step, batch in enumerate(train_loader):
            inputs = batch_to_device(batch, device)
            outputs = model(**inputs)
            loss = outputs[0] / args.gradient_accumulation_steps
            if not torch.isfinite(loss):
                raise FloatingPointError("reference baseline produced non-finite training loss")
            scaler.scale(loss).backward()
            # This retains the upstream source's update condition (step % grad_accum == 0).
            if step % args.gradient_accumulation_steps == 0:
                num_steps += 1
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                model.zero_grad()
                train_history.append({"epoch": epoch, "step": step, "update": num_steps, "loss": float(loss.item())})
                if args.evaluation_steps > 0 and num_steps % args.evaluation_steps == 0:
                    valid_result = evaluate(
                        model, features["valid"], utils_module.collate_fn, torch, np,
                        importlib.import_module("evaluation").get_f1, label_names, device, args.test_batch_size,
                    )
                    evaluation_points.append({"update": num_steps, "valid": valid_result["metrics"]})
                    print(json.dumps({"event": "valid", "update": num_steps, "metrics": valid_result["metrics"]}, sort_keys=True), flush=True)

    split_results = {
        split: evaluate(
            model, features[split], utils_module.collate_fn, torch, np,
            importlib.import_module("evaluation").get_f1, label_names, device, args.test_batch_size,
        )
        for split in ("train", "valid", "test")
    }
    tensor_store: dict[str, Any] = {}
    case_samples: list[dict[str, Any]] = []
    for split in ("train", "test"):
        for index in choose_case_indices(split_results[split], args.case_study_per_split):
            case_samples.append(
                sample_trace(
                    model, features[split][index], raw_records[split][index], tokenizer,
                    label_names, torch, device, split, index, tensor_store,
                )
            )
    torch.save(tensor_store, output_dir / "case_study_tensors.pt")
    case_study = {
        "schema": "q-attention.reference-case-study.v1",
        "status": "complete",
        "tensor_artifact": "case_study_tensors.pt",
        "samples": case_samples,
    }
    (output_dir / "case_study.json").write_text(json.dumps(case_study, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tensor_manifest = {
        "schema": "q-attention.reference-case-study-tensor-manifest.v1",
        "path": "case_study_tensors.pt",
        "sha256": sha256_file(output_dir / "case_study_tensors.pt"),
        "samples": sorted(tensor_store),
        "keys": sorted({key for payload in tensor_store.values() for key in payload}),
    }
    (output_dir / "case_study_tensor_manifest.json").write_text(json.dumps(tensor_manifest, indent=2) + "\n", encoding="utf-8")

    checkpoint_dir = output_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.to_dict(),
            "args": vars(args),
            "upstream_commit": git_output("rev-parse", "HEAD", cwd=upstream_dir),
        },
        checkpoint_dir / "model.pt",
    )
    (checkpoint_dir / "label_to_id.json").write_text(json.dumps(label_to_id, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary: dict[str, Any] = {
        "schema": "q-attention.retacred-reference-improved-baseline.v1",
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "device": str(device),
        "test_used_for_training_or_selection": False,
        "selection_rule": "final_upstream_source_semantics",
        "configuration": {
            "model_name_or_path": args.model_name_or_path,
            "input_format": args.input_format,
            "max_seq_length": args.max_seq_length,
            "train_batch_size": args.train_batch_size,
            "test_batch_size": args.test_batch_size,
            "learning_rate": args.learning_rate,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "adam_epsilon": args.adam_epsilon,
            "max_grad_norm": args.max_grad_norm,
            "warmup_ratio": args.warmup_ratio,
            "num_train_epochs": args.num_train_epochs,
            "evaluation_steps": args.evaluation_steps,
            "num_class": args.num_class,
        },
        "upstream": {
            "directory": str(upstream_dir),
            "commit": git_output("rev-parse", "HEAD", cwd=upstream_dir),
            "origin": git_output("remote", "get-url", "origin", cwd=upstream_dir),
            "source_files": {
                name: sha256_file(upstream_dir / name)
                for name in ("train_retacred.py", "prepro.py", "model.py", "utils.py", "evaluation.py")
            },
        },
        "data_manifest": manifest,
        "splits": {},
        "training": {"total_steps": total_steps, "warmup_steps": warmup_steps, "updates": num_steps, "history_tail": train_history[-100:]},
        "evaluation_points": evaluation_points,
        "case_study": {"status": "complete", "samples": len(case_samples), "tensor_manifest": "case_study_tensor_manifest.json"},
        "provenance": {
            "project_git_commit": git_output("rev-parse", "HEAD"),
            "project_git_branch": git_output("branch", "--show-current"),
            "project_git_dirty": bool(git_output("status", "--porcelain")),
            "torch": torch.__version__,
            "transformers": importlib.import_module("transformers").__version__,
            "cuda_available": torch.cuda.is_available(),
        },
    }
    for split, result in split_results.items():
        result.pop("predictions", None)
        result.pop("labels", None)
        summary["splits"][split] = result
    summary["checkpoint_sha256"] = sha256_file(checkpoint_dir / "model.pt")
    summary["case_study_tensor_sha256"] = tensor_manifest["sha256"]
    (output_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps({"splits": summary["splits"], "schema": summary["schema"]}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    data_lines = [
        "schema=q-attention.retacred-reference-improved-baseline-run-summary-data.v1",
        "status=complete",
        f"seed={args.seed}",
        f"upstream_commit={summary['upstream']['commit']}",
        f"checkpoint_sha256={summary['checkpoint_sha256']}",
        f"case_study_tensor_sha256={summary['case_study_tensor_sha256']}",
        "test_used_for_training_or_selection=false",
    ]
    for split, info in manifest["splits"].items():
        data_lines.extend([
            f"split.{split}.records={info['records']}",
            f"split.{split}.source_sha256={info['source_sha256']}",
            f"split.{split}.prepared_sha256={info['output_sha256']}",
            f"split.{split}.evaluated_items={summary['splits'][split]['items']}",
        ])
    (output_dir / "run_summary.data").write_text("\n".join(data_lines) + "\n", encoding="utf-8")
    write_markdown(summary, output_dir / "run_summary.md")
    print(json.dumps({"event": "reference_baseline_complete", "output_dir": str(output_dir), "status": "complete"}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
