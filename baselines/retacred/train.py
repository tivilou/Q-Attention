"""Common declared protocol, exact post-update recovery and live sample traces."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

from .common import ROOT, atomic_json, cohort, digest, fingerprint, git, metrics, pin_source, select_models
from .improved import SampleCapture
from .adapters import build, metadata_identity

PAUSED = 75


def save_torch(path, value):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def rng_state():
    import numpy as np
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(value):
    import numpy as np
    import torch
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])
    if value["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(value["cuda"])


def restore_checkpoint(path, contract, model, optimizer, scheduler, scaler):
    import torch
    value = torch.load(path, map_location="cpu", weights_only=False)  # private, self-generated trusted checkpoint
    if value["contract"] != contract:
        raise ValueError("checkpoint contract differs: source/data/config/model/seed must match")
    model.load_state_dict(value["model"], strict=True)
    optimizer.load_state_dict(value["optimizer"])
    scheduler.load_state_dict(value["scheduler"])
    scaler.load_state_dict(value["scaler"])
    restore_rng(value["rng"])
    return value


def update_groups(permutation, batch_size):
    """Logical batch stays fixed; each update consumes its stored index group."""
    return [permutation[i:i + batch_size] for i in range(0, len(permutation), batch_size)]


def trace(adapter, batch, record, capture, *, phase, sample_index, step, directory, loss, probabilities, prediction):
    import torch
    key = f"{phase}-index{sample_index}-update{step}"
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in batch.items():
        capture.save("task_input." + name, value)
    capture.save("task_output.logits", prediction)
    capture.save("task_output.probabilities", probabilities)
    capture.values["task_output.loss"] = loss.detach().cpu().clone()
    if phase == "train":
        capture.values.update(adapter.parameter_trace())
    path = directory / (key + ".pt")
    save_torch(path, capture.values)
    row = capture.row
    gold = int(batch["labels"][row])
    guessed = int(prediction[row].argmax())
    payload = {"schema": "sample-trace.v2", "status": "complete", "phase": phase,
               "capture_origin": "actual_training_forward_backward" if phase == "train" else "actual_final_test_forward",
               "sample_index": sample_index, "optimizer_update": step,
               "raw_input": record, "token_ids": batch["input_ids"][row].detach().cpu().tolist(),
               "token_strings": adapter.tokenizer.convert_ids_to_tokens(batch["input_ids"][row].detach().cpu().tolist()),
               "target": adapter.labels[gold], "predicted_relation": adapter.labels[guessed],
               "target_access": "training_loss_only_or_post_prediction_evaluation",
               "tensor_artifact": path.name, "sha256": digest(path), "size_bytes": path.stat().st_size,
               "batch_context": {"physical_batch_size": len(batch["labels"]), "selected_row": row},
               "loss_scope": "actual_physical_batch_mean_cross_entropy_not_selected_sample_only",
               **adapter.trace_fields(batch, row),
               **capture.metadata()}
    atomic_json(directory / (key + ".json"), payload)
    return payload


def evaluate(adapter, features, records, *, batch_size, case_index=None, step=0, trace_dir=None):
    import torch
    gold, predicted, losses, captures = [], [], 0., []
    adapter.model.eval()
    for start in range(0, len(features), batch_size):
        batch = adapter.batch(features[start:start + batch_size])
        row = case_index-start if case_index is not None and start <= case_index < start + len(batch["labels"]) else None
        capture = SampleCapture(adapter.model, row, len(batch["labels"])) if row is not None else None
        with torch.no_grad():
            if capture:
                with capture:
                    loss, logits = adapter.forward(batch)
                captures.append(trace(adapter, batch, records[case_index], capture, phase="test", sample_index=case_index,
                                      step=step, directory=trace_dir, loss=loss, probabilities=logits.float().softmax(-1), prediction=logits))
            else:
                loss, logits = adapter.forward(batch)
        losses += float(loss) * len(batch["labels"])
        gold.extend(batch["labels"].tolist())
        predicted.extend(logits.argmax(-1).tolist())
    return metrics(gold, predicted, adapter.labels, losses/len(gold)), captures


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--logical-batch", type=int, default=64)
    parser.add_argument("--micro-batch", type=int, default=32)
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--canary", action="store_true", help="synthetic scheduler child only; does not train")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.canary:
        args.run_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(args.run_dir / "heartbeat.json", {"phase": "canary", "gpu": os.environ.get("CUDA_VISIBLE_DEVICES")})
        if os.environ.get("RETACRED_CANARY_FAIL") == args.model:
            raise RuntimeError("injected child failure")
        atomic_json(args.run_dir / "metrics.json", {"synthetic": True})
        (args.run_dir / "RUN_COMPLETE").touch()
        return
    import numpy as np
    import torch
    from transformers import get_linear_schedule_with_warmup
    row = select_models(args.model)[0]
    if min(args.epochs, args.logical_batch, args.micro_batch, args.checkpoint_every) <= 0 or args.seed < 0:
        raise ValueError("positive epochs/batch/checkpoint interval and nonnegative seed required")
    if args.run_dir.exists() and any(args.run_dir.iterdir()) and not args.resume:
        raise ValueError("existing run requires --resume")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    source = pin_source(row["source"], fetch=False)
    paths = {"train": args.data_dir / "train.json", "valid": args.data_dir / "dev.json", "test": args.data_dir / "test.json"}
    data_identity = {name: digest(path) for name, path in paths.items()}
    source_identity = {name: digest(source/name) for name in cohort()["sources"][row["source"]]["files"]}
    is_ptr = row["id"] == "ptr"
    from .ptr import PTR
    from .improved import Improved
    kind = PTR if is_ptr else Improved
    weight_identity = {path.name: digest(path) for path in sorted(args.checkpoint.iterdir()) if path.is_file()}
    contract = {"schema": "reference-training-contract.v1", "model": args.model, "seed": args.seed,
                "data": data_identity, "source": source_identity, "weights": weight_identity,
                "epochs": args.epochs, "logical_batch": args.logical_batch,
                "selection": "final_epoch", "protocol": kind.protocol,
                "project_code": {name: digest(Path(__file__).parent/name) for name in (
                    "train.py", "improved.py", "common.py", "adapters.py", "ptr.py", "updates.py", "ptr_resources.py")},
                "optimizer": kind.optimizer_contract, "metadata": metadata_identity(row),
                "physical_partition": args.micro_batch, "checkpointing": args.checkpointing,
                "attention_implementation": "eager_upstream_style_with_actual_softmax_trace",
                "precision": kind.precision_contract}
    contract["environment"] = {"torch":torch.__version__, "transformers":__import__('transformers').__version__, "numpy":np.__version__}
    contract_path = args.run_dir / "contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("immutable run contract mismatch; physical dropout partition cannot silently change")
    atomic_json(contract_path, contract)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    adapter = build(row, source, args.checkpoint, device=args.device, checkpointing=args.checkpointing)
    # This trainer intentionally uses the common logical-batch update convention,
    # not upstream's step % accumulation == 0 off-by-one. Difference is explicit.
    try:
        from transformers.optimization import AdamW
    except ImportError as exc:
        raise RuntimeError("locked Transformers optimizer unavailable; no silent optimizer fallback") from exc
    raw = {split: json.loads(path.read_text()) for split, path in paths.items()}
    features = {split: adapter.encode(items) for split, items in raw.items()}
    adapter.finalize_vocabulary()
    total = math.ceil(len(features["train"])/args.logical_batch)*args.epochs
    if is_ptr:
        optimizer, scheduler = adapter.make_updates(math.ceil(len(features["train"])/args.logical_batch), args.epochs)
    else:
        optimizer = AdamW(adapter.model.parameters(), lr=3e-5, eps=1e-6, weight_decay=0, correct_bias=True)
        scheduler = get_linear_schedule_with_warmup(optimizer, int(total*.1), total)
    scaler = torch.amp.GradScaler("cuda", enabled=args.device == "cuda" and adapter.amp_enabled)
    saved = args.run_dir / "checkpoints/latest.pt"
    state = {"epoch": 0, "cursor": 0, "permutation": None, "update": 0, "traces": []}
    if args.resume:
        if not saved.is_file():
            raise ValueError("resume checkpoint missing; refusing epoch-level restart")
        state = restore_checkpoint(saved, contract, adapter.model, optimizer, scheduler, scaler)["cursor"]
        # Only remove obsolete status markers after a validated durable restore.
        (args.run_dir/"RUN_PAUSED").unlink(missing_ok=True)
    pause = [False]
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: pause.__setitem__(0, True))
    def checkpoint():
        save_torch(saved, {"contract": contract, "model": adapter.model.state_dict(), "optimizer": optimizer.state_dict(),
                          "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "rng": rng_state(),
                          "cursor": copy.deepcopy(state)})
    if not saved.exists():
        checkpoint()
    train_case = next((i for i, item in enumerate(features["train"]) if item["labels"] != 0), 0)
    trace_dir = args.run_dir / "case_study"
    start_time, start_update = time.monotonic(), state["update"]
    while state["epoch"] < args.epochs:
        if state["permutation"] is None:
            state["permutation"] = torch.randperm(len(features["train"])).tolist()
        groups = update_groups(state["permutation"], args.logical_batch)
        adapter.model.train()
        while state["cursor"] < len(groups):
            indices = groups[state["cursor"]]
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.
            pending_trace = None
            for offset in range(0, len(indices), args.micro_batch):
                selected = indices[offset:offset+args.micro_batch]
                batch = adapter.batch([features["train"][i] for i in selected])
                capture_this = train_case in selected and not any(item["phase"] == "train" for item in state["traces"])
                if capture_this:
                    with SampleCapture(adapter.model, selected.index(train_case), len(selected)) as capture:
                        loss, logits = adapter.forward(batch)
                        scaled_loss = loss * len(selected)/len(indices)
                        scaler.scale(scaled_loss).backward()
                    pending_trace = (batch, capture, loss.detach(), logits.detach())
                else:
                    loss, logits = adapter.forward(batch)
                    scaler.scale(loss * len(selected)/len(indices)).backward()
                loss_value += float(loss.detach())*len(selected)/len(indices)
                if not math.isfinite(loss_value):
                    raise FloatingPointError("non-finite loss")
                del loss, logits, batch
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(adapter.model.parameters(), 1.)
            if pending_trace:
                trace_batch, capture, trace_loss, trace_logits = pending_trace
                result = trace(adapter, trace_batch, raw["train"][train_case], capture, phase="train", sample_index=train_case,
                               step=state["update"]+1, directory=trace_dir, loss=trace_loss,
                               probabilities=trace_logits.float().softmax(-1), prediction=trace_logits)
                result["activation_gradient_scale"] = scaler.get_scale()
                result["parameter_gradient_semantics"] = "unscaled_clipped_complete_logical_batch"
                atomic_json(trace_dir / result["tensor_artifact"].replace(".pt", ".json"), result)
                state["traces"].append({"phase":"train", "path":result["tensor_artifact"].replace(".pt", ".json"), "captured_update":state["update"]+1})
                del pending_trace, trace_batch, capture, trace_loss, trace_logits
            previous_scale = scaler.get_scale()
            scaler.step(optimizer); scaler.update()
            # AMP skipped updates consume the batch but do not advance LR.
            if scaler.get_scale() >= previous_scale:
                scheduler.step()
            state["update"] += 1
            state["cursor"] += 1
            elapsed = time.monotonic()-start_time
            speed = (state["update"]-start_update)/max(elapsed, 1e-9)
            heartbeat = {"phase":"train", "epoch":state["epoch"]+1, "epochs":args.epochs,
                         "batch":state["cursor"], "batches":len(groups), "update":state["update"],
                         "total_updates":total, "loss":loss_value, "batches_per_second":speed,
                         "eta_seconds":(total-state["update"])/max(speed, 1e-9),
                         "cuda_allocated_mib":torch.cuda.memory_allocated()/2**20 if args.device=="cuda" else 0}
            atomic_json(args.run_dir / "heartbeat.json", heartbeat)
            if state["update"] % args.checkpoint_every == 0 or pause[0] or (state["traces"] and state["traces"][-1].get("captured_update") == state["update"]):
                checkpoint()
            if state["update"] == 1 or state["update"] % 20 == 0:
                print(f"[{args.model} seed{args.seed}] epoch {state['epoch']+1}/{args.epochs} | batch {state['cursor']}/{len(groups)} | {speed:.2f} batch/s | loss {loss_value:.4f}", flush=True)
            if pause[0]:
                (args.run_dir / "RUN_PAUSED").touch()
                raise SystemExit(PAUSED)
        state["epoch"] += 1; state["cursor"] = 0; state["permutation"] = None
        checkpoint()
    # Final epoch selection is declared; test is touched only after fitting.
    splits = {}
    for split in ("train", "valid", "test"):
        case_index = next((i for i,item in enumerate(features[split]) if item["labels"] != 0), 0) if split == "test" else None
        result, captures = evaluate(adapter, features[split], raw[split], batch_size=args.micro_batch,
                                   case_index=case_index, step=state["update"], trace_dir=trace_dir)
        splits[split] = result
        for item in captures:
            state["traces"].append({"phase":"test", "path":item["tensor_artifact"].replace(".pt", ".json")})
    if {item["phase"] for item in state["traces"]} != {"train", "test"}:
        raise ValueError("required actual train/test trace absent")
    summary = {"schema":"retacred-reference-result.v1", "status":"complete", "model":args.model, "seed":args.seed,
               "contract":contract, "splits":splits, "updates":state["update"],
               "data_manifest":json.loads((args.data_dir/"data_manifest.json").read_text()),
               "case_study": {"status":"complete_local", "schema":"sample-trace.v2", "traces":state["traces"]},
               "adaptations":["declared_common_logical_batches_preserve_source_objective_and_update_components",
                              "final_epoch_selection_no_intermediate_test_access", "real_train_test_observability"],
               "fidelity":"upstream_forward_imported_runtime_certificate_separate",
               "environment":{"torch":torch.__version__, "transformers":__import__('transformers').__version__}}
    atomic_json(args.run_dir/"metrics.json", splits)
    atomic_json(args.run_dir/"run_summary.json", summary)
    save_torch(args.run_dir/"checkpoints/final.pt", {"model":adapter.model.state_dict(), "contract":contract})
    (args.run_dir/"RUN_COMPLETE").touch()  # final task marker; no export/push implied


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Never downgrade OOM by silently changing dropout/physical-batch semantics.
        print(f"[reference worker failed] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
