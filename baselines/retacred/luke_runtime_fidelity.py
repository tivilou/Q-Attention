"""Bounded LUKE runtime/trace/recovery checks, never a published-score certificate."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from types import ModuleType

from .common import ROOT, atomic_json, digest, fingerprint
from .luke import LUKE, checked_file, lock, require_runtime_memory, verify_source

MEMORY_MINIMUM = {"tiny": 768 * 1024**2, "full": 24 * 1024**3}
PENDING = ["actual_AllenNLP_wrapper_indexer_equivalence", "official_training_recipe_binding",
           "upstream_optimizer_scheduler_equivalence", "CoRE_CFIE_resample_focal_mechanisms",
           "full_data_published_score_reproduction"]


def description(mode, device):
    return {"schema": "luke-runtime-gate.v1", "mode": mode, "device": device,
            "formal_training": False, "formal_comparison_admitted": False,
            "minimum_host_headroom_bytes": MEMORY_MINIMUM[mode], "synthetic_examples": 2,
            "optimizer_updates": 2, "replayed_updates": 1, "precision": "FP32",
            "provenance": "pinned_author_HF_classifier_and_installed_HF_LUKE_backbone",
            "weight_scope": "random_tiny_architecture" if mode == "tiny" else "hash_locked_luke_large_pretrained",
            "diagnostic_optimizer": {"type": "torch.optim.AdamW", "lr": 1e-5, "eps": 1e-8,
                                     "weight_decay": 0.01, "warmup_steps": 0, "schedule_steps": 4},
            "optimizer_claim": "diagnostic_nonzero_update_and_recovery_only_not_source_recipe",
            "pending": list(PENDING)}


def source_classifier(source):
    verify_source(source)
    spec = lock()["source"]["files"]["examples/relation_classification/luke_for_entity_pair_classification.py"]
    path = checked_file(source, spec["file"])
    module = ModuleType("retacred_luke_runtime_source")
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module.LukeForEntityPairClassification


def tensor_identity(value):
    import torch
    tensor = value.detach().cpu().contiguous()
    # NumPy exposes the contiguous storage without making another weight-sized copy.
    raw = memoryview(tensor.numpy()).cast("B")
    result = hashlib.sha256(raw).hexdigest()
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "sha256": result}


def state_identity(value):
    import torch
    if torch.is_tensor(value):
        return tensor_identity(value)
    if isinstance(value, dict):
        return {str(key): state_identity(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [state_identity(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def gradient_identity(model):
    import torch
    result = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():
                raise FloatingPointError("non-finite gradient: " + name)
            result[name] = {**tensor_identity(parameter.grad),
                            "nonzero": bool(torch.count_nonzero(parameter.grad))}
    required = ("luke.embeddings.word_embeddings.weight", "luke.entity_embeddings.entity_embeddings.weight",
                "luke.encoder.layer.0.attention.self.w2e_query.weight", "classifier.weight")
    if any(name not in result or not result[name]["nonzero"] for name in required):
        raise AssertionError("word/entity/entity-aware-attention/head gradient path missing")
    return result


def validate_capture(capture, *, phase):
    keys = set(capture.values)
    required = ("luke.embeddings.call0.output.tensor", "luke.entity_embeddings.call0.output.tensor",
                "functional.entity_readout.subject", "functional.entity_readout.object",
                "functional.entity_readout.concatenated", "classifier.call0.input.0",
                "classifier.call0.output.tensor")
    if any(key not in keys for key in required):
        raise AssertionError("required numeric LUKE trace path absent: " + str(sorted(keys)))
    if not any(key.startswith("functional.attention_scores_to_probabilities.") for key in keys):
        raise AssertionError("actual entity-aware attention probabilities absent")
    if phase == "train" and not any(key.endswith(".gradient") for key in keys):
        raise AssertionError("actual training activation gradients absent")
    if not capture.values or any(value.numel() == 0 for value in capture.values.values()):
        raise AssertionError("empty numeric trace")


def make_adapter(source, checkpoint, mode, device):
    import torch
    from transformers import LukeConfig
    labels = ["no_relation"] + ["synthetic_relation_" + str(i) for i in range(1, 40)]
    records = [{"id": "runtime-" + str(i), "token": ["Alice", "joined", "Acme", "."],
                "subj_start": 0 if i == 0 else 2, "subj_end": 0 if i == 0 else 2,
                "obj_start": 2 if i == 0 else 0, "obj_end": 2 if i == 0 else 0,
                "relation": labels[i+1]} for i in range(2)]
    if mode == "full":
        adapter = LUKE(source, checkpoint, labels, device=device)
        features = adapter.encode(records)
        adapter.finalize_vocabulary()
        batches = [adapter.batch([feature]) for feature in features]
        inputs = [{"raw_input": raw, "token_strings": adapter.tokenizer.convert_ids_to_tokens(feature["input_ids"]),
                   "tokenization": "actual_pinned_checkpoint_HF_BPE"}
                  for raw, feature in zip(records, features)]
    else:
        config = LukeConfig(vocab_size=64, entity_vocab_size=16, hidden_size=32, entity_emb_size=16,
                            num_hidden_layers=2, num_attention_heads=4, intermediate_size=64,
                            max_position_embeddings=64, num_labels=40, classifier_bias=True,
                            hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
                            use_entity_aware_attention=True)
        adapter = object.__new__(LUKE)
        adapter.source, adapter.torch, adapter.device, adapter.labels = source, torch, torch.device(device), labels
        adapter.model = source_classifier(source)(config).to(device)
        adapter.model.classifier.reset_parameters()
        batches, inputs = [], []
        for i in range(2):
            spans = [[1, 2, 3], [5, 6, 7]] if i == 0 else [[5, 6, 7], [1, 2, 3]]
            fields = {"input_ids": [[0, 10, 11, 12, 13, 14, 15, 16, 17, 2]], "attention_mask": [[1]*10],
                      "entity_ids": [[2, 3]], "entity_attention_mask": [[1, 1]],
                      "entity_position_ids": [spans], "labels": [i+1]}
            batches.append({key: torch.tensor(value, device=device) for key, value in fields.items()})
            inputs.append({"raw_input": None, "token_strings": None,
                           "tokenization": "synthetic_tensor_ids_not_BPE_or_raw_sentence_alignment",
                           "entity_roles": ["subject", "object"]})
    return adapter, batches, inputs


def record_capture(directory, capture, batch, adapter, raw, *, phase, loss, logits):
    from .train import save_torch
    for key, value in batch.items():
        capture.save("task_input." + key, value)
    capture.save("task_output.logits", logits)
    capture.save("task_output.probabilities", logits.softmax(-1))
    capture.values["task_output.loss"] = loss.detach().cpu().clone()
    if phase == "train":
        capture.values.update(adapter.parameter_trace())
    validate_capture(capture, phase=phase)
    target = directory / (phase + ".pt")
    save_torch(target, capture.values)
    import torch
    reloaded = torch.load(target, map_location="cpu", weights_only=True)
    if state_identity(reloaded) != state_identity(capture.values):
        raise AssertionError("numeric trace roundtrip mismatch")
    payload = {"schema": "luke-runtime-sample-trace.v1", "status": "complete_diagnostic_scope",
               "phase": phase, "capture_origin": "actual_forward_backward" if phase == "train" else "actual_forward",
               **raw, **adapter.trace_fields(batch, 0), "tensor_artifact": target.name,
               "sha256": digest(target), "size_bytes": target.stat().st_size,
               "target": adapter.labels[int(batch["labels"][0])],
               "predicted_relation": adapter.labels[int(logits[0].argmax())],
               "loss": float(loss), **capture.metadata()}
    atomic_json(target.with_suffix(".json"), payload)
    return {"json": target.with_suffix(".json").name, "pt": target.name,
            "sha256": payload["sha256"], "tensors": len(capture.values)}


def exercise(adapter, batches, raw, output):
    import torch
    from transformers import get_linear_schedule_with_warmup
    from .improved import SampleCapture
    from .train import restore_checkpoint, rng_state, save_torch

    model, batch = adapter.model, batches[0]
    model.eval()
    with torch.no_grad():
        direct = model(**batch, return_dict=True)
        loss, logits = adapter.forward(batch)
        torch.testing.assert_close(loss, direct.loss, rtol=0, atol=0)
        torch.testing.assert_close(logits, direct.logits, rtol=0, atol=0)
        changed = dict(batch, labels=(batch["labels"]+1) % 40)
        _, changed_logits = adapter.forward(changed)
        torch.testing.assert_close(logits, changed_logits, rtol=0, atol=0)

    model.train()
    model.zero_grad(set_to_none=True)
    initial_rng = torch.get_rng_state()
    initial_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    direct = model(**batch, return_dict=True)
    direct.loss.backward()
    expected_gradients = gradient_identity(model)
    expected_rng = state_identity(rng_state())
    expected_loss, expected_logits = direct.loss.detach(), direct.logits.detach()
    del direct
    model.zero_grad(set_to_none=True)
    torch.set_rng_state(initial_rng)
    if initial_cuda:
        torch.cuda.set_rng_state_all(initial_cuda)
    with SampleCapture(model, 0, 1) as capture:
        loss, logits = adapter.forward(batch)
        loss.backward()
    torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
    torch.testing.assert_close(logits, expected_logits, rtol=0, atol=0)
    if gradient_identity(model) != expected_gradients or state_identity(rng_state()) != expected_rng:
        raise AssertionError("tracing changed gradients or dropout RNG")
    train_trace = record_capture(output, capture, batch, adapter, raw[0], phase="train", loss=loss, logits=logits)
    del capture, expected_gradients, loss, logits

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, eps=1e-8, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, 4)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    before = tensor_identity(model.classifier.weight)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step(); scheduler.step()
    if before == tensor_identity(model.classifier.weight):
        raise AssertionError("diagnostic optimizer update was vacuous")
    contract = {"scope": "diagnostic_only", "batch": state_identity(batch),
                "source": verify_source(adapter.source), "config": model.config.to_dict()}
    checkpoint = output / "post-update1.pt"
    save_torch(checkpoint, {"contract": contract, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                            "rng": rng_state(), "cursor": {"update": 1, "next_batch": 1}})

    def continuation():
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, logits = adapter.forward(batches[1])
        loss.backward()
        if not math.isfinite(float(loss)):
            raise FloatingPointError("non-finite continuation loss")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step(); scheduler.step()
        return {"model": fingerprint(state_identity(model.state_dict())),
                "optimizer": fingerprint(state_identity(optimizer.state_dict())),
                "scheduler": state_identity(scheduler.state_dict()), "rng": fingerprint(state_identity(rng_state())),
                "loss": float(loss), "logits": tensor_identity(logits)}

    expected = continuation()
    restored = restore_checkpoint(checkpoint, contract, model, optimizer, scheduler, scaler)
    if (restored["cursor"] != {"update": 1, "next_batch": 1}
            or optimizer.param_groups[0]["lr"] != 1e-5 * 0.75):
        raise AssertionError("post-update cursor or scheduler recovery mismatch")
    del restored
    resumed = continuation()
    if expected != resumed:
        raise AssertionError("post-update resume differs from uninterrupted continuation")
    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad(), SampleCapture(model, 0, 1) as capture:
        loss, logits = adapter.forward(batches[1])
    test_trace = record_capture(output, capture, batches[1], adapter, raw[1], phase="test", loss=loss, logits=logits)
    return {"fixed_input_forward_loss": "passed", "gold_input_isolation": "passed",
            "word_entity_attention_head_gradients": "passed", "dropout_trace_transparency": "passed",
            "nonzero_diagnostic_update": "passed", "diagnostic_scheduler_recovery": "passed",
            "exact_post_update_resume": "passed", "resume_state": resumed,
            "actual_numeric_train_test_trace_roundtrip": "passed", "traces": [train_trace, test_trace],
            "rtol": 0, "atol": 0, "checkpoint": {"path": checkpoint.name, "sha256": digest(checkpoint)},
            "source_optimizer_equivalence": "pending_not_tested"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", type=Path, default=ROOT/".external/retacred/luke-foundation-20261008")
    parser.add_argument("--checkpoint", type=Path, default=ROOT/".external/assets/luke"/lock()["checkpoint"]["revision"]/"model")
    parser.add_argument("--mode", choices=("tiny", "full"), default="full")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--describe", action="store_true", help="read-only plan; no backend, weights, network or writes")
    args = parser.parse_args(argv)
    result = description(args.mode, args.device)
    if args.describe:
        print(json.dumps(result, indent=2))
        return 0
    if args.output_dir is None:
        parser.error("--output-dir required for runtime checks")
    output = args.output_dir.absolute()
    if output.resolve() != output or (output.exists() and any(output.iterdir())):
        parser.error("runtime output must be a non-symlinked new or empty directory")
    try:
        result["source_files_verified"] = len(verify_source(args.source_cache))
        try:
            result["host_headroom_bytes"] = require_runtime_memory(MEMORY_MINIMUM[args.mode])
        except RuntimeError as error:
            result.update(status="pending_memory_gate", reason=str(error), backend_imported=False, weights_read=False)
            atomic_json(output/"runtime-evidence.json", result)
            print(json.dumps(result, indent=2))
            return 2
        # Offline-only even when run in a collaborator environment with HF credentials.
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_TF="0", USE_FLAX="0")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        import random
        import numpy as np
        torch.set_num_threads(1)
        random.seed(13)
        np.random.seed(13)
        torch.manual_seed(13)
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("requested CUDA device unavailable")
        torch.use_deterministic_algorithms(True)
        output.mkdir(parents=True, exist_ok=True)
        adapter, batches, raw = make_adapter(args.source_cache, args.checkpoint, args.mode, args.device)
        result["checks"] = exercise(adapter, batches, raw, output)
        result.update(status="tiny_runtime_verified" if args.mode == "tiny" else "full_foundation_runtime_verified",
                      backend_imported=True, weights_read=args.mode == "full",
                      environment={"torch": torch.__version__, "transformers": __import__("transformers").__version__})
        if args.mode == "tiny":
            result["pending"].append("full_weight_loading_forward_autograd_update_resume_CUDA")
        else:
            result["loading_info"] = adapter.loading_info
            result["checkpoint_asset_hashes"] = adapter.checkpoint_hashes
        atomic_json(output/"runtime-evidence.json", result)
        print(json.dumps({key: value for key, value in result.items() if key != "checks"}, indent=2))
        return 0
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__, reason=str(error))
        atomic_json(output/"runtime-evidence.json", result)
        print(json.dumps(result, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
