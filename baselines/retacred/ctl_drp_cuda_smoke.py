"""Bounded single-GPU CTL diagnostic, with explicit tiny/pretrained identity."""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import time

from .common import atomic_json, fingerprint
from .ctl_drp_neural import (RENDERING, collate_groups, descriptions_from_native,
                            encode_groups, joint_model, tiny_model, tokenizer_from_cache)
from .ctl_drp_pretrained import load_encoder, sha
from .ctl_drp_smoke import examples, synthetic_table


def state_hash(value):
    """Streaming exact tensor bytes; never retain a second large GPU model."""
    import torch
    h = hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            h.update(str(x.dtype).encode()); h.update(str(tuple(x.shape)).encode())
            t = x.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
            h.update(memoryview(t.numpy()))
        elif isinstance(x, dict):
            h.update(b"dict")
            for k in sorted(x, key=lambda a: (type(a).__name__, str(a))):
                visit(k); visit(x[k])
        elif isinstance(x, (tuple, list)):
            h.update(type(x).__name__.encode())
            for item in x: visit(item)
        else:
            h.update((type(x).__name__ + ":" + repr(x) + "\0").encode())
    visit(value)
    return h.hexdigest()


def describe():
    return {"schema": "ctl-drp-single-cuda-smoke.v1", "formal_training": False,
            "formal_comparison_admitted": False, "synthetic_sentences": 2,
            "updates": 2, "replay_updates": 1, "device": "single_visible_cuda",
            "precision": "FP32_no_AMP", "default_initialization": "none_explicit_flag_required",
            "checkpoint_scope": "same_GPU_runtime_post_update_AdamW_not_topology_migration"}


def run(output, cache, native, *, tiny=False, weight=None, variant="sup"):
    # Must precede torch import/any CUDA initialization.
    required = ":4096:8"
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG", required) != required:
        raise ValueError("explicit deterministic cuBLAS configuration drift")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = required
    import torch
    import numpy as np
    from .ctl_drp_cuda_checkpoint import load, numpy_rng, runtime, save
    from .improved import SampleCapture
    if torch.cuda.device_count() != 1:
        raise ValueError("select exactly one GPU with CUDA_VISIBLE_DEVICES for this diagnostic")
    if tiny == bool(weight):
        raise ValueError("choose exactly one: explicit random --tiny or locked --weight")
    output = Path(output).absolute()
    if output.exists() or any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("fresh regular private output path required")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(13); np.random.seed(13); torch.manual_seed(13); torch.cuda.manual_seed_all(13)
    if tiny:
        model = tiny_model(seed=13, dropout=.1)
        identity = {"initialization": "random_two_layer_width32_not_pretrained"}
    else:
        encoder, receipt = load_encoder(weight, cache, variant=variant, marker_seed=13)
        model = joint_model(encoder, head_seed=13)
        identity = {"initialization": "hash_locked_official_SimCSE_candidate", **receipt}
    model = model.to("cuda:0")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0., eps=1e-8,
                                 foreach=False, fused=False)
    tok = tokenizer_from_cache(cache, variant=variant)
    desc = descriptions_from_native(native)
    table = synthetic_table(); records = examples()
    groups = encode_groups(records, table, tok, desc, split="train")
    batch = collate_groups(groups, device="cuda:0"); gold = [g["gold"] for g in groups]
    contract = {"identity": identity, "rendering": RENDERING, "table": fingerprint(table),
                "architecture": model.encoder.config.to_dict(), "synthetic_only": True,
                "batch_input_hash": state_hash(batch), "optimizer": "unfused_AdamW_FP32_lr1e-5"}
    output.mkdir(parents=True)
    atomic_json(output / "initialization.json", identity)
    start = time.monotonic(); traces = []
    torch.cuda.reset_peak_memory_stats()

    def trace(phase, selected, tensors, labels):
        model.train(phase == "train")
        optimizer.zero_grad(set_to_none=True)
        with SampleCapture(model, row=0, batch_size=-1) as capture:
            capture.save("input.input_ids", tensors["input_ids"])
            capture.save("input.attention_mask", tensors["attention_mask"])
            result = model(**tensors, gold=labels)
            capture.save("functional.first_token", result["first_token"])
            capture.save("functional.candidate_scores", result["scores"])
            for i, p in enumerate(result["probabilities"]):
                capture.save(f"functional.sentence{i}.probabilities", p)
            capture.save("functional.per_sentence_loss", result["per_sentence_loss"])
            if phase == "train":
                result["loss"].backward()
                for name, param in model.named_parameters():
                    if param.grad is None or not torch.isfinite(param.grad).all():
                        raise ValueError("missing/nonfinite live parameter gradient: " + name)
                    capture.save("parameter_gradient." + name, param.grad)
                optimizer.step()
                capture.save("post_update.head.weight", model.head.weight)
            values = {k: v.contiguous() for k, v in capture.values.items()}
        path = output / (phase + "-trace.pt")
        torch.save(values, path)
        loaded = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        assert set(loaded) == set(values) and all(torch.equal(v, loaded[k]) for k, v in values.items())
        loss = float(result["loss"].detach())
        metadata = {"phase": phase, "scope": "complete_selected_synthetic_GPU_diagnostic_not_formal_cases",
                    "samples": selected, "gold": labels, "predictions": result["predictions"], "loss": loss,
                    "nodes": capture.nodes, "pt": path.name, "sha256": sha(path),
                    "tensor_count": len(values), "actual_numeric_roundtrip": True,
                    "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype),
                                    "numel": v.numel()} for k, v in values.items()}}
        for part in ("word_embeddings", "query", "attention_scores_to_probabilities"):
            assert any(part in key for key in values)
        atomic_json(output / (phase + "-trace.json"), metadata)
        traces.append({k: metadata[k] for k in ("phase", "pt", "sha256", "tensor_count", "actual_numeric_roundtrip")})
        del loaded, values, capture, result
        gc.collect()
        return loss

    first_loss = trace("train", groups, batch, gold)
    optimizer.zero_grad(set_to_none=True)
    checkpoint = output / "checkpoint.pt"
    save(checkpoint, model, optimizer, contract=contract, cursor=1)

    def update():
        model.train(); optimizer.zero_grad(set_to_none=True)
        result = model(**batch, gold=gold)
        result["loss"].backward(); optimizer.step()
        return float(result["loss"].detach())

    second_loss = update()
    expected = {"model": state_hash(model.state_dict()), "optimizer": state_hash(optimizer.state_dict()),
                "cpu_rng": state_hash(torch.get_rng_state()), "cuda_rng": state_hash(torch.cuda.get_rng_state(0)),
                "python_rng": random.getstate(), "numpy_rng": numpy_rng()}
    optimizer.zero_grad(set_to_none=True)
    # One model only. Clear old optimizer allocations before restoring its state.
    optimizer.state.clear(); gc.collect(); torch.cuda.empty_cache()
    assert load(checkpoint, model, optimizer, contract=contract) == 1
    repeated_loss = update()
    assert second_loss == repeated_loss, "dropout next loss differs after resume"
    assert expected["model"] == state_hash(model.state_dict()), "post-resume parameter bytes differ"
    assert expected["optimizer"] == state_hash(optimizer.state_dict()), "post-resume AdamW bytes differ"
    assert expected["cpu_rng"] == state_hash(torch.get_rng_state())
    assert expected["cuda_rng"] == state_hash(torch.cuda.get_rng_state(0))
    assert expected["python_rng"] == random.getstate() and expected["numpy_rng"] == numpy_rng()
    testing = copy.deepcopy(records)
    testing[0]["token"][0] = "Carol"; testing[1]["token"][0] = "Dave"
    testing_groups = encode_groups(testing, table, tok, desc, split="test")
    with torch.no_grad():
        test_loss = trace("test", testing_groups, collate_groups(testing_groups, device="cuda:0"),
                          [r["relation"] for r in testing])
    torch.cuda.synchronize()
    result = {**describe(), "status": "single_cuda_diagnostic_verified", "identity": identity,
              "runtime": runtime(), "first_loss": first_loss, "second_loss": second_loss,
              "repeated_loss": repeated_loss, "test_loss": test_loss,
              "elapsed_seconds": time.monotonic() - start,
              "candidate_pair_shape": list(batch["input_ids"].shape),
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
              "checks": {"actual_cuda_forward_backward": True, "all_parameters_finite_gradients": True,
                         "exact_post_update_model_and_optimizer_resume": True,
                         "exact_dropout_next_loss_and_RNG": True, "train_test_numeric_roundtrip": True},
              "checkpoint": {"path": checkpoint.name, "sha256": sha(checkpoint)}, "traces": traces,
              "pending": ["full_task_adapter_and_data_identity", "full_recipe_and_runner",
                          "AMP_and_GPU_topology_resume", "formal_tensor_delivery", "independent_review"]}
    atomic_json(output / "cuda-evidence.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--describe", action="store_true")
    p.add_argument("--cache", type=Path); p.add_argument("--paper-source", type=Path)
    p.add_argument("--output", type=Path)
    init = p.add_mutually_exclusive_group()
    init.add_argument("--tiny", action="store_true"); init.add_argument("--weight", type=Path)
    p.add_argument("--variant", choices=("sup", "unsup"), default="sup")
    a = p.parse_args()
    if a.describe:
        print(json.dumps(describe())); return
    if not all((a.cache, a.paper_source, a.output)) or not (a.tiny or a.weight):
        p.error("private --cache --paper-source --output and explicit --tiny or --weight required")
    print(json.dumps(run(a.output, a.cache, a.paper_source, tiny=a.tiny, weight=a.weight, variant=a.variant)))


if __name__ == "__main__":
    main()
