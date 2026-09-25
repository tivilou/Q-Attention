#!/usr/bin/env python3
"""Run one Q-EPVG selector worker for the formal scheduler."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EXPERIMENTS = ROOT / "experiments"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from q_attention.experiments.relation_steering import (  # noqa: E402
    choose_device,
    load_relation_run,
    make_relation_loader,
    move_batch,
)
from q_attention.adapters.q_epvg_attention import QEPVGAttentionAdapter  # noqa: E402
from q_attention.adapters.encoder import resolve_module  # noqa: E402
from q_attention.experiments.batch_resume import (  # noqa: E402
    PAUSED_EXIT_CODE,
    TrainingMemoryPressure,
    TrainingPaused,
    atomic_write_json,
)
from q_attention.tasks.relation import load_relation_jsonl  # noqa: E402
from run_q_epvg_transfer_base import (  # noqa: E402
    train_kernel,
)
from run_q_epvg_formal_single_seed import (  # noqa: E402
    CUDA_OOM_EXIT_CODE,
    MEMORY_PRESSURE_EXIT_CODE,
    build_kernel,
    evaluate_selector,
    is_cuda_oom_error,
    selector_resume_contract,
    selector_resume_contract_compatible,
    metric_delta,
)


MIN_WORKER_FREE_MIB = 8 * 1024
MEMORY_PRESSURE_POLL_INTERVAL_STEPS = 20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class CudaMemoryPressureMonitor:
    """Reclaim only this worker's idle allocator cache after complete updates."""

    def __init__(
        self,
        *,
        selector: str,
        enabled: bool,
        restart_on_pressure: bool,
        poll_interval_steps: int = MEMORY_PRESSURE_POLL_INTERVAL_STEPS,
    ) -> None:
        self.selector = selector
        self.enabled = enabled
        self.restart_on_pressure = restart_on_pressure
        self.poll_interval_steps = max(1, int(poll_interval_steps))

    @staticmethod
    def _mib(value: int) -> int:
        return int(value // (1024 * 1024))

    def _snapshot(self) -> dict[str, int]:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        return {
            "free_mib": self._mib(free_bytes),
            "total_mib": self._mib(total_bytes),
            "allocated_mib": self._mib(allocated),
            "reserved_mib": self._mib(reserved),
        }

    @staticmethod
    def _minimum_free_mib(total_mib: int) -> int:
        # Keep a modest allocation margin, capped so large cards are not
        # needlessly treated as under pressure.
        return max(512, min(2 * 1024, total_mib // 20))

    def __call__(
        self, *, epoch: int, total_batches: int, cursor: Any
    ) -> dict[str, Any] | None:
        if (
            not self.enabled
            or not torch.cuda.is_available()
            or int(cursor.global_step) % self.poll_interval_steps != 0
        ):
            return None
        try:
            before = self._snapshot()
        except Exception as exc:  # diagnostics must never stop a valid update loop
            print(
                json.dumps(
                    {
                        "event": "memory_pressure_sample_failed",
                        "selector": self.selector,
                        "error": str(exc),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return None
        minimum_free_mib = self._minimum_free_mib(before["total_mib"])
        cached_mib = max(0, before["reserved_mib"] - before["allocated_mib"])
        under_pressure = before["free_mib"] < minimum_free_mib
        fragmented = (
            before["free_mib"] < 2 * minimum_free_mib
            and cached_mib >= max(512, before["total_mib"] // 20)
        )
        if not under_pressure and not fragmented:
            return None

        # gc collects only unreachable objects; empty_cache returns only idle
        # blocks from this process's PyTorch allocator. Neither can release
        # active training tensors or memory owned by another CUDA process.
        gc.collect()
        torch.cuda.empty_cache()
        try:
            after = self._snapshot()
        except Exception as exc:  # reclaim succeeded even if the second sample is unavailable
            print(
                json.dumps(
                    {
                        "event": "memory_pressure_sample_failed",
                        "selector": self.selector,
                        "phase": "after_reclaim",
                        "error": str(exc),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return None
        event = {
            "event": "memory_pressure_reclaim",
            "selector": self.selector,
            "epoch": int(epoch),
            "batch": int(cursor.next_batch_index),
            "batches": int(total_batches),
            "global_step": int(cursor.global_step),
            "minimum_free_mib": minimum_free_mib,
            "trigger": "low_free" if under_pressure else "fragmented_cache",
            "before": before,
            "after": after,
            "reclaimed_reserved_mib": max(
                0, before["reserved_mib"] - after["reserved_mib"]
            ),
            "reclaimed_free_mib": max(0, after["free_mib"] - before["free_mib"]),
        }
        print(json.dumps(event, sort_keys=True), flush=True)
        if after["free_mib"] >= minimum_free_mib or not self.restart_on_pressure:
            return None
        return event


def check_worker_gpu_capacity(device_name: str) -> None:
    """Reject a worker start when its assigned physical GPU is already busy."""
    if device_name != "cuda":
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.isdigit():
        return
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"worker GPU capacity check failed: {result.stderr.strip()}")
    rows = {}
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 2)]
        if len(fields) != 3:
            continue
        try:
            rows[int(fields[0])] = (int(fields[1]), int(fields[2]))
        except ValueError:
            continue
    physical_id = int(visible)
    if physical_id not in rows:
        raise RuntimeError(f"worker GPU capacity check could not find physical GPU {physical_id}")
    free_mib, total_mib = rows[physical_id]
    if free_mib < MIN_WORKER_FREE_MIB:
        raise RuntimeError(
            f"worker GPU {physical_id} has only {free_mib} MiB free of {total_mib} MiB; "
            f"at least {MIN_WORKER_FREE_MIB} MiB is required. "
            "A competing CUDA process is using the assigned GPU."
        )


def write_case_study(
    *,
    model: torch.nn.Module,
    kernel: torch.nn.Module,
    records: dict[str, list[Any]],
    artifacts: Any,
    device: torch.device,
    config: dict[str, Any],
    config_path: Path,
    output_dir: Path,
    selector: str,
    initial_state: dict[str, Any] | None = None,
    final_state: dict[str, Any] | None = None,
) -> None:
    """Emit semantic train/valid/test traces plus detached tensor evidence.

    The trace is replay-only: samples and checkpoints are frozen in config, and
    no target label is passed to the selector.  Full tensors are written to a
    private per-selector directory; the JSON payload contains only safe scalar
    projections and manifests that the portal/exporter may consume.
    """
    case_config = config.get("case_study", {})
    split_records = {
        "train": list(case_config.get("records", {}).get("train", [])),
        "valid": list(case_config.get("records", {}).get("valid", [])),
        "test": list(case_config.get("records", {}).get("test", [])),
    }
    if not any(split_records.values()):
        legacy_split = str(case_config.get("split", "valid"))
        legacy_indices = list(case_config.get("record_indices", [0, 1, 2]))
        # Older configs froze validation examples only.  Replicate the
        # same deterministic positions in each split so the upgraded writer
        # remains callable for historical controls while new configs must
        # declare split-specific samples explicitly.
        for split in split_records:
            split_records[split] = list(legacy_indices)
    for split, indices in split_records.items():
        split_records[split] = [int(index) for index in indices]
        if any(index < 0 or index >= len(records[split]) for index in indices):
            raise ValueError(f"case_study.records.{split} contains an out-of-range index")
    if not all(split_records.values()):
        raise ValueError("case_study must freeze at least one sample for train, valid, and test")

    def _tensor_manifest(
        tensor: torch.Tensor,
        *,
        capture_root: Path,
        name: str,
        axis_semantics: list[str],
    ) -> dict[str, Any]:
        del capture_root
        value = tensor.detach().to(device="cpu").contiguous()
        payload = io.BytesIO()
        torch.save(value, payload, _use_new_zipfile_serialization=False)
        raw = payload.getvalue()
        relative = Path("case_study_tensors") / f"{name}.pt"
        target = output_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        preview_value = value.real if value.is_complex() else value
        preview_float = preview_value.float()
        return {
            "id": name.rsplit("__", 1)[-1],
            "path": str(relative),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "axis_semantics": axis_semantics,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "byte_count": len(raw),
            "preview": {
                "values_first_32": preview_value.flatten()[:32].tolist(),
                "min": float(preview_float.min().item()) if preview_float.numel() else 0.0,
                "max": float(preview_float.max().item()) if preview_float.numel() else 0.0,
                "mean": float(preview_float.mean().item()) if preview_float.numel() else 0.0,
                "l2_norm": float(preview_float.norm().item()),
            },
        }

    def _detach_map(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {key: _detach_map(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_detach_map(item) for item in value]
        return value

    def _forward_capture(
        selected: list[Any],
        split: str,
        checkpoint_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        loader = make_relation_loader(
            selected, artifacts.vocab, artifacts.label_to_id, batch_size=len(selected)
        )
        batch = move_batch(next(iter(loader)), device)
        captures: dict[str, Any] = {
            "token_embeddings": None,
            "hidden_states": {},
            "qkv": {},
            "baseline_scores": {},
            "steered_scores": {},
            "epvg_trace": {},
        }

        def _register_common(mode: str) -> list[torch.utils.hooks.RemovableHandle]:
            handles: list[torch.utils.hooks.RemovableHandle] = []
            embedding = resolve_module(model, "encoder.token_embedding")
            handles.append(
                embedding.register_forward_hook(
                    lambda _m, _i, out: captures.__setitem__(
                        "token_embeddings", out.detach()
                    )
                )
            )
            for layer_index in range(int(model.config.num_layers)):
                layer_path = f"encoder.layers.{layer_index}"
                layer = resolve_module(model, layer_path)
                handles.append(
                    layer.register_forward_hook(
                        lambda _m, _i, out, index=layer_index: captures["hidden_states"].__setitem__(
                            index, out.detach()
                        )
                    )
                )
                for projection in ("query_proj", "key_proj", "value_proj"):
                    path = f"encoder.layers.{layer_index}.attn.{projection}"
                    module = resolve_module(model, path)
                    handles.append(
                        module.register_forward_hook(
                            lambda _m, _i, out, index=layer_index, name=projection: captures["qkv"].__setitem__(
                                (index, name), out.detach()
                            )
                        )
                    )
            return handles

        model.eval()
        kernel.eval()
        with torch.no_grad():
            baseline_handles = _register_common("baseline")
            try:
                baseline_logits = model(
                    batch["input_ids"], batch["attention_mask"],
                    batch["subject_mask"], batch["object_mask"]
                )
            finally:
                for handle in baseline_handles:
                    handle.remove()
            captures["hidden_states"] = {}
            captures["qkv"] = {}
            adapter = QEPVGAttentionAdapter(model, list(kernel))
            adapter.attach()
            steered_handles = _register_common("steered")
            try:
                selector_logits = model(
                    batch["input_ids"], batch["attention_mask"],
                    batch["subject_mask"], batch["object_mask"]
                )
            finally:
                for handle in steered_handles:
                    handle.remove()
                adapter.remove()
                captures["epvg_trace"] = {
                    int(index): {name: _detach_map(value) for name, value in trace.items()}
                    for index, trace in enumerate(adapter.traces)
                    if isinstance(trace, dict)
                }
                # The explicit intervention API does not expose a score-only
                # module hook. Reconstruct the baseline score tensor from the
                # captured projected Q/K and use the EPVG trace adjustment for
                # the steered score projection.
                for layer_index in range(int(model.config.num_layers)):
                    query = captures["qkv"][(layer_index, "query_proj")]
                    key = captures["qkv"][(layer_index, "key_proj")]
                    batch_size, tokens, _ = query.shape
                    heads = int(model.config.num_heads)
                    head_dim = int(model.config.dim // heads)
                    q = query.view(batch_size, tokens, heads, head_dim).transpose(1, 2)
                    k = key.view(batch_size, tokens, heads, head_dim).transpose(1, 2)
                    base_scores = torch.matmul(q, k.transpose(-1, -2)) / (head_dim ** 0.5)
                    trace = captures["epvg_trace"].get(layer_index, {})
                    adjustment = trace.get("score_adjustment", torch.zeros_like(base_scores))
                    captures["baseline_scores"][layer_index] = {"input": base_scores}
                    captures["steered_scores"][layer_index] = {"input": base_scores + adjustment}

        final_hidden = captures["hidden_states"].get(int(model.config.num_layers) - 1)
        if final_hidden is None:
            raise RuntimeError("case-study hidden-state hook did not capture final encoder output")
        subject_mask = batch["subject_mask"].to(dtype=final_hidden.dtype)
        object_mask = batch["object_mask"].to(dtype=final_hidden.dtype)
        attention_mask = batch["attention_mask"].to(dtype=final_hidden.dtype)
        pooled = torch.cat(
            (
                (final_hidden * subject_mask.unsqueeze(-1)).sum(1) / subject_mask.sum(1, keepdim=True).clamp_min(1),
                (final_hidden * object_mask.unsqueeze(-1)).sum(1) / object_mask.sum(1, keepdim=True).clamp_min(1),
                (final_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True).clamp_min(1),
            ),
            dim=-1,
        )
        captures["subject_object_pooled_states"] = pooled.detach()
        captures["input_ids"] = batch["input_ids"].detach()
        captures["attention_mask"] = batch["attention_mask"].detach()
        captures["subject_mask"] = batch["subject_mask"].detach()
        captures["object_mask"] = batch["object_mask"].detach()
        captures["probabilities_baseline"] = torch.softmax(baseline_logits, dim=-1).detach()
        captures["probabilities_selector"] = torch.softmax(selector_logits, dim=-1).detach()
        captures["baseline_logits"] = baseline_logits.detach()
        captures["selector_logits"] = selector_logits.detach()
        captures["batch"] = batch
        captures["split"] = split
        captures["checkpoint"] = checkpoint_name
        return baseline_logits.detach(), selector_logits.detach(), captures

    def _indices_for(split: str) -> list[int]:
        return split_records[split]

    initial_state = copy.deepcopy(initial_state or kernel.state_dict())
    best_state = copy.deepcopy(kernel.state_dict())
    checkpoint_states: list[tuple[str, dict[str, Any], str]] = [
        ("initial_or_pre_training", initial_state, "initial"),
        ("best_valid_or_declared_selection_checkpoint", best_state, "best"),
        ("final", copy.deepcopy(final_state or best_state), "final"),
    ]
    # The caller invokes this function after train_kernel has loaded the best
    # checkpoint; all three captures are therefore deterministic, with the
    # initial state retained for the pre-training replay.
    all_cases: list[dict[str, Any]] = []
    stages_by_case: list[dict[str, Any]] = []
    tensor_manifest: list[dict[str, Any]] = []
    for checkpoint_label, state, checkpoint_slug in checkpoint_states:
        kernel.load_state_dict(state)
        for split in ("train", "valid", "test"):
            selected = [records[split][index] for index in _indices_for(split)]
            baseline_logits, selector_logits, captures = _forward_capture(
                selected, split, checkpoint_slug
            )
            batch = captures["batch"]
            for local_index, record_index in enumerate(_indices_for(split)):
                record = selected[local_index]
                case_id = f"{selector}:{split}:{record_index}:{checkpoint_slug}"
                prefix = f"{split}_{record_index}_{checkpoint_slug}"
                reps: dict[str, Any] = {}

                def add_rep(rep_id: str, value: torch.Tensor, axes: list[str]) -> None:
                    manifest = _tensor_manifest(
                        value,
                        capture_root=output_dir,
                        name=f"{prefix}__{rep_id}",
                        axis_semantics=axes,
                    )
                    reps[rep_id] = manifest
                    tensor_manifest.append(manifest)

                add_rep("token_embeddings", captures["token_embeddings"][local_index], ["tokens", "hidden_dim"])
                hidden = torch.stack(
                    [captures["hidden_states"][index][local_index] for index in sorted(captures["hidden_states"])],
                    dim=0,
                )
                add_rep("encoder_hidden_states", hidden, ["layers", "tokens", "hidden_dim"])
                add_rep("subject_object_pooled_states", captures["subject_object_pooled_states"][local_index], ["pooled_features"])
                qkv = torch.stack(
                    [torch.stack([captures["qkv"][(index, name)][local_index] for name in ("query_proj", "key_proj", "value_proj")], dim=0) for index in range(int(model.config.num_layers))],
                    dim=0,
                )
                add_rep("attention_qkv", qkv, ["layers", "qkv", "tokens", "model_dim"])
                base_scores = torch.stack([captures["baseline_scores"][index]["input"][local_index] for index in sorted(captures["baseline_scores"])], dim=0)
                steered_scores = torch.stack([captures["steered_scores"][index]["input"][local_index] for index in sorted(captures["steered_scores"])], dim=0)
                add_rep("baseline_attention_scores", base_scores, ["layers", "heads", "query_tokens", "key_tokens"])
                add_rep("steered_attention_scores", steered_scores, ["layers", "heads", "query_tokens", "key_tokens"])
                trace_by_layer = captures.get("epvg_trace", {})
                for rep_id, trace_key, axes in (
                    ("q_epvg_query", "query", ["layers", "heads", "query_tokens", "head_dim"]),
                    ("q_epvg_key", "key", ["layers", "heads", "key_tokens", "head_dim"]),
                    ("q_epvg_value", "value", ["layers", "heads", "key_tokens", "value_dim"]),
                    ("q_epvg_zz", "zz", ["layers", "heads", "query_tokens", "key_tokens"]),
                    ("q_epvg_xx", "xx", ["layers", "heads", "query_tokens", "key_tokens"]),
                    ("q_epvg_observable", "observable", ["layers", "heads", "query_tokens", "key_tokens"]),
                    ("q_epvg_theta", "theta", ["layers", "heads", "query_tokens", "key_tokens"]),
                    ("q_epvg_gate", "gate", ["layers", "heads", "query_tokens", "key_tokens"]),
                    ("q_epvg_base_attention", "base_attention", ["heads", "query_tokens", "key_tokens"]),
                    ("q_epvg_score_adjustment", "score_adjustment", ["layers", "heads", "query_tokens", "key_tokens"]),
                    ("q_epvg_attention", "attention", ["layers", "heads", "query_tokens", "key_tokens"]),
                    ("q_epvg_routed_values", "routed_values", ["layers", "heads", "query_tokens", "key_tokens", "value_dim"]),
                    ("q_epvg_query_update", "query_update", ["layers", "heads", "query_tokens", "head_dim"]),
                    ("q_epvg_output", "output", ["layers", "heads", "query_tokens", "value_dim"]),
                ):
                    rows = []
                    for layer_index in range(int(model.config.num_layers)):
                        trace = trace_by_layer.get(layer_index)
                        if not isinstance(trace, dict) or trace_key not in trace:
                            continue
                        rows.append(trace[trace_key][local_index])
                    if rows:
                        add_rep(rep_id, torch.stack(rows, dim=0), axes)
                add_rep("classifier_logits_probabilities", torch.stack((captures["baseline_logits"][local_index], captures["selector_logits"][local_index], captures["probabilities_baseline"][local_index], captures["probabilities_selector"][local_index])), ["variant_probability_or_logit", "labels"])
                labels = batch["labels"].cpu().tolist()
                baseline_prediction = int(baseline_logits.argmax(-1)[local_index].item())
                selector_prediction = int(selector_logits.argmax(-1)[local_index].item())
                case = {
                    "case_id": case_id,
                    "split": split,
                    "checkpoint": checkpoint_label,
                    "record_index": int(record_index),
                    "sentence": " ".join(record.tokens),
                    "tokens": list(record.tokens),
                    "token_ids": batch["input_ids"][local_index].cpu().tolist(),
                    "attention_mask": batch["attention_mask"][local_index].cpu().tolist(),
                    "subject": {"text": " ".join(record.tokens[record.subject[0]:record.subject[1]]), "span": list(record.subject), "token_positions": torch.nonzero(batch["subject_mask"][local_index], as_tuple=False).flatten().cpu().tolist(), "entity_type": dict(record.metadata).get("subject_type", dict(record.metadata).get("subj_type"))},
                    "object": {"text": " ".join(record.tokens[record.object[0]:record.object[1]]), "span": list(record.object), "token_positions": torch.nonzero(batch["object_mask"][local_index], as_tuple=False).flatten().cpu().tolist(), "entity_type": dict(record.metadata).get("object_type", dict(record.metadata).get("obj_type"))},
                    "metadata": dict(record.metadata),
                    "gold_relation": record.label,
                    "label_access": "post_evaluation_only",
                    "prediction_labels": {
                        "baseline": artifacts.id_to_label.get(baseline_prediction, str(baseline_prediction)),
                        "selector": artifacts.id_to_label.get(selector_prediction, str(selector_prediction)),
                    },
                    "label_id": int(labels[local_index]),
                    "baseline_prediction": baseline_prediction,
                    "selector_prediction": selector_prediction,
                    "baseline_correct": baseline_prediction == int(labels[local_index]),
                    "selector_correct": selector_prediction == int(labels[local_index]),
                    "baseline_logits": captures["baseline_logits"][local_index].cpu().tolist(),
                    "selector_logits": captures["selector_logits"][local_index].cpu().tolist(),
                    "baseline_probabilities": captures["probabilities_baseline"][local_index].cpu().tolist(),
                    "selector_probabilities": captures["probabilities_selector"][local_index].cpu().tolist(),
                    "representations": reps,
                }
                all_cases.append(case)
                stages_by_case.append({
                    "sample_id": case_id,
                    "split_position": int(record_index),
                    "checkpoint": checkpoint_label,
                    "stages": [
                        {"stage": "data", "status": "observed", "observed_fields": {"sentence": case["sentence"], "tokens": case["tokens"], "subject": case["subject"], "object": case["object"]}},
                        {"stage": "preprocess", "status": "observed", "observed_fields": {"token_ids": case["token_ids"], "attention_mask": case["attention_mask"], "representations": ["token_embeddings", "encoder_hidden_states", "subject_object_pooled_states"]}},
                        {"stage": "training", "status": "observed", "observed_fields": {"selector": selector, "checkpoint": checkpoint_label, "epochs": int(config["kernel"]["epochs"]), "replay_only": True}},
                        {"stage": "retrieval", "status": "not_applicable", "observed_fields": {"reason": "relation extraction has no retrieval stage"}},
                        {"stage": "scoring", "status": "observed", "observed_fields": {"representations": ["attention_qkv", "baseline_attention_scores", "q_epvg_alignment_real", "q_epvg_alignment_imag", "q_epvg_gate"]}},
                        {"stage": "selection", "status": "observed", "observed_fields": {"representations": ["q_epvg_score_adjustment", "steered_attention_scores"]}},
                        {"stage": "context", "status": "observed", "observed_fields": {"representations": ["q_epvg_routed_values", "q_epvg_output"], "selector_prediction": selector_prediction}},
                        {"stage": "generation", "status": "not_applicable", "observed_fields": {"reason": "relation extraction outputs a class label"}},
                        {"stage": "evaluation", "status": "observed", "observed_fields": {"gold_relation": record.label, "baseline_prediction": baseline_prediction, "selector_prediction": selector_prediction, "representations": ["classifier_logits_probabilities"]}},
                        {"stage": "diagnosis", "status": "observed", "observed_fields": {"kernel_metadata": kernel.metadata(), "tensor_manifest_count": len(reps)}},
                    ],
                })
    kernel.load_state_dict(checkpoint_states[-1][1])
    payload = {
        "schema_version": "q-attention.Q-EPVG-case-study.v1",
        "selector": selector,
        "status": "observed",
        "selection_rule": "Train/valid/test indices and checkpoint names are frozen in the formal config before execution; replay is post-training and never used for optimization or model selection.",
        "required_splits": ["train", "valid", "test"],
        "checkpoint_policy": [item[0] for item in checkpoint_states],
        "kernel_metadata": kernel.metadata(),
        "representation_inventory": sorted({manifest["id"] for manifest in tensor_manifest}),
        "tensor_manifest": tensor_manifest,
        "provenance": {
            "git_revision": _git_revision(),
            "config_sha256": _sha256(config_path),
            "plugin_sha256": _sha256(ROOT / "src" / "q_attention" / "plugins" / "q_epvg.py"),
            "full_trace": "case_study_tensors/",
            "portal_projection": "case_study.json",
        },
        "cases": all_cases,
    }
    (output_dir / "case_study.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    sample_trace = {
        "schema_version": "sample-trace.v1",
        "trace_id": f"{output_dir.name}:{selector}",
        "experiment": {"run_id": output_dir.parent.parent.name, "dataset": "retacred.train+valid+test", "code_revision": _git_revision(), "config_sha256": _sha256(config_path), "model_identity": "relation-transformer-Q-EPVG", "seed": int(config["seed"])},
        "sample_selection": {"rule": payload["selection_rule"], "population_scope": "retacred.train+valid+test", "seed": int(config["seed"]), "selected_count": len(stages_by_case), "selected_sample_ids": [item["sample_id"] for item in stages_by_case]},
        "coverage": {"data": "observed", "preprocess": "observed", "training": "observed", "retrieval": "not_applicable", "scoring": "observed", "selection": "observed", "context": "observed", "generation": "not_applicable", "evaluation": "observed", "diagnosis": "observed"},
        "semantic_contract": {"version": "q-attention.case-study-trace-contract.v2", "required_roles": ["source_sample", "sequence_or_features", "task_objects", "target_or_gold", "prediction", "diagnosis"], "required_splits": ["train", "valid", "test"], "required_checkpoints": [item[0] for item in checkpoint_states], "representation_inventory": payload["representation_inventory"]},
        "samples": stages_by_case,
    }
    (output_dir / "sample_trace.json").write_text(json.dumps(sample_trace, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selector", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--log-every-batches", type=int, default=50)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume this selector from its compatible post-update checkpoint",
    )
    parser.add_argument("--checkpoint-every-batches", type=int, default=50)
    parser.add_argument(
        "--pair-chunk-size",
        default="all",
        help="maximum pair chunk, or all for every pair in the physical micro-batch",
    )
    parser.add_argument(
        "--pair-chunk-divisor", type=int, default=1,
        help="divide an all-pairs chunk by this positive power-of-two fallback divisor",
    )
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--activation-checkpointing", type=int, choices=(0, 1), default=None)
    parser.add_argument(
        "--adaptive-memory",
        action="store_true",
        help="allow the parent scheduler to resume this worker with a lower memory tier",
    )
    parser.add_argument(
        "--elastic-resume",
        action="store_true",
        help="allow an explicit execution-only contract migration during resume",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.log_every_batches <= 0:
        raise ValueError("--log-every-batches must be positive")
    if args.checkpoint_every_batches <= 0:
        raise ValueError("--checkpoint-every-batches must be positive")
    if args.pair_chunk_size == "all":
        args.pair_chunk_size = None
    else:
        try:
            args.pair_chunk_size = int(args.pair_chunk_size)
        except ValueError as exc:
            raise ValueError("--pair-chunk-size must be a positive integer or all") from exc
        if args.pair_chunk_size <= 0:
            raise ValueError("--pair-chunk-size must be positive")
    if args.pair_chunk_divisor <= 0:
        raise ValueError("--pair-chunk-divisor must be positive")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    check_worker_gpu_capacity(args.device)
    device = choose_device(args.device)
    artifacts = load_relation_run(args.baseline_dir, device)
    data_dir = args.data_dir
    train_records = load_relation_jsonl(data_dir / "train.jsonl")
    valid_records = load_relation_jsonl(data_dir / "valid.jsonl")
    test_records = load_relation_jsonl(data_dir / "test.jsonl")
    kernel_config = config["kernel"]
    logical_batch_size = int(kernel_config["batch_size"])
    if args.micro_batch_size is None:
        args.micro_batch_size = logical_batch_size
    if args.micro_batch_size <= 0 or args.micro_batch_size > logical_batch_size:
        raise ValueError("--micro-batch-size must be in (0, logical batch size]")
    if args.micro_batch_size * args.gradient_accumulation_steps != logical_batch_size:
        raise ValueError(
            "--micro-batch-size * --gradient-accumulation-steps must equal the logical batch size"
        )
    valid_loader = make_relation_loader(
        valid_records,
        artifacts.vocab,
        artifacts.label_to_id,
        batch_size=int(kernel_config["batch_size"]),
    )
    test_loader = make_relation_loader(
        test_records,
        artifacts.vocab,
        artifacts.label_to_id,
        batch_size=int(kernel_config["batch_size"]),
    )
    for parameter in artifacts.model.parameters():
        parameter.requires_grad_(False)
    baseline_test = json.loads(
        (args.data_dir.parent / "baseline_eval.json").read_text(encoding="utf-8")
    )["test"]
    kernel = build_kernel(
        args.selector,
        artifacts.model,
        args.seed,
        config,
        pair_chunk_size=args.pair_chunk_size,
        pair_chunk_divisor=args.pair_chunk_divisor,
        activation_checkpointing=(
            None
            if args.activation_checkpointing is None
            else bool(args.activation_checkpointing)
        ),
    ).to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    initial_state_path = args.output_dir / "initial_kernel_with_metadata.pt"
    if args.resume and initial_state_path.is_file():
        initial_payload = torch.load(initial_state_path, map_location="cpu", weights_only=True)
        initial_kernel_state = initial_payload["state_dict"]
    else:
        initial_kernel_state = copy.deepcopy(kernel.state_dict())
        torch.save(
            {"state_dict": initial_kernel_state, "metadata": kernel.metadata()},
            initial_state_path,
        )
    train_args = argparse.Namespace(
        batch_size=logical_batch_size,
        epochs=int(kernel_config["epochs"]),
        kernel_lr=float(kernel_config["lr"]),
        log_every_batches=args.log_every_batches,
        batch_resume=True,
        resume=args.resume,
        checkpoint_every_batches=args.checkpoint_every_batches,
        resume_contract=selector_resume_contract(
            config_path=args.config,
            baseline_dir=args.baseline_dir,
            data_dir=data_dir,
            selector=args.selector,
            seed=args.seed,
            pair_chunk_size=args.pair_chunk_size,
            pair_chunk_divisor=args.pair_chunk_divisor,
            micro_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            activation_checkpointing=args.activation_checkpointing,
            adaptive_memory=args.adaptive_memory,
        ),
        resume_contract_compatible=(
            selector_resume_contract_compatible if args.elastic_resume else None
        ),
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        query_chunk_size=(
            None
            if args.pair_chunk_size is None
            else max(1, int(args.pair_chunk_size) // max(1, int(args.pair_chunk_divisor)))
        ),
        memory_pressure_monitor=CudaMemoryPressureMonitor(
            selector=args.selector,
            enabled=device.type == "cuda",
            restart_on_pressure=args.adaptive_memory,
        ),
    )
    try:
        train_result = train_kernel(
            artifacts.model,
            kernel,
            train_records,
            valid_loader,
            artifacts,
            device,
            args.selector,
            args.seed,
            train_args,
            args.output_dir,
        )
        final_kernel_state_path = args.output_dir / "final_kernel.pt"
        final_kernel_state = (
            torch.load(final_kernel_state_path, map_location="cpu", weights_only=True)
            if final_kernel_state_path.is_file()
            else None
        )
    except TrainingPaused:
        print(
            json.dumps(
                {"event": "run_paused", "selector": args.selector}, sort_keys=True
            ),
            flush=True,
        )
        return PAUSED_EXIT_CODE
    except TrainingMemoryPressure as exc:
        event = {
            "event": "memory_pressure_restart",
            "selector": args.selector,
            "checkpoint": str(args.output_dir / "checkpoints" / "latest.pt"),
            "diagnostics": exc.diagnostics,
        }
        atomic_write_json(args.output_dir / "memory_pressure_event.json", event)
        print(json.dumps(event, sort_keys=True), flush=True)
        return MEMORY_PRESSURE_EXIT_CODE
    valid_result = evaluate_selector(
        artifacts.model,
        valid_loader,
        device,
        len(artifacts.label_to_id),
        kernel,
        f"{args.selector}_valid_final",
    )
    test_result = evaluate_selector(
        artifacts.model,
        test_loader,
        device,
        len(artifacts.label_to_id),
        kernel,
        f"{args.selector}_test",
    )
    metadata: dict[str, Any] = kernel.metadata()
    trainable_parameters = sum(parameter.numel() for parameter in kernel.parameters())
    torch.save(
        {"state_dict": kernel.state_dict(), "metadata": metadata},
        args.output_dir / "best_kernel_with_metadata.pt",
    )
    write_case_study(
        model=artifacts.model,
        kernel=kernel,
        records={"train": train_records, "valid": valid_records, "test": test_records},
        artifacts=artifacts,
        device=device,
        config=config,
        config_path=args.config,
        output_dir=args.output_dir,
        selector=args.selector,
        initial_state=initial_kernel_state,
        final_state=final_kernel_state,
    )
    torch.save(
        {"state_dict": kernel.state_dict(), "metadata": metadata},
        args.output_dir / "final_kernel_with_metadata.pt",
    )
    row = {
        "selector": args.selector,
        "seed": args.seed,
        "valid": valid_result,
        "test": {
            **test_result,
            "delta_vs_baseline": metric_delta(test_result["metrics"], baseline_test["metrics"]),
        },
        "train": train_result,
        "metadata": metadata,
        "trainable_parameters": trainable_parameters,
        "finite": all(
            torch.isfinite(torch.tensor(value))
            for value in list(valid_result["metrics"].values())
            + list(test_result["metrics"].values())
        ),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(row, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "event": "selector_complete",
                "selector": args.selector,
                "device": str(device),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as exc:
        if is_cuda_oom_error(exc):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            selector = sys.argv[sys.argv.index("--selector") + 1] if "--selector" in sys.argv else "<unknown>"
            print(
                json.dumps(
                    {"event": "cuda_oom", "selector": selector, "error": str(exc)},
                    sort_keys=True,
                ),
                flush=True,
            )
            raise SystemExit(CUDA_OOM_EXIT_CODE) from exc
        raise
