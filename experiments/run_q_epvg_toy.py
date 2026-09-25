#!/usr/bin/env python3
"""Run the bounded Q-EPVG observable/path preflight and emit sample-trace.v1."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q_attention.plugins.q_epvg import (  # noqa: E402
    EPVG_CONTROLS,
    EPVG_OBSERVABLES,
    EPVG_PATHS,
    QEPVGConfig,
    build_q_epvg,
)


def tensor_json(value: torch.Tensor) -> dict[str, Any]:
    value = value.detach().cpu()
    return {"shape": list(value.shape), "dtype": str(value.dtype).replace("torch.", ""), "values": value.tolist()}


def make_batch(seed: int, size: int = 4) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    query = torch.randn(size, 1, 5, 4, generator=generator)
    key = torch.randn(size, 1, 5, 4, generator=generator)
    value = torch.randn(size, 1, 5, 4, generator=generator)
    scores = torch.randn(size, 1, 5, 5, generator=generator) * 0.1
    attention_mask = torch.ones(size, 5, dtype=torch.bool)
    attention_mask[:, -1] = False
    query_mask = torch.ones(size, 5, dtype=torch.bool)
    query_mask[:, -1] = False
    return {"query": query, "key": key, "value": value, "scores": scores, "attention_mask": attention_mask, "query_mask": query_mask}


def run_variant(seed: int, observable: str, path: str, control: str) -> dict[str, Any]:
    batch = make_batch(seed)
    kernel = build_q_epvg(QEPVGConfig(observable=observable, path=path, seed=seed + 17))
    output, trace = kernel(**batch, control=control, return_trace=True)
    return {
        "seed": seed,
        "observable": observable,
        "path": path,
        "control": control,
        "finite": bool(torch.isfinite(output).all() and all(torch.isfinite(v).all() for v in trace.values())),
        "masked_attention_zero": bool(torch.allclose(trace["attention"][..., -1], torch.zeros_like(trace["attention"][..., -1]), atol=1e-7)),
        "row_sum": float(trace["attention"].sum(dim=-1).mean().item()),
        "output_rms": float(output.square().mean().sqrt().item()),
        "trace": {name: tensor_json(value) for name, value in trace.items()},
    }


def sample_trace(result: dict[str, Any], split: str, checkpoint: str) -> dict[str, Any]:
    return {
        "schema": "sample-trace.v1",
        "selector": f"q_epvg_{result['observable']}_{result['path']}_{result['control']}",
        "seed": result["seed"],
        "split": split,
        "checkpoint": checkpoint,
        "sample_index": 0,
        "semantic": {
            "sentence": "Alice relates to Acme.",
            "tokens": ["[CLS]", "Alice", "relates", "Acme", "[SEP]"],
            "entity_pair": {"subject": "Alice", "object": "Acme"},
            "entity_spans": {"subject": [1, 1], "object": [3, 3]},
            "target_relation": None,
            "predicted_relation": None,
            "target_boundary": "post_evaluation_only",
        },
        "intermediate": result["trace"],
        "metadata": {
            "observable": result["observable"],
            "path": result["path"],
            "control": result["control"],
            "fixed_reducer": True,
            "fixed_linear_mapping": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="13")
    parser.add_argument("--output-root", default="runs/q_epvg_toy")
    args = parser.parse_args()
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    traces = []
    for seed in seeds:
        for observable in EPVG_OBSERVABLES:
            for path in EPVG_PATHS:
                for control in EPVG_CONTROLS:
                    result = run_variant(seed, observable, path, control)
                    results.append({k: v for k, v in result.items() if k != "trace"})
                    traces.extend(sample_trace(result, split, checkpoint) for split, checkpoint in (("train", "initial"), ("valid", "best"), ("test", "final")))
    finite = all(item["finite"] for item in results)
    masked = all(item["masked_attention_zero"] for item in results)
    gates = {
        "all_finite": finite,
        "all_masks_respected": masked,
        "all_row_sums_unit": all(abs(item["row_sum"] - 1.0) < 1e-5 for item in results),
        "matrix_complete": len(results) == len(seeds) * 3 * 3 * 3,
        "control_outputs_present": {item["control"] for item in results} == set(EPVG_CONTROLS),
    }
    summary = {
        "schema": "q-epvg-toy.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {"seeds": seeds, "observables": list(EPVG_OBSERVABLES), "paths": list(EPVG_PATHS), "controls": list(EPVG_CONTROLS)},
        "results": results,
        "toy_gates": gates,
        "trace_contract": {"schema": "sample-trace.v1", "required_splits": ["train", "valid", "test"], "required_checkpoints": ["initial", "best", "final"], "target_boundary": "post_evaluation_only"},
    }
    summary["payload_sha256"] = hashlib.sha256(json.dumps(summary, sort_keys=True).encode()).hexdigest()
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    trace_payload = {"schema": "sample-trace.v1", "experiment": "q_epvg_toy", "traces": traces, "coverage": {"splits": ["train", "valid", "test"], "checkpoints": ["initial", "best", "final"]}}
    (output / "sample-trace.json").write_text(json.dumps(trace_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "toy_gates": gates}, sort_keys=True))


if __name__ == "__main__":
    main()
