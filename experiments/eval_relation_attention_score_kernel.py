from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q_attention.adapters import AttentionScoreKernelAdapter  # noqa: E402
from q_attention.experiments import (  # noqa: E402
    choose_device,
    diagnose_relation_counterfactual_evidence,
    diagnose_relation_evidence_task_alignment,
    diagnose_relation_expert_routing,
    diagnose_relation_routing_task_alignment,
    evaluate_relation_attention_score_kernel,
    load_relation_run,
    make_relation_loader,
)
from q_attention.plugins import load_relation_attention_score_kernel_checkpoint  # noqa: E402
from q_attention.tasks.relation import load_relation_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an attention-score kernel checkpoint.")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--random_repeats", type=int, default=4)
    parser.add_argument("--random_seed", type=int, default=101)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def metric_delta(steered: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {key: steered[key] - baseline[key] for key in steered.keys() & baseline.keys()}


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = load_relation_run(Path(args.model_dir), device)
    kernel, checkpoint_metadata = load_relation_attention_score_kernel_checkpoint(
        args.checkpoint,
        map_location=device,
    )
    kernel.to(device).eval()
    expected_paths = checkpoint_metadata.get("score_module_paths")
    if expected_paths is not None and tuple(expected_paths) != tuple(artifacts.model.score_module_paths):
        raise ValueError("score-kernel checkpoint module paths do not match the base model")
    loader = make_relation_loader(
        load_relation_jsonl(Path(args.data_path)),
        artifacts.vocab,
        artifacts.label_to_id,
        batch_size=args.batch_size,
        shuffle=False,
    )
    adapter = AttentionScoreKernelAdapter(
        artifacts.model,
        artifacts.model.score_module_paths,
        kernel,
    )
    baseline = evaluate_relation_attention_score_kernel(
        artifacts.model,
        loader,
        device,
        len(artifacts.label_to_id),
        adapter=None,
    )
    steered = evaluate_relation_attention_score_kernel(
        artifacts.model,
        loader,
        device,
        len(artifacts.label_to_id),
        adapter=adapter,
    )
    uniform_routing = None
    routing_diagnostics = None
    routing_alignment = None
    if kernel.expert_router is not None:
        uniform_routing = evaluate_relation_attention_score_kernel(
            artifacts.model,
            loader,
            device,
            len(artifacts.label_to_id),
            adapter=adapter,
            routing_mode="uniform",
        )
        routing_diagnostics = diagnose_relation_expert_routing(
            artifacts.model,
            loader,
            device,
            adapter=adapter,
        )
        routing_alignment = diagnose_relation_routing_task_alignment(
            artifacts.model,
            loader,
            device,
            adapter=adapter,
        )
    evidence_diagnostics = None
    evidence_alignment = None
    if kernel.evidence_selector is not None:
        evidence_diagnostics = diagnose_relation_counterfactual_evidence(
            artifacts.model,
            loader,
            device,
            adapter=adapter,
            random_repeats=args.random_repeats,
            random_seed=args.random_seed,
        )
        evidence_alignment = diagnose_relation_evidence_task_alignment(
            artifacts.model,
            loader,
            device,
            adapter=adapter,
        )
    payload = {
        "baseline": baseline,
        "steered": steered,
        "delta_vs_baseline": metric_delta(steered, baseline),
        "kernel_metadata": kernel.metadata(),
        "checkpoint_metadata": checkpoint_metadata,
        "counterfactual_evidence": evidence_diagnostics,
        "evidence_task_alignment": evidence_alignment,
        "uniform_routing": uniform_routing,
        "expert_routing": routing_diagnostics,
        "routing_task_alignment": routing_alignment,
        "data_path": args.data_path,
        "device": str(device),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
