from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q_attention.adapters import AttentionScoreKernelAdapter  # noqa: E402
from q_attention.experiments import (  # noqa: E402
    choose_device,
    diagnose_relation_counterfactual_evidence,
    diagnose_relation_evidence_measurement_frames,
    diagnose_relation_evidence_task_alignment,
    evaluate_relation_attention_score_kernel,
    load_relation_run,
    make_relation_loader,
)
from q_attention.plugins import (  # noqa: E402
    load_relation_attention_score_kernel_checkpoint,
)
from q_attention.tasks.relation import load_relation_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose Z/X measurement frames in a trained Q-RES bank."
    )
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--valid_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--random_repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def _mean(summary: dict[str, Any]) -> float | None:
    value = summary["mean"]
    return None if value is None else float(value)


def _layer_alignment(payload: dict[str, Any]) -> float | None:
    values = [
        _mean(layer["evidence_descent_cosine"])
        for layer in payload["layers"]
    ]
    active = [value for value in values if value is not None]
    return sum(active) / len(active) if active else None


def _format(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6g}"


def _render_report(payload: dict[str, Any]) -> str:
    rows = [
        "# Relation Measurement Frame Diagnostics",
        "",
        f"- Selector: `{payload['selector_type']}`",
        "- Correlation mode: "
        f"`{payload['frame_statistics']['correlation_mode']}`",
        f"- Checkpoint: `{payload['checkpoint']}`",
        "- All views use the same frozen checkpoint; no parameters are retrained.",
        "",
        "## Frozen Frame Views",
        "",
        "| View | Loss | Macro-F1 | Keep advantage | Drop advantage | Keep/drop win | Alignment | Pass |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for view, result in payload["views"].items():
        metrics = result["metrics"]
        selectivity = result["selectivity"]
        keep = _mean(selectivity["metrics"]["keep_advantage"])
        drop = _mean(selectivity["metrics"]["drop_advantage"])
        keep_win = _mean(selectivity["metrics"]["keep_win"])
        drop_win = _mean(selectivity["metrics"]["drop_win"])
        alignment = _layer_alignment(result["alignment"])
        rows.append(
            f"| {view} | {metrics['loss']:.6f} | {metrics['macro_f1']:.6f} | "
            f"{_format(keep)} | {_format(drop)} | "
            f"{_format(keep_win)}/{_format(drop_win)} | {_format(alignment)} | "
            f"{selectivity['selectivity_pass']} |"
        )

    rows.extend(
        [
            "",
            "## Frame Contributions",
            "",
            "| Layer | Head | Frame | Abs contribution | Contribution L2 | Abs task gradient | Gradient L2 | Descent effect |",
            "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for layer in payload["frame_statistics"]["layers"]:
        for head in layer["heads"]:
            for frame, statistics in head["frames"].items():
                rows.append(
                    f"| {layer['layer_index']} | {head['head_index']} | {frame} | "
                    f"{_format(_mean(statistics['absolute_contribution']))} | "
                    f"{_format(_mean(statistics['contribution_l2']))} | "
                    f"{_format(_mean(statistics['absolute_task_gradient']))} | "
                    f"{_format(_mean(statistics['task_gradient_l2']))} | "
                    f"{_format(_mean(statistics['task_descent_effect']))} |"
                )
    if payload["frame_statistics"]["correlation_mode"] == "born_reliability":
        rows.extend(
            [
                "",
                "## Born Reliability",
                "",
                "| Layer | Head | Frame | Quality | Effective gate |",
                "| ---: | ---: | --- | ---: | ---: |",
            ]
        )
        for layer in payload["frame_statistics"]["layers"]:
            for head in layer["heads"]:
                for frame, statistics in head["born_reliability"].items():
                    rows.append(
                        f"| {layer['layer_index']} | {head['head_index']} | "
                        f"{frame} | {_format(_mean(statistics['quality']))} | "
                        f"{_format(_mean(statistics['effective_gate']))} |"
                    )
    rows.extend(
        [
            "",
            "## Correlation Channel Decomposition",
            "",
            "| Layer | Head | Frame | Channel | Abs contribution | Contribution L2 | Descent effect |",
            "| ---: | ---: | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for layer in payload["frame_statistics"]["layers"]:
        for head in layer["heads"]:
            for channel, frame_statistics in head[
                "correlation_channels"
            ].items():
                for frame, statistics in frame_statistics.items():
                    rows.append(
                        f"| {layer['layer_index']} | {head['head_index']} | "
                        f"{frame} | {channel} | "
                        f"{_format(_mean(statistics['absolute_contribution']))} | "
                        f"{_format(_mean(statistics['contribution_l2']))} | "
                        f"{_format(_mean(statistics['task_descent_effect']))} |"
                    )
    rows.append("")
    return "\n".join(rows)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.random_repeats <= 0:
        raise ValueError("batch_size and random_repeats must be positive")
    device = choose_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = load_relation_run(Path(args.model_dir), device)
    kernel, checkpoint_metadata = load_relation_attention_score_kernel_checkpoint(
        args.checkpoint,
        map_location=device,
    )
    selector = kernel.evidence_selector
    if selector is None:
        raise ValueError("checkpoint must contain an evidence selector")
    if selector.config.evidence_measurement_mode not in {
        "relation_frame_bank",
        "relation_frame_coherent",
    }:
        raise ValueError("checkpoint must contain a banked relation-frame selector")
    expected_paths = checkpoint_metadata.get("score_module_paths")
    if expected_paths is not None and tuple(expected_paths) != tuple(
        artifacts.model.score_module_paths
    ):
        raise ValueError("checkpoint module paths do not match the base model")
    kernel.to(device).eval()
    model = artifacts.model.eval()
    adapter = AttentionScoreKernelAdapter(model, model.score_module_paths, kernel)
    loader = make_relation_loader(
        load_relation_jsonl(Path(args.valid_path)),
        artifacts.vocab,
        artifacts.label_to_id,
        batch_size=args.batch_size,
        shuffle=False,
    )

    views: dict[str, Any] = {}
    for view in ("full", "z", "x"):
        with selector.use_measurement_frame_view(view):
            views[view] = {
                "metrics": evaluate_relation_attention_score_kernel(
                    model,
                    loader,
                    device,
                    len(artifacts.label_to_id),
                    adapter=adapter,
                ),
                "selectivity": diagnose_relation_counterfactual_evidence(
                    model,
                    loader,
                    device,
                    adapter=adapter,
                    random_repeats=args.random_repeats,
                    random_seed=args.seed + 8009,
                ),
                "alignment": diagnose_relation_evidence_task_alignment(
                    model,
                    loader,
                    device,
                    adapter=adapter,
                ),
            }

    payload = {
        "args": vars(args),
        "checkpoint": str(Path(args.checkpoint)),
        "checkpoint_metadata": checkpoint_metadata,
        "selector_type": selector.selector_type,
        "kernel_metadata": kernel.metadata(),
        "views": views,
        "frame_statistics": diagnose_relation_evidence_measurement_frames(
            model,
            loader,
            device,
            adapter=adapter,
        ),
        "test_evaluated": False,
    }
    (output_dir / "measurement_frame_diagnostics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "measurement_frame_diagnostics.md").write_text(
        _render_report(payload),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "selector_type": selector.selector_type,
                "views": list(views),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
