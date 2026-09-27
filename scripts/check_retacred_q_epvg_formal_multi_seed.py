from __future__ import annotations

"""Preflight and parent-to-child canary for the Q-EPVG multi-seed runner."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_retacred_q_epvg_formal_multi_seed as scheduler


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        python_bin=sys.executable,
        hardware_profile="adaptive",
        log_every_batches=50,
        checkpoint_every_batches=50,
        allow_gpu_topology_change=False,
    )


def _run_canary(command: list[str], label: str) -> None:
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    output, _ = process.communicate(timeout=60)
    if process.returncode != 0:
        raise RuntimeError(f"{label} canary failed with exit {process.returncode}: {output[-2000:]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "retacred_q_epvg_formal_single_seed.json",
    )
    parser.add_argument("--canary", action="store_true", help="launch exact child --help commands")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise SystemExit(f"missing config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    scheduler.configure_protocol(config_path, config)
    if scheduler.SELECTORS[0] != "disabled" or len(scheduler.SELECTORS) < 2:
        raise SystemExit("config must declare disabled plus at least one selector")
    for target in (scheduler.SINGLE_RUNNER, scheduler.SELECTOR_WORKER):
        if not target.is_file():
            raise SystemExit(f"missing launch target: {target}")
    launch_args = _args()
    fake_config = ROOT / ".q_epvg_canary_seed_13.json"
    fake_run = ROOT / ".q_epvg_canary_run"
    baseline = scheduler.build_child_command(
        seed=13,
        gpu_id=0,
        config_path=fake_config,
        run_dir=fake_run,
        args=launch_args,
        baseline_only=True,
    )
    selector = scheduler.build_selector_command(
        seed=13,
        selector=scheduler.SELECTOR_TASKS[0],
        gpu_id=0,
        config_path=fake_config,
        seed_dir=fake_run,
        selector_dir=fake_run / "selectors" / scheduler.SELECTOR_TASKS[0],
        args=launch_args,
        profile={
            "pair_chunk_size": None,
            "pair_chunk_divisor": 1,
            "micro_batch_size": 256,
            "gradient_accumulation_steps": 1,
            "activation_checkpointing": False,
        },
        adaptive=True,
        resume=False,
    )
    for label, command, target in (
        ("baseline", baseline, scheduler.SINGLE_RUNNER),
        ("selector", selector, scheduler.SELECTOR_WORKER),
    ):
        if Path(command[1]).resolve() != target.resolve():
            raise SystemExit(f"{label} command target mismatch: {command[1]}")
        if command[0] != sys.executable:
            raise SystemExit(f"{label} interpreter mismatch: {command[0]}")
        if args.canary:
            _run_canary(command + ["--help"], label)
    print("Q-EPVG multi-seed preflight passed")
    if args.canary:
        print("Q-EPVG parent-to-child canary passed for baseline and selector")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
