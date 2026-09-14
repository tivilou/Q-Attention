from __future__ import annotations

"""Run the predeclared Q-CEASC Re-TACRED replication seeds.

The runner owns scheduling and lifecycle only.  Each seed is executed by the
same single-seed runner with a seed-specific copy of the frozen config; no
scientific parameter is changed.  It deliberately keeps all raw artifacts
under ``runs/`` and leaves report publication to the audited exporter.
"""

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (13, 29, 53)
MIN_FREE_MIB = 8 * 1024
SINGLE_RUNNER = ROOT / "experiments" / "run_retacred_qceasc_formal_single_seed.py"
CONFIG_PATH = ROOT / "configs" / "retacred_qceasc_formal_single_seed.json"
GROUP_ROOT = ROOT / "runs" / "retacred_qceasc_formal_multi_seed"
SELECTORS = ("disabled", "q_ceasc", "classical_ceasc")


def parse_int_list(value: str, *, label: str) -> list[int]:
    items = [part.strip() for part in value.split(",")]
    if not items or any(not item.isdigit() for item in items):
        raise ValueError(f"{label} must be a comma-separated list of non-negative integers")
    values = [int(item) for item in items]
    if not values or len(set(values)) != len(values):
        raise ValueError(f"{label} must contain at least one unique integer")
    return values


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def query_gpu_inventory() -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free,memory.used",
            "--format=csv,noheader,nounits",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
    inventory: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        fields = [part.strip() for part in line.split(",", 4)]
        if len(fields) != 5:
            raise RuntimeError(f"unexpected nvidia-smi row: {line!r}")
        index, name, total, free, used = fields
        inventory.append(
            {
                "index": int(index),
                "name": name,
                "memory_total_mib": int(total),
                "memory_free_mib": int(free),
                "memory_used_mib": int(used),
            }
        )
    if not inventory:
        raise RuntimeError("nvidia-smi reported no GPUs")
    return inventory


def resolve_gpu_ids(spec: str, inventory: list[dict[str, Any]]) -> list[int]:
    if spec.strip().lower() == "auto":
        ids = [
            int(item["index"])
            for item in inventory
            if int(item["memory_free_mib"]) >= MIN_FREE_MIB
        ]
    else:
        ids = parse_int_list(spec, label="gpus")
    known = {int(item["index"]): item for item in inventory}
    missing = [gpu_id for gpu_id in ids if gpu_id not in known]
    if missing:
        raise ValueError(f"requested GPU IDs are unavailable: {missing}")
    insufficient = [
        gpu_id for gpu_id in ids if int(known[gpu_id]["memory_free_mib"]) < MIN_FREE_MIB
    ]
    if insufficient:
        detail = ", ".join(
            f"GPU {gpu_id} free={known[gpu_id]['memory_free_mib']} MiB"
            for gpu_id in insufficient
        )
        raise RuntimeError(f"selected GPU capacity is below 8 GiB: {detail}")
    if not ids:
        raise RuntimeError("no GPU with at least 8 GiB free was selected")
    return ids


def canonical_config(config: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(json.dumps(config))
    value["seed"] = 0
    value.pop("replication", None)
    return value


def protocol_fingerprint(config: dict[str, Any]) -> str:
    encoded = json.dumps(canonical_config(config), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_seed_config(base_config: dict[str, Any], seed: int, destination: Path) -> dict[str, Any]:
    config = json.loads(json.dumps(base_config))
    config["seed"] = seed
    config["replication"] = {
        "schema_version": "q-attention.q-ceasc-replication-child.v1",
        "seed_set": list(DEFAULT_SEEDS),
        "child_seed": seed,
        "parent_contract": protocol_fingerprint(base_config),
    }
    write_json(destination, config)
    return config


def build_child_command(
    *,
    seed: int,
    gpu_id: int,
    config_path: Path,
    run_dir: Path,
    args: argparse.Namespace,
    resume: bool = False,
) -> list[str]:
    command = [
        args.python_bin,
        str(SINGLE_RUNNER),
        "--config",
        str(config_path),
        "--resume" if resume else "--output-dir",
        str(run_dir),
        "--device",
        "cuda",
        "--gpus",
        str(gpu_id),
        "--seed",
        str(seed),
        "--hardware-profile",
        args.hardware_profile,
        "--log-every-batches",
        str(args.log_every_batches),
        "--checkpoint-every-batches",
        str(args.checkpoint_every_batches),
        "--replication-child",
    ]
    return command


def latest_heartbeat(run_dir: Path) -> dict[str, Any] | None:
    candidates = [run_dir / "baseline" / "heartbeat.json"]
    candidates.extend(run_dir.glob("selectors/*/heartbeat.json"))
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        return None
    path = max(existing, key=lambda item: item.stat().st_mtime)
    try:
        value = load_json(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    value["heartbeat_file"] = str(path)
    return value


def _format_duration(seconds: Any) -> str:
    try:
        value = max(0, int(float(seconds)))
    except (TypeError, ValueError):
        return "--"
    hours, value = divmod(value, 3600)
    minutes, seconds = divmod(value, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def render_dashboard(statuses: dict[int, dict[str, Any]], group_dir: Path) -> str:
    counts = {name: 0 for name in ("queued", "running", "complete", "failed", "not_started")}
    lines = []
    for seed in sorted(statuses):
        item = statuses[seed]
        counts[item["status"]] = counts.get(item["status"], 0) + 1
        progress = item.get("progress") or {}
        if item["status"] == "running" and progress:
            batch = progress.get("batch", progress.get("completed_batches", "?"))
            total = progress.get("batches", progress.get("total_batches", "?"))
            rate = progress.get("batches_per_second")
            rate_text = f" {float(rate):.2f} batch/s" if isinstance(rate, (int, float)) else ""
            lines.append(
                f"  seed {seed:<3} GPU {item.get('gpu_id', '?')} RUNNING "
                f"batch {batch}/{total}{rate_text} ETA {_format_duration(progress.get('eta_seconds'))}"
            )
        else:
            suffix = f" GPU {item['gpu_id']}" if item.get("gpu_id") is not None else ""
            lines.append(f"  seed {seed:<3}{suffix} {item['status'].upper()}")
    header = (
        f"[Q-CEASC multi-seed] {group_dir.name} | "
        f"complete {counts.get('complete', 0)}/{len(statuses)} | "
        f"running {counts.get('running', 0)} | queued {counts.get('queued', 0)} | "
        f"failed {counts.get('failed', 0)}"
    )
    return "\n".join([header, *lines])


def write_command_artifact(
    run_dir: Path, *, seed: int, gpu_id: int, command: list[str], commit: str
) -> None:
    write_json(
        run_dir / "command.json",
        {
            "schema_version": "q-attention.child-command.v1",
            "seed": seed,
            "physical_gpu_id": gpu_id,
            "cwd": str(ROOT),
            "target": str(SINGLE_RUNNER.relative_to(ROOT)),
            "argv": command,
            "git_revision": commit,
        },
    )


def _terminate(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
            process.wait(timeout=30)


def run_group(
    args: argparse.Namespace,
    *,
    group_dir: Path,
    seeds: list[int],
    gpu_ids: list[int],
    commit: str,
    base_config: dict[str, Any],
    resume_group: bool = False,
) -> int:
    configs_dir = group_dir / "configs"
    if resume_group:
        if not group_dir.is_dir() or not (group_dir / "multi_seed_manifest.json").is_file():
            raise ValueError("--resume-group requires an existing multi-seed manifest")
        previous = load_json(group_dir / "multi_seed_manifest.json")
        if [int(seed) for seed in previous.get("seeds", [])] != seeds:
            raise ValueError("resume group seed set differs from the frozen 13,29,53 contract")
        if previous.get("git_commit") != commit:
            raise ValueError("resume group was created by a different code revision")
        if previous.get("base_config_sha256") != sha256(CONFIG_PATH):
            raise ValueError("resume group was created from a different formal config")
        if (group_dir / "MULTI_SEED_COMPLETE").is_file():
            raise ValueError("multi-seed group is already complete")
        for seed in seeds:
            if not (configs_dir / f"seed_{seed}.json").is_file():
                raise ValueError(f"resume group is missing configs/seed_{seed}.json")
        (group_dir / "MULTI_SEED_FAILED").unlink(missing_ok=True)
    else:
        group_dir.mkdir(parents=True, exist_ok=False)
    if not resume_group:
        for seed in seeds:
            build_seed_config(base_config, seed, configs_dir / f"seed_{seed}.json")
        manifest = {
            "schema_version": "q-attention.q-ceasc.formal-multiseed-manifest.v1",
            "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": commit,
            "base_config": str(CONFIG_PATH),
            "base_config_sha256": sha256(CONFIG_PATH),
            "protocol_fingerprint": protocol_fingerprint(base_config),
            "seeds": seeds,
            "gpus": gpu_ids,
            "gpu_inventory": args.gpu_inventory,
            "worker_count": min(len(seeds), len(gpu_ids)),
            "hardware_profile": args.hardware_profile,
            "checkpoint_every_batches": args.checkpoint_every_batches,
        }
        write_json(group_dir / "multi_seed_manifest.json", manifest)
    statuses: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        seed_dir = group_dir / f"seed_{seed}"
        if resume_group and (seed_dir / "RUN_COMPLETE").is_file() and (seed_dir / "run_summary.json").is_file():
            statuses[seed] = {"seed": seed, "status": "complete", "gpu_id": None, "resumed_skip": True}
        else:
            statuses[seed] = {"seed": seed, "status": "queued", "gpu_id": None}
    write_json(group_dir / "multi_seed_status.json", {"group_dir": str(group_dir), "assignments": list(statuses.values())})
    pending: deque[int] = deque(seed for seed in seeds if statuses[seed]["status"] != "complete")
    available = deque(gpu_ids)
    active: dict[int, dict[str, Any]] = {}
    failed = False
    stop_requested = False
    previous_handlers: dict[int, Any] = {}

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"[scheduler] received signal {signum}; requesting safe child stop", flush=True)

    for signal_name in (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGHUP", None)):
        if signal_name is not None:
            previous_handlers[signal_name] = signal.getsignal(signal_name)
            signal.signal(signal_name, request_stop)

    def save_status() -> None:
        payload = {
            "schema_version": "q-attention.q-ceasc.formal-multiseed-status.v1",
            "updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "group_dir": str(group_dir),
            "assignments": [statuses[seed] for seed in seeds],
        }
        write_json(group_dir / "multi_seed_status.json", payload)

    def stop_active() -> None:
        for entry in active.values():
            _terminate(entry["process"])

    previous_dashboard = 0.0
    try:
        while pending or active:
            if stop_requested:
                raise KeyboardInterrupt
            while pending and available and not failed:
                seed = pending.popleft()
                gpu_id = available.popleft()
                seed_dir = group_dir / f"seed_{seed}"
                if resume_group:
                    seed_dir.mkdir(parents=True, exist_ok=True)
                else:
                    seed_dir.mkdir(parents=True, exist_ok=False)
                command = build_child_command(
                    seed=seed,
                    gpu_id=gpu_id,
                    config_path=configs_dir / f"seed_{seed}.json",
                    run_dir=seed_dir,
                    args=args,
                    resume=resume_group and any(seed_dir.iterdir()),
                )
                write_command_artifact(seed_dir, seed=seed, gpu_id=gpu_id, command=command, commit=commit)
                log_handle = (seed_dir / "parent-child.log").open("w", encoding="utf-8")
                environment = os.environ.copy()
                environment.update(
                    {
                        "PYTHONUNBUFFERED": "1",
                        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                        "CUDA_VISIBLE_DEVICES": str(gpu_id),
                        "Q_ATTENTION_PROGRESS_FORMAT": "json",
                    }
                )
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    text=True,
                )
                statuses[seed].update(
                    {
                        "status": "running",
                        "gpu_id": gpu_id,
                        "pid": process.pid,
                        "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "log_file": str(seed_dir / "parent-child.log"),
                    }
                )
                active[seed] = {
                    "process": process,
                    "log_handle": log_handle,
                    "gpu_id": gpu_id,
                    "started_monotonic": time.monotonic(),
                    "run_dir": seed_dir,
                }
                save_status()
                print(f"[scheduler] started seed={seed} gpu={gpu_id}", flush=True)

            for seed, entry in list(active.items()):
                progress = latest_heartbeat(entry["run_dir"])
                if progress:
                    statuses[seed]["progress"] = progress
                process = entry["process"]
                return_code = process.poll()
                if return_code is None:
                    continue
                entry["log_handle"].close()
                elapsed = round(time.monotonic() - entry["started_monotonic"], 3)
                success = return_code == 0 and (entry["run_dir"] / "RUN_COMPLETE").is_file() and (entry["run_dir"] / "run_summary.json").is_file()
                statuses[seed].update(
                    {
                        "status": "complete" if success else "failed",
                        "return_code": return_code,
                        "finished_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "duration_seconds": elapsed,
                    }
                )
                if not success:
                    failed = True
                available.append(entry["gpu_id"])
                del active[seed]
                save_status()
                print(f"[scheduler] finished seed={seed} success={str(success).lower()} elapsed={_format_duration(elapsed)}", flush=True)

            if failed and pending:
                while pending:
                    seed = pending.popleft()
                    statuses[seed].update({"status": "not_started", "reason": "sibling seed failed"})
                save_status()
            now = time.monotonic()
            if now - previous_dashboard >= args.dashboard_interval:
                previous_dashboard = now
                print(render_dashboard(statuses, group_dir), flush=True)
            if active:
                time.sleep(0.2)
    except KeyboardInterrupt:
        stop_active()
        failed = True
        for seed, entry in active.items():
            return_code = entry["process"].poll()
            statuses[seed].update({"status": "paused" if return_code == 75 else "failed", "return_code": return_code, "reason": "parent interrupted"})
        for seed in list(pending):
            statuses[seed].update({"status": "not_started", "reason": "parent interrupted"})
        save_status()
    finally:
        for entry in active.values():
            entry["log_handle"].close()
        for signal_name, handler in previous_handlers.items():
            signal.signal(signal_name, handler)

    completed = all(statuses[seed]["status"] == "complete" for seed in seeds)
    summary = {
        "schema_version": "q-attention.q-ceasc.formal-multiseed-run.v1",
        "group_dir": str(group_dir),
        "seeds": seeds,
        "success": completed and not failed,
        "assignments": [statuses[seed] for seed in seeds],
    }
    write_json(group_dir / "multi_seed_run_summary.json", summary)
    if completed and not failed:
        summary_command = [
            args.python_bin,
            str(ROOT / "scripts" / "summarize_retacred_qceasc_formal_multi_seed.py"),
            "--group-dir",
            str(group_dir),
            "--output-json",
            str(group_dir / "multi_seed_summary.json"),
            "--output-md",
            str(group_dir / "multi_seed_summary.md"),
        ]
        subprocess.run(summary_command, cwd=ROOT, check=True)
        (group_dir / "MULTI_SEED_COMPLETE").write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
        print(f"[scheduler] complete group_dir={group_dir}", flush=True)
        if not args.no_export:
            export_command = ["bash", "scripts/export_retacred_qceasc_formal_multi_seed_report.sh", "--group-dir", str(group_dir)]
            if args.report_dir:
                export_command.extend(["--report-dir", args.report_dir])
            result = subprocess.run(export_command, cwd=ROOT, check=False)
            if result.returncode != 0:
                print("[scheduler] experiment complete but audited report export failed", file=sys.stderr, flush=True)
                return result.returncode
        return 0
    (group_dir / "MULTI_SEED_FAILED").write_text(datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8")
    print(f"[scheduler] failed group_dir={group_dir}", file=sys.stderr, flush=True)
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--gpus", "--gpu", dest="gpus", default="auto")
    parser.add_argument("--hardware-profile", choices=("adaptive", "auto", "low_memory", "balanced", "high_memory"), default="adaptive")
    parser.add_argument("--log-every-batches", type=int, default=50)
    parser.add_argument("--checkpoint-every-batches", type=int, default=50)
    parser.add_argument("--dashboard-interval", type=float, default=30.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume-group", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--no-export", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python-bin", default=sys.executable, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        seeds = parse_int_list(args.seeds, label="seeds")
        if seeds != sorted(seeds) or seeds != list(DEFAULT_SEEDS):
            raise ValueError("L2 replication seed set is frozen to 13,29,53")
        if args.log_every_batches <= 0 or args.checkpoint_every_batches <= 0:
            raise ValueError("batch intervals must be positive")
        if args.dashboard_interval <= 0:
            raise ValueError("--dashboard-interval must be positive")
        if not CONFIG_PATH.is_file() or not SINGLE_RUNNER.is_file():
            raise ValueError("Q-CEASC formal config or single-seed runner is missing")
        base_config = load_json(CONFIG_PATH)
        if base_config.get("schema_version") != "q-attention.q-ceasc-formal-single-seed.v1":
            raise ValueError("unsupported Q-CEASC formal config")
        inventory = query_gpu_inventory()
        gpu_ids = resolve_gpu_ids(args.gpus, inventory)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc
    args.gpu_inventory = inventory
    commit_result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
    commit = commit_result.stdout.strip() if commit_result.returncode == 0 else "unknown"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.output_dir and args.resume_group:
        raise SystemExit("--output-dir and --resume-group are mutually exclusive")
    group_dir = args.resume_group or args.output_dir or GROUP_ROOT / stamp
    if not group_dir.is_absolute():
        group_dir = ROOT / group_dir
    group_dir = group_dir.resolve()
    if not group_dir.is_relative_to(GROUP_ROOT.resolve()):
        raise SystemExit("output directory must be under runs/retacred_qceasc_formal_multi_seed/")
    if group_dir.exists() and not args.dry_run and not args.resume_group:
        raise SystemExit(f"refusing to reuse output directory: {group_dir}")
    commands = [
        build_child_command(
            seed=seed,
            gpu_id=gpu_ids[index % len(gpu_ids)],
            config_path=group_dir / "configs" / f"seed_{seed}.json",
            run_dir=group_dir / f"seed_{seed}",
            args=args,
        )
        for index, seed in enumerate(seeds)
    ]
    print(f"[scheduler] seeds={','.join(map(str, seeds))} gpus={','.join(map(str, gpu_ids))} workers={min(len(seeds), len(gpu_ids))}", flush=True)
    if args.dry_run:
        for seed, command in zip(seeds, commands, strict=True):
            print(f"[scheduler] seed={seed} command={' '.join(command)}", flush=True)
        return 0
    if not args.skip_preflight:
        preflight = [
            "bash",
            "scripts/check_retacred_qceasc_formal_single_seed.sh",
            "--fresh",
            "--gpus",
            ",".join(map(str, gpu_ids)),
            "--device",
            "cuda",
            "--hardware-profile",
            args.hardware_profile,
        ]
        subprocess.run(preflight, cwd=ROOT, check=True)
    return run_group(
        args,
        group_dir=group_dir,
        seeds=seeds,
        gpu_ids=gpu_ids,
        commit=commit,
        base_config=base_config,
        resume_group=bool(args.resume_group),
    )


if __name__ == "__main__":
    raise SystemExit(main())
