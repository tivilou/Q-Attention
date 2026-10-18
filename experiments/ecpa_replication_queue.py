"""One dependency-aware queue for every new (seed, stage); one worker/GPU."""
from __future__ import annotations

from datetime import datetime, timezone
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import textwrap
import time

from q_attention.experiments import ecpa_protocol as p
from q_attention.experiments.ecpa_runner import completed, duration, gpu_ids, stop
import ecpa_replication_contract as c


def utc():
    return datetime.now(timezone.utc).isoformat()


def ready(jobs):
    return [key for key, job in jobs.items() if job["status"] == "pending"
        and (job["mode"] == "baseline" or jobs[f"{job['seed']}:baseline"]["status"] == "complete")]


def render(root, slots, jobs, active, started, tty):
    complete = [k for k, j in jobs.items() if j["status"] == "complete"]
    pending = [k for k, j in jobs.items() if j["status"] == "pending"]
    failed = [k for k, j in jobs.items() if j["status"] in ("failed", "blocked")]
    lines = [f"Q-ECPA replication | seed13 imported | {len(complete)}/{len(jobs)} new jobs complete | {len(active)} running | {duration(time.monotonic()-started)}"]
    for gpu in slots:
        key = next((k for k, v in active.items() if v["gpu"] == gpu), None)
        if key is None:
            lines.append(f"GPU {gpu} | waiting")
            continue
        job = jobs[key]
        heartbeat = Path(job["run"]) / "jobs" / job["mode"] / "heartbeat.json"
        event = p.read(heartbeat) if heartbeat.exists() else {}
        done, total, rate = event.get("done", 0), event.get("total", 0), event.get("rate", 0)
        bar = min(16, max(0, int(16*done/max(1, total))))
        lines.append(f"GPU {gpu} | s{job['seed']} {job['mode']} | {event.get('phase', 'initializing')} | [{'#'*bar}{'-'*(16-bar)}] {done}/{total} | {rate:.3f}/s | ETA {duration((total-done)/rate) if rate else '--:--:--'} | VRAM {event.get('allocated_gib', 0):.2f}G | T{job['tier']} m{event.get('micro', '-')} save@{event.get('checkpoint_update', '-')}")
    width, height = shutil.get_terminal_size((140, 40))
    for title, values in (("PENDING", pending), ("COMPLETED", list(reversed(complete))), ("FAILED", failed)):
        lines.extend(textwrap.wrap(f"{title} ({len(values)}) | " + (", ".join(values) or "empty"),
            width=max(20, width-1), subsequent_indent="  ", break_long_words=False, break_on_hyphens=False))
    if tty:
        print("\x1b[H\x1b[2J" + "\n".join(line[:max(20, width-1)] for line in lines[:height-1]), end="", flush=True)
    else:
        print("\n".join(lines), flush=True)
    p.atomic_json(root / "scheduler_heartbeat.json", dict(updated_at=utc(), complete=complete, pending=pending, failed=failed))


def build_jobs(seed_runs):
    jobs = {}
    # Both upstream jobs enter the queue before any downstream selectors.
    for mode in ("baseline", *p.MODES):
        for seed, run in seed_runs.items():
            manifest = p.read(run / "run_manifest.json")
            path = run / "jobs" / mode / "memory_state.json"
            if not path.exists():
                p.atomic_json(path, dict(tier=0, retries=0, events=[]))
            jobs[f"{seed}:{mode}"] = dict(seed=seed, mode=mode, run=str(run),
                status="complete" if completed(run, mode, manifest) else "pending", tier=p.read(path)["tier"])
    return jobs


def execute(root, seed_runs, args, slots):
    jobs = build_jobs(seed_runs)
    active, paused, pause_started = {}, False, None
    started, last_frame = time.monotonic(), 0
    def pause(signum, frame):
        nonlocal paused
        paused = True
    signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    old = {sig: signal.signal(sig, pause) for sig in signals}
    tty = sys.stdout.isatty() and os.environ.get("TERM") != "dumb"
    if tty:
        print("\x1b[?1049h\x1b[?25l", end="", flush=True)
    try:
        while True:
            if paused and pause_started is None:
                pause_started = time.monotonic()
                for item in active.values():
                    try:
                        if item["process"].poll() is None:
                            os.killpg(item["process"].pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            if paused and time.monotonic()-pause_started > 600:
                for item in active.values():
                    stop(item["process"], timeout=1)
            for key, job in jobs.items():
                if job["status"] == "pending" and jobs[f"{job['seed']}:baseline"]["status"] == "failed":
                    job.update(status="blocked", error="this seed baseline failed")
            occupied = {item["gpu"] for item in active.values()}
            for gpu in slots:
                if gpu in occupied or paused:
                    continue
                candidates = ready(jobs)
                if not candidates:
                    break
                key, device = candidates[0], "cpu" if gpu.startswith("cpu") else "cuda"
                job = jobs[key]
                run = Path(job["run"])
                spec = c.worker_spec(run, job["mode"], device, inject_oom=args.inject_oom_once,
                    stop_after=args.stop_after if job["mode"] == "quantum" and job["seed"] == 29 else None)
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES="" if device == "cpu" else gpu, CUDA_DEVICE_ORDER="PCI_BUS_ID")
                logs = run / "logs"
                logs.mkdir(exist_ok=True)
                p.atomic_json(logs / f"{job['mode']}.command.json", dict(**spec, gpu=gpu, seed=job["seed"]))
                stream = (logs / f"{job['mode']}.log").open("a")
                try:
                    process = subprocess.Popen(spec["argv"], cwd=spec["cwd"], env=env,
                        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                except BaseException:
                    stream.close()
                    raise
                active[key] = dict(process=process, stream=stream, gpu=gpu)
                job.update(status="running", gpu=gpu, pid=process.pid, started_at=utc())
            for key, item in list(active.items()):
                code = item["process"].poll()
                if code is None:
                    continue
                item["stream"].close()
                del active[key]
                job = jobs[key]
                job.update(exit_code=code, finished_at=utc())
                directory = Path(job["run"]) / "jobs" / job["mode"]
                if code == p.OOM_EXIT:
                    state = p.read(directory / "memory_state.json")
                    config = p.read(Path(job["run"]) / "run_config.json")
                    tiers = p.memory_tiers(config["baseline" if job["mode"] == "baseline" else "kernel"]["batch_size"])
                    if state["tier"]+1 < len(tiers):
                        event = p.read(directory / "OOM.json")
                        state["events"].append(dict(from_tier=state["tier"], to_tier=state["tier"]+1, **event))
                        state.update(tier=state["tier"]+1, retries=state["retries"]+1)
                        p.atomic_json(directory / "memory_state.json", state)
                        job.update(status="pending", tier=state["tier"])
                    else:
                        job.update(status="failed", error="all finite memory tiers exhausted")
                elif code == p.PAUSED_EXIT:
                    job["status"], paused = "paused", True
                elif code == 0 and completed(Path(job["run"]), job["mode"], p.read(Path(job["run"]) / "run_manifest.json")):
                    job["status"] = "complete"
                else:
                    job.update(status="failed", error=f"worker exit {code}; see this seed log")
            p.atomic_json(root / "gpu_assignments.json", dict(requested=args.gpu, resolved=slots, jobs=jobs,
                scheduler="global first-free seed/stage DAG, one worker per GPU"))
            if time.monotonic()-last_frame > (2 if tty else 30):
                render(root, slots, jobs, active, started, tty)
                last_frame = time.monotonic()
            if not active and (paused or all(j["status"] in ("complete", "failed", "blocked") for j in jobs.values())):
                break
            time.sleep(.2)
    finally:
        for item in active.values():
            stop(item["process"])
            item["stream"].close()
        for sig, handler in old.items():
            signal.signal(sig, handler)
        if tty:
            print("\x1b[?25h\x1b[?1049l", end="", flush=True)
    if paused:
        p.atomic_json(root / "RUN_PAUSED.json", dict(jobs=jobs, at=utc()))
        return p.PAUSED_EXIT
    if not all(job["status"] == "complete" for job in jobs.values()):
        p.atomic_json(root / "RUN_FAILED.json", dict(jobs=jobs, at=utc()))
        raise RuntimeError(f"replication has failed/blocked tasks; inspect {root}/gpu_assignments.json")
    return 0
