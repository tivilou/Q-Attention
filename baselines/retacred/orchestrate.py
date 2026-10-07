"""One heavy independent task per GPU; topology-independent task identities."""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .common import ROOT, PACKAGE, atomic_json, cohort, digest, fingerprint, git, pin_source, select_models


def gpu_ids(raw: str) -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True)
    physical = [(a.strip(), b.strip()) for a,b in (line.split(",") for line in output.strip().splitlines())]
    if visible is not None:
        allowed = [part.strip() for part in visible.split(",") if part.strip()]
    else:
        allowed = [index for index, _ in physical]
    known = {value for pair in physical for value in pair}
    requested = allowed if raw == "auto" else [part.strip() for part in raw.split(",")]
    if not requested or len(requested) != len(set(requested)) or any(item not in known for item in requested):
        raise ValueError("GPU list empty, duplicate, malformed or unavailable")
    identities = {index: uuid for index, uuid in physical} | {uuid: uuid for _,uuid in physical}
    if len({identities[item] for item in requested}) != len(requested):
        raise ValueError("duplicate physical GPU aliases")
    if visible is not None and any(identities[item] not in {identities[x] for x in allowed if x in identities} for item in requested):
        raise ValueError("requested GPU outside CUDA_VISIBLE_DEVICES")
    return requested


def build_command(job, args, checkpoint, data):
    target = ROOT / "baselines/retacred/train.py"
    if not target.is_file():
        raise FileNotFoundError(target)
    command = [sys.executable, "-m", "baselines.retacred.train", "--model", job["model"], "--seed", str(job["seed"]),
               "--data-dir", str(data), "--checkpoint", str(checkpoint), "--run-dir", str(job["directory"]),
               "--micro-batch", str(args.micro_batch), "--logical-batch", str(args.logical_batch)]
    if args.checkpointing:
        command.append("--checkpointing")
    if (job["directory"] / "checkpoints/latest.pt").exists():
        command.append("--resume")
    if args.canary:
        command.append("--canary")
    return command


def validate_job(path: Path, *, canary: bool = False):
    if not (path / "RUN_COMPLETE").is_file() or not (path / "metrics.json").is_file():
        raise ValueError(f"missing child completion artifacts: {path}")
    if canary:
        return
    summary = json.loads((path / "run_summary.json").read_text())
    if summary["status"] != "complete" or summary["case_study"]["status"] != "complete_local":
        raise ValueError("child metrics or traces incomplete")
    for item in summary["case_study"]["traces"]:
        trace = json.loads((path / "case_study" / item["path"]).read_text())
        tensor = path / "case_study" / trace["tensor_artifact"]
        if not tensor.is_file() or digest(tensor) != trace["sha256"]:
            raise ValueError("trace tensor missing or wrong SHA-256")
    if {item["phase"] for item in summary["case_study"]["traces"]} != {"train", "test"}:
        raise ValueError("actual train or test trace absent")


def run_queue(jobs: list[dict], devices: list[str], args, checkpoint: Path, data: Path):
    pending, active, completed = deque(jobs), {}, []
    assignments = []
    pause = [False]
    saved_handlers = {sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: pause.__setitem__(0, True))
    failure = None
    last_print = 0.
    try:
        while pending or active:
            if pause[0] and not failure:
                failure = "pause_requested"
            for device in devices:
                if device in active or not pending or failure:
                    continue
                job = pending.popleft()
                job["directory"].parent.mkdir(parents=True, exist_ok=True)
                command = build_command(job, args, checkpoint, data)
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = device
                environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                log_path = args.run_dir / "logs" / (job["id"] + ".log")
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log = log_path.open("a", encoding="utf-8")
                spec = {"argv":command, "cwd":str(ROOT), "physical_gpu":device, "target":"baselines.retacred.train"}
                atomic_json(log_path.with_suffix(".command.json"), spec)
                try:
                    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                except Exception:
                    log.close()
                    raise
                assignment = {"task":job["id"], "gpu":device, "pid":process.pid, "status":"running"}
                assignments.append(assignment)
                active[device] = (process, job, log, assignment)
            for device, (process, job, log, assignment) in list(active.items()):
                rc = process.poll()
                if rc is None:
                    continue
                log.close()
                del active[device]
                assignment["exit_code"] = rc
                if rc == 0:
                    try:
                        validate_job(job["directory"], canary=args.canary)
                        completed.append(job["id"])
                        assignment["status"] = "complete"
                    except Exception as exc:
                        assignment["status"] = "failed"
                        failure = str(exc)
                else:
                    assignment["status"] = "paused" if rc == 75 else "failed"
                    failure = f"{job['id']} exited {rc}; inspect {args.run_dir / 'logs' / (job['id']+'.log')}"
            atomic_json(args.run_dir / "gpu_assignments.json", {"scheduler":"dynamic_one_heavy_task_per_gpu", "devices":devices,
                       "tasks":assignments, "pending":[job["id"] for job in pending]})
            if time.monotonic()-last_print >= 10 or not active:
                print(f"[references] complete {len(completed)}/{len(jobs)} | running {len(active)} | pending {len(pending)}", flush=True)
                for device, (_, job, _, _) in active.items():
                    heartbeat = job["directory"] / "heartbeat.json"
                    value = json.loads(heartbeat.read_text()) if heartbeat.exists() else {"phase":"initializing"}
                    print(f"  GPU {device} | {job['id']} | {value.get('phase')} | {value.get('batch','-')}/{value.get('batches','-')} | {value.get('batches_per_second',0):.2f} batch/s", flush=True)
                last_print = time.monotonic()
            if failure:
                break
            time.sleep(.2)
    finally:
        # Pause only processes launched by this parent; bounded grace then reap.
        for process, *_ in active.values():
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic()+45
        for process, job, log, assignment in active.values():
            try:
                process.wait(timeout=max(.01, deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            assignment["exit_code"] = process.returncode
            assignment["status"] = "paused" if process.returncode == 75 else "stopped"
            log.close()
        atomic_json(args.run_dir / "gpu_assignments.json", {"devices":devices,"tasks":assignments,"pending":[j["id"] for j in pending]})
        for sig, handler in saved_handlers.items():
            signal.signal(sig,handler)
    if failure:
        atomic_json(args.run_dir / "RUN_PAUSED.json" if pause[0] else args.run_dir / "RUN_FAILED.json", {"reason":failure})
        raise RuntimeError(failure)
    return completed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="ready", help="ready (four marker variants + normal PTR), explicit IDs or all (rejects unavailable entries)")
    parser.add_argument("--seeds", default="13")
    parser.add_argument("--gpu", default="auto")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/relation/retacred")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--logical-batch", type=int, default=64)
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = select_models(args.models)  # complete admission check before launching anything
    seeds = [int(item) for item in args.seeds.split(",")]
    if not seeds or len(seeds) != len(set(seeds)) or min(seeds) < 0:
        raise ValueError("invalid seeds")
    if min(args.micro_batch,args.logical_batch) <= 0 or args.micro_batch > args.logical_batch:
        raise ValueError("require 0 < micro-batch <= logical-batch")
    if args.export_only and (not args.resume or args.canary):
        raise ValueError("--export-only requires an actual completed --resume run")
    if args.resume and args.run_dir:
        raise ValueError("choose --run-dir or --resume")
    default_dir = ROOT / "runs/retacred_reference_suite" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.run_dir = (args.resume or args.run_dir or default_dir).resolve()
    if not args.run_dir.is_relative_to(ROOT / "runs"):
        raise ValueError("run directory must be inside project runs/")
    jobs = [{"id":f"{row['id']}_seed{seed}", "model":row["id"], "seed":seed,
             "directory":args.run_dir / "tasks" / f"{row['id']}_seed{seed}"} for row in rows for seed in seeds]
    contract = {"schema":"retacred-reference-suite.v1", "jobs":[job["id"] for job in jobs],
                "cohort_sha256":digest(PACKAGE/"cohort.json"), "micro_batch":args.micro_batch,
                "logical_batch":args.logical_batch,"checkpointing":args.checkpointing,
                "canary":args.canary,
                "data_identity": {split:digest(args.data_dir/(split+".jsonl")) for split in ("train","valid","test")} if not args.canary and not args.dry_run else {}}
    if args.dry_run:
        print(json.dumps({"contract":contract, "jobs":[job["id"] for job in jobs], "requested_gpu":args.gpu,
                          "not_the_full_top15":True}, indent=2))
        return
    if not args.canary:
        from .report import publication_preflight
        expected_report = ROOT / "reports/retacred_reference_suite" / args.run_dir.name
        publication_preflight(no_push=args.no_push, allowed_report=expected_report if args.export_only else None)
        if not args.export_only:
            from .preflight import check
            print(json.dumps(check(args.data_dir,args.gpu)),flush=True)
    if args.resume:
        if json.loads((args.run_dir / "run_manifest.json").read_text()) != contract:
            raise ValueError("suite manifest mismatch; only GPU topology may change")
    elif args.run_dir.exists():
        raise FileExistsError(args.run_dir)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.run_dir / "run_manifest.json", contract)
    if args.export_only:
        from .report import export_and_publish
        for job in jobs:
            validate_job(job["directory"], canary=args.canary)
        export_and_publish(args.run_dir, jobs, no_push=args.no_push)
        return
    devices = ["canary-0", "canary-1"] if args.canary else gpu_ids(args.gpu)
    checkpoint, prepared = Path("canary"), Path("canary")
    if not args.canary:
        from .assets import fetch
        for source_key in sorted({row["source"] for row in rows}):
            pin_source(source_key)
        if any(row["id"] == "ptr" for row in rows):
            from .ptr_resources import fetch_resources
            fetch_resources()
        checkpoint = fetch()
        prepared = args.run_dir / "prepared_data"
        if not args.resume:
            command = [sys.executable, str(ROOT/"experiments/prepare_retacred_reference_improved_baseline.py"),
                       "--train-path", str(args.data_dir/"train.jsonl"), "--valid-path",str(args.data_dir/"valid.jsonl"),
                       "--test-path",str(args.data_dir/"test.jsonl"), "--output-dir",str(prepared)]
            subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
            (prepared/"valid.json").rename(prepared/"dev.json")
            manifest = json.loads((prepared/"data_manifest.json").read_text())
            manifest["splits"]["valid"]["output_path"] = str(prepared/"dev.json")
            atomic_json(prepared/"data_manifest.json",manifest)
            if [manifest["splits"][split]["records"] for split in ("train","valid","test")] != [58465,19584,13418]:
                raise ValueError("formal split counts do not match Re-TACRED")
            atomic_json(args.run_dir/"dataset_qualification.json", {"status":"qualified", "task":"supervised_sentence_entity_pair_relation_classification",
                        "labels":40,"null_relation":"no_relation","splits":manifest["splits"],
                        "verification":"converter_validates_labels_entity_spans_types_and_full_split_counts"})
        for row in rows:
            certificate = args.run_dir / "fidelity" / (row["id"] + ".json")
            if not certificate.exists():
                subprocess.run([sys.executable,"-m","baselines.retacred.fidelity","--model",row["id"],
                                "--checkpoint",str(checkpoint),"--output",str(certificate)],cwd=ROOT,check=True)
    remaining = []
    for job in jobs:
        if (job["directory"] / "RUN_COMPLETE").exists():
            validate_job(job["directory"], canary=args.canary)
        else:
            remaining.append(job)
    run_queue(remaining, devices, args, checkpoint, prepared)
    for job in jobs:
        validate_job(job["directory"], canary=args.canary)
    atomic_json(args.run_dir / "run_summary.json", {"status":"complete", "requested_jobs":[job["id"] for job in jobs],
               "scope":"selected_configurations_only", "full_top15_completed":False, "synthetic":args.canary})
    (args.run_dir / "RUN_COMPLETE").touch()
    if not args.canary:
        from .report import export_and_publish
        export_and_publish(args.run_dir, jobs, no_push=args.no_push)


if __name__ == "__main__":
    main()
