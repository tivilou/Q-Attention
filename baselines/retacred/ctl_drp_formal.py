"""Gated CTL-only production bridge. Never use the marker/PTR batch64 path."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from .common import PACKAGE, ROOT, atomic_json, cohort, digest, fingerprint, git, select_models
from .ctl_drp_runner import VARIANT_MODELS, safe
from .ctl_drp_task import COMPLETE_COUNTS, RECIPE, UpdatePlan, regular

REVIEW_FILES = (
    "cohort.json", "ctl_drp_formal.py", "ctl_drp_qualification.py", "ctl_drp_task.py",
    "ctl_drp_runner.py", "ctl_drp_trace.py", "ctl_drp_export.py", "ctl_drp_neural.py",
    "ctl_drp_pretrained.py", "ctl_drp_cuda_checkpoint.py", "ctl_drp_candidates.py",
    "ctl_drp_paper.lock.json", "ctl_drp_simcse.lock.json", "ctl_drp_runtime_choices.json",
    "ctl_drp_preflight.py", "ctl_drp_simcse.py", "ctl_drp_paper.py", "orchestrate.py",
    "improved.py", "common.py", "report.py", "assets.py",
)


def source_identity():
    return {**{name: digest(regular(PACKAGE / name)) for name in REVIEW_FILES},
            "scripts/collab/lib/exchange_upload.py": digest(regular(ROOT / "scripts/collab/lib/exchange_upload.py"))}


def require_review(path, model):
    value = json.loads(regular(path).read_text(encoding="utf-8"))
    if (value.get("schema") != "ctl-drp-independent-review.v1" or value.get("status") != "accepted"
            or value.get("reviewer_role") != "independent" or not value.get("reviewer")
            or model not in value.get("models", []) or value.get("fidelity") != "paper_interpretive"
            or value.get("source_sha256") != source_identity()):
        raise ValueError("accepted independent mechanism/release review must bind current source and model")
    return {"sha256": digest(path), "fidelity": "paper_interpretive", "strict_comparison_admitted": False}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--describe", action="store_true")
    p.add_argument("--variant", choices=tuple(VARIANT_MODELS), default="sup")
    p.add_argument("--seed", type=int, choices=tuple(RECIPE["seeds"]), default=13)
    p.add_argument("--micro-batch", type=int, default=4)
    for name in ("qualification", "prepared-data", "review-receipt", "cache", "paper-source", "weight", "run-dir"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--gpu", default="auto", help="one selected visible GPU, not topology migration")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--export-only", action="store_true", help="no model import/load/update/evaluation")
    p.add_argument("--pause-after", type=int)
    p.add_argument("--no-push", action="store_true", help="still requires delivery/export/report-only commit")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return p


def describe(args):
    model = VARIANT_MODELS[args.variant]
    row = next(r for r in cohort()["models"] if r["id"] == model)
    return {"schema": "ctl-drp-formal-bridge.v1", "model": model, "recipe": RECIPE,
            "complete_counts": COMPLETE_COUNTS, "budget_updates": UpdatePlan(COMPLETE_COUNTS["train"], args.seed).total,
            "registry_status": row["status"], "formal_execution_admitted": row["status"] == "implemented_runtime_gate_required",
            "independent_review_required": True, "fidelity": "paper_interpretive",
            "strict_comparison_admitted": False, "default_cohort_changed": False,
            "execution": "one_variant_one_seed_one_visible_GPU_FP32",
            "publication": "required_exchange_then_atomic_whitelist_then_report_only_commit_and_optional_push",
            "export_only": "no_backend_or_training", "review_source_sha256": source_identity()}


def report_path(run):
    return safe(ROOT / "reports/retacred_ctl_drp_reference" / run.name)


def validate_options(args):
    if not 1 <= args.micro_batch <= 512 or args.pause_after is not None and args.pause_after < 0:
        raise ValueError("immutable micro-batch1..512 and nonnegative pause cursor required")
    if args.describe:
        return
    if not args.run_dir or not args.review_receipt:
        raise ValueError("explicit run-dir and independent review-receipt required")
    if args.export_only and (args.worker or args.pause_after is not None):
        raise ValueError("export-only never launches a training worker or pauses training")
    args.run_dir = safe(args.run_dir)
    if args.run_dir.is_relative_to(ROOT / "reports"):
        raise ValueError("run evidence belongs outside public reports")
    for key in ("qualification", "prepared_data", "review_receipt", "cache", "paper_source", "weight"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, safe(value))
    if not args.export_only and not all((args.qualification, args.cache, args.paper_source, args.weight)):
        raise ValueError("training requires qualification, cache, paper-source and explicit pinned weight")


def control_path(args):
    return safe(args.run_dir.with_name(args.run_dir.name + ".ctl-formal-control"))


def gates(args):
    model = VARIANT_MODELS[args.variant]
    select_models(model)  # before GPU/backend, qualification writes, upload or Git
    review = require_review(args.review_receipt, model)
    return model, review


def require_disk(args, weight):
    # Two rolling + two event pins + one atomic staging generation, with
    # model/two AdamW moments each. Margin includes shards and full trace values.
    minimum = 18 * weight["size_bytes"] + 2 * 1024**3
    parent = args.run_dir.parent
    while not parent.exists():
        parent = parent.parent
    free = shutil.disk_usage(parent).free
    if free < minimum:
        raise ValueError("formal CTL checkpoint and trace disk headroom below conservative estimate")
    return {"free_bytes": free, "minimum_bytes": minimum, "throughput_certified": False}


def worker(args):
    model, review = gates(args)
    control = control_path(args)
    bound = json.loads(regular(control / "contract.json").read_text())
    if bound["model"] != model or bound["review"] != review or bound["seed"] != args.seed or bound["micro_batch"] != args.micro_batch:
        raise ValueError("worker independent review/config binding drift")
    from .ctl_drp_preflight import verify_weight
    if verify_weight(args) != bound["weight"]:
        raise ValueError("worker pinned weight identity drift")
    if str(args.cache) != bound["cache"] or digest(regular(args.paper_source)) != bound["paper_source_sha256"]:
        raise ValueError("worker tokenizer/paper interface identity drift")
    from .ctl_drp_qualification import materialize
    native = materialize(args.qualification, prepared=args.prepared_data, output=control / "physical-data")
    if digest(native) != bound["physical_qualification_sha256"]:
        raise ValueError("worker physical qualification identity drift")
    from .ctl_drp_runner import run, seed_runtime
    from .ctl_drp_task import CTLDRPTask
    seed_runtime(args.seed)
    adapter = CTLDRPTask.from_assets(native, args.cache, args.paper_source, args.weight,
                                     variant=args.variant, seed=args.seed, micro_batch=args.micro_batch, device="cuda:0")
    result = run(adapter, args.run_dir, resume=args.resume, pause_after=args.pause_after)
    print(json.dumps({"status": result["status"], "updates": result["updates"], "model": model}), flush=True)
    return 75 if result["status"] == "paused" else 0


def validate_compute(args, bound):
    from .ctl_drp_export import validate_run
    summary, contract, _ = validate_run(args.run_dir)
    task = contract["task"]
    if (contract["diagnostic"] or summary["model"] != bound["model"] or summary["seed"] != bound["seed"]
            or task["recipe"] != RECIPE or task["physical_micro_batch_sentences"] != bound["micro_batch"]
            or summary["updates"] != UpdatePlan(COMPLETE_COUNTS["train"], bound["seed"]).total
            or task["initialization"]["weight_sha256"] != bound["weight"]["sha256"]
            or task["initialization"]["id"] != bound["weight"]["id"]
            or task["data_H"]["data"]["qualification_sha256"] != bound["physical_qualification_sha256"]
            or summary["data"] != task["data_H"]["data"]):
        raise ValueError("formal child recipe/budget/initialization/physical data result mismatch")
    return summary


def build_command(args):
    command = [sys.executable, "-u", "-m", "baselines.retacred.ctl_drp_formal", "--worker",
               "--variant", args.variant, "--seed", str(args.seed), "--micro-batch", str(args.micro_batch)]
    for key in ("qualification", "prepared_data", "review_receipt", "cache", "paper_source", "weight", "run_dir"):
        if getattr(args, key) is not None:
            command += ["--" + key.replace("_", "-"), str(getattr(args, key))]
    if args.resume and (args.run_dir / "contract.json").is_file():
        command.append("--resume")
    if args.pause_after is not None:
        command += ["--pause-after", str(args.pause_after)]
    return command


def launch(args, control, selected):
    from .ctl_drp_preflight import stop_child
    command = build_command(args)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": selected, "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
           "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONDONTWRITEBYTECODE": "1",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "USE_TF": "0", "USE_FLAX": "0",
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    record = {"schema": "ctl-formal-command.v1", "argv": command, "cwd": str(ROOT),
              "selected_gpu": selected, "status": "starting"}
    atomic_json(control / "command.json", record)
    handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    interrupted = [False]
    for sig in handlers:
        signal.signal(sig, lambda *_: interrupted.__setitem__(0, True))
    child = None
    try:
        with (control / "worker.log").open("a", encoding="utf-8") as output:
            offset = output.tell()
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=output,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            record.update(pid=child.pid, status="running")
            atomic_json(control / "command.json", record)
            with (control / "worker.log").open(encoding="utf-8") as stream:
                stream.seek(offset)
                while child.poll() is None:
                    print(stream.read(), end="", flush=True)
                    if interrupted[0]:
                        stop_child(child)
                        break
                    time.sleep(.2)
                print(stream.read(), end="", flush=True)
            code = child.wait()
            record.update(child_exit_status=code, exit_status=75 if interrupted[0] else code,
                          status="paused" if interrupted[0] or code == 75 else "exited")
            atomic_json(control / "command.json", record)
            return record["exit_status"]
    finally:
        if child is not None:
            stop_child(child)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def publish(run, report, *, no_push):
    from .ctl_drp_export import export, validate_report
    from .report import publication_preflight
    publication_preflight(no_push=no_push, allowed_report=report)
    export(run, report)  # upload/readiness first; failures preserve COMPUTE_COMPLETE
    validate_report(report)
    if not (run / "RUN_COMPLETE").is_file():
        raise ValueError("required delivery and atomic export completion marker absent")
    relative = report.relative_to(ROOT).as_posix()
    subprocess.run(["git", "add", "--", relative], cwd=ROOT, check=True)
    staged = git("diff", "--cached", "--name-only").splitlines()
    if any(not name.startswith(relative + "/") for name in staged):
        raise ValueError("foreign staged paths outside exact CTL report whitelist")
    if staged:
        subprocess.run(["git", "commit", "-m", f"reports: CTL-DRP reference {run.name}"], cwd=ROOT, check=True)
    atomic_json(run / "publication.json", {"status": "committed", "report": relative, "commit": git("rev-parse", "HEAD")})
    if not no_push:
        subprocess.run(["git", "-c", "http.version=HTTP/1.1", "push", "origin", "HEAD:1.1"], cwd=ROOT, check=True)
        atomic_json(run / "publication.json", {"status": "pushed", "report": relative, "commit": git("rev-parse", "HEAD")})
    return report


def dispatch(args):
    from .report import publication_preflight
    model, review = gates(args)
    control = control_path(args)
    report = report_path(args.run_dir)
    publication_preflight(no_push=args.no_push, allowed_report=report)
    resources = None
    if args.export_only:
        bound = json.loads(regular(control / "contract.json").read_text())
        if bound["model"] != model or bound["review"] != review or bound["seed"] != args.seed:
            raise ValueError("export-only review/model/seed contract drift")
    else:
        from .ctl_drp_preflight import verify_weight
        from .ctl_drp_qualification import materialize
        weight = verify_weight(args)
        resources = require_disk(args, weight)
        # Fresh/resume collisions fail before creating a data bridge or child.
        if args.run_dir.exists() and not args.resume:
            raise ValueError("existing formal run requires explicit resume or export-only")
        if args.resume and args.run_dir.exists() and not (args.run_dir / "contract.json").is_file():
            raise ValueError("existing formal run contract absent")
        if control.exists() and not args.resume:
            raise ValueError("existing formal parent requires explicit resume")
        if args.resume and not control.exists():
            raise ValueError("resume parent control contract absent")
        control.mkdir(parents=True, exist_ok=True)
        # Acquire the parent lock before any qualification/contract mutation.
        bound = None
    import fcntl
    with (control / "parent.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("formal parent already active") from error
        if not args.export_only:
            native = materialize(args.qualification, prepared=args.prepared_data, output=control / "physical-data")
            bound = {"schema": "ctl-formal-parent.v1", "model": model, "review": review,
                     "seed": args.seed, "micro_batch": args.micro_batch, "recipe": RECIPE,
                     "weight": weight, "physical_qualification_sha256": digest(native),
                     "cache": str(args.cache), "paper_source_sha256": digest(regular(args.paper_source)),
                     "qualification_sha256": digest(regular(args.qualification)), "run_dir": str(args.run_dir)}
            path = control / "contract.json"
            if path.exists():
                if json.loads(regular(path).read_text()) != bound:
                    raise ValueError("immutable formal parent contract drift")
            else:
                atomic_json(path, bound)
        phase = "validate_compute" if args.export_only else "training"
        try:
            if not args.export_only:
                from .orchestrate import gpu_ids
                code = launch(args, control, gpu_ids(args.gpu)[0])
                if code != 0:
                    atomic_json(control / "status.json", {"status": "paused" if code == 75 else "failed", "phase": phase,
                                                           "exit_status": code, "resources": resources})
                    return code
            phase = "validate_compute"
            validate_compute(args, bound)
            phase = "delivery_export_publication"
            publish(args.run_dir, report, no_push=args.no_push)
            atomic_json(control / "status.json", {"status": "committed" if args.no_push else "pushed", "phase": phase, "exit_status": 0})
            return 0
        except Exception as error:
            atomic_json(control / "status.json", {"status": "failed", "phase": phase, "error_type": type(error).__name__,
                                                   "exit_status": 1, "recovery": "resume" if phase == "training" else "export-only"})
            raise


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        validate_options(args)
        if args.describe:
            print(json.dumps(describe(args), indent=2))
            return 0
        return worker(args) if args.worker else dispatch(args)
    except Exception as error:
        print(f"[CTL formal] {type(error).__name__}; admission/review/data or private control record needs attention", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
