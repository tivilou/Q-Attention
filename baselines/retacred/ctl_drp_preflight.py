"""Bounded CTL dispatcher: diagnostic lanes never admit a formal cohort job."""
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

from .common import PACKAGE, ROOT, atomic_json, digest, fingerprint


def description(args):
    return {"schema": "ctl-drp-dispatch-preflight.v1", "mode": args.mode,
            "variant": args.variant, "diagnostic_updates": 0 if args.mode == "data" else args.updates,
            "formal_training": False, "formal_comparison_admitted": False,
            "downloads": False, "publishing": False,
            "data_scope": "explicit_external_receipt" if args.mode == "data" else "synthetic_fixture_only",
            "checkpoint_scope": "single_visible_GPU_FP32_or_tiny_CPU",
            "pending": ["real_data_qualification_and_whole_split_audit", "independent_review",
                        "formal_dispatch_export_publication", "actual_exchange_and_portal_binding"]}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("tiny", "pretrained", "data"), default="tiny")
    p.add_argument("--variant", choices=("sup", "unsup"), default="sup")
    p.add_argument("--describe", action="store_true")
    p.add_argument("--cache", type=Path)
    p.add_argument("--paper-source", type=Path)
    p.add_argument("--weight", type=Path)
    p.add_argument("--qualification", type=Path)
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--seed", type=int, choices=(13, 29, 53), default=13)
    p.add_argument("--micro-batch", type=int, default=2)
    p.add_argument("--updates", type=int, choices=(1, 2))
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--gpu", default="auto")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--pause-after", type=int, choices=(0, 1, 2))
    p.add_argument("--local-export-dir", type=Path)
    p.add_argument("--timeout-seconds", type=int, default=1800)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return p


def validate_options(args):
    if args.updates is None:
        args.updates = 1 if args.mode == "pretrained" else 2
    if not 1 <= args.micro_batch <= 512 or not 1 <= args.timeout_seconds <= 7200:
        raise ValueError("micro-batch1..512 and timeout1..7200 required")
    if args.mode == "pretrained" and args.device != "cuda":
        raise ValueError("pretrained runner diagnostic requires one CUDA GPU")
    if args.mode != "pretrained" and args.weight is not None:
        raise ValueError("--weight belongs to pretrained mode only")
    if args.mode != "data" and args.qualification is not None:
        raise ValueError("real qualification belongs to data audit; training diagnostics use synthetic fixtures only")
    if args.mode == "data" and (args.resume or args.pause_after is not None or args.local_export_dir):
        raise ValueError("data audit has no training resume/pause/export lifecycle")
    if args.describe:
        return
    if not all((args.cache, args.paper_source, args.run_dir)):
        raise ValueError("explicit --cache --paper-source --run-dir required")
    if args.mode == "pretrained" and args.weight is None:
        raise ValueError("pretrained mode requires an explicit pinned --weight")
    if args.mode == "data" and args.qualification is None:
        raise ValueError("data audit requires an external --qualification; no automatic qualification")
    from .ctl_drp_runner import safe
    args.run_dir = safe(args.run_dir)
    if args.run_dir.is_relative_to(ROOT / "reports"):
        raise ValueError("diagnostics belong in private runs, never public reports")
    if args.local_export_dir:
        args.local_export_dir = safe(args.local_export_dir)
        if args.local_export_dir.is_relative_to(ROOT / "reports"):
            raise ValueError("private diagnostic export must remain outside public reports")
    for key in ("cache", "paper_source", "weight", "qualification"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, safe(value))


def verify_weight(args):
    """Read bytes and locked JSON only; reject mismatch before model/GPU imports."""
    from .ctl_drp_pretrained import regular
    from .ctl_drp_simcse import load_lock
    lock = load_lock()
    identity = lock["facts"]["models"][args.variant]
    weight = regular(args.weight)
    if weight.stat().st_size != identity["weight"]["bytes"] or digest(weight) != identity["weight"]["sha256"]:
        raise ValueError("pinned pretrained weight size/hash drift before backend loading")
    url = f'https://huggingface.co/{identity["id"]}/resolve/{identity["revision"]}/config.json'
    item = next(row for row in lock["artifacts"] if row["url"] == url)
    config = regular(args.cache / item["path"])
    if digest(config) != item["sha256"] or json.loads(config.read_text()) != identity["config"]:
        raise ValueError("pinned pretrained configuration drift before backend loading")
    return {"id": identity["id"], "revision": identity["revision"],
            "sha256": identity["weight"]["sha256"], "size_bytes": weight.stat().st_size,
            "paper_exact_checkpoint_identity": False}


def require_disk(args, weight):
    # Bounded updates retain model+two AdamW moments per positive generation.
    # Include an additional weight-sized margin plus complete sampled traces.
    minimum = (3 * args.updates + 1) * weight["size_bytes"] + 512 * 1024**2
    parent = args.run_dir.parent
    while not parent.exists():
        parent = parent.parent
    free = shutil.disk_usage(parent).free
    if free < minimum:
        raise ValueError(f"pretrained diagnostic disk headroom {free} below conservative {minimum}")
    return {"free_bytes": free, "minimum_bytes": minimum, "estimate_not_formal_batch_capacity": True}


def worker(args):
    from .ctl_drp_task import PhysicalSplits, CTLDRPTask, audit_encodings
    from .ctl_drp_neural import tokenizer_from_cache, descriptions_from_native, tiny_model, joint_model
    from .ctl_drp_runner import run, seed_runtime, synthetic_receipt
    tokenizer = tokenizer_from_cache(args.cache, variant=args.variant)
    descriptions = descriptions_from_native(args.paper_source)
    if args.mode == "data":
        if args.run_dir.exists():
            raise FileExistsError("fresh data audit directory required; existing evidence preserved")
        data = PhysicalSplits(args.qualification)
        audit = audit_encodings(data, tokenizer, descriptions)
        result = {**description(args), "status": "diagnostic_data_verified" if data.identity["synthetic"] else "qualified_data_audit_verified",
                  "training_executed": False, "backend_loaded": False, "data_identity": data.identity,
                  "H_binding": data.binding, "coverage": data.coverage, "encoding_audit": audit}
        # binding already contains only data identities and H fingerprint, not raw rows.
        atomic_json(args.run_dir / "data-preflight.json", result)
        print(json.dumps({"status": result["status"], "formal_training": False}), flush=True)
        return 0
    seed_runtime(args.seed)
    data = PhysicalSplits(synthetic_receipt(args.run_dir.with_name(args.run_dir.name + ".synthetic-data")))
    audit = audit_encodings(data, tokenizer, descriptions)
    initialization = None
    if args.mode == "pretrained":
        verify_weight(args)
        from .ctl_drp_pretrained import load_encoder
        encoder, initialization = load_encoder(args.weight, args.cache, variant=args.variant, marker_seed=args.seed)
        model = joint_model(encoder, head_seed=args.seed)
    else:
        model = tiny_model(seed=args.seed, dropout=.1)
    adapter = CTLDRPTask(data, tokenizer, descriptions, model, seed=args.seed, micro_batch=args.micro_batch,
                        device="cuda:0" if args.device == "cuda" else "cpu",
                        initialization=initialization, diagnostic=True, encoding_audit=audit)
    result = run(adapter, args.run_dir, resume=args.resume, pause_after=args.pause_after, diagnostic_budget=args.updates)
    if result["status"] != "paused" and args.local_export_dir:
        from .ctl_drp_export import export
        export(args.run_dir, args.local_export_dir, allow_diagnostic=True)
    print(json.dumps({"status": result["status"], "updates": result["updates"],
                      "weight_scope": args.mode, "formal_training": False}), flush=True)
    return 75 if result["status"] == "paused" else 0


def stop_child(process, *, grace=45):
    """Only this parent's dedicated child session; never unrelated GPU processes."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def build_worker_command(args):
    command = [sys.executable, "-u", "-m", "baselines.retacred.ctl_drp_preflight", "--worker",
               "--mode", args.mode, "--variant", args.variant, "--seed", str(args.seed),
               "--micro-batch", str(args.micro_batch), "--updates", str(args.updates),
               "--device", args.device, "--timeout-seconds", str(args.timeout_seconds)]
    for key in ("cache", "paper_source", "weight", "qualification", "run_dir", "local_export_dir"):
        value = getattr(args, key)
        if value is not None:
            command += ["--" + key.replace("_", "-"), str(value)]
    if args.resume:
        command.append("--resume")
    if args.pause_after is not None:
        command += ["--pause-after", str(args.pause_after)]
    return command


def validate_child(args):
    if args.mode == "data":
        result = json.loads((args.run_dir / "data-preflight.json").read_text())
        if (result.get("formal_training") is not False or result.get("formal_comparison_admitted") is not False
                or result.get("training_executed") is not False or result.get("backend_loaded") is not False
                or result.get("status") not in ("diagnostic_data_verified", "qualified_data_audit_verified")):
            raise ValueError("data child scope/result mismatch")
    else:
        from .ctl_drp_export import validate_run
        summary, contract, _ = validate_run(args.run_dir)
        if (not contract["diagnostic"] or not summary["data"]["synthetic"]
                or summary["seed"] != args.seed or summary["updates"] != args.updates):
            raise ValueError("diagnostic child budget/data/seed mismatch")
        if args.mode == "pretrained":
            expected = verify_weight(args)
            if contract["task"]["initialization"]["weight_sha256"] != expected["sha256"]:
                raise ValueError("pretrained child initialization identity mismatch")


def dispatch(args, original):
    from .ctl_drp_runner import safe
    weight = verify_weight(args) if args.mode == "pretrained" else None
    resources = require_disk(args, weight) if weight else None
    if args.mode == "data" or args.device == "cpu":
        selected = "-1"
    else:
        from .orchestrate import gpu_ids
        choices = gpu_ids(args.gpu)
        # A bounded single-GPU lane deliberately chooses only one visible device.
        selected = choices[0]
    if args.run_dir.exists() and not args.resume:
        raise FileExistsError("existing diagnostic run needs explicit --resume")
    if args.resume and not (args.run_dir / "contract.json").is_file():
        raise ValueError("runner resume contract missing")
    control = safe(args.run_dir.with_name(args.run_dir.name + ".ctl-control"))
    contract = {"schema": "ctl-preflight-parent.v1", "mode": args.mode, "variant": args.variant,
                "seed": args.seed, "micro_batch": args.micro_batch, "updates": args.updates,
                "device": args.device if args.mode != "data" else "none",
                "run_dir": str(args.run_dir), "cache": str(args.cache),
                "paper_source_sha256": digest(args.paper_source), "weight": weight,
                "qualification_sha256": digest(args.qualification) if args.qualification else None,
                "local_export_dir": str(args.local_export_dir) if args.local_export_dir else None,
                "source_sha256": {name: digest(PACKAGE / name) for name in ("ctl_drp_preflight.py", "orchestrate.py")}}
    if control.exists():
        if not args.resume or json.loads((control / "contract.json").read_text()) != contract:
            raise ValueError("preflight control collision or immutable parent contract drift")
    else:
        if args.resume:
            raise ValueError("resume requires the matching preflight parent control")
        control.mkdir(parents=True)
        atomic_json(control / "contract.json", contract)
    import fcntl
    with (control / "parent.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("preflight parent already active") from error
        return launch(args, control, contract, selected, resources)


def launch(args, control, contract, selected, resources):
    command = build_worker_command(args)
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": selected, "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                   "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONDONTWRITEBYTECODE": "1",
                   "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "USE_TF": "0", "USE_FLAX": "0"}
    record = {**description(args), "argv": command, "cwd": str(ROOT), "selected_gpu": selected,
              "contract_sha256": fingerprint(contract), "resources": resources, "status": "starting"}
    atomic_json(control / "command.json", record)
    handlers = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    interrupted = [False]
    for s in handlers:
        signal.signal(s, lambda *_: interrupted.__setitem__(0, True))
    process = None
    try:
        with (control / "worker.log").open("a", encoding="utf-8") as output:
            offset = output.tell()
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=output,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            record.update(pid=process.pid, status="running")
            atomic_json(control / "command.json", record)
            started = time.monotonic()
            with (control / "worker.log").open(encoding="utf-8") as stream:
                # Emit only the new attempt, not the previous successful/failed log.
                stream.seek(offset)
                while process.poll() is None:
                    text = stream.read()
                    if text:
                        print(text, end="", flush=True)
                    if interrupted[0] or time.monotonic() - started > args.timeout_seconds:
                        stop_child(process)
                        break
                    time.sleep(.2)
                print(stream.read(), end="", flush=True)
            code = process.wait()
            record["child_exit_status"] = code
            if interrupted[0]:
                code = 75
            elif time.monotonic() - started > args.timeout_seconds:
                code = 124
            if code == 0:
                validate_child(args)
            record.update(status="diagnostic_complete" if code == 0 else "paused" if code == 75 else "failed", exit_status=code)
            atomic_json(control / "command.json", record)
            return code
    except Exception as error:
        record.update(status="failed", exit_status=1, error_type=type(error).__name__)
        atomic_json(control / "command.json", record)
        raise
    finally:
        if process is not None:
            stop_child(process)
        for s, handler in handlers.items():
            signal.signal(s, handler)


def main(argv=None):
    original = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(original)
    try:
        validate_options(args)
        if args.describe:
            print(json.dumps(description(args), indent=2))
            return 0
        return worker(args) if args.worker else dispatch(args, original)
    except Exception as error:
        # No exception text that might echo raw input samples or environment tokens.
        print(f"[CTL preflight] {type(error).__name__}; check arguments/private worker log", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
