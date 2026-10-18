#!/usr/bin/env python3
"""Frozen Q-ECPA seeds29/53, reusing the audited full seed13 result."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from q_attention.experiments import ecpa_protocol as p
from q_attention.experiments.ecpa_runner import gpu_ids, publication_preflight
import ecpa_replication_contract as c
from ecpa_replication_queue import execute
from ecpa_replication_stats import aggregate


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    group = cli.add_mutually_exclusive_group()
    group.add_argument("--output-dir", type=Path)
    group.add_argument("--resume", type=Path)
    cli.add_argument("--gpu", default="auto")
    cli.add_argument("--data-dir", type=Path)
    cli.add_argument("--hardware-profile", choices=["adaptive"], default="adaptive")
    cli.add_argument("--check", action="store_true")
    cli.add_argument("--dry-run", action="store_true")
    cli.add_argument("--export-only", action="store_true")
    cli.add_argument("--no-push", action="store_true")
    cli.add_argument("--canary", action="store_true")
    cli.add_argument("--canary-seed13", type=Path)
    cli.add_argument("--inject-oom-once", action="store_true")
    cli.add_argument("--stop-after", type=int)
    return cli


def validate_run(root):
    manifest, config = p.read(root / "run_manifest.json"), c.protocol()
    if manifest["protocol"] != config or manifest["execution_source"] != c.source_contract():
        raise ValueError("replication source/config changed; resume/export refused")
    if manifest["runtime_sha256"] != p.identity(p.source_contract()):
        raise ValueError("scientific runtime changed")
    if p.sha(root / "imported_seed13_summary.json") != manifest["imported_seed13"]["summary_sha256"]:
        raise ValueError("imported seed13 summary changed")
    if p.identity(manifest["contract"]) != manifest["fingerprint"]:
        raise ValueError("cohort identity changed")
    expected = dict(protocol=config, imported_summary_sha256=manifest["imported_seed13"]["summary_sha256"],
        data=manifest["data"], execution_source=manifest["execution_source"],
        runtime_sha256=manifest["runtime_sha256"], canary=manifest["canary"])
    if manifest["contract"] != expected or p.read(root / "run_config.json") != config:
        raise ValueError("cohort manifest fields disagree with its immutable contract")
    if manifest["data"] != manifest["imported_seed13"]["data"]:
        raise ValueError("cohort data differs from seed13")
    if not manifest["canary"]:
        imported, provenance, _ = c.seed13_import(config)
        if provenance != manifest["imported_seed13"] or imported != p.read(root / "imported_seed13_summary.json"):
            raise ValueError("import no longer matches the pinned audited seed13")
    return manifest


def run(args):
    if (args.inject_oom_once or args.stop_after is not None or args.canary_seed13 is not None) and not args.canary:
        raise ValueError("fault injection and canary fixture are synthetic-only")
    if args.canary and args.canary_seed13 is None and args.resume is None:
        raise ValueError("synthetic canary requires a completed --canary-seed13 fixture")
    if args.export_only:
        if args.resume is None or args.canary or args.check or args.dry_run:
            raise ValueError("export-only requires completed formal --resume")
        from ecpa_replication_export import finish
        return finish(args.resume.resolve(), push=not args.no_push)
    spec = c.protocol()
    if not args.canary:
        publication_preflight()
        if not os.environ.get("PROJECT_EXCHANGE_URL") or not os.environ.get("PROJECT_EXCHANGE_TOKEN"):
            raise ValueError("configure PROJECT_EXCHANGE_URL and PROJECT_EXCHANGE_TOKEN before formal execution")
    slots = gpu_ids(args.gpu, canary=args.canary)
    # Exercise the same launch contract that Popen will use, before any run is made.
    worker = c.worker_spec(Path("."), "baseline", "cpu")
    subprocess.run(worker["argv"]+["--help"], cwd=worker["cwd"], check=True, stdout=subprocess.DEVNULL)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = Path(args.resume or args.output_dir or ROOT / "runs" / c.EXPERIMENT / stamp).resolve()
    if args.resume:
        manifest = validate_run(root)
        if manifest["canary"] != args.canary:
            raise ValueError("formal/canary mode differs from original cohort")
        data_dir = Path(manifest["data_dir"])
        imported, provenance = p.read(root / "imported_seed13_summary.json"), manifest["imported_seed13"]
    else:
        if root.exists():
            raise ValueError("fresh output directory must not already exist")
        imported, provenance, fixture_data = c.seed13_import(spec, canary_run=args.canary_seed13 if args.canary else None)
        data_dir = Path(fixture_data or args.data_dir or ROOT / "data/relation/retacred").resolve()
    recipe = p.load_config(args.canary)
    _, _, _, _, _, data = p.load_data(data_dir, recipe)
    if data != provenance["data"]:
        raise ValueError("replication data differs from imported seed13")
    if args.check or args.dry_run:
        print(dict(status="preflight_passed", new_seeds=spec["new_seeds"], imported_seed=13,
            data=data, slots=slots, worker=worker, scientific_runtime=spec["frozen_runtime_sha256"]), flush=True)
        return 0
    if not args.resume:
        contract = dict(protocol=spec, imported_summary_sha256=provenance["summary_sha256"],
            data=data, execution_source=c.source_contract(), runtime_sha256=spec["frozen_runtime_sha256"], canary=args.canary)
        manifest = dict(schema_version="ecpa-replication-run.v1", run_id=root.name,
            fingerprint=p.identity(contract), contract=contract, protocol=spec, imported_seed13=provenance,
            execution_source=contract["execution_source"], runtime_sha256=contract["runtime_sha256"],
            canary=args.canary, data=data, data_dir=str(data_dir), git_revision=p.revision())
    root.mkdir(parents=True, exist_ok=True)
    with p.exclusive(root / ".scheduler.lock"):
        if not args.resume:
            if any(path.name != ".scheduler.lock" for path in root.iterdir()):
                raise ValueError("fresh run was concurrently initialized; refuse overwrite")
            p.atomic_json(root / "run_manifest.json", manifest)
            p.atomic_json(root / "run_config.json", spec)
            p.atomic_json(root / "imported_seed13_summary.json", imported)
        args.resolved_run = root
        seed_runs = {}
        for seed in spec["new_seeds"]:
            run = root / "seeds" / f"seed{seed}"
            config = c.seed_config(seed, args.canary)
            fingerprint = p.identity(dict(config=config, data=data, source_sha256=spec["frozen_runtime_sha256"]))
            if (run / "run_manifest.json").exists():
                if p.read(run / "run_config.json") != config or p.read(run / "run_manifest.json")["fingerprint"] != fingerprint:
                    raise ValueError("seed checkpoint contract changed")
            else:
                p.atomic_json(run / "run_manifest.json", dict(schema_version="ecpa-run.v1", run_id=f"{root.name}_seed{seed}",
                    git_revision=manifest["git_revision"], fingerprint=fingerprint, source_sha256=spec["frozen_runtime_sha256"],
                    source_files=p.source_contract(), data=data, data_dir=str(data_dir),
                    environment=dict(python=sys.version.split()[0], torch=__import__("torch").__version__), profile=config["profile"]))
                p.atomic_json(run / "run_config.json", config)
            seed_runs[seed] = run
        code = execute(root, seed_runs, args, slots)
        if code:
            print(f"Paused safely. Resume this cohort: --resume {root}", flush=True)
            return code
        summaries = {13: imported, **{seed: c.summarize_seed(run) for seed, run in seed_runs.items()}}
        summary = aggregate(summaries, p.MODES)
        summary["fingerprint"] = manifest["fingerprint"]
        p.atomic_json(root / "run_summary.json", summary)
        p.atomic_json(root / "run_summary.data", summary)
        text = "# Q-ECPA seeds13/29/53\n\nSeed13 imported unchanged; new seeds29/53. Values are fractions, mean ± sample SD.\n\n"
        text += "\n".join(f"- {mode}: {row['metrics']['macro_f1']['mean']:.8f} ± {row['metrics']['macro_f1']['std']:.8f}" for mode, row in summary["rows"].items())
        (root / "run_summary.md").write_text(text+"\n")
        (root / "data.sha256").write_text("".join(f"{v['sha256']}  {k}.jsonl\n" for k, v in data.items()))
        (root / "data_counts.txt").write_text("".join(f"{k}: {v['records']}\n" for k, v in data.items()))
        p.atomic_json(root / "RUN_COMPLETE.json", dict(fingerprint=manifest["fingerprint"], summary_sha256=p.sha(root / "run_summary.json")))
        for name in ("RUN_PAUSED.json", "RUN_FAILED.json"):
            (root / name).unlink(missing_ok=True)
    if not args.canary:
        from ecpa_replication_export import finish
        return finish(root, push=not args.no_push)
    print(f"Synthetic replication complete: {root}", flush=True)
    return 0


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        root = getattr(args, "resolved_run", None)
        if root is not None and not (root / "RUN_COMPLETE.json").exists():
            p.atomic_json(root / "RUN_FAILED.json", dict(error_type=type(exc).__name__,
                retry="inspect per-seed logs, fix the cause, then --resume this cohort"))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
