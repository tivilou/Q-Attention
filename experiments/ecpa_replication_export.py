"""Strict three-seed audit, private numeric exchange and report-only publication."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile

from q_attention.analysis.ecpa_formal_trace import validate
from q_attention.experiments import ecpa_protocol as p
from q_attention.experiments import ecpa_export as legacy
from q_attention.experiments.ecpa_runner import completed
import ecpa_replication_contract as c
from ecpa_replication_stats import aggregate

PUBLIC = ("run_summary.json", "run_summary.data", "run_summary.md", "run_config.json", "data.sha256", "data_counts.txt")


def audit(run):
    # Import lazily: the entry point also imports finish for export-only retries.
    from run_ecpa_replication import validate_run
    run = Path(run).resolve()
    manifest = validate_run(run)
    marker = p.read(run / "RUN_COMPLETE.json")
    if marker != dict(fingerprint=manifest["fingerprint"], summary_sha256=p.sha(run / "run_summary.json")):
        raise ValueError("completed cohort identity/hash mismatch")
    summaries = {13: p.read(run / "imported_seed13_summary.json")}
    for seed in manifest["protocol"]["new_seeds"]:
        root = run / "seeds" / f"seed{seed}"
        config, identity = p.read(root / "run_config.json"), p.read(root / "run_manifest.json")
        fingerprint = p.identity(dict(config=config, data=manifest["data"], source_sha256=manifest["runtime_sha256"]))
        if (config != c.seed_config(seed, manifest["canary"]) or identity["fingerprint"] != fingerprint
                or identity["source_files"] != p.source_contract() or identity["data"] != manifest["data"]
                or identity["source_sha256"] != manifest["runtime_sha256"]
                or identity["run_id"] != f"{manifest['run_id']}_seed{seed}"):
            raise ValueError("per-seed scientific contract changed")
        summary = p.read(root / "run_summary.json")
        if (p.read(root / "RUN_COMPLETE.json") != dict(fingerprint=fingerprint, summary_sha256=p.sha(root / "run_summary.json"))
                or p.read(root / "run_summary.data") != summary):
            raise ValueError("incomplete per-seed summary")
        c.validate_summary(summary, config, fingerprint)
        for mode in ("baseline", *p.MODES):
            job = root / "jobs" / mode
            if not completed(root, mode, identity):
                raise ValueError(f"missing completed job: seed{seed}/{mode}")
            metrics = p.read(job / "metrics.json")
            if metrics["fingerprint"] != fingerprint or metrics["test_used_for_training_or_selection"] is not False:
                raise ValueError("invalid job identity/selection flag")
            if mode != "baseline" and summary["rows"][mode] != metrics:
                raise ValueError("summary differs from immutable job metrics")
            # Reads and verifies every full numeric NPY, shape/hash and semantic lineage.
            result = validate(job / "sample_trace.json", mode, config)
            if result != p.read(job / "trace_validation.json"):
                raise ValueError("stored trace validation differs from actual numeric evidence")
        summaries[seed] = summary
    expected = aggregate(summaries, p.MODES)
    expected["fingerprint"] = manifest["fingerprint"]
    if p.read(run / "run_summary.json") != expected or p.read(run / "run_summary.data") != expected:
        raise ValueError("aggregate differs from the three complete seed results")
    for name in PUBLIC:
        if not (run / name).is_file() or not (run / name).stat().st_size:
            raise ValueError(f"missing public evidence: {name}")
    expected_hashes = "".join(f"{v['sha256']}  {k}.jsonl\n" for k, v in manifest["data"].items())
    expected_counts = "".join(f"{k}: {v['records']}\n" for k, v in manifest["data"].items())
    if (run / "data.sha256").read_text() != expected_hashes or (run / "data_counts.txt").read_text() != expected_counts:
        raise ValueError("aggregate data population/provenance differs")
    return manifest, summaries


def verified_receipt(receipt):
    if (receipt.get("status") != "uploaded" or not receipt.get("parts")
            or not all(part.get("verified") for part in receipt["parts"])):
        raise ValueError("full numeric evidence lacks verified private exchange receipts")
    return receipt


def export_report(run, *, exchange_fn=legacy.exchange, destination=None, inject_failure=False):
    run = Path(run).resolve()
    with p.exclusive(run / ".export.lock"):
        manifest, summaries = audit(run)
        canary = manifest["canary"]
        if not canary and (exchange_fn is not legacy.exchange or destination is not None or inject_failure):
            raise ValueError("custom destination/transport and fault injection are canary-only")
        final = Path(destination or p.ROOT / "reports" / c.EXPERIMENT / run.name).resolve()
        if not canary:
            legacy.publication_preflight(final)
        stage = Path(tempfile.mkdtemp(prefix="ecpa-replication-export-"))
        try:
            imported = manifest["imported_seed13"]
            receipts = {"13": imported.get("exchange_manifest", dict(status="synthetic_fixture"))}
            if not canary:
                verified_receipt(receipts["13"])
            case_index = {"13": imported["case_study"]}
            for seed in manifest["protocol"]["new_seeds"]:
                root = run / "seeds" / f"seed{seed}"
                # Unique per-seed run IDs keep the frozen private exchange writer unchanged.
                receipts[str(seed)] = verified_receipt(exchange_fn(root, p.read(root / "run_manifest.json")))
                case_index[str(seed)] = dict(jobs={mode: dict(
                    validation=p.read(root / "jobs" / mode / "trace_validation.json"),
                    trace_sha256=p.sha(root / "jobs" / mode / "sample_trace.json"),
                    evidence_path=f"jobs/{mode}/sample_trace.json") for mode in ("baseline", *p.MODES)})
            for name in PUBLIC:
                shutil.copy2(run / name, stage / name)
            for seed, summary in summaries.items():
                p.atomic_json(stage / "seeds" / f"seed{seed}" / "run_summary.json", summary)
                if seed != 13:
                    root = run / "seeds" / f"seed{seed}"
                    p.atomic_json(stage / "seeds" / f"seed{seed}" / "run_config.json", p.read(root / "run_config.json"))
            safe = {key: manifest[key] for key in ("schema_version", "run_id", "fingerprint", "contract",
                "protocol", "execution_source", "runtime_sha256", "canary", "data", "git_revision")}
            safe["seed13_provenance"] = {key: imported[key] for key in ("seed", "kind", "summary_sha256", "fingerprint")}
            if not canary:
                safe["seed13_provenance"].update(commit=imported["commit"], report=imported["report"])
            p.atomic_json(stage / "run_manifest.json", safe)
            p.atomic_json(stage / "exchange_manifest.json", dict(schema_version="ecpa-replication-exchange.v1", seeds=receipts))
            p.atomic_json(stage / "case_study.json", dict(schema_version="ecpa-replication-private-case-index.v1",
                trace_completeness="complete", numeric_values="complete lossless private NPY",
                public_preview="not_included", seeds=case_index))
            if inject_failure:
                raise RuntimeError("injected replication export validation failure")
            files = {path.relative_to(stage).as_posix(): p.sha(path) for path in sorted(stage.rglob("*")) if path.is_file()}
            allowed = {*PUBLIC, "run_manifest.json", "exchange_manifest.json", "case_study.json",
                *(f"seeds/seed{seed}/run_summary.json" for seed in (13, 29, 53)),
                *(f"seeds/seed{seed}/run_config.json" for seed in (29, 53))}
            if set(files) != allowed:
                raise ValueError("public export whitelist differs")
            report_identity = dict(schema_version="ecpa-replication-report.v1", fingerprint=manifest["fingerprint"], files=files)
            p.atomic_json(stage / "REPORT_COMPLETE.json", report_identity)
            if final.exists():
                actual = {path.relative_to(final).as_posix() for path in final.rglob("*") if path.is_file()}
                if (any(path.is_symlink() for path in final.rglob("*")) or actual != {*files, "REPORT_COMPLETE.json"}
                        or p.read(final / "REPORT_COMPLETE.json") != report_identity
                        or any(p.sha(final / name) != digest for name, digest in files.items())):
                    raise ValueError("existing report changed; refuse overwrite")
            else:
                final.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(prefix=".ecpa-replication-stage-", dir=final.parent) as sibling:
                    content = Path(sibling) / "report"
                    shutil.copytree(stage, content)
                    os.replace(content, final)
            p.atomic_json(run / "EXPORT_COMPLETE.json", dict(report=str(final), identity=p.identity(report_identity)))
            (run / "EXPORT_FAILED.json").unlink(missing_ok=True)
            return final
        except Exception as exc:
            p.atomic_json(run / "EXPORT_FAILED.json", dict(error_type=type(exc).__name__,
                retry="--export-only --resume same_completed_cohort; no training"))
            raise
        finally:
            shutil.rmtree(stage)


def finish(run, *, push=True):
    run = Path(run).resolve()
    report = export_report(run)
    if push:
        try:
            git, relative = legacy.publication_preflight(report)
            git("add", "--", relative)
            if git("diff", "--cached", "--name-only"):
                git("commit", "-m", f"reports: Q-ECPA seeds13/29/53 {run.name}")
            legacy.retry(lambda: git("push", "origin", "HEAD:refs/heads/1.1"))
        except Exception as exc:
            p.atomic_json(run / "PUBLISH_FAILED.json", dict(error_type=type(exc).__name__,
                retry="--export-only --resume same_completed_cohort; no training"))
            raise
        (run / "PUBLISH_FAILED.json").unlink(missing_ok=True)
    print(f"Audited replication report: {report}; published={push}", flush=True)
    return 0
