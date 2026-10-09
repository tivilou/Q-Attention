"""CTL report whitelist and required-delivery hook; no implicit Git action."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import tempfile

from .common import ROOT, atomic_json, digest, fingerprint
from .ctl_drp_runner import safe, sync_dir
from .ctl_drp_trace import validate


def validate_run(run):
    run=safe(run);summary=json.loads((run/"run_summary.json").read_text());contract=json.loads((run/"contract.json").read_text())
    diagnostic=contract["diagnostic"]
    if not diagnostic and (contract["task"]["diagnostic"] or contract["task"]["data_H"]["data"]["synthetic"]):
        raise ValueError("diagnostic task identity cannot be relabeled as a formal run")
    marker="DIAGNOSTIC_COMPLETE" if diagnostic else "COMPUTE_COMPLETE"
    if not (run/marker).is_file() or summary["contract_sha256"]!=fingerprint(contract) or summary["updates"]!=contract["budget_updates"]:
        raise ValueError("complete immutable compute gate absent")
    if summary["status"]!=("diagnostic_complete" if diagnostic else "compute_complete") or {t["phase"] for t in summary["case_study"]["traces"]}!={"train","test"}:
        raise ValueError("completion status or phase coverage differs")
    layers=contract["task"]["model_configuration"]["num_hidden_layers"]
    files={}
    for item in summary["case_study"]["traces"]:
        path=safe(run/"case_study"/item["name"])
        if path.parent!=run/"case_study" or digest(path)!=item["sha256"]:raise ValueError("trace manifest integrity drift")
        validate(path,layers=layers)
    for path in (run/"case_study").iterdir():
        if not path.is_file() or path.suffix not in (".json",".pt") or safe(path).parent!=run/"case_study":raise ValueError("foreign private evidence")
        files[path.name]=digest(path)
    return summary,contract,files


def export(run, report, *, deliver=None, allow_diagnostic=False):
    """Real reports require verified exchange, diagnostics remain private local-only.

    Transport defaults to the existing hash/size-verified sharding client. It is
    injected only in focused tests; no success receipt is synthesized in the CLI.
    """
    run=safe(run);report=safe(report)
    summary,contract,files=validate_run(run)
    diagnostic=contract["diagnostic"]
    if diagnostic and (not allow_diagnostic or report.is_relative_to(ROOT/"reports")):
        raise ValueError("synthetic evidence must not become a public formal report")
    identity=digest(run/"run_summary.json")
    if report.exists():
        validate_report(report)
        manifest=json.loads((report/"export_manifest.json").read_text())
        if manifest["run_identity"]!=identity:raise ValueError("existing export belongs to different results")
        return report
    delivery={"status":"diagnostic_local_only_not_delivered","artifact_identity":files}
    if not diagnostic:
        if deliver is None:
            from .report import upload_evidence
            def deliver():return upload_evidence(run,[{"id":f"ctl_drp_seed{summary['seed']}","directory":run}])
        delivery=deliver_required(run,summary,files,deliver)
    report.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix=report.name+".stage-",dir=report.parent))
    try:
        atomic_json(stage/"metrics.json",summary["splits"])
        public={k:summary[k] for k in ("schema","status","model","seed","updates","formal_training","formal_comparison_admitted","contract_sha256")}
        public["fidelity"]="paper_interpretive_release_review_pending"
        public["case_study"]={"status":"diagnostic_local_values_only" if diagnostic else "complete_exchanged",
                              "schema":"sample-trace.v2","private_files":files,
                              "exchange_namespace":delivery.get("target_dir"),"exchange_configuration_ref":"PROJECT_EXCHANGE_URL"}
        atomic_json(stage/"run_summary.json",public)
        atomic_json(stage/"data_identity.json",summary["data"])
        (stage/"data.sha256").write_text("".join(f"{row['sha256']}  {split}\n" for split,row in summary["data"]["splits"].items()))
        (stage/"data_counts.txt").write_text("".join(f"{split} {row['count']}\n" for split,row in summary["data"]["splits"].items()))
        (stage/"run_summary.data").write_text(f"status={summary['status']}\nmodel=ctl_drp\nseed={summary['seed']}\nupdates={summary['updates']}\ncase_study={public['case_study']['status']}\n")
        hashes={p.name:digest(p) for p in stage.iterdir()}
        atomic_json(stage/"export_manifest.json",{"schema":"ctl-report-export.v1","run_identity":identity,"files":hashes,"diagnostic":diagnostic})
        validate_report(stage);sync_dir(stage);os.rename(stage,report);sync_dir(report.parent)
    finally:
        if stage.exists():
            if safe(stage).parent!=report.parent:raise ValueError("staging cleanup boundary changed")
            shutil.rmtree(stage)
    if not diagnostic:
        (run/"RUN_COMPLETE").touch();sync_dir(run)
    return report


def deliver_required(run,summary,files,deliver):
    # Write pending before the external side effect; a retry never retrains.
    atomic_json(run/"ctl_delivery_pending.json",{"status":"pending","artifact_identity":files})
    delivery=deliver()
    expected={f"ctl_drp_seed{summary['seed']}/{name}":hash_ for name,hash_ in files.items()}
    if delivery.get("status")!="complete" or delivery.get("artifact_identity")!=expected or not delivery.get("files") or not all(f.get("uploaded") for f in delivery["files"]):
        raise ValueError("required tensor delivery failed; preserve compute and retry export only")
    atomic_json(run/"ctl_delivery_pending.json",{"status":"complete","artifact_identity":files})
    return delivery


def validate_report(report):
    report=safe(report);manifest=json.loads((report/"export_manifest.json").read_text())
    expected={"metrics.json","run_summary.json","data_identity.json","data.sha256","data_counts.txt","run_summary.data"}
    if set(manifest["files"])!=expected or {p.name for p in report.iterdir()}!=expected|{"export_manifest.json"}:
        raise ValueError("export whitelist differs")
    for name,hash_ in manifest["files"].items():
        path=safe(report/name)
        if path.parent!=report or not path.is_file() or path.stat().st_size>1024*1024 or digest(path)!=hash_:raise ValueError("export size/integrity drift")
    return {"status":"whitelist_verified","files":len(expected)}
