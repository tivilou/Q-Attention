"""CTL report whitelist and required-delivery hook; no implicit Git action."""
from __future__ import annotations
import json
import hashlib
import os
from pathlib import Path
import shutil
import tempfile

from .common import ROOT, atomic_json, digest, fingerprint
from .ctl_drp_runner import safe, sync_dir
from .ctl_drp_trace import validate

SHARD_AT = 1024**3


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
        if not diagnostic:
            # A crash after rename but before RUN_COMPLETE must be recoverable
            # without another model load or optimizer update. Recheck delivery.
            if deliver is None:
                from .report import upload_evidence
                def deliver():return upload_evidence(run,[{"id":f"ctl_drp_seed{summary['seed']}","directory":run}])
            delivery=deliver_required(run,summary,files,deliver)
            public=json.loads((report/"run_summary.json").read_text())
            if (public["case_study"]["private_files"]!=files
                    or public["case_study"]["exchange_namespace"]!=delivery["target_dir"]
                    or public["model"]!=summary["model"]
                    or json.loads((report/"metrics.json").read_text())!=summary["splits"]
                    or json.loads((report/"data_identity.json").read_text())!=summary["data"]):
                raise ValueError("existing report no longer matches compute/delivery evidence")
            (run/"RUN_COMPLETE").touch();sync_dir(run)
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
        public["fidelity"]="paper_interpretive_release_review_pending" if diagnostic else "paper_interpretive"
        public["case_study"]={"status":"diagnostic_local_values_only" if diagnostic else "complete_exchanged",
                              "schema":"sample-trace.v2","private_files":files,
                              "exchange_namespace":delivery.get("target_dir"),
                              "job_namespace":f"ctl_drp_seed{summary['seed']}",
                              "traces":summary["case_study"]["traces"],
                              "exchange_configuration_ref":"PROJECT_EXCHANGE_URL"}
        if not diagnostic:
            public["delivery"]={"status":"complete", "files":delivery["files"]}
        atomic_json(stage/"run_summary.json",public)
        atomic_json(stage/"data_identity.json",summary["data"])
        (stage/"data.sha256").write_text("".join(f"{row['sha256']}  {split}\n" for split,row in summary["data"]["splits"].items()))
        (stage/"data_counts.txt").write_text("".join(f"{split} {row['count']}\n" for split,row in summary["data"]["splits"].items()))
        (stage/"run_summary.data").write_text(f"status={summary['status']}\nmodel={summary['model']}\nseed={summary['seed']}\nupdates={summary['updates']}\ncase_study={public['case_study']['status']}\n")
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
    saved=run/"ctl_delivery_receipt.json"
    delivery=json.loads(saved.read_text()) if saved.exists() else deliver()
    expected={f"ctl_drp_seed{summary['seed']}/{name}":hash_ for name,hash_ in files.items()}
    inventory=delivery_inventory(run,summary,files)
    received=delivery.get("files",[])
    if (delivery.get("status")!="complete" or delivery.get("artifact_identity")!=expected
            or delivery.get("target_dir")!=f"q-attention/retacred-reference-suite/{run.name}"
            or not isinstance(received,list) or len(received)!=len(inventory)
            or any(not isinstance(item,dict) or item.get("uploaded") is not True
                   or item.get("path") not in inventory
                   or {k:item.get(k) for k in ("sha256","size_bytes")}!=inventory[item["path"]] for item in received)
            or len({item["path"] for item in received})!=len(inventory)):
        raise ValueError("required tensor delivery failed; preserve compute and retry export only")
    delivery={"status":"complete", "target_dir":delivery["target_dir"], "artifact_identity":expected,
              "files":[{k:item[k] for k in ("path","sha256","size_bytes","uploaded")} for item in received]}
    atomic_json(saved,delivery)
    atomic_json(run/"ctl_delivery_pending.json",{"status":"complete","artifact_identity":files})
    return delivery


def delivery_inventory(run,summary,files):
    """Bind every transport receipt to exact private bytes, including shard joins."""
    run=safe(run);job=f"ctl_drp_seed{summary['seed']}";expected={}
    for name, hash_ in files.items():
        file=safe(run/"case_study"/name)
        if file.parent!=run/"case_study" or not file.is_file() or digest(file)!=hash_:
            raise ValueError("required tensor delivery failed; private file identity drift")
        if file.stat().st_size < SHARD_AT:
            expected[f"{job}/{name}"]={"sha256":hash_, "size_bytes":file.stat().st_size}
            continue
        manifest=safe(run/"exchange_parts"/job/(name+".parts.json"))
        spec=json.loads(manifest.read_text())
        if spec.get("name")!=name or spec.get("sha256")!=hash_ or spec.get("size_bytes")!=file.stat().st_size or not spec.get("parts"):
            raise ValueError("required tensor delivery failed; shard identity drift")
        joined=hashlib.sha256();size=0
        for i, part in enumerate(spec["parts"]):
            if part["name"]!=f"{name}.part{i:04d}":raise ValueError("required tensor delivery failed; shard order drift")
            path=safe(manifest.parent/part["name"])
            if path.parent!=manifest.parent or digest(path)!=part["sha256"] or path.stat().st_size!=part["size_bytes"]:
                raise ValueError("required tensor delivery failed; shard bytes drift")
            with path.open("rb") as source:
                for block in iter(lambda:source.read(1024*1024),b""):joined.update(block);size+=len(block)
            expected[f"{job}/{part['name']}"]={k:part[k] for k in ("sha256","size_bytes")}
        if joined.hexdigest()!=hash_ or size!=spec["size_bytes"]:raise ValueError("required tensor delivery failed; shard join drift")
        expected[f"{job}/{manifest.name}"]={"sha256":digest(manifest),"size_bytes":manifest.stat().st_size}
    return expected


def validate_report(report):
    report=safe(report);manifest=json.loads((report/"export_manifest.json").read_text())
    expected={"metrics.json","run_summary.json","data_identity.json","data.sha256","data_counts.txt","run_summary.data"}
    if set(manifest["files"])!=expected or {p.name for p in report.iterdir()}!=expected|{"export_manifest.json"}:
        raise ValueError("export whitelist differs")
    for name,hash_ in manifest["files"].items():
        path=safe(report/name)
        if path.parent!=report or not path.is_file() or path.stat().st_size>1024*1024 or digest(path)!=hash_:raise ValueError("export size/integrity drift")
    return {"status":"whitelist_verified","files":len(expected)}
