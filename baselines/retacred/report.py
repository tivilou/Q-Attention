"""Private tensor exchange, atomic public whitelist, report-only commit/push retry."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from .assets import CA, shard_file, uploader
from .common import ROOT, atomic_json, digest, git

PUBLIC_LIMIT = 1024 * 1024


def publication_preflight(*, no_push: bool, allowed_report: Path | None = None):
    if git("branch", "--show-current") != "1.1":
        raise ValueError("formal collaborator execution/publication must be on branch 1.1")
    for ref in ("origin/1.1", "origin/main"):
        subprocess.run(["git","merge-base","--is-ancestor",ref,"HEAD"],cwd=ROOT,check=True)
    staged = subprocess.check_output(["git","diff","--cached","--name-only"],cwd=ROOT,text=True).splitlines()
    prefix = str(allowed_report.relative_to(ROOT)).replace("\\","/")+"/" if allowed_report else None
    if staged and (not prefix or any(not name.startswith(prefix) for name in staged)):
        raise ValueError("Git index contains unrelated pre-existing staged changes")
    dirty = subprocess.check_output(["git","status","--porcelain","--untracked-files=all"],cwd=ROOT,text=True).rstrip("\n")
    if dirty:
        prefix = str(allowed_report.relative_to(ROOT)).replace("\\","/")+"/" if allowed_report else None
        if not prefix or any(not line[3:].startswith(prefix) for line in dirty.splitlines()):
            raise ValueError("worktree has unrelated changes; publication must never stage them")
    if not git("config","user.name") or not git("config","user.email"):
        raise ValueError("normal repository Git identity is required")
    if not no_push:
        subprocess.run(["git","ls-remote","--exit-code","origin","refs/heads/1.1"],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)


def upload_evidence(run: Path, jobs: list[dict]) -> dict:
    if jobs and all((j['directory']/'run_summary.json').is_file() and json.loads((j['directory']/'run_summary.json').read_text()).get('schema')=='retacred-reference-result.v2' for j in jobs):
        from .reference_archive import aggregate
        return aggregate(run,jobs)
    receipt_path = run / "exchange_manifest.json"
    declared = []
    identities = {}
    for job in jobs:
        for path in sorted((job["directory"] / "case_study").iterdir()):
            if path.suffix not in (".json", ".pt") or path.is_symlink():
                raise ValueError("undeclared or unsafe case-study evidence file")
            name = f"{job['id']}/{path.name}"
            identities[name] = digest(path)
            if path.stat().st_size >= 1024*1024*1024:
                parts = shard_file(path, run / "exchange_parts" / job["id"])
                manifest = run / "exchange_parts" / job["id"] / (path.name + ".parts.json")
                atomic_json(manifest,{"name":path.name,"sha256":digest(path),"size_bytes":path.stat().st_size,"parts":parts})
                declared.append((manifest,f"{job['id']}/{manifest.name}"))
                declared.extend((manifest.parent/part["name"],f"{job['id']}/{part['name']}") for part in parts)
            else:
                declared.append((path,name))
    if receipt_path.exists():
        existing = json.loads(receipt_path.read_text())
        if existing.get("artifact_identity") != identities or existing.get("status") != "complete":
            raise ValueError("exchange receipt does not match current completed traces")
        return existing
    url = os.environ.get("PROJECT_EXCHANGE_URL", "")
    token = os.environ.get("PROJECT_EXCHANGE_TOKEN", "")
    if not url.startswith("https://") or not token:
        raise ValueError("required private tensor upload: configure HTTPS PROJECT_EXCHANGE_URL and PROJECT_EXCHANGE_TOKEN")
    target_dir = f"q-attention/retacred-reference-suite/{run.name}"
    pending = {"schema":"retacred-evidence-delivery.v1", "status":"pending",
               "target_dir":target_dir, "artifact_identity":identities,
               "files":[{"name":name,"size_bytes":path.stat().st_size,"sha256":digest(path)} for path,name in declared]}
    atomic_json(run/"exchange_pending.json",pending)
    try:
        receipt = uploader().upload_declared_files(base_url=url, token=token,
                        target_dir=target_dir, files=declared, ca_file=CA)
    except Exception as exc:
        atomic_json(run/"exchange_pending.json", {**pending,"status":"failed","error_type":type(exc).__name__})
        raise
    if receipt.get("status") != "uploaded" or not all(item.get("uploaded") for item in receipt["files"]):
        raise ValueError("upload receipt is incomplete")
    receipt["transport_status"] = receipt["status"]
    receipt["status"] = "complete"
    receipt["artifact_identity"] = identities
    atomic_json(receipt_path, receipt)
    atomic_json(run/"exchange_pending.json", {**pending,"status":"complete"})
    return receipt


def validate_export(report: Path):
    manifest = json.loads((report / "export_manifest.json").read_text())
    files = {path.relative_to(report).as_posix() for path in report.rglob("*") if path.is_file()}
    if files != set(manifest["files"]) | {"export_manifest.json"}:
        raise ValueError("report contains missing or extra artifacts")
    for name, expected in manifest["files"].items():
        path = report / name
        if path.is_symlink() or path.stat().st_size > PUBLIC_LIMIT or digest(path) != expected:
            raise ValueError("public report violates hash/size/symlink policy")


def export_report(run: Path, jobs: list[dict], exchange: dict, report: Path):
    run_identity = {job["id"]:digest(job["directory"] / "run_summary.json") for job in jobs}
    if report.exists():
        validate_export(report)
        old = json.loads((report / "export_manifest.json").read_text())
        if old["run_identity"] != run_identity:
            raise FileExistsError("existing report identity belongs to different results")
        return
    report.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=report.name+".stage-",dir=report.parent))
    try:
        rows = []
        for job in jobs:
            destination = staging / job["id"]
            destination.mkdir()
            summary = json.loads((job["directory"] / "run_summary.json").read_text())
            atomic_json(destination / "metrics.json", summary["splits"])
            public_summary = {key:summary[key] for key in ("schema","status","model","seed","updates","adaptations","fidelity","environment")}
            public_summary["contract"] = summary["contract"]
            public_summary["case_study"] = {"status":"complete_exchanged", "schema":"sample-trace.v2",
                       "target_dir":exchange["target_dir"]+"/"+job["id"], "exchange_url":exchange["exchange_url"],
                       "traces":summary["case_study"]["traces"]}
            atomic_json(destination / "run_summary.json", public_summary)
            # Absolute private source paths are not useful in public reports.
            split_identity = {split:{key:value[key] for key in ("source_sha256","output_sha256","records")}
                              for split,value in summary["data_manifest"]["splits"].items()}
            atomic_json(destination / "data_manifest.json", split_identity)
            (destination/"data.sha256").write_text("".join(f"{value['source_sha256']}  {split}.jsonl\n" for split,value in split_identity.items()))
            (destination/"data_counts.txt").write_text("".join(f"{split} {value['records']}\n" for split,value in split_identity.items()))
            (destination/"run_summary.data").write_text(f"status=complete\nmodel={job['model']}\nseed={job['seed']}\ncase_study=complete_exchanged\n")
            test = summary["splits"]["test"]
            loss_text=f"{test['loss']:.6f}" if test.get('loss') is not None else 'N/A (see loss_scope)'
            rows.append(f"| {job['model']} | {job['seed']} | {test['micro_f1']:.6f} | {test['accuracy']:.6f} | {test['macro_f1']:.6f} | {loss_text} |")
        # Only compact receipts; no token, raw samples or tensors enter Git.
        receipt = {key:exchange[key] for key in ("schema_version","status","exchange_url","target_dir","files","artifact_identity")}
        atomic_json(staging / "exchange_manifest.json", receipt)
        atomic_json(staging / "run_manifest.json", json.loads((run / "run_manifest.json").read_text()))
        upgrade=run/'execution_upgrade.json'
        if upgrade.is_file():atomic_json(staging/'execution_upgrade.json',json.loads(upgrade.read_text()))
        (staging/"run_summary.md").write_text("# Re-TACRED reference suite\n\nOnly requested configurations completed; not a full top-15 reproduction.\n\n| model | seed | benchmark micro-F1 | accuracy | macro-F1 | loss |\n| --- | ---: | ---: | ---: | ---: | ---: |\n"+"\n".join(rows)+"\n")
        files = {path.relative_to(staging).as_posix():digest(path) for path in staging.rglob("*") if path.is_file()}
        atomic_json(staging/"export_manifest.json", {"schema":"retacred-report-export.v1","run_identity":run_identity,"files":files})
        validate_export(staging)
        os.rename(staging, report)
    finally:
        if staging.exists():
            shutil.rmtree(staging)  # exact newly-created staging directory, never a run or existing report


def export_and_publish(run: Path, jobs: list[dict], *, no_push: bool):
    report = ROOT / "reports/retacred_reference_suite" / run.name
    publication_preflight(no_push=no_push, allowed_report=report)
    if not (run/"RUN_COMPLETE").is_file():
        raise ValueError("parent completion gate absent")
    from .orchestrate import validate_job
    for job in jobs:
        validate_job(job["directory"])
    exchange = upload_evidence(run,jobs)
    export_report(run,jobs,exchange,report)
    relative = report.relative_to(ROOT).as_posix()
    subprocess.run(["git","add","--",relative],cwd=ROOT,check=True)
    staged = git("diff","--cached","--name-only").splitlines()
    if any(not name.startswith(relative+"/") for name in staged):
        raise ValueError("staged paths outside the exact report whitelist")
    if staged:
        subprocess.run(["git","commit","-m",f"reports: Re-TACRED references {run.name}"],cwd=ROOT,check=True)
    atomic_json(run/"publication.json", {"status":"committed", "report":relative,"commit":git("rev-parse","HEAD")})
    if not no_push:
        subprocess.run(["git","-c","http.version=HTTP/1.1","push","origin","HEAD:1.1"],cwd=ROOT,check=True)
        atomic_json(run/"publication.json", {"status":"pushed", "report":relative,"commit":git("rev-parse","HEAD")})
    print(f"[references] {'committed (push disabled)' if no_push else 'exported, committed and pushed'}: {relative}",flush=True)
