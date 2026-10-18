"""Verified private trace exchange and atomic, report-only publication."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import time

from q_attention.analysis.ecpa_formal_trace import validate
from q_attention.experiments import ecpa_protocol as p

PUBLIC = ("run_summary.json","run_summary.data","run_summary.md","run_config.json","data.sha256","data_counts.txt")
PART_BYTES = 64*1024**2


def audit(run):
    run = Path(run).resolve()
    config,manifest,marker = (p.read(run/name) for name in ("run_config.json","run_manifest.json","RUN_COMPLETE.json"))
    if marker != dict(fingerprint=manifest["fingerprint"],summary_sha256=p.sha(run/"run_summary.json")):
        raise ValueError("immutable completed summary marker mismatch")
    summary = p.read(run/"run_summary.json")
    if summary["fingerprint"] != manifest["fingerprint"] or p.read(run/"run_summary.data") != summary:
        raise ValueError("summary identity mismatch")
    if config != p.load_config(config["profile"]=="synthetic_canary") or config["selectors"] != list(p.MODES):
        raise ValueError("report contract differs from frozen configuration")
    if summary["test_used_for_training_or_selection"] is not False:
        raise ValueError("test leakage gate failed")
    for mode in ("baseline",*p.MODES):
        directory = run/"jobs"/mode
        complete,metrics = p.read(directory/"RUN_COMPLETE.json"),p.read(directory/"metrics.json")
        if (complete["fingerprint"] != manifest["fingerprint"] or complete["metrics_sha256"] != p.sha(directory/"metrics.json")
                or complete["trace_sha256"] != p.sha(directory/"sample_trace.json")):
            raise ValueError(f"incomplete/damaged job: {mode}")
        if metrics["fingerprint"] != manifest["fingerprint"] or metrics["test_used_for_training_or_selection"] is not False:
            raise ValueError("job identity/leakage gate failed")
        if mode != "baseline" and summary["rows"][mode] != metrics:
            raise ValueError("summary differs from selected job metrics")
        validate(directory/"sample_trace.json",mode,config)
    for name in PUBLIC:
        if not (run/name).is_file() or not (run/name).stat().st_size:
            raise ValueError(f"missing nonempty report identity file: {name}")
    return config,manifest


def retry(operation, attempts=3):
    for attempt in range(attempts):
        try:
            return operation()
        except Exception:
            if attempt+1 == attempts:
                raise
            time.sleep(attempt+1)


def exchange(run, manifest):
    """Reuse hash-verified part receipts; never delete the source trace values."""
    base_url,token = os.environ.get("PROJECT_EXCHANGE_URL"),os.environ.get("PROJECT_EXCHANGE_TOKEN")
    if not base_url or not token:
        raise ValueError("artifact exchange environment is not configured")
    spec = importlib.util.spec_from_file_location("ecpa_exchange",p.ROOT/"scripts/collab/lib/exchange_upload.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    scheme,*_ = helper.parse_base_url(base_url)
    if scheme != "https":
        raise ValueError("formal private evidence requires HTTPS")
    context = helper.ssl_context_for(scheme,helper.DEFAULT_CA_FILE)
    target = helper.safe_relative(f"q-attention/{p.EXPERIMENT}/{manifest['run_id']}",label="target")
    retry(lambda:helper.create_directory(base_url,token,target,context=context))
    cache = run/"export_state"
    cache.mkdir(exist_ok=True)
    receipt_path = cache/"exchange.json"
    receipt = p.read(receipt_path) if receipt_path.exists() else dict(fingerprint=manifest["fingerprint"],target_dir=target,parts=[])
    if receipt["fingerprint"] != manifest["fingerprint"] or receipt["target_dir"] != target:
        raise ValueError("exchange receipt belongs to a different run")
    # Deterministic uncompressed tar allows byte-identical bounded-part retries.
    # No archive on the critical training path. At most one extra case archive.
    archive = cache/"cases.tar"
    with tarfile.open(archive,"w",format=tarfile.PAX_FORMAT) as tar:
        for mode in ("baseline",*p.MODES):
            directory = run/"jobs"/mode
            files = [directory/"sample_trace.json",directory/"trace_validation.json",*sorted((directory/"values").rglob("*.npy"))]
            for path in files:
                if path.is_symlink() or not path.resolve().is_relative_to(run):
                    raise ValueError("unsafe evidence path")
                info = tar.gettarinfo(str(path),arcname=str(path.relative_to(run)))
                info.uid=info.gid=info.mtime=0
                info.uname=info.gname=""
                with path.open("rb") as stream:
                    tar.addfile(info,stream)
    archive_sha = p.sha(archive)
    if receipt.get("archive_sha256",archive_sha) != archive_sha:
        raise ValueError("completed case archive changed since previous upload")
    receipt.update(archive_sha256=archive_sha,archive_bytes=archive.stat().st_size)
    previous = {part["name"]:part for part in receipt["parts"]}
    receipt["parts"] = []
    with archive.open("rb") as stream:
        index=0
        while block:=stream.read(PART_BYTES):
            part_path=cache/f"cases.part{index:05d}"
            part_path.write_bytes(block)
            digest=p.sha(part_path)
            old=previous.get(part_path.name)
            if old is None or old["sha256"] != digest or old["size_bytes"] != len(block):
                remote=target+"/"+part_path.name
                result=retry(lambda:helper._upload_file(base_url,token,part_path,remote,len(block),digest,context=context))
                if (result.get("path"),result.get("size_bytes"),result.get("sha256")) != (remote,len(block),digest):
                    raise ValueError("unverified exchange receipt")
            receipt["parts"].append(dict(name=part_path.name,sha256=digest,size_bytes=len(block),verified=True))
            p.atomic_json(receipt_path,receipt)
            part_path.unlink()
            index+=1
    receipt["status"]="uploaded"
    p.atomic_json(receipt_path,receipt)
    # URL/token never enter receipts or public evidence. Keep all source NPYs.
    archive.unlink()
    return receipt


def export_report(run, *, exchange_fn=exchange, inject_failure=False, destination=None):
    run=Path(run).resolve()
    with p.exclusive(run/".export.lock"):
        config,manifest=audit(run)
        final=Path(destination or p.ROOT/"reports"/p.EXPERIMENT/run.name).resolve()
        if destination is not None and config["profile"] != "synthetic_canary":
            raise ValueError("custom export destination is canary-only")
        if config["profile"] != "synthetic_canary":
            if exchange_fn is not exchange or inject_failure:
                raise ValueError("injected export behavior is canary-only")
            publication_preflight(final)
        final.parent.mkdir(parents=True,exist_ok=True)
        stage=Path(tempfile.mkdtemp(prefix="ecpa-export-"))  # outside Git
        try:
            receipt=exchange_fn(run,manifest)
            if receipt.get("status") != "uploaded" or not receipt.get("parts") or not all(x.get("verified") for x in receipt["parts"]):
                raise ValueError("private full-value evidence has no verified exchange receipt")
            for name in PUBLIC:
                shutil.copy2(run/name,stage/name)
            safe_manifest={key:manifest[key] for key in ("schema_version","run_id","git_revision","fingerprint","source_sha256","source_files","data","environment","profile")}
            p.atomic_json(stage/"run_manifest.json",safe_manifest)
            p.atomic_json(stage/"exchange_manifest.json",receipt)
            cases={mode:dict(validation=p.read(run/"jobs"/mode/"trace_validation.json"),
                trace_sha256=p.sha(run/"jobs"/mode/"sample_trace.json"),
                evidence_path=f"jobs/{mode}/sample_trace.json") for mode in ("baseline",*p.MODES)}
            p.atomic_json(stage/"case_study.json",dict(schema_version="ecpa-private-case-index.v1",trace_completeness="complete",
                evidence_location="authenticated project exchange; concatenate ordered parts and open tar",numeric_values="complete lossless private NPY",
                public_preview="not_included",jobs=cases))
            if inject_failure:
                raise RuntimeError("injected export validation failure")
            files={path.name:p.sha(path) for path in sorted(stage.iterdir())}
            report_identity=dict(schema_version="ecpa-report.v1",fingerprint=manifest["fingerprint"],files=files)
            p.atomic_json(stage/"REPORT_COMPLETE.json",report_identity)
            if final.exists():
                if p.read(final/"REPORT_COMPLETE.json") != report_identity or {x.name for x in final.iterdir()} != {*files,"REPORT_COMPLETE.json"}:
                    raise ValueError("existing report identity/whitelist differs; refuse overwrite")
                if any(p.sha(final/name)!=digest for name,digest in files.items()):
                    raise ValueError("existing report files changed")
            else:
                # Cross-filesystem safe staging uses a sibling only AFTER all
                # validation. Clean it on any copy/rename failure.
                with tempfile.TemporaryDirectory(prefix=".ecpa-stage-",dir=final.parent) as sibling:
                    content=Path(sibling)/"report"
                    shutil.copytree(stage,content)
                    os.replace(content,final)
            p.atomic_json(run/"EXPORT_COMPLETE.json",dict(report=str(final),identity=p.identity(report_identity)))
            (run/"EXPORT_FAILED.json").unlink(missing_ok=True)
            return final
        except Exception as exc:
            p.atomic_json(run/"EXPORT_FAILED.json",dict(stage="export",error_type=type(exc).__name__,retry="--export-only --resume same_run; no training"))
            raise
        finally:
            shutil.rmtree(stage)


def publication_preflight(report):
    relative=str(Path(report).resolve().relative_to(p.ROOT))
    def git(*args, **kwargs):
        return subprocess.run(["git","-c","http.version=HTTP/1.1",*args],cwd=p.ROOT,text=True,capture_output=True,timeout=90,check=True,**kwargs).stdout.strip()
    if git("branch","--show-current") != "1.1":
        raise ValueError("report publication requires collaborator branch 1.1")
    for upstream in ("origin/1.1","origin/main"):
        git("merge-base","--is-ancestor",upstream,"HEAD")
    # Clean status alone does not prevent pushing unrelated local history.
    # Public main/1.1 ancestry may be merged; every remaining unpublished
    # commit must modify only this report, including merge commit resolution.
    outgoing=git("rev-list","HEAD","--not","origin/1.1","origin/main").splitlines()
    for commit in outgoing:
        parents=git("rev-list","--parents","-n","1",commit).split()[1:]
        if len(parents)>1:
            if len(parents)!=2:
                raise ValueError("unsupported unpublished octopus merge")
            # Replay the public merge in a disposable shared clone. This works
            # on Git 2.34 too and distinguishes clean same-file hunk merges
            # from new, unrelated conflict resolutions or merge edits.
            with tempfile.TemporaryDirectory(prefix="ecpa-merge-audit-") as temporary:
                clone=Path(temporary)/"repo"
                git("clone","--shared","--no-checkout","--quiet",str(p.ROOT),str(clone))
                def probe(*args):
                    return git("-C",str(clone),"-c","core.hooksPath=/dev/null",
                        "-c","user.name="+git("config","user.name"),"-c","user.email="+git("config","user.email"),*args)
                probe("checkout","--quiet","--detach",parents[0])
                probe("merge","--no-commit","--no-ff",parents[1])
                expected=probe("write-tree")
                paths=probe("diff-tree","--no-commit-id","--name-only","-r","-z",expected,commit)
        else:
            paths=git("diff-tree","--root","--no-commit-id","--name-only","-r","-z",commit)
        if any(not name.startswith(relative+"/") for name in filter(None,paths.split("\0"))):
            raise ValueError("unrelated unpublished commit blocks report-only publication")
    # -z and explicit pathspec avoid quoting ambiguities, unrelated staged edits.
    for command in (("diff","--cached","--name-only","-z"),("ls-files","--modified","--others","--exclude-standard","-z")):
        for name in filter(None,git(*command).split("\0")):
            if not name.startswith(relative+"/"):
                raise ValueError("unrelated changes block report-only publication")
    return git,relative


def publish(report):
    git,relative=publication_preflight(report)
    git("add","--",relative)
    changed=git("diff","--cached","--name-only")
    if changed:
        git("commit","-m",f"reports: Q-ECPA seed 13 {Path(report).name}")
    retry(lambda:git("push","origin","HEAD:refs/heads/1.1"))


def finish(run, *, push=True):
    report=export_report(run)
    if push:
        try:
            publish(report)
        except Exception as exc:
            p.atomic_json(Path(run)/"PUBLISH_FAILED.json",dict(error_type=type(exc).__name__,report=str(report),retry="--export-only --resume same_run"))
            raise
        (Path(run)/"PUBLISH_FAILED.json").unlink(missing_ok=True)
    print(f"Audited report: {report}; published={push}",flush=True)
    return 0
