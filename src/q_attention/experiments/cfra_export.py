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

from q_attention.experiments.cfra_worker import validate_job as validate
from q_attention.experiments import cfra_protocol as p

PUBLIC = ("run_summary.json","run_summary.data","run_summary.md","run_config.json","data.sha256","data_counts.txt")
PART_BYTES = 64*1024**2


def audit(run):
    run=Path(run).resolve()
    config,manifest,marker=(p.read(run/name) for name in ('run_config.json','run_manifest.json','RUN_COMPLETE.json'))
    expected=dict(fingerprint=manifest['fingerprint'],summary_sha256=p.sha(run/'run_summary.json'))
    if marker!=expected:raise ValueError('completed summary marker differs')
    summary=p.read(run/'run_summary.json')
    if p.read(run/'run_summary.data')!=summary:raise ValueError('summary.data differs')
    if (run/'run_summary.md').read_text(encoding='utf-8')!=render_markdown(summary):raise ValueError('human-readable summary differs')
    rebuilt=build_summary(run)
    if summary!=rebuilt:raise ValueError('summary/gates differ from independently audited jobs')
    expected_hashes=''.join(f"{v['sha256']}  {s}.jsonl\n" for s,v in manifest['data'].items())
    expected_counts=''.join(f"{s}: {v['records']}\n" for s,v in manifest['data'].items())
    if (run/'data.sha256').read_text()!=expected_hashes or (run/'data_counts.txt').read_text()!=expected_counts:
        raise ValueError('public population receipt differs')
    for name in PUBLIC:
        if not (run/name).is_file() or not (run/name).stat().st_size:raise ValueError('missing report identity: '+name)
    return config,manifest


def build_summary(run):
    """Rebuild selection, metrics, trace and intervention gates from completed jobs."""
    import math
    import hashlib
    import statistics
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from q_attention.experiments import cfra_checkpoint as cp
    from q_attention.experiments.cfra_runner import completed
    run=Path(run);config,manifest=p.read(run/'run_config.json'),p.read(run/'run_manifest.json')
    if config!=p.load_config(config['profile']=='synthetic_canary'):raise ValueError('unfrozen config')
    if p.identity(p.source_contract())!=manifest['source_sha256']:raise ValueError('executed source changed')
    if manifest['source_files']!=p.source_contract():raise ValueError('source file manifest differs')
    if p.inspect_data(manifest['data_dir'],config)!=manifest['data']:raise ValueError('source data changed')
    if p.identity(dict(config=config,data=manifest['data'],source_sha256=manifest['source_sha256']))!=manifest['fingerprint']:
        raise ValueError('scientific fingerprint differs')
    rows={};warm_state=None;case_inputs={};diagnostics=None
    for mode in ('baseline',*p.MODES):
        directory=run/'jobs'/mode
        if not completed(run,mode,manifest):raise ValueError('incomplete job: '+mode)
        report=p.read(directory/'metrics.json');state=cp.load(directory)
        budget=config['baseline' if mode=='baseline' else 'kernel']
        total=math.ceil(manifest['data']['train']['records']/budget['batch_size'])*budget['epochs']
        if (state['fingerprint']!=manifest['fingerprint'] or state['mode']!=mode or
            state['next_update']!=total or report['updates']!=total or
            state['best_update']!=report['best_update'] or state['best_metric']!=report['reports']['valid']['metrics']['macro_f1'] or
            state['selection_update']!=total or report['test_used_for_training_or_selection'] is not False):
            raise ValueError('job checkpoint/selection/metric boundary differs')
        plugin=0 if mode in ('baseline','disabled') else 4*(config['model']['dim']+1)+18
        if report['plugin_parameters']!=plugin:raise ValueError('nominal parameter budget differs')
        if mode!='baseline':
            warm_state=warm_state or report['base_state_sha256']
            if report['base_state_sha256']!=warm_state:raise ValueError('unpaired shared warm-start')
            if mode=='disabled':classifier=report['trainable_parameters']
            if report['trainable_parameters']!=classifier+plugin:raise ValueError('unmatched classifier/plugin budget')
        for split in ('valid','test'):
            values=report['reports'][split];classes=list(values['per_class'].values())
            supports={label:manifest['data'][split]['supports'].get(label,0) for label in manifest['data']['train']['supports']}
            if ({v['label']:v['support'] for v in classes}!=supports or
                sum(v['predicted_support'] for v in classes)!=manifest['data'][split]['records']):
                raise ValueError('metric population/label support differs')
            for c in classes:
                tp,fp,fn=c['true_positive'],c['false_positive'],c['false_negative']
                if c['support']!=tp+fn or c['predicted_support']!=tp+fp:raise ValueError('invalid class counts')
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
                if not math.isclose(f1,c['f1'],abs_tol=1e-14):raise ValueError('class F1 differs from counts')
            if (not math.isclose(statistics.mean(c['f1'] for c in classes),values['metrics']['macro_f1'],abs_tol=1e-14) or
                not math.isclose(sum(c['true_positive'] for c in classes)/sum(supports.values()),values['metrics']['accuracy'],abs_tol=1e-14)):
                raise ValueError('aggregate metric differs from counts')
        readiness=validate(directory/'sample_trace.json',mode,config)
        if readiness!=p.read(directory/'trace_validation.json'):raise ValueError('trace validation receipt differs')
        trace=p.read(directory/'sample_trace.json')
        def model_hash(tensors):
            h=hashlib.sha256()
            for name,t in sorted(tensors.items()):
                h.update(name.encode())
                if mode=='baseline':h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode())
                else:h.update(str((tuple(t.shape),t.dtype)).encode())
                h.update(t.detach().cpu().contiguous().numpy().tobytes())
            return h.hexdigest()
        best=torch.load(directory/'best.pt',map_location='cpu',weights_only=False)
        if best['fingerprint']!=manifest['fingerprint'] or best['update']!=state['best_update'] or best['metric']!=state['best_metric'] or model_hash(best['model'])!=model_hash(state['best_model']):
            raise ValueError('best checkpoint differs from committed selection')
        for sample in trace['samples']:
            if sample['checkpoint'] in ('best','final'):
                chosen=state['best_model'] if sample['checkpoint']=='best' else state['model']
                if sample['roles']['checkpoint_identity']['model_state_sha256']!=model_hash(chosen):
                    raise ValueError('trace is not bound to selected/final model')
            roles={k:sample['roles'][k] for k in ('sentence','tokens','subject','object','correct_relation')}
            digest=p.identity(roles);key=(sample['split'],sample['checkpoint'],sample['lane'])
            if mode!='baseline':
                if case_inputs.setdefault(key,digest)!=digest:raise ValueError('unmatched fixed cases')
            if mode=='quantum' and sample['lane']=='training':
                reps={r['id']:r for r in sample['representations']}
                def array(name):return np.load(directory/reps[name]['artifact_refs'][0]['path'],allow_pickle=False)
                gradient=array('parameter_gradient');groups={k:[] for k in ('projection','channel','readout')};cursor=0
                for item in trace['parameter_order']:
                    part=gradient[cursor:cursor+item['numel']];cursor+=item['numel']
                    for group in groups:
                        if item['name'].startswith('kernel.'+group):groups[group].extend(part.tolist())
                if cursor!=gradient.size or not all(groups.values()):raise ValueError('incomplete mechanism gradients')
                norms={k:float(np.linalg.norm(v)) for k,v in groups.items()}
                mask=array('attention_mask').astype(bool);raw=array('base_scores').copy();raw[...,~mask]=-1e30
                reference=torch.from_numpy(raw).softmax(-1).numpy()
                shift=float(np.abs(array('attention')-reference)[...,mask,:].sum(-1).max())
                diagnostics=dict(plugin_group_gradient_norms=norms,max_valid_attention_l1_shift=shift,
                    active=min(norms.values())>1e-12 and shift>1e-8)
        rows[mode]=report
    if diagnostics is None:raise ValueError('missing actual first-update mechanism evidence')
    base=rows['disabled']['reports']['test']['metrics'][config['primary']]
    delta={m:rows[m]['reports']['test']['metrics'][config['primary']]-base for m in p.MODES}
    formal=config['profile']=='formal_seed13'
    return dict(schema_version='cfra-summary.v1',experiment=config['name'],profile=config['profile'],
        fingerprint=manifest['fingerprint'],data=manifest['data'],primary=config['primary'],rows={m:rows[m] for m in p.MODES},
        delta_vs_disabled=delta,relative_delta_vs_disabled={m:v/max(abs(base),1e-12) for m,v in delta.items()},
        mechanism_diagnostics=diagnostics,gates=dict(single_seed_L1_utility_candidate=formal and delta['quantum']>0 and diagnostics['active'],
            classical_QI_candidates={m:formal and delta[m]/max(abs(base),1e-12)>.01 for m in ('classical_cosine','classical_rbf')},
            quantum_specific_advantage_claimed=False,automatic_multiseed_authorized=False),
        control_differences={m:rows['quantum']['reports']['test']['metrics'][config['primary']]-rows[m]['reports']['test']['metrics'][config['primary']] for m in p.MODES if m!='quantum'},
        claim_ceiling='Owner audit required; formal single-seed utility candidate at most; controls only nominal-budget matched; no L2/novelty/quantum advantage.',
        test_used_for_training_or_selection=False)

def render_markdown(value):
    return '# Q-CFRA seed13\n\n'+ '\n'.join(
        f"- {mode}: test Macro-F1 {row['reports']['test']['metrics']['macro_f1']:.9f}; delta {value['delta_vs_disabled'][mode]:+.9f}"
        for mode,row in value['rows'].items())+'\n\n'+value['claim_ceiling']+'\n'


def summarize(run):
    run=Path(run);value=build_summary(run)
    for receipt in (run/'jobs').glob('*/completed_checkpoint_recovery.json'):
        old=p.read(receipt)['old_summary']
        if old is not None and old!=value:raise ValueError('completed repair changed frozen summary; stop for owner audit')
    p.atomic_json(run/'run_summary.json',value);p.atomic_json(run/'run_summary.data',value)
    (run/'run_summary.md').write_text(render_markdown(value),encoding='utf-8')
    p.atomic_json(run/'RUN_COMPLETE.json',dict(fingerprint=value['fingerprint'],summary_sha256=p.sha(run/'run_summary.json')))
    return value

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
    spec = importlib.util.spec_from_file_location("cfra_exchange",p.ROOT/"scripts/collab/lib/exchange_upload.py")
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
        attempt_path=run/'export_state/attempts.json'
        history=p.read(attempt_path) if attempt_path.exists() else []
        attempt=dict(attempt=len(history)+1,source_revision=manifest['git_revision'],source_sha256=manifest['source_sha256'],
            exporter_sha256=p.sha(Path(__file__)),status='started')
        history.append(attempt);p.atomic_json(attempt_path,history)
        final=Path(destination or p.ROOT/"reports"/p.EXPERIMENT/run.name).resolve()
        if destination is not None and config["profile"] != "synthetic_canary":
            raise ValueError("custom export destination is canary-only")
        if config["profile"] != "synthetic_canary":
            if exchange_fn is not exchange or inject_failure:
                raise ValueError("injected export behavior is canary-only")
            publication_preflight(final)
        final.parent.mkdir(parents=True,exist_ok=True)
        stage=Path(tempfile.mkdtemp(prefix="cfra-export-"))  # outside Git
        try:
            receipt=exchange_fn(run,manifest)
            if receipt.get("status") != "uploaded" or not receipt.get("parts") or not all(x.get("verified") for x in receipt["parts"]):
                raise ValueError("private full-value evidence has no verified exchange receipt")
            for name in PUBLIC:
                shutil.copy2(run/name,stage/name)
            safe_manifest={key:manifest[key] for key in ("schema_version","run_id","git_revision","fingerprint","source_sha256","source_files","data","environment","profile","execution")}
            p.atomic_json(stage/"run_manifest.json",safe_manifest)
            p.atomic_json(stage/"exchange_manifest.json",receipt)
            cases={mode:dict(validation=p.read(run/"jobs"/mode/"trace_validation.json"),
                trace_sha256=p.sha(run/"jobs"/mode/"sample_trace.json"),
                evidence_path=f"jobs/{mode}/sample_trace.json") for mode in ("baseline",*p.MODES)}
            p.atomic_json(stage/"case_study.json",dict(schema_version="cfra-private-case-index.v1",trace_completeness="complete",
                evidence_location="authenticated project exchange; concatenate ordered parts and open tar",numeric_values="complete lossless private NPY",
                public_preview="not_included",jobs=cases))
            if inject_failure:
                raise RuntimeError("injected export validation failure")
            files={path.name:p.sha(path) for path in sorted(stage.iterdir())}
            report_identity=dict(schema_version="cfra-report.v1",fingerprint=manifest["fingerprint"],files=files)
            p.atomic_json(stage/"REPORT_COMPLETE.json",report_identity)
            if final.exists():
                if p.read(final/"REPORT_COMPLETE.json") != report_identity or {x.name for x in final.iterdir()} != {*files,"REPORT_COMPLETE.json"}:
                    raise ValueError("existing report identity/whitelist differs; refuse overwrite")
                if any(p.sha(final/name)!=digest for name,digest in files.items()):
                    raise ValueError("existing report files changed")
            else:
                # Cross-filesystem safe staging uses a sibling only AFTER all
                # validation. Clean it on any copy/rename failure.
                with tempfile.TemporaryDirectory(prefix=".cfra-stage-",dir=final.parent) as sibling:
                    content=Path(sibling)/"report"
                    shutil.copytree(stage,content)
                    os.replace(content,final)
            attempt.update(status='complete',report_identity=p.identity(report_identity));p.atomic_json(attempt_path,history)
            p.atomic_json(run/"EXPORT_COMPLETE.json",dict(report=str(final),identity=p.identity(report_identity),
                source_revision=manifest['git_revision'],exporter_sha256=attempt['exporter_sha256'],attempt=attempt['attempt'],retry_count=attempt['attempt']-1))
            (run/"EXPORT_FAILED.json").unlink(missing_ok=True)
            return final
        except Exception as exc:
            attempt.update(status='failed',error_type=type(exc).__name__);p.atomic_json(attempt_path,history)
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
            with tempfile.TemporaryDirectory(prefix="cfra-merge-audit-") as temporary:
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
        git("commit","-m",f"reports: Q-CFRA seed 13 {Path(report).name}")
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
