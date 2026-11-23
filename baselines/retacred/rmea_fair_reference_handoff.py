"""Thin collaborator CLI around the unchanged shared training and report APIs."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from types import SimpleNamespace
from datetime import datetime

REVIEW='docs/evidence/rmea/20261124-fair-reference/handoff-review.json'
TRUST_ROOT=Path('/etc/q-attention/owner-handoffs')
DEFAULT_BINDING=str(TRUST_ROOT/'fair-reference.json')
ATTEMPTS_ROOT=Path('/hy-tmp/Q-Attention/owner/attempts')
CONTROL_ROOT=Path('/hy-tmp/Q-Attention/control/full-protocol-fair-reference-20261124')


def require(ok,message):
    if not ok:raise ValueError(message)


def compact(p,path):
    path=Path(path)
    require(not path.is_symlink() and path.is_file() and path.stat().st_size<2*1024**2,
            'Expected regular compact handoff file: '+str(path))
    return p.read(path)


def owner_controlled(path):
    info=path.lstat()
    return info.st_uid==0 and info.st_mode&0o022==0 and not path.is_symlink()


def trusted_binding(path):
    path=Path(path).absolute();anchor=TRUST_ROOT.absolute()
    require(path.parent==anchor and re.fullmatch(r'[A-Za-z0-9_-]+\.json',path.name) is not None,
            'Binding must be installed in the owner-controlled handoff directory')
    require(all(owner_controlled(p) for p in (path,*path.parents)),
            'Binding and its ancestors require trusted root ownership, no writable group/other or symlinks')
    return path


def publication_protocol(p):
    from .rmea_token_transport_recovery import run_lock
    values=dict(vars(p));values.update(REPOSITORY_ORIGIN='https://github.com/tivilou/Q-Attention.git',
        REPORT_SUBDIRECTORY='q_rmea_fair_reference',lock=run_lock)
    return SimpleNamespace(**values)


def gpu_profile(assigned):
    from .rmea_token_transport_parallel import devices
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free',
                                 '--format=csv,noheader,nounits'],text=True)
    free={u.strip():int(m.strip())*1024**2 for u,m in (row.split(',') for row in raw.splitlines())}
    selected=devices('auto',assigned,[free[u] for u in assigned])
    weakest=min(free[u] for u in selected)
    return dict(profile='high' if weakest>=22*1024**3 else 'balanced' if weakest>=12*1024**3 else 'low-memory',
                eligible_gpu_uuids=selected,assigned_gpu_uuids=assigned,weakest_free_bytes=weakest,
                scientific_config_unchanged=True)


def binding_context(study,path,action):
    p=study.p;b=compact(p,trusted_binding(path));root=Path(p.ROOT).resolve()
    require(b.get('schema')=='q-rmea.fair-reference.execution-binding.v1'
            and b.get('formal_executor')=='collaborator','Prepared collaborator binding required')
    require(Path(b['source']).resolve()==root,'Binding belongs to another checkout')
    a=b['allocation'];attempt=a['runner_attempt_id']
    require(a['project_id']=='q-attention' and re.fullmatch(r'[0-9a-f]{32}',attempt) is not None
            and re.fullmatch(r'fair-reference-seed13-[A-Za-z0-9_-]+',a['run_id']) is not None,
            'Q-Attention fair-reference allocation required')
    require(re.fullmatch(r'[0-9]{8}T[0-9]{6}Z_seed13',b['report_id']) is not None,'Startup report identity differs')
    datetime.strptime(b['report_id'],'%Y%m%dT%H%M%SZ_seed13')
    output=ATTEMPTS_ROOT/attempt/'output'
    run_root=output/b['report_id'];private_run=run_root/a['run_id'];discovery=private_run/'discovery'
    require(all(not path.is_symlink() and path.resolve()==path.absolute()
                for path in (output.parent,output,run_root,private_run,discovery)),
            'Attempt output hierarchy must not redirect to another directory')
    require(not run_root.resolve().is_relative_to(root),'Private run must remain outside Git')
    require(b['source_sha256']==p.identity(p.sources())
            and b['config_sha256']==p.identity(p.config('discovery')),'Binding scientific source/config differs')
    require(p.git('merge-base','--is-ancestor',b['revision'],'HEAD')=='','Training revision is not in checkout')
    args=SimpleNamespace(output=run_root,run_id=a['run_id'],inputs=Path(b['inputs']),
        unit_output=Path(b['unit_output']),admission=Path(b['admission']),admission_sha256=b['admission_sha256'])
    if action!='report':
        require(p.git('rev-parse','HEAD')==b['revision'],'Prepared training revision changed')
        permit=compact(p,a['permit_path'])
        require(p.digest(Path(a['permit_path']))==a['permit_sha256'],'Allocation permit changed')
        keys=('allocation_id','generation','project_id','host_id','run_id','runner_attempt_id','gpu_uuids')
        require(all(permit[k]==a[k] for k in keys),'Allocation identity differs')
        state=compact(p,Path(a['permit_path']).parent/'state.json')
        require(state.get('state')=='unknown' and state.get('allocation_only') is True,
                'Allocation has returned or changed')
        attachment=b['publisher_attachment']
        require(attachment['attempt']==attempt and attachment['state'] in ('attached','verified')
                and attachment['source_sha256']==b['source_sha256']
                and attachment['config_sha256']==b['config_sha256'],
                'Prepared owner publisher attachment differs')
        require(p.digest(args.admission)==args.admission_sha256,'Admission file changed')
        require(p.digest(args.inputs/'manifest.json')==p.BASE_MANIFEST,'Fixed input manifest differs')
        gpu_uuids=a['gpu_uuids']
        require(gpu_uuids and len(set(gpu_uuids))==len(gpu_uuids)
                and all(re.fullmatch(r'GPU-[0-9a-fA-F-]+',u) for u in gpu_uuids),'Assigned GPU identities required')
        args.execution_profile=gpu_profile(gpu_uuids)
        os.environ.update(CUDA_VISIBLE_DEVICES=','.join(args.execution_profile['eligible_gpu_uuids']),QATTN_COLLABORATOR_FORMAL='1',
            RESEARCH_OUTPUT_DIR=str(output),QATTN_PROGRESS_PATH=str(output/'progress.json'))
        p.review_gate(action='run')
        study.runtime.verify_admission(args,study.execution_environment())
    return b,args,discovery



def resident_context(p,binding,binding_path,args):
    a=binding['allocation'];attempt=a['runner_attempt_id']
    control=CONTROL_ROOT/('formal-'+attempt)
    resident=binding['resident'];script=control/'full_host_run.py';launch=control/'full-launch.json'
    require(str(script)==resident['path'] and str(launch)==resident['launch_path'],
            'Resident control namespace differs')
    require(all(not path.is_symlink() and path.resolve()==path.absolute() for path in (control,script,launch)),
            'Resident control paths must not redirect')
    require(p.digest(script)==resident['sha256'] and p.digest(launch)==resident['launch_sha256'],
            'Prepared resident code/launch changed')
    require(len(a['gpu_uuids'])==2,'Prepared resident requires the exact two-card allocation')
    require(Path(binding_path).absolute()==TRUST_ROOT/'fair-reference.json','Resident requires the fixed fair-reference binding filename')
    v=compact(p,launch)
    require(type(v.get('resume',False)) is bool and type(v.get('restore_pin',False)) is bool
            and v.get('resume',False)==args.resume and v.get('restore_pin',False)==args.restore_pin,
            'Prepared resume/restore fields must match the explicit requested action')
    expected=[sys.executable,'-B','-m','baselines.retacred.rmea_fair_reference_handoff',
              'run','--binding',str(trusted_binding(binding_path)),'--resident-child']
    if args.resume:expected.append('--resume')
    if args.restore_pin:expected.append('--restore-pin')
    output=ATTEMPTS_ROOT/attempt/'output'/binding['report_id']
    require(v['experiment']=='fair-reference' and v['allocation']==a and v['argv']==expected
            and v['python']==sys.executable and v['source']==str(Path(p.ROOT).resolve())
            and v['revision']==binding['revision'] and v['output']==str(output)
            and v['run_root']==str(output/a['run_id'])
            and v['identity']['source']==binding['source_sha256']
            and v['identity']['config']==binding['config_sha256'],
            'Resident launch does not match the collaborator binding/explicit action')
    return control,resident


def start_resident(p,binding,binding_path,args):
    control,resident=resident_context(p,binding,binding_path,args)
    attempt=binding['allocation']['runner_attempt_id'];script=control/'full_host_run.py'
    marker=control/'resident-start.json'
    require(not marker.exists() and not (control/'resident-claim.json').exists(),
            'Resident already started; retain its state and request explicit recovery')
    with (control/'resident.stdout.log').open('xb') as stdout,(control/'resident.stderr.log').open('xb') as stderr:
        proc=subprocess.Popen([sys.executable,'-B',str(script),'--launch-sha256',resident['launch_sha256']],
            cwd=p.ROOT,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True)
    deadline=time.monotonic()+5
    while time.monotonic()<deadline:
        if marker.exists():
            row=compact(p,marker)
            require(row['pid']==proc.pid and row['attempt']==attempt
                    and row['launch_sha256']==resident['launch_sha256'], 'Resident startup belongs to another process')
            return dict(status='STARTED',resident_pid=proc.pid,attempt=attempt,
                        report_id=binding['report_id'],control=str(control),formal_executor='collaborator')
        require(proc.poll() is None,'Resident exited before startup; retain control logs')
        time.sleep(.05)
    raise RuntimeError('Resident startup not confirmed; inspect the existing process, do not relaunch blindly')



def resident_child_guard(p,binding):
    attempt=binding['allocation']['runner_attempt_id'];control=CONTROL_ROOT/('formal-'+attempt)
    row=compact(p,control/'resident-start.json');parent=os.getppid()
    require(row['pid']==parent and row['attempt']==attempt
            and row['launch_sha256']==binding['resident']['launch_sha256']
            and row['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            and row['owner_start_ticks']==Path('/proc',str(parent),'stat').read_text().split(') ')[1].split()[19]
            and os.environ.get('QATTN_RESIDENT_ATTEMPT')==attempt,
            'Resident child parent/start identity differs')


def main(argv=None):
    from .rmea_experiments import build
    from .rmea_report_git import branch_gate,publish_report,origin_gate
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',nargs='?',default='run',choices=('check','run','report'))
    parser.add_argument('--binding',type=Path,default=Path(os.environ.get('QATTN_HANDOFF_BINDING',DEFAULT_BINDING)))
    parser.add_argument('--resident-child',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--restore-pin',action='store_true')
    args=parser.parse_args(argv)
    require(not args.resident_child or args.action=='run','Resident child must be a run')
    require(not args.restore_pin or (args.action in ('run','check') and args.resume),'Pinned restore requires an explicit run/check --resume')
    require(not args.resume or args.action in ('run','check'),'Resume applies only to run/check')
    study=build('fair-reference');p=study.p;publication=publication_protocol(p)
    review=compact(p,Path(p.ROOT)/REVIEW)
    require(review.get('verdict')=='PASS' and review.get('review_scope')=='lifecycle'
            and not review.get('must_fix'),'Exact handoff lifecycle review required')
    required={'baselines/retacred/rmea_fair_reference_handoff.py',
              'baselines/retacred/rmea_report_git.py','scripts/run_q_rmea_fair_reference.sh'}
    require(set(review['source_hashes'])==required
            and all(not (Path(p.ROOT)/n).is_symlink() and p.digest(Path(p.ROOT)/n)==h
                    for n,h in review['source_hashes'].items()),'Handoff source revision differs')
    if args.action!='report':
        origin_gate(publication)
        p.git('fetch','--no-tags','origin','+refs/heads/main:refs/remotes/origin/main',
              '+refs/heads/1.1:refs/remotes/origin/1.1')
        branch_gate(publication,set())
    binding,run,output=binding_context(study,args.binding,args.action)
    if args.action!='report':
        require(binding['resident']['sha256']==review['resident_sha256'],'Resident differs from independent review')
        resident_context(p,binding,args.binding,args)
    if args.action=='check':
        print(json.dumps(dict(status='PREPARED',cloud_protection='First optimizer update waits for authenticated COS preparation ACK',experiment=p.EXPERIMENT,report_id=binding['report_id'],
                             formal_executor='collaborator',training_started=False,
                             execution_profile=run.execution_profile)));return
    if args.action=='run' and not args.resident_child:
        print(json.dumps(start_resident(p,binding,args.binding,args)));return
    if args.action=='run':
        resident_child_guard(p,binding)
        run.output.mkdir(exist_ok=True)
        control=Path('/hy-tmp/Q-Attention/control/full-protocol-fair-reference-20261124')/(
            'formal-'+binding['allocation']['runner_attempt_id'])
        p.atomic_json(control/'execution-profile.json',run.execution_profile)
        command=['run','--output',str(run.output),'--inputs',str(run.inputs),
            '--run-id',run.run_id,'--unit-output',str(run.unit_output),
            '--admission',str(run.admission),'--admission-sha256',run.admission_sha256]
        if args.resume:command.append('--resume')
        if args.restore_pin:command.append('--restore-pin')
        study.main(command)
    require(compact(p,output/'manifest.json')['git_revision']==binding['revision'],
            'Completed discovery does not belong to the bound training revision')
    print(json.dumps(publish_report(publication,study.runtime,output,run.output)))


if __name__=='__main__':main()
