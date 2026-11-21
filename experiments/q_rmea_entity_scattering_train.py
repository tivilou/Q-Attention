"""Owner-managed single-seed experiment using existing parallel/recovery components."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from baselines.retacred import rmea_entity_scattering_protocol as p
from baselines.retacred import rmea_entity_scattering_runtime as runtime

INPUT_RUN=Path('/hy-tmp/Q-Attention/worker/attempts/6a2bf35b8d9d4ac4a6b4fe2acb6bc116/output/20261121T034306Z_seed13/discovery')


def require(ok,message):
    if not ok:raise ValueError(message)


def execution_environment():
    import platform,numpy,torch
    return dict(python=platform.python_version(),implementation=platform.python_implementation(),numpy=numpy.__version__,
        torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
        torch_build_sha256=hashlib.sha256(torch.__config__.show().encode()).hexdigest(),cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))


def metadata_temporary(f,root):
    """Interrupted atomic metadata writes are non-authoritative and kept as evidence."""
    import re
    rel=f.relative_to(root).as_posix()
    names=('run-identity.json','.preparing.json','handoff.json','discovery/manifest.json','recovery-inputs/input-recovery.json')
    return not f.is_symlink() and f.is_file() and f.stat().st_size<=2*1024**2 and any(re.fullmatch(re.escape(n)+r'\.[a-z0-9_]{8}',rel) for n in names)


def fresh_attempt(folder):
    require(folder.is_dir(),'Existing scheduler output directory required')
    names={f.name for f in folder.iterdir() if f.name!='.run.lock' and not metadata_temporary(f,folder)}
    if not names:return
    require(names=={'progress.json'},'Preserve existing attempt output')
    row=p.read(folder/'progress.json');expected=Path(os.environ['RESEARCH_OUTPUT_DIR']).parent.name
    require(row.get('schema')=='research.progress.v1' and row.get('attempt')==expected and row.get('phase')=='experiment'
        and type(row.get('current')) is int and row['current']==0,'Existing progress is not an initial scheduler record')


def sync_directory(directory):
    fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def verify_installed_pin(folder,root,profile):
    phase=p.read(folder/'restore-install.json');pin=p.read(root/'restore-pin.json')
    require(phase['status']=='complete' and phase['pin_sha256']==p.digest(root/'restore-pin.json') and phase['run_id']==root.name
        and phase['profile']==profile and phase['source_sha256']==p.identity(p.sources()),'Pinned installation is incomplete or different')
    for row in pin['acknowledged_generations']:
        directory=root/'cos-channel'/row['request']['generation']
        require(p.read(directory/'request.json')==row['request'] and p.read(directory/'ack.json')==row['ack']
            and p.digest(directory/'request.json')==row['ack']['request_sha256'],'Installed pinned prefix differs; use --restore-pin')
    pointer=p.read(root/'cos-channel/cloud-latest.json')
    require(pointer['done']>=pin['done'],'Installed cloud pointer precedes pin')
    if pointer['generation']==pin['generation']:
        require(pointer['request_sha256']==pin['acknowledged_generations'][-1]['ack']['request_sha256'],'Pinned cloud pointer differs')


def durable_inputs(root):
    """One preparation/restore barrier, including linked or extracted inputs."""
    for f in root.rglob('*'):
        require(not f.is_symlink(),'Regular recovery inputs required')
        if f.is_file():
            with f.open('rb') as handle:os.fsync(handle.fileno())
    for d in sorted((f for f in root.rglob('*') if f.is_dir()),key=lambda f:len(f.parts),reverse=True):sync_directory(d)
    sync_directory(root);sync_directory(root.parent)


def restore_identity(folder,root,profile,*,inject_failure=False):
    """Same identity admission for the formal pinned restore and its tiny canary."""
    from baselines.retacred.rmea_token_transport_parallel import pin_receipts,install_pinned_workers
    output=root/'discovery';m=prepared(output);cfg=p.config(profile);handoff=p.read(root/'handoff.json');pin=p.read(root/'restore-pin.json')
    require(m['config']==cfg and handoff['run_id']==root.name and handoff['profile']==profile
        and handoff['config']==cfg and handoff['source_sha256']==p.identity(p.sources()),'Restored run/source/profile differs')
    total=len(p.MODES)*cfg['epochs']*((cfg['counts']['train']+31)//32)
    latest=output/'checkpoints/latest.pt';best=output/'checkpoints/best.pt'
    receipts=pin_receipts(pin,run_id=root.name,identity=dict(config=p.identity(cfg),source=p.identity(p.sources())),total=total,
        sha256=p.digest(latest),latest_bytes=latest.stat().st_size,best_sha256=p.digest(best),best_bytes=best.stat().st_size)
    intent=dict(run_id=root.name,profile=profile,source_sha256=p.identity(p.sources()),config_sha256=p.identity(cfg))
    phase=folder/'restore-install.json';installation=dict(intent,pin_sha256=p.digest(root/'restore-pin.json'),status='pending')
    if phase.exists():require(all(p.read(phase).get(k)==value for k,value in installation.items() if k!='status'),'Different interrupted restore installation')
    else:p.atomic_json(phase,installation)
    for index,row in enumerate(receipts):
        directory=root/'cos-channel'/row['request']['generation']
        for name,key in (('request.json','request'),('ack.json','ack')):
            f=directory/name
            if f.exists():require(p.read(f)==row[key],'Different interrupted COS prefix')
            else:p.atomic_json(f,row[key])
        require(p.digest(directory/'request.json')==row['ack']['request_sha256'],'Installed request bytes differ')
        sync_directory(directory);sync_directory(directory.parent);sync_directory(root)
        if inject_failure and index==0:raise OSError('Injected partially installed pinned receipt interruption')
    last=receipts[-1]
    for f in (latest,best):
        fd=os.open(f,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    sync_directory(latest.parent);sync_directory(output)
    p.atomic_json(root/'cos-channel/cloud-latest.json',dict(generation=last['request']['generation'],done=last['request']['done'],
        index_commit=last['ack']['index_commit'],request_sha256=last['ack']['request_sha256']))
    sync_directory(root/'cos-channel');sync_directory(root)
    install_pinned_workers(output,cfg,p)
    durable_inputs(root)
    p.atomic_json(phase,dict(installation,status='complete'))
    sync_directory(folder)
    f=folder/'run-identity.json'
    if f.exists():require(p.read(f)==intent,'Existing outer run identity differs')
    else:p.atomic_json(f,intent)
    sync_directory(folder);verify_installed_pin(folder,root,profile)
    return intent


def prepared(output):
    """Authenticate non-checkpoint inputs again before local or pinned continuation."""
    output=Path(output);root=output.parent;m=p.read(output/'manifest.json');cfg=m['config']
    profile=cfg['profile'];require(cfg==p.config(profile),'Frozen configuration differs')
    require(m['execution_environment']==execution_environment(),'Pinned execution environment differs')
    source=root/'recovery-inputs/source-manifest.json';original=root/'recovery-inputs/original-manifest.json'
    require(p.digest(original)==p.BASE_MANIFEST and p.digest(source)==(p.BASE_MANIFEST if profile=='discovery' else p.CANARY_MANIFEST),'Pinned source manifest differs')
    names=['metadata.json']+[s+'-'+k+'.npy' for s in cfg['counts'] for k in ('h','roles','offsets','targets')]
    cache={n:p.read(source)['cache'][n] for n in names}
    require(m['cache']==cache and m['source_sha256']==p.identity(p.sources()) and m['source_hashes']==p.sources()
        and m['git_revision']==p.git('rev-parse','HEAD') and m['source_input_manifest']==p.BASE_MANIFEST,'Training source/cache provenance differs')
    require(p.digest(root/'recovery-inputs/labels.json')==p.LABELS_SHA,'Pinned labels differ')
    summary=ROOT/'reports/q_rmea_token_transport/20261121T034306Z_seed13/summary.json'
    require(p.digest(summary)==p.BASE_SUMMARY,'Pinned public reference differs');old=p.read(summary)
    expected=dict(summary_sha256=p.BASE_SUMMARY,result=old['results']['linear_coherent'],initial_state_sha256=old['common_initial_state_sha256'],
        order_sha256=old['common_epoch_order_sha256'],comparison='Pinned completed full-data linear baseline; not retrained')
    require(m['reference']==expected,'Historical reference metrics differ')
    ref=root/'recovery-inputs/input-recovery.json';d=p.read(ref);dep=d['dependency']
    raw=(json.dumps(dep['request'],ensure_ascii=False,indent=2)+'\n').encode()
    require(hashlib.sha256(raw).hexdigest()==p.INPUT_REQUEST_SHA and dep['ack']['request_sha256']==p.INPUT_REQUEST_SHA
        and p.identity(dep['ack'])==p.INPUT_ACK_ID and set(dep['request']['files'])==set(dep['ack']['objects'])
        and d['cache']==cache and d['source_manifest_sha256']==p.BASE_MANIFEST
        and m['input_recovery_reference_sha256']==p.digest(ref),'Immutable input dependency differs')
    require(set(x.name for x in (output/'cache').iterdir())==set(names),'Minimal cache inventory differs')
    return m


def initialize(root,profile,inputs):
    """Link immutable cache bytes once; never rewrite a historical input/run."""
    root=Path(root).resolve();inputs=Path(inputs).resolve();cfg=p.config(profile)
    require(not(root.is_relative_to(inputs) or inputs.is_relative_to(root) or root.is_relative_to(ROOT)), 'Independent run root required')
    output=root/'discovery'
    if not (output/'manifest.json').exists():
        marker=root/'.preparing.json';intent=dict(run_id=root.name,profile=profile,source_sha256=p.identity(p.sources()),config_sha256=p.identity(cfg))
        if marker.exists():require(p.read(marker)==intent,'Interrupted preparation identity differs')
        else:
            require(not root.exists() or all(f.name=='.lock' or metadata_temporary(f,root) for f in root.iterdir()),'Preparation identity missing')
            p.atomic_json(marker,intent)
            sync_directory(root);sync_directory(root.parent)
    require(p.digest(inputs/'manifest.json')==p.BASE_MANIFEST,'Original discovery manifest differs')
    summary=inputs/'summary.json'
    if not summary.exists():summary=ROOT/'reports/q_rmea_token_transport/20261121T034306Z_seed13/summary.json'
    require(p.digest(summary)==p.BASE_SUMMARY,'Pinned baseline summary differs')
    old=p.read(summary)
    source=inputs if profile=='discovery' else inputs.parent/'canary'
    require(p.digest(source/'manifest.json')==(p.BASE_MANIFEST if profile=='discovery' else p.CANARY_MANIFEST),'Fixture identity differs')
    original=p.read(source/'manifest.json')
    names=['metadata.json']+[s+'-'+k+'.npy' for s in cfg['counts'] for k in ('h','roles','offsets','targets')]
    cache={}
    for n in names:
        f=source/'cache'/n;ref=original['cache'][n]
        require(not f.is_symlink() and f.is_file() and f.stat().st_size==ref['bytes'] and p.digest(f)==ref['sha256'],'Protected cache differs: '+n)
        cache[n]=ref
    labels=inputs.parent/'recovery-inputs/labels.json'
    require(p.digest(labels)==p.LABELS_SHA,'Train-derived labels differ')
    req=inputs.parent/'cos-channel/000-preparation/request.json';ack=req.parent/'ack.json'
    request,receipt=p.read(req),p.read(ack)
    require(receipt['request_sha256']==p.digest(req)==p.INPUT_REQUEST_SHA and p.identity(receipt)==p.INPUT_ACK_ID
        and receipt['index_commit'] and set(receipt['objects'])==set(request['files']),'Original input recovery receipt differs')
    for n,row in request['files'].items():
        obj=receipt['objects'][n]
        require(obj.get('version_id') and obj['sha256']==row['sha256'] and obj['bytes']==row['bytes'],'Original pinned input object differs')
    output=root/'discovery'
    if (output/'manifest.json').exists():prepared(output);return output
    # Only an interrupted preparation may be continued. Never infer a fresh fit from missing state.
    allowed={'handoff.json','.lock','.preparing.json'}|{'discovery/cache/'+n for n in names}|{'recovery-inputs/'+n for n in ('labels.json','input-recovery.json','source-manifest.json','original-manifest.json')}
    require(not root.exists() or all(f.relative_to(root).as_posix() in allowed or metadata_temporary(f,root) for f in root.rglob('*') if f.is_file()),'Preparation has training or unknown state')
    (output/'cache').mkdir(parents=True,exist_ok=True);(root/'recovery-inputs').mkdir(exist_ok=True)
    def link(src,dest):
        if dest.exists():require(not dest.is_symlink() and p.digest(dest)==p.digest(src),'Interrupted preparation bytes differ')
        else:os.link(src,dest)
    for n in names:link(source/'cache'/n,output/'cache'/n)
    link(labels,root/'recovery-inputs/labels.json')
    link(source/'manifest.json',root/'recovery-inputs/source-manifest.json')
    link(inputs/'manifest.json',root/'recovery-inputs/original-manifest.json')
    p.atomic_json(root/'recovery-inputs/input-recovery.json',dict(schema='q-rmea.scattering.input-reference.v1',
        source_manifest_sha256=p.BASE_MANIFEST,source_manifest_canary=p.CANARY_MANIFEST,baseline_summary=p.BASE_SUMMARY,
        dependency=dict(request=request,ack=receipt),cache=cache,
        restore='Download pinned original preparation parts to managed storage, verify all bytes, use existing restore_preparation; install only named cache files by identity. Restore new minimal preparation and pinned latest/best; run explicit resume. No raw/model data to owner-PC.'))
    p.atomic_json(root/'handoff.json',dict(schema='q-rmea.scattering.owner-run.v1',run_id=root.name,profile=profile,
        source_sha256=p.identity(p.sources()),config=cfg,environment='Existing pinned qattn Python/torch environment; no full environment image',
        recovery='Full-state local latest/best, global10percent COS; immutable original input reference'))
    durable_inputs(root)
    p.atomic_json(output/'manifest.json',dict(schema='q-rmea.scattering.manifest.v1',config=cfg,
        execution_environment=execution_environment(),
        git_revision=p.git('rev-parse','HEAD'),source_sha256=p.identity(p.sources()),source_hashes=p.sources(),cache=cache,
        source_input_manifest=p.BASE_MANIFEST,canary_input_manifest=p.CANARY_MANIFEST if profile=='canary' else None,
        input_recovery_reference_sha256=p.digest(root/'recovery-inputs/input-recovery.json'),
        reference=dict(summary_sha256=p.BASE_SUMMARY,result=old['results']['linear_coherent'],
            initial_state_sha256=old['common_initial_state_sha256'],order_sha256=old['common_epoch_order_sha256'],
            comparison='Pinned completed full-data linear baseline; not retrained')))
    sync_directory(output);prepared(output);return output


def worker(args):
    require(ctypes.CDLL(None,use_errno=True).prctl(1,signal.SIGKILL,0,0,0)==0,'Parent-death fencing failed')
    require(os.getppid()==args.parent_pid,'Parent identity changed')
    import torch
    from baselines.retacred.rmea_token_transport_parallel import condition_recovery
    from baselines.retacred.rmea_token_transport_recovery import run_lock
    from baselines.retacred.rmea_token_transport_study import train
    require(torch.cuda.device_count()==1 and bool(os.environ.get('CUDA_VISIBLE_DEVICES')),'One allocated GPU required')
    output=args.output.resolve();manifest=p.read(output/'manifest.json');cfg=manifest['config']
    require(cfg==p.config(cfg['profile']),'Frozen config differs')
    require(manifest['execution_environment']==execution_environment(),'Worker execution environment differs')
    folder=output/'.workers'/args.condition
    with run_lock(folder/'.lock'):
        assignment=p.read(folder/'assignment.json')
        require(assignment==dict(token=args.assignment_token,parent_pid=args.parent_pid,condition=args.condition,
            device=os.environ['CUDA_VISIBLE_DEVICES'],source=p.identity(p.sources())),'Active parent assignment differs')
        require(not (output/'RUN_COMPLETE').exists() and not (folder/'WORKER_COMPLETE').exists(),'Preserve sealed run/condition')
        require(args.resume or not (folder/'latest.pt').exists(),'Explicit worker resume required')
        recovery=condition_recovery(output,cfg,args.condition,protocol=p,resume=args.resume)
        def pause(*_):recovery.stop=True
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,pause)
        result=train(cfg,output,p.read(output.parent/'recovery-inputs/labels.json'),recovery=recovery,
            modes=(args.condition,),finalize=False,runtime=runtime)
        p.atomic_json(folder/'result.json',result);(folder/'RUN_PAUSED').unlink(missing_ok=True)
        sync_directory(output/'analysis');sync_directory(output)
        sync_directory(folder);sync_directory(folder.parent)
        p.atomic_json(folder/'WORKER_COMPLETE',dict(result_sha256=p.digest(folder/'result.json'),
            terminal_sha256=p.digest(folder/'latest.pt'),checkpoint_sha256=p.digest(output/'checkpoints'/(args.condition+'-final.pt')),
            source=p.identity(p.sources()),condition=args.condition,
            anchors={s:p.digest(output/'analysis'/(args.condition+'-'+s+'.json')) for s in ('train','valid','test')}))
        sync_directory(folder)


def parallel(output,devices,*,resume=False,pause_after=0,restore_pin=False):
    from baselines.retacred.rmea_token_transport_parallel import parallel_train
    m=prepared(output);cfg=m['config']
    require(cfg==p.config(cfg['profile']) and m['source_sha256']==p.identity(p.sources()),'Run config/source differs')
    parallel_train(cfg,output,p.read(output.parent/'recovery-inputs/labels.json'),devices,
        cos=False,resume=resume,pause_after=pause_after,restore_pin=restore_pin,runtime=runtime)


def unit(out):
    import torch
    from experiments import q_rmea_entity_scattering as old
    from baselines.retacred.rmea_token_transport_study import state_identity
    require(not os.environ.get('CUDA_VISIBLE_DEVICES'),'CPU unit recipe required')
    from experiments.q_rmea_query_alignment import fresh_output
    fresh_output(out)
    import subprocess
    regression=subprocess.run([sys.executable,'-B','-m','unittest','tests.test_rmea_token_transport_parallel','tests.test_rmea_token_transport_recovery'],cwd=ROOT,capture_output=True,text=True)
    (out/'default-regression.txt').write_text(regression.stdout+regression.stderr)
    require(regression.returncode==0,'Default runner regression failed')
    checks=out/'identities';checks.mkdir();old.unit(checks)
    torch.set_num_threads(1);torch.manual_seed(20261122)
    h=torch.randn(8,5,1024,dtype=torch.float64)*.1
    roles=torch.zeros(8,5,4,dtype=torch.float64);roles[:,0,2]=1;roles[:,2,3]=1
    mask=torch.ones(8,5,dtype=torch.bool);mask[-3:,3:]=False;y=torch.arange(8)%40
    errors={}
    for kind in p.MODES:
        torch.manual_seed(13);a=runtime.make_head(kind).double();b=runtime.make_head(kind).double();b.load_state_dict(a.state_dict())
        require(sum(x.numel() for x in a.parameters())==81028,'Parameter budget differs')
        full=torch.nn.functional.cross_entropy(a(h,roles,mask),y);full.backward()
        chunk=runtime.backward(b,(h,roles,mask),y)
        error=max(float((x.grad-z.grad).abs().max()) for x,z in zip(a.parameters(),b.parameters()))
        require(error<1e-9 and abs(float(full.detach()-chunk))<1e-10,'Microbatch/full objective or gradient differs')
        errors[kind]=error
        if kind=='constant_onsite':
            d=a(h,roles,mask,details=True)
            expected=((d['utilities'].tanh()*mask[:,None,:]).sum(-1,keepdim=True)/mask.sum(-1)[:,None,None]).expand_as(d['onsite'])
            require(torch.equal(d['onsite'],expected),'Constant onsite control differs')
    torch.manual_seed(13)
    from baselines.retacred.rmea import RMEAHead
    require(state_identity(RMEAHead('quantum').double().state_dict())==p.BASE_INITIAL,'Initialization differs from reused baseline')
    p.atomic_json(out/'unit-validation.json',dict(gate=True,source_sha256=p.identity(p.sources()),contract_sha256=p.identity(p.contract()),
        gradient_errors=errors,numeric_fixture_sha256=p.digest(checks/'unit-validation.json'),baseline_initial=p.BASE_INITIAL))
    p.atomic_json(out/'RUN_COMPLETE',dict(artifacts={'unit-validation.json':p.digest(out/'unit-validation.json')}))


def require_unit(folder):
    require(folder is not None,'Explicit completed unit output required')
    a=p.read(folder/'unit-validation.json');b=p.read(folder/'RUN_COMPLETE')
    require(a.get('gate') is True and a['source_sha256']==p.identity(p.sources()) and a['contract_sha256']==p.identity(p.contract())
        and b['artifacts']['unit-validation.json']==p.digest(folder/'unit-validation.json'),'Unit identity/success differs')


def replay(output,condition,split):
    import io
    import numpy as np
    import torch
    from baselines.retacred.rmea_token_transport_study import state_identity
    m=prepared(output);ready=runtime.closed_analysis(output);af=output/'analysis'/(condition+'-'+split+'.json');a=p.read(af)
    receipt=p.read(output/'.workers'/condition/'WORKER_COMPLETE');cp=output/'checkpoints'/(condition+'-final.pt')
    metadata=p.read(output/'cache/metadata.json')
    authenticated,values=runtime.unit_inputs(output,split,[0,1],buffers=True)
    require(cp.stat().st_size<16*1024**2,'Bounded replay checkpoint required')
    checkpoint=cp.read_bytes();checkpoint_sha=hashlib.sha256(checkpoint).hexdigest()
    require(receipt['anchors'][split]==p.digest(af) and a['tolerance']==1e-8 and a['indices']==[0,1]
        and a['condition']==condition and a['split']==split and a['checkpoint']=='checkpoints/'+condition+'-final.pt'
        and a['sample_hashes']==[metadata[split][i]['id_hash'] for i in (0,1)]
        and a['inputs']==authenticated
        and checkpoint_sha==a['checkpoint_sha256']==receipt['checkpoint_sha256'],'Replay model/anchor identity differs')
    head=runtime.make_head(condition).double();head.load_state_dict(torch.load(io.BytesIO(checkpoint),map_location='cpu',weights_only=True));head=head.to('cuda').eval()
    result=[]
    with torch.no_grad():
        for h,roles in values:
            inputs=tuple(torch.from_numpy(np.array(x,dtype=np.float64,copy=True))[None].to('cuda') for x in (h,roles))
            result.append(head(*inputs,torch.ones(1,len(h),dtype=torch.bool,device='cuda')).cpu())
    error=float((torch.cat(result)-torch.tensor(a['logits'],dtype=torch.float64)).abs().max())
    require(error<=a['tolerance'],'Replay numerical tolerance failed')
    return dict(condition=condition,split=split,error=error,checkpoint_sha256=a['checkpoint_sha256'])


def cloud_canary(target,devices,output):
    """Publish one paused aggregate, fetch pinned bytes to a fresh directory, resume."""
    import math,shutil,urllib.request,urllib.parse
    import torch
    from baselines.retacred.rmea_token_transport_recovery import Recovery,restore_preparation
    r=Recovery(target,p.config('canary'),enabled=True,protocol=p)
    r.state=torch.load(r.latest,map_location='cpu',weights_only=True)
    done=r.state['done']
    require(math.ceil(r.total/10)<=done<r.total,'Nonterminal committed COS canary required')
    runtime.prepare(r,p.read(r.parent/'recovery-inputs/labels.json'),None)
    directory=r.channel/'010-progress';directory.mkdir()
    for f in (r.latest,r.best):os.link(f,directory/f.name)
    r._publish('010-progress',[directory/'latest.pt',directory/'best.pt'],done)
    receipts=[];downloads=output/'pinned-downloads';downloads.mkdir()
    for name in ('000-preparation','010-progress'):
        request=p.read(r.channel/name/'request.json');ack=p.read(r.channel/name/'ack.json')
        receipts.append(dict(request=request,ack=ack))
        for rel,row in request['files'].items():
            obj=ack['objects'][rel];url=ack['restore_urls'][rel]
            u=urllib.parse.urlsplit(url)
            require(u.scheme=='https' and urllib.parse.parse_qs(u.query).get('versionId')==[obj['version_id']],'Version-pinned HTTPS restore required')
            f=downloads/Path(rel).name;require(not f.exists(),'Unique download role required')
            size=0;h=hashlib.sha256()
            with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(url,timeout=90) as src,f.open('xb') as dst:
                while True:
                    b=src.read(1024**2)
                    if not b:break
                    size+=len(b);require(size<=row['bytes'],'Restore exceeds sealed size');h.update(b);dst.write(b)
            require(size==row['bytes'] and h.hexdigest()==row['sha256'],'Pinned restored bytes differ')
    restored=output/'restored'/r.parent.name
    restore_preparation(sorted(downloads.glob('*.tarpart')),restored)
    dest=restored/'discovery';(dest/'checkpoints').mkdir()
    for n in ('latest.pt','best.pt'):shutil.copyfile(downloads/n,dest/'checkpoints'/n)
    obj=receipts[-1]['ack']['objects']['cos-channel/010-progress/latest.pt']
    p.atomic_json(restored/'restore-pin.json',dict(run_id=r.parent.name,identity=r.identity,total=r.total,done=done,
        generation='010-progress',sha256=obj['sha256'],version_id=obj['version_id'],acknowledged_generations=receipts))
    from baselines.retacred.rmea_token_transport_recovery import run_lock
    with run_lock(restored.parent/'.run.lock'):
        try:restore_identity(restored.parent,restored,'canary',inject_failure=True)
        except OSError:pass
        else:raise ValueError('Pinned install interruption not injected')
        require(not (restored.parent/'run-identity.json').exists() and p.read(restored.parent/'restore-install.json')['status']=='pending','Interrupted restore admitted too early')
        restore_identity(restored.parent,restored,'canary')
    # Exercise the enabled preparation/history branch after interrupted pin installation.
    # It must consume the protected prefix without rebuilding/reuploading the tar.
    cloud=Recovery(dest,p.config('canary'),enabled=True,protocol=p)
    before=p.digest(restored/'cos-channel/000-preparation/request.json')
    runtime.prepare(cloud,p.read(restored/'recovery-inputs/labels.json'),None)
    cloud.drain_pending();cloud.verify_cloud_history()
    require(p.digest(restored/'cos-channel/000-preparation/request.json')==before,'Restored preparation was repackaged')
    # Same exact parent/child runner, fresh directory and checkpoint/RNG/AdamW state.
    with run_lock(restored.parent/'.run.lock'):
        verify_installed_pin(restored.parent,restored,'canary')
        parallel(dest,devices,resume=True)
    return dest,dict(generation='010-progress',done=done,total=r.total,objects=sum(len(x['request']['files']) for x in receipts),
        bytes=sum(x['bytes'] for v in receipts for x in v['request']['files'].values()),
        interrupted_pin_install_retry=True,enabled_preparation_reused=True,
        scope='Live COS VersionId/hash download, interrupted prefix installation retry, enabled preparation reuse and fresh-directory local continuation on the same managed node; not a different-provider drill')


def canary(args):
    """Serial/parallel pause-resume uses real tiny cache; no full-data training."""
    import torch
    from baselines.retacred.orchestrate import gpu_ids
    from baselines.retacred.rmea_token_transport_parallel import equal_state
    from experiments.q_rmea_query_alignment import fresh_output
    require_unit(args.unit_output);fresh_output(args.output)
    require(bool(os.environ.get('CUDA_VISIBLE_DEVICES','').strip()),'Explicit scheduler GPU allocation required')
    devices=gpu_ids('auto');require(len(devices)==3,'Three-GPU group required')
    suffix=args.output.parent.name
    serial=initialize(args.output/('serial-'+suffix),'canary',args.inputs)
    parallel(serial,devices[:1])
    target=initialize(args.output/('canary-'+suffix),'canary',args.inputs)
    try:parallel(target,devices,pause_after=1)
    except SystemExit as exc:require(exc.code==75,'Expected safe pause')
    else:raise ValueError('Pause fault did not execute')
    paused=[]
    for kind in p.MODES:
        x=torch.load(target/'.workers'/kind/'latest.pt',map_location='cpu',weights_only=True)
        paused.append(x['done'])
        require(not x['done'] or x['optimizer'] is not None or x['completed'],'Paused state must preserve optimizer or completed work')
    require(0<sum(paused)<9 and any(0<n<3 for n in paused),'Pause must exercise actual optimizer continuation')
    restored,cloud=cloud_canary(target,devices,args.output)
    parallel(target,devices,resume=True)
    for kind in p.MODES:
        for part in ('latest.pt','best.pt'):
            a=torch.load(serial/'.workers'/kind/part,map_location='cpu',weights_only=True)
            b=torch.load(target/'.workers'/kind/part,map_location='cpu',weights_only=True)
            # Unused RNG streams differ by independent process; scientific state is exact.
            keys=('active','optimizer','completed','best','cursor','done','initial_sha','order_sha') if part=='latest.pt' else None
            require(equal_state({k:a[k] for k in keys},{k:b[k] for k in keys}) if keys else equal_state(a,b),'Serial/parallel resumed scientific state differs')
            c=torch.load(restored/'.workers'/kind/part,map_location='cpu',weights_only=True)
            require(equal_state({k:a[k] for k in keys},{k:c[k] for k in keys}) if keys else equal_state(a,c),'Pinned COS resumed scientific state differs')
    replays=[replay(target,k,s) for k in p.MODES for s in ('train','valid','test')]
    public=args.output/'canary-report'
    before={n:p.digest(target/n) for n in ('manifest.json','summary.json','RUN_COMPLETE')}
    try:runtime.export(target,public,inject_failure=True)
    except OSError:pass
    else:raise ValueError('Exporter fault was not injected')
    require(not public.exists() and not list(public.parent.glob('.'+public.name+'-*')),'Failed export residue')
    export=runtime.export(target,public)
    require(runtime.export(target,public)==export,'Existing installed report retry differs')
    require(before=={n:p.digest(target/n) for n in before},'Export retried with mutated run')
    p.atomic_json(args.output/'canary-validation.json',dict(gate=True,source_sha256=p.identity(p.sources()),contract_sha256=p.identity(p.contract()),
        serial_parallel_resume_equal=True,replays=replays,export_retry=True,report=export,
        cos_live_pinned_restore=True,cloud_restore=cloud,
        input_dependency_versions_verified=False,scope='Local and live COS continuation passed; original full-input dependency verification is separate'))
    p.atomic_json(args.output/'RUN_COMPLETE',dict(artifacts={'canary-validation.json':p.digest(args.output/'canary-validation.json')}))


def run_location(args):
    import re
    run_id=args.run_id or 'entity-scattering-seed13-'+args.output.parent.name
    require(re.fullmatch(r'entity-scattering-seed13-[A-Za-z0-9_-]+',run_id) is not None,'Safe unique run ID required')
    return args.output/run_id/'discovery'


def deliver(args):
    output=run_location(args);prepared(output)
    report=runtime.export(output,args.output/'entity-scattering-safe-report')
    record=dict(run=str(output),report=report,source_sha256=p.identity(p.sources()))
    final=args.output/'run-complete.json'
    if final.exists():require(p.read(final)==record,'Existing delivery identity differs')
    else:p.atomic_json(final,record)
    return record


def preserve_admission(root,args):
    """Keep only the compact already-verified gates needed for replacement-node CLI."""
    sources={'unit/unit-validation.json':args.unit_output/'unit-validation.json','unit/RUN_COMPLETE':args.unit_output/'RUN_COMPLETE'}
    sources.update({'canary/'+n:args.admission.parent/n for n in ('admission.json','dependency-verification.json','canary-validation.json','RUN_COMPLETE')})
    for name,source in sources.items():
        target=root/'recovery-inputs/admission'/name;data=source.read_bytes();require(len(data)<2*1024**2,'Compact admission required')
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():require(target.read_bytes()==data,'Protected admission differs')
        else:
            temp=target.with_name('.'+target.name+'.installing')
            with temp.open('wb') as handle:handle.write(data);handle.flush();os.fsync(handle.fileno())
            os.replace(temp,target)
    durable_inputs(root/'recovery-inputs')


def run(args):
    from baselines.retacred.orchestrate import gpu_ids
    from baselines.retacred.rmea_token_transport_parallel import parallel_train
    from baselines.retacred.rmea_token_transport_recovery import run_lock
    from experiments.q_rmea_query_alignment import fresh_output
    require_unit(args.unit_output)
    require(args.admission is not None,'Exact canary/COS admission required')
    require(args.admission_sha256==p.digest(args.admission),'Owner-approved admission digest differs')
    a=p.read(args.admission)
    validation=args.admission.parent/'canary-validation.json';seal=args.admission.parent/'RUN_COMPLETE'
    require(p.read(seal)==dict(artifacts={'canary-validation.json':p.digest(validation)})
        and a.get('canary_validation_sha256')==p.digest(validation) and a.get('canary_seal_sha256')==p.digest(seal),'Completed canary admission seal differs')
    c=p.read(validation)
    require(all(a.get(k)==c.get(k) for k in ('source_sha256','contract_sha256','serial_parallel_resume_equal','export_retry','cos_live_pinned_restore')),'Canary gate differs from admission')
    require(a.get('source_sha256')==p.identity(p.sources()) and a.get('contract_sha256')==p.identity(p.contract())
        and a.get('serial_parallel_resume_equal') is True and a.get('export_retry') is True and a.get('cos_live_pinned_restore') is True
        and a.get('input_dependency_versions_verified') is True
        and a.get('input_request_sha256')==p.INPUT_REQUEST_SHA and a.get('input_acknowledgement_sha256')==p.INPUT_ACK_ID,'Formal training admission incomplete')
    depfile=args.admission.parent/'dependency-verification.json';dep=p.read(depfile)
    require(a.get('dependency_verification_sha256')==p.digest(depfile) and dep.get('verified') is True
        and dep.get('request_sha256')==p.INPUT_REQUEST_SHA and dep.get('acknowledgement_sha256')==p.INPUT_ACK_ID
        and dep.get('objects')==13 and dep.get('total_pinned_bytes')==6947440640 and dep.get('payload_bytes_read')==0,'Input dependency acceptance record differs')
    require(bool(os.environ.get('CUDA_VISIBLE_DEVICES','').strip()),'Explicit scheduler GPU allocation required')
    devices=gpu_ids('auto');require(len(devices)==3,'Three allocated GPUs required')
    output=run_location(args);root=output.parent
    identity=args.output/'run-identity.json';intent=dict(run_id=root.name,profile='discovery',source_sha256=p.identity(p.sources()),config_sha256=p.identity(p.config('discovery')))
    if args.resume and not args.restore_pin:
        require(identity.is_file() and p.read(identity)==intent,'Resume requires an existing identified run')
    elif args.resume:require(root.is_dir(),'Pinned restore requires an existing restored run')
    with run_lock(args.output/'.run.lock'):
        if args.resume and args.restore_pin:require(restore_identity(args.output,root,'discovery')==intent,'Pinned identity differs')
        elif args.resume:
            require(p.read(identity)==intent,'Run identity changed')
            if (args.output/'restore-install.json').exists():verify_installed_pin(args.output,root,'discovery')
        else:
            fresh_attempt(args.output);p.atomic_json(identity,intent)
            sync_directory(args.output)
        root.mkdir(exist_ok=True)
        if not (output/'manifest.json').exists():initialize(root,'discovery',args.inputs)
        require(prepared(output)['config']['profile']=='discovery','Formal run requires discovery profile')
        preserve_admission(root,args)
        if (output/'RUN_COMPLETE').exists():deliver(args);return
        parallel_train(p.config('discovery'),output,p.read(root/'recovery-inputs/labels.json'),devices,
            cos=True,resume=args.resume,restore_pin=False,runtime=runtime)
        deliver(args)


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('action',choices=('unit','canary','run','worker','replay','export'))
    a.add_argument('--output',type=Path,required=True);a.add_argument('--inputs',type=Path,default=INPUT_RUN)
    a.add_argument('--unit-output',type=Path);a.add_argument('--admission',type=Path);a.add_argument('--admission-sha256')
    a.add_argument('--condition',choices=p.MODES);a.add_argument('--split',choices=('train','valid','test'));a.add_argument('--run-id')
    a.add_argument('--parent-pid',type=int);a.add_argument('--assignment-token');a.add_argument('--resume',action='store_true');a.add_argument('--restore-pin',action='store_true')
    args=a.parse_args()
    require(not args.restore_pin or args.action=='run' and args.resume,'--restore-pin requires run --resume')
    p.review_gate()
    if args.action=='worker':worker(args)
    elif args.action=='unit':unit(args.output)
    elif args.action=='canary':canary(args)
    elif args.action=='replay':print(json.dumps(replay(args.output,args.condition,args.split)))
    elif args.action=='export':
        from baselines.retacred.rmea_token_transport_recovery import run_lock
        with run_lock(args.output/'.run.lock'):print(json.dumps(deliver(args)))
    else:run(args)


if __name__=='__main__':main()
