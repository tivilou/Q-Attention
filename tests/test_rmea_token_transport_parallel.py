"""Pure contract tests and a bounded real one/three-GPU parity acceptance."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baselines.retacred import rmea_token_transport_protocol as p
from baselines.retacred.rmea_token_transport_parallel import devices,launch_spec,pin_receipts


def fixture_pin(root,state,sha,latest_bytes=1,best_sha256=None,best_bytes=1):
    """Synthetic receipt identities for local restore tests; no COS claim."""
    import hashlib
    rows=[]
    for percent in range(0,101,10):
        name='000-preparation' if percent==0 else '%03d-progress'%percent
        key='cos-channel/'+name+('/preparation.000.tarpart' if percent==0 else '/latest.pt')
        files={key:dict(bytes=latest_bytes,sha256=sha)}
        if percent:files['cos-channel/'+name+'/best.pt']=dict(bytes=best_bytes,sha256=best_sha256 or sha)
        req=dict(schema='q-rmea.cos-request.v1',run_id=root.name,generation=name,identity=state['identity'],
                 done=state['done'] if percent else 0,total=state['total'],files=files)
        raw=(json.dumps(req,ensure_ascii=False,indent=2)+'\n').encode()
        ack=dict(request_sha256=hashlib.sha256(raw).hexdigest(),done=req['done'],index_commit='synthetic-canary-only',
                 objects={k:dict(v,version_id='synthetic-canary-only') for k,v in files.items()})
        rows.append(dict(request=req,ack=ack))
    return dict(run_id=root.name,sha256=sha,version_id='synthetic-canary-only',generation='100-progress',
                identity=state['identity'],done=state['done'],total=state['total'],acknowledged_generations=rows)


class Contracts(unittest.TestCase):
    def test_pin_requires_matching_version_and_complete_history(self):
        import copy
        pin=fixture_pin(Path('/fixture'),dict(identity={'source':'test'},done=4,total=4),'a'*64)
        check=lambda row:pin_receipts(row,run_id='fixture',identity=pin['identity'],total=4,sha256='a'*64,
            latest_bytes=1,best_sha256='a'*64,best_bytes=1)
        self.assertEqual(len(check(pin)),11)
        bad=copy.deepcopy(pin);bad['version_id']='wrong'
        with self.assertRaises(ValueError):check(bad)
        bad=copy.deepcopy(pin);bad['acknowledged_generations'].pop(3)
        with self.assertRaises(ValueError):check(bad)
    def test_auto_uses_all_eligible_allocated_devices(self):
        self.assertEqual(devices('auto',['GPU-a','GPU-b','GPU-c'],[12*2**30]*3),['GPU-a','GPU-b','GPU-c'])
    def test_auto_never_escapes_allocation(self):
        self.assertEqual(devices('auto',['GPU-b'],[12*2**30]),['GPU-b'])
    def test_busy_device_not_selected(self):
        self.assertEqual(devices('auto',['0','1','2'],[12*2**30,2**20,12*2**30]),['0','2'])
    def test_invalid_indexes(self):
        for value in ['',',','-1','0,0','2','0,a']:
            with self.subTest(value=value),self.assertRaises(ValueError):devices(value,['GPU-a','GPU-b'],[12*2**30]*2)
    def test_explicit_busy_is_error(self):
        with self.assertRaises(ValueError):devices('0,1',['0','1'],[12*2**30,2**20])
    def test_explicit_subset_and_mapping(self):
        self.assertEqual(devices('1,0',['GPU-x','GPU-y'],[12*2**30]*2),['GPU-y','GPU-x'])
    def test_empty_or_duplicate_allocation(self):
        for allocation in ([],['0','0']):
            with self.assertRaises(ValueError):devices('auto',allocation,[12*2**30]*len(allocation))
    def test_single_launch_contract(self):
        with patch.object(p,'git',return_value='f'*40):spec=launch_spec('/tmp/qattn-test-output','native','GPU-x',resume=True)
        self.assertEqual(spec['argv'][3],'baselines.retacred.rmea_token_transport_parallel')
        self.assertEqual(spec['cwd'],str(p.ROOT));self.assertIn('--resume',spec['argv'])
        self.assertEqual(spec['argv'][-2:],[str(os.getpid()),'--resume'])
    def test_worker_help_uses_real_launch_target(self):
        result=subprocess.run([sys.executable,'-B','-m','baselines.retacred.rmea_token_transport_parallel','--help'],cwd=p.ROOT,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr.decode())

    def test_ack_replay_after_deleted_preparation_parts(self):
        from baselines.retacred.rmea_token_transport_recovery import Recovery
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);output=root/'discovery';(output/'cache').mkdir(parents=True)
            (output/'cache/input').write_bytes(b'fixed')
            recovery=Recovery(output,p.config('canary'),enabled=True)
            recovery.progress=lambda stage:None
            generation=recovery.channel/'000-preparation';generation.mkdir(parents=True)
            part=generation/'preparation.000.tarpart';part.write_bytes(b'archive fixture')
            rel=part.relative_to(root).as_posix()
            request=dict(schema='q-rmea.cos-request.v1',run_id=root.name,generation='000-preparation',
                identity=recovery.identity,done=0,total=recovery.total,files={rel:dict(bytes=part.stat().st_size,sha256=p.digest(part))})
            p.atomic_json(generation/'request.json',request)
            p.atomic_json(generation/'ack.json',dict(request_sha256=p.digest(generation/'request.json'),done=0,
                index_commit='a'*40,objects={rel:dict(request['files'][rel],version_id='v1')}))
            part.unlink();recovery.drain_pending()
            self.assertEqual(p.read(recovery.channel/'cloud-latest.json')['generation'],'000-preparation')
            (recovery.channel/'cloud-latest.json').unlink();recovery.prepare([],None)
            self.assertTrue((recovery.channel/'cloud-latest.json').exists())
            bad=p.read(generation/'ack.json');bad['objects'][rel]['sha256']='b'*64;p.atomic_json(generation/'ack.json',bad)
            with self.assertRaises(ValueError):recovery.drain_pending()


def gpu_acceptance(args):
    import shutil
    import signal
    import time
    import numpy as np
    import torch
    if (torch.cuda.device_count()!=3 or not os.environ.get('CUDA_VISIBLE_DEVICES')
            or len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))!=3):raise ValueError('This acceptance expects an explicit actual three-GPU allocation')
    args.output.mkdir(parents=True,exist_ok=False)
    records=[];runs={}
    for name,execution,gpu,pause in [('serial','serial','0',False),('one','auto','0',False),
                                     ('three','auto','auto',False),('resumed','auto','auto',True)]:
        root=args.output/(name+'_canary')
        argv=[sys.executable,'-B','experiments/q_rmea_token_transport_handoff.py','run',
              '--development-canary','--canary-only','--development-cache',str(args.fixture),
              '--run-dir',str(root),'--execution',execution,'--gpu',gpu]
        if pause:argv+=['--pause-after','1']
        result=subprocess.run(argv,cwd=p.ROOT,capture_output=True,text=True)
        records.append(dict(argv=argv,exit=result.returncode,stdout=result.stdout[-3000:],stderr=result.stderr[-3000:]))
        if pause:
            if result.returncode!=75 or not (root/'canary/RUN_PAUSED').exists():raise AssertionError(('Actual parent did not propagate pause75',records[-1]))
            saved=torch.load(root/'canary/checkpoints/latest.pt',map_location='cpu',weights_only=True)
            if saved['schema']!='q-rmea.parallel-resume.v1' or not 0<saved['done']<=4:raise AssertionError('Parallel snapshot differs')
            argv=argv[:-2]+['--resume'];result=subprocess.run(argv,cwd=p.ROOT,capture_output=True,text=True)
            records.append(dict(argv=argv,exit=result.returncode,stdout=result.stdout[-3000:],stderr=result.stderr[-3000:]))
        if result.returncode:raise AssertionError(records[-1])
        manifest,summary=p.audit(root/'canary');runs[name]=summary
        if execution=='auto':
            roster=p.read(root/'canary/parallel.json');assigned={r['physical_gpu'] for r in roster['commands']}
            if len(assigned)!=(3 if gpu=='auto' else 1):raise AssertionError('Requested GPU parallelism not exercised')
            for row in roster['commands']:
                # A zombie is dead; a reused PID is not this original command.
                cmd=Path('/proc')/str(row['pid'])/'cmdline'
                if cmd.exists() and b'rmea_token_transport_parallel' in cmd.read_bytes():raise AssertionError('Worker leaked')
    for name,row in runs.items():
        if row['common_initial_state_sha256']!=runs['serial']['common_initial_state_sha256'] or row['common_epoch_order_sha256']!=runs['serial']['common_epoch_order_sha256']:
            raise AssertionError('Initialization/order changed')
        for kind in p.MODES:
            for field in ('final_state_sha256','updates','history','valid','test'):
                if row['results'][kind][field]!=runs['serial']['results'][kind][field]:raise AssertionError((name,kind,field))
        for name2 in ('one','three','resumed'):
            for ref in (args.output/'serial_canary/canary/trace').glob('*.npz'):
                with np.load(ref,allow_pickle=False) as a,np.load(args.output/(name2+'_canary')/'canary/trace'/ref.name,allow_pickle=False) as b:
                    if set(a.files)!=set(b.files) or any(not np.array_equal(a[k],b[k]) for k in a.files):raise AssertionError('Trace parity differs')
    def resume_copy(name,pinned=False,rebuild=False):
        target=args.output/(name+'_canary');shutil.copytree(args.output/'three_canary',target)
        (target/'canary/RUN_COMPLETE').unlink()
        old_commands=p.read(target/'canary/parallel.json')['commands']
        if pinned:
            shutil.rmtree(target/'canary/.workers')
            for f in (target/'canary/checkpoints').glob('*-final.pt'):f.unlink()
            shutil.rmtree(target/'canary/trace')
            state=torch.load(target/'canary/checkpoints/latest.pt',map_location='cpu',weights_only=True)
            latest=target/'canary/checkpoints/latest.pt';best=target/'canary/checkpoints/best.pt'
            pin=fixture_pin(target,state,p.digest(latest),latest.stat().st_size,p.digest(best),best.stat().st_size)
            p.atomic_json(target/'restore-pin.json',pin)
            # Emulate interruption after installing the first receipt and worker.
            from baselines.retacred.rmea_token_transport_recovery import atomic_torch
            p.atomic_json(target/'canary/.workers/restore-pin.json',pin)
            receipt=pin['acknowledged_generations'][0]
            p.atomic_json(target/'cos-channel/000-preparation/request.json',receipt['request'])
            atomic_torch(target/'canary/.workers/native/latest.pt',state['workers']['native'])
        elif rebuild:
            folder=target/'canary/.workers/native'
            (folder/'WORKER_COMPLETE').unlink();(folder/'result.json').unlink()
            for f in (target/'canary/trace').glob('native-*'):f.unlink()
        argv=[sys.executable,'-B','experiments/q_rmea_token_transport_handoff.py','run','--development-canary',
              '--canary-only','--run-dir',str(target),'--execution','auto','--resume']
        if pinned:argv+=['--restore-pin']
        x=subprocess.run(argv,cwd=p.ROOT,capture_output=True,text=True)
        records.append(dict(argv=argv,exit=x.returncode,stdout=x.stdout[-2000:],stderr=x.stderr[-3000:]))
        if x.returncode:raise AssertionError(records[-1])
        new_commands=p.read(target/'canary/parallel.json')['commands'][len(old_commands):]
        if not pinned and {r['condition'] for r in new_commands}!=({'native'} if rebuild else set()):
            raise AssertionError('Completed worker was relaunched')
        _,row=p.audit(target/'canary')
        if any(row['results'][k]['final_state_sha256']!=runs['serial']['results'][k]['final_state_sha256'] for k in p.MODES):
            raise AssertionError('Restored/reused checkpoint differs')
        for ref in (args.output/'three_canary/canary/trace').glob('*.npz'):
            with np.load(ref,allow_pickle=False) as a,np.load(target/'canary/trace'/ref.name,allow_pickle=False) as b:
                if any(not np.array_equal(a[k],b[k]) for k in a.files):raise AssertionError('Rebuilt trace differs')
        actual=p.read(target/'canary/sample-trace.json')
        reference=p.read(args.output/'three_canary/canary/sample-trace.json')
        def mapping(trace):
            return [(s['sample_id'],s['roles']['provenance'],[(r['id'],[f['path'] for f in r['artifact_refs']],[f['path'] for f in r['value_capture']['artifact_refs']],r['value_capture']['reader_projection']['reader_ref'])
                    for r in s['representations']]) for s in trace['samples']]
        if mapping(actual)!=mapping(reference):raise AssertionError('Rebuilt source/checkpoint/reader mapping differs')
    resume_copy('reuse')
    resume_copy('rebuild',rebuild=True)
    resume_copy('pin',pinned=True)
    # Reject malformed active and completed payloads before restore/publication.
    import copy
    from baselines.retacred.rmea_token_transport_recovery import Recovery
    validator=object.__new__(Recovery);validator.cfg=p.config('canary');validator.modes=('native',);validator.total=1
    final=torch.load(args.output/'three_canary/canary/.workers/native/latest.pt',map_location='cpu',weights_only=True)
    validator._validate_cursor(final)
    for field in ('rng','optimizer','model','history'):
        bad=copy.deepcopy(final)
        if field=='rng':bad['rng']['cuda']=[]
        elif field=='optimizer':bad['completed']['native']['optimizer']={}
        elif field=='model':bad['completed']['native']['state']={}
        else:bad['completed']['native']['history']=[]
        try:validator._validate_cursor(bad)
        except (ValueError,KeyError):pass
        else:raise AssertionError('Malformed terminal '+field+' accepted')
    active=next(r for r in saved['workers'].values() if r and r['active'] is not None)
    validator.modes=(active['identity']['condition'],)
    bad=copy.deepcopy(active);bad['optimizer']={}
    try:validator._validate_cursor(bad)
    except (ValueError,KeyError):pass
    else:raise AssertionError('Malformed active optimizer accepted')
    from baselines.retacred.rmea_token_transport_parallel import equal_state
    serial_state=torch.load(args.output/'serial_canary/canary/checkpoints/latest.pt',map_location='cpu',weights_only=True)
    for name in ('one','three','resumed'):
        state=torch.load(args.output/(name+'_canary')/'canary/checkpoints/latest.pt',map_location='cpu',weights_only=True)
        for kind in p.MODES:
            if not equal_state(serial_state['completed'][kind]['optimizer'],state['workers'][kind]['completed'][kind]['optimizer']):
                raise AssertionError('Terminal AdamW state differs')
    # Existing two-step active restore verifies the next AdamW update and all RNGs.
    code='import tempfile;from pathlib import Path;from tests.test_rmea_token_transport_recovery import local_fixture;\nwith tempfile.TemporaryDirectory() as d: print(local_fixture(Path(d)))'
    # tests is intentionally not a package; load the exact existing test module.
    code=code.replace('from tests.test_rmea_token_transport_recovery import local_fixture','import runpy;local_fixture=runpy.run_path("tests/test_rmea_token_transport_recovery.py")["local_fixture"]')
    x=subprocess.run([sys.executable,'-B','-c',code],cwd=p.ROOT,capture_output=True,text=True,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0],CUBLAS_WORKSPACE_CONFIG=':4096:8'))
    records.append(dict(argv=[sys.executable,'-B','-c',code],exit=x.returncode,stdout=x.stdout,stderr=x.stderr))
    if x.returncode:raise AssertionError(records[-1])
    # The real parent-death route: kill only the top handoff process.
    target=args.output/'death_canary'
    argv=[sys.executable,'-B','experiments/q_rmea_token_transport_handoff.py','run','--development-canary',
          '--canary-only','--development-cache',str(args.fixture),'--run-dir',str(target),'--gpu','auto']
    with (args.output/'death.log').open('wb') as log:
        child=subprocess.Popen(argv,cwd=p.ROOT,stdout=log,stderr=subprocess.STDOUT)
        roster=target/'canary/parallel.json';deadline=time.monotonic()+90
        while child.poll() is None and time.monotonic()<deadline:
            if roster.exists() and len(p.read(roster)['commands'])>=3:break
            time.sleep(.1)
        if not roster.exists() or len(p.read(roster)['commands'])<3:
            child.kill();child.wait();raise AssertionError('Death test never reached three actual workers')
        launched=p.read(roster)['commands'];child.kill();child.wait(timeout=10);time.sleep(2)
        for row in launched:
            cmd=Path('/proc')/str(row['pid'])/'cmdline'
            if cmd.exists() and b'rmea_token_transport_parallel' in cmd.read_bytes():raise AssertionError('Parent death leaked worker')
    # Worker failure is injected after the real parent starts three workers.
    target=args.output/'failure_canary';argv[argv.index('--run-dir')+1]=str(target)
    with (args.output/'failure.log').open('wb') as log:
        child=subprocess.Popen(argv,cwd=p.ROOT,stdout=log,stderr=subprocess.STDOUT)
        roster=target/'canary/parallel.json';deadline=time.monotonic()+90
        while child.poll() is None and time.monotonic()<deadline:
            if roster.exists() and len(p.read(roster)['commands'])>=3:break
            time.sleep(.1)
        if not roster.exists() or len(p.read(roster)['commands'])<3:
            child.kill();child.wait();raise AssertionError('Failure test never reached three actual workers')
        launched=p.read(roster)['commands'];os.kill(launched[0]['pid'],signal.SIGKILL)
        try:
            if child.wait(timeout=150)==0 or (target/'canary/RUN_COMPLETE').exists():raise AssertionError('Worker failure falsely completed')
        finally:
            if child.poll() is None:child.kill();child.wait(timeout=10)
        launched=p.read(roster)['commands']
        for row in launched:
            cmd=Path('/proc')/str(row['pid'])/'cmdline'
            if cmd.exists() and b'rmea_token_transport_parallel' in cmd.read_bytes():raise AssertionError('Failure leaked sibling')
    # Existing export retry machinery, now against the actual parallel parent output.
    from baselines.retacred.rmea_token_transport_handoff import export,validate_report
    source=args.output/'three_canary';target=args.output/'injected-safe-report'
    try:export(source,target,inject_failure=True)
    except RuntimeError:pass
    else:raise AssertionError('Exporter fault not injected')
    if target.exists() or list(args.output.glob('.injected-safe-report-*')):raise AssertionError('Failed export leaked')
    export(source,target);validate_report(target);export(source,target);validate_report(target)
    out=dict(accepted=True,scope='bounded16/8/8 single-update canary, not formal experiment',
             serial_one_three_resume_bitwise=True,exact_four_budgets=True,three_physical_devices=True,
             child_cleanup=True,export_failure_cleanup_retry=True,commands=records)
    out.update(trace_array_parity=True,completed_worker_reuse=True,terminal_capture_rebuild=True,
               fresh_pinned_restore=True,pinned_restore_scope='synthetic receipts, no live COS transfer',
               parent_death_cleanup=True,worker_failure_cleanup=True)
    p.atomic_json(args.output/'acceptance.json',out);p.atomic_json(args.output/'RUN_COMPLETE',dict(accepted=True))
    print(json.dumps(out),flush=True)


if __name__=='__main__':
    if '--gpu-acceptance' in sys.argv:
        parser=argparse.ArgumentParser();parser.add_argument('--gpu-acceptance',action='store_true')
        parser.add_argument('--output',type=Path,required=True);parser.add_argument('--fixture',type=Path,required=True)
        gpu_acceptance(parser.parse_args())
    else:unittest.main()
