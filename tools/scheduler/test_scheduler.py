import copy
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import sys

import queue_store as q
from controller import tick,leader,HealthMonitor,fresh_health

REV='a'*40
def recipe(gpus=0):
    return dict(kind='smoke',command=['{python}','-c','print(1)'],timeout=5,max_attempts=2,replay_safe=True,
        resources=dict(cpu=1,ram_mb=128,disk_mb=16,gpus=gpus,gpu_mb=256 if gpus else 0),outputs=['result.json'])
def configuration():
    projects={key:dict(path='/'+key,revision=REV,allowed_kinds=['smoke'],nodes=['node'],recipes={'smoke':recipe()}) for key in ('alpha','beta')}
    return dict(projects=projects,nodes={'node':dict(enabled=True,projects=['alpha','beta'],capacity=dict(cpu=2,ram_mb=512))})
def probe():return dict(schema='research.node.v1',node='node',boot_id='boot',available_ram_mb=512,available_disk_mb=512,gpus=[dict(uuid='GPU-a',free_mb=512),dict(uuid='GPU-b',free_mb=512)])
def receipt(req,state='complete'):
    return dict(schema='research.receipt.v1',attempt=req['attempt'],task=req['task'],token=req['token'],node=req['node'],
        state=state,quiescent=True,revision=REV,returncode=0,outputs={'result.json':dict(sha256='b'*64,bytes=2)})

class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.store=q.Store(self.root/'queue.db');self.config=configuration()
    def tearDown(self):self.temp.cleanup()
    def submit(self,project='alpha',key='one'):return self.store.submit(self.config,project,'smoke',key)
    def claim(self):return self.store.claim(self.config,{'node':probe()})
    def test_idempotent_and_conflicting_submission(self):
        row=self.submit();self.assertEqual(row['id'],self.submit()['id'])
        self.config['projects']['alpha']['recipes']['smoke']['timeout']=6
        with self.assertRaisesRegex(ValueError,'conflicts'):self.submit()
    def test_review_gate_reads_explicit_utf8(self):
        review=dict(review_scope='lifecycle',verdict='PASS',must_fix=[],source_sha256='test',note='审查 ✓')
        (self.root/'review.json').write_text(json.dumps(review,ensure_ascii=False),encoding='utf-8')
        with patch.object(q,'__file__',str(self.root/'queue_store.py')),patch.object(q,'source_identity',return_value='test'):
            q.review_gate()
    def test_policy_and_revision_gate(self):
        self.config['projects']['alpha']['allowed_kinds']=[]
        with self.assertRaisesRegex(ValueError,'policy'):self.submit()
        self.config['projects']['alpha']['allowed_kinds']=['smoke'];self.config['projects']['alpha']['revision']=None
        with self.assertRaisesRegex(ValueError,'revision'):self.submit()
    def test_unbound_project_can_queue_but_cannot_dispatch(self):
        self.config['projects']['alpha']['nodes']=[];self.submit();self.assertIsNone(self.claim())
    def test_cpu_jobs_mask_all_gpus(self):
        self.submit();self.assertEqual(self.claim()['gpu_ids'],[])
    def test_gpu_group_reserved_atomically(self):
        for p in self.config['projects'].values():p['recipes']['smoke']=recipe(2)
        self.submit();self.submit('beta');a=self.claim()
        self.assertEqual(len(a['gpu_ids']),2);self.assertIsNone(self.claim())
        self.store.observe(a,receipt(a));self.assertEqual(len(self.claim()['gpu_ids']),2)
    def test_ram_disk_and_gpu_memory_gate(self):
        self.submit();p=probe();p['available_ram_mb']=127
        self.assertIsNone(self.store.claim(self.config,{'node':p}))
        p=probe();p['available_disk_mb']=15;self.assertIsNone(self.store.claim(self.config,{'node':p}))
        self.config['nodes']['node']['capacity']['ram_mb']=127;self.assertIsNone(self.claim())
    def test_unknown_execution_holds_capacity(self):
        self.config['nodes']['node']['capacity']['cpu']=1;self.submit();a=self.claim();self.submit('beta')
        self.store.observe(a,dict(state='unknown',token=a['token']))
        self.assertIsNone(self.claim());self.assertEqual(len(self.store.attempts()),1)
    def test_project_fairness_and_one_active_limit(self):
        self.submit(key='one');self.submit(key='two');self.submit('beta')
        a=self.claim();b=self.claim();self.assertEqual({a['project'],b['project']},{'alpha','beta'});self.assertIsNone(self.claim())
    def test_expired_lease_retry_is_bounded_and_fenced(self):
        self.submit();a=self.claim();self.store.observe(a,receipt(a,'lease_expired'));b=self.claim()
        self.assertNotEqual(a['attempt'],b['attempt']);self.assertNotEqual(a['token'],b['token'])
        with self.assertRaisesRegex(ValueError,'stale'):self.store.observe(a,receipt(a))
        self.store.observe(b,receipt(b,'lease_expired'));self.assertEqual(self.store.tasks()[0]['state'],'failed');self.assertIsNone(self.claim())
    def test_deterministic_failure_not_retried(self):
        self.submit();a=self.claim();self.store.observe(a,receipt(a,'failed'));self.assertEqual(self.store.tasks()[0]['state'],'failed')
    def test_non_replayable_lease_expiration_not_retried(self):
        r=self.config['projects']['alpha']['recipes']['smoke'];r.update(replay_safe=False,max_attempts=1)
        self.submit();a=self.claim();self.store.observe(a,receipt(a,'lease_expired'));self.assertEqual(self.store.tasks()[0]['state'],'failed')
    def test_missing_outputs_and_wrong_node_retain_reservation(self):
        self.submit();a=self.claim();bad=receipt(a);bad['outputs']={}
        with self.assertRaisesRegex(ValueError,'output'):self.store.observe(a,bad)
        bad=receipt(a);bad['node']='wrong'
        with self.assertRaises(ValueError):self.store.observe(a,bad)
        self.assertEqual(len(self.store.attempts()),1)
    def test_cancel_queue_and_active_without_retry(self):
        row=self.submit();self.store.cancel(row['id']);self.assertEqual(self.store.tasks()[0]['state'],'cancelled')
        row=self.submit('beta');a=self.claim();self.store.cancel(row['id']);self.store.observe(a,receipt(a,'lease_expired'))
        self.assertEqual(self.store.tasks()[1]['state'],'cancelled');self.assertIsNone(self.claim())
    def test_restart_and_consistent_database_backup(self):
        self.submit();a=self.claim();backup=self.root/'backup.db';self.store.backup(backup)
        reopened=q.Store(backup);self.assertEqual(reopened.attempts()[0]['token'],a['token'])
        reopened.observe(a,receipt(a));self.assertEqual(reopened.tasks()[0]['state'],'complete')
        self.assertEqual(self.store.tasks()[0]['state'],'launching')
    def test_parallel_claims_do_not_overallocate(self):
        self.config['nodes']['node']['capacity']['cpu']=1;self.submit();self.submit('beta');results=[]
        threads=[threading.Thread(target=lambda:results.append(self.claim())) for _ in range(4)]
        for t in threads:t.start()
        for t in threads:t.join()
        self.assertEqual(sum(x is not None for x in results),1)
    def test_leader_lock_is_exclusive(self):
        with leader(self.root/'leader.lock'):
            with self.assertRaises(OSError):
                with leader(self.root/'leader.lock'):self.fail('second controller acquired leadership')
    def test_lost_launch_ack_is_reconciled_without_second_start(self):
        self.submit();calls=[];current={}
        def rpc(node,message,project=None):
            calls.append(message['action'])
            if message['action']=='describe':return probe()
            if message['action']=='start':
                current.update(message['request']);raise TimeoutError('lost acknowledgement')
            return receipt(current)
        tick(self.store,self.config,rpc);self.assertEqual(self.store.tasks()[0]['state'],'unknown')
        tick(self.store,self.config,rpc);self.assertEqual(self.store.tasks()[0]['state'],'complete')
        self.assertEqual(calls.count('start'),1)
    def test_invalid_resources_paths_and_unbounded_retry(self):
        for change in [dict(outputs=['../private']),dict(max_attempts=9),dict(command=['bash','-c','echo unsafe']),dict(replay_safe=False),dict(timeout=True)]:
            r=recipe();r.update(change)
            with self.assertRaises(ValueError):q.recipe_check(r)
    def enable_health(self):
        self.config['health']=dict(enabled=True,max_age_seconds=180,interval_seconds=60)
        self.config['nodes']['node']['server_id']='registered-node'
    def test_health_blocks_new_dispatch_but_keeps_unknown_reservations(self):
        self.submit();request=self.claim();self.submit('beta');self.enable_health();calls=[]
        def rpc(node,message,project=None):
            calls.append(message['action'])
            raise TimeoutError('node unavailable')
        tick(self.store,self.config,rpc,{'registered-node':dict(status='offline',checked_at=time.time(),reason='timeout')})
        self.assertEqual(calls,['renew'])
        self.assertEqual(self.store.attempts()[0]['id'],request['attempt'])
        self.assertEqual(self.store.attempts()[0]['state'],'unknown')
        self.assertEqual(self.store.tasks()[1]['state'],'queued')
        observations=json.loads((self.root/'worker-health.json').read_text())
        self.assertEqual(observations['nodes']['node']['status'],'blocked_health')
    def test_healthy_recovery_dispatches_only_explicit_registered_node(self):
        self.submit();self.enable_health();calls=[]
        def rpc(node,message,project=None):
            calls.append(message['action'])
            if message['action']=='describe':return probe()
            return dict(state='running',token=message['request']['token'])
        tick(self.store,self.config,rpc,{})
        self.assertEqual(calls,[])
        tick(self.store,self.config,rpc,{'unregistered-node':dict(status='online',checked_at=time.time())})
        self.assertEqual(calls,[])
        tick(self.store,self.config,rpc,{'registered-node':dict(status='online',checked_at=time.time())})
        self.assertEqual(calls,['describe','start'])
        self.assertEqual(self.store.tasks()[0]['state'],'running')
    def test_stale_suspect_and_future_health_fail_closed(self):
        now=time.time()
        self.assertTrue(fresh_health(dict(status='online',checked_at=now),now=now))
        for row in (dict(status='online',checked_at=now-181),dict(status='online',checked_at=now+1),dict(status='suspect',checked_at=now),dict(status='online')):
            self.assertFalse(fresh_health(row,now=now))
    def test_slow_fleet_scan_does_not_block_lease_renewal(self):
        self.submit();request=self.claim();self.enable_health()
        entered=threading.Event();release=threading.Event();calls=[]
        def scan(*args,**kwargs):
            entered.set();release.wait(5);return dict(servers=[])
        def rpc(node,message,project=None):
            calls.append(message['action']);return dict(state='running',token=request['token'])
        with patch('controller.serverctl_json',side_effect=scan):
            monitor=HealthMonitor(self.config)
            try:
                before=time.monotonic();health=monitor.poll()
                self.assertLess(time.monotonic()-before,1)
                self.assertTrue(entered.wait(1))
                tick(self.store,self.config,rpc,health)
                self.assertEqual(calls,['renew'])
                self.assertEqual(self.store.attempts()[0]['state'],'running')
            finally:release.set();monitor.close()

@unittest.skipUnless(os.name=='posix','POSIX worker contract')
class NodeContractTests(unittest.TestCase):
    def setUp(self):
        import node_worker as worker
        self.worker=worker;self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.config=dict(id='node',state_root=str(self.root/'attempts'),lease_seconds=10,capacity=dict(cpu=1,ram_mb=256),
            projects={'alpha':dict(repo=str(self.root),python=sys.executable,allowed_kinds=['smoke'],recipes={'smoke':recipe()})})
        file=self.root/'config.json';file.write_text(json.dumps(self.config));self.node=worker.Node(file)
        self.req=dict(project='alpha',recipe_id='smoke',recipe=recipe(),revision=REV,task='c'*32,attempt='d'*32,
            token='e'*32,node='node',boot_id=worker.boot_id(),gpu_ids=[])
        self.node.describe=lambda:dict(probe(),boot_id=worker.boot_id())
    def tearDown(self):self.temp.cleanup()
    def test_duplicate_start_is_not_second_launch(self):
        with patch.object(self.worker.subprocess,'Popen') as popen:
            self.node.start(self.req);self.node.start(self.req);self.assertEqual(popen.call_count,1)
            argv=popen.call_args.args[0]
            self.assertEqual(Path(argv[1]).name,'node_worker.py');self.assertEqual(argv[2],'run')
        with self.assertRaisesRegex(ValueError,'stale'):self.node.renew(self.req['attempt'],'wrong')
    def test_node_also_enforces_resource_reservations(self):
        with patch.object(self.worker.subprocess,'Popen'):
            self.node.start(self.req);other=dict(self.req,attempt='f'*32)
            with self.assertRaisesRegex(ValueError,'reserved'):self.node.start(other)
    def test_stale_supervisor_is_not_renewed(self):
        with patch.object(self.worker.subprocess,'Popen'):self.node.start(self.req)
        directory=self.node.directory(self.req['attempt']);before=(directory/'lease.json').read_bytes()
        self.worker.atomic(directory/'state.json',dict(state='running',token=self.req['token'],heartbeat=0))
        self.assertEqual(self.node.renew(self.req['attempt'],self.req['token'])['state'],'unknown')
        self.assertEqual((directory/'lease.json').read_bytes(),before)
    def test_unregistered_recipe_is_rejected_before_launch(self):
        bad=copy.deepcopy(self.req);bad['recipe']['command']=['{python}','-c','raise SystemExit(9)']
        with patch.object(self.worker.subprocess,'Popen') as popen:
            with self.assertRaisesRegex(ValueError,'unregistered'):self.node.start(bad)
            popen.assert_not_called()

if __name__=='__main__':unittest.main()
