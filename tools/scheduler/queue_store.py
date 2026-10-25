"""Single-owner transactional queue. No model/runtime dependencies."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time
import uuid

ACTIVE=('launching','running','unknown')
KINDS={'unit','smoke','diagnostic','preflight','audit','experiment'}
SOURCE_FILES=('queue_store.py','node_worker.py','controller.py','researchctl.py','test_scheduler.py','README.zh.md')

def source_identity():
    root=Path(__file__).resolve().parent
    return digest({name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in SOURCE_FILES})
def review_gate():
    review=json.loads((Path(__file__).resolve().parent/'review.json').read_text(encoding='utf-8'))
    if review.get('review_scope') not in ('lifecycle','both') or review.get('verdict')!='PASS' or review.get('must_fix') or review.get('source_sha256')!=source_identity():raise ValueError('exact-source lifecycle PASS required')

def encoded(value):return json.dumps(value,sort_keys=True,separators=(',',':'))
def digest(value):return hashlib.sha256(encoded(value).encode()).hexdigest()
def ident(value):
    if not isinstance(value,str) or not re.fullmatch('[a-z0-9][a-z0-9-]{0,62}',value):raise ValueError('invalid identifier')
    return value

def recipe_check(recipe):
    if recipe['kind'] not in KINDS:raise ValueError('invalid kind')
    cmd=recipe['command']
    if not isinstance(cmd,list) or not cmd or cmd[0]!='{python}' or not all(isinstance(x,str) and x and '\0' not in x for x in cmd):raise ValueError('explicit trusted Python argv required')
    if type(recipe['timeout']) is not int or not 1<=recipe['timeout']<=86400:raise ValueError('invalid timeout')
    if type(recipe['max_attempts']) is not int or not 1<=recipe['max_attempts']<=3:raise ValueError('bounded attempts required')
    if type(recipe['replay_safe']) is not bool:raise ValueError('explicit replay policy required')
    if not recipe['replay_safe'] and recipe['max_attempts']!=1:raise ValueError('non-replayable job cannot retry')
    r=recipe['resources']
    if set(r)!={'cpu','ram_mb','disk_mb','gpus','gpu_mb'} or any(type(v) is not int or v<0 for v in r.values()) or min(r['cpu'],r['ram_mb'],r['disk_mb'])<1:raise ValueError('invalid resource budget')
    if r['gpus'] and not r['gpu_mb']:raise ValueError('GPU memory budget required')
    if not isinstance(recipe['outputs'],list) or len(recipe['outputs'])!=len(set(recipe['outputs'])):raise ValueError('unique output allowlist required')
    for name in recipe['outputs']:
        if not isinstance(name,str) or not name or '\\' in name or name.startswith('/') or any(x in ('','..','.') for x in name.split('/')):raise ValueError('invalid output path')
    if set(recipe)-{'kind','command','timeout','max_attempts','replay_safe','resources','outputs'}:raise ValueError('unknown recipe fields')
    return recipe

class Store:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.transaction() as db:
            db.executescript('''CREATE TABLE IF NOT EXISTS tasks(
                id TEXT PRIMARY KEY,project TEXT NOT NULL,key TEXT NOT NULL,spec TEXT NOT NULL,
                state TEXT NOT NULL,active_attempt TEXT,created REAL NOT NULL,updated REAL NOT NULL,
                UNIQUE(project,key));
              CREATE TABLE IF NOT EXISTS attempts(id TEXT PRIMARY KEY,task TEXT NOT NULL,node TEXT NOT NULL,
                token TEXT NOT NULL,state TEXT NOT NULL,spec TEXT NOT NULL,receipt TEXT,created REAL NOT NULL,updated REAL NOT NULL);
              CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT,task TEXT,attempt TEXT,state TEXT,at REAL);
              CREATE TABLE IF NOT EXISTS control(id INTEGER PRIMARY KEY CHECK(id=1),heartbeat REAL,pid INTEGER);
            ''')
    @contextmanager
    def transaction(self):
        db=sqlite3.connect(self.path,timeout=15);db.row_factory=sqlite3.Row
        db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
        try:
            db.execute('BEGIN IMMEDIATE');yield db;db.commit()
        except BaseException:db.rollback();raise
        finally:db.close()
    def event(self,db,task,attempt,state):db.execute('INSERT INTO events(task,attempt,state,at) VALUES(?,?,?,?)',(task,attempt,state,time.time()))
    def submit(self,config,project,recipe_id,key):
        project=ident(project);policy=config['projects'][project]
        recipe=recipe_check(policy['recipes'][recipe_id])
        if recipe['kind'] not in policy['allowed_kinds']:raise ValueError('project policy forbids this workload')
        if not re.fullmatch('[0-9a-f]{40}',policy['revision'] or ''):raise ValueError('register an exact committed project revision first')
        if not isinstance(key,str) or not 1<=len(key)<=128:raise ValueError('explicit bounded idempotency key required')
        spec=dict(project=project,recipe_id=recipe_id,recipe=recipe,revision=policy['revision'])
        with self.transaction() as db:
            row=db.execute('SELECT * FROM tasks WHERE project=? AND key=?',(project,key)).fetchone()
            if row:
                if row['spec']!=encoded(spec):raise ValueError('idempotency key conflicts with another contract')
                return dict(row)
            task=uuid.uuid4().hex;now=time.time()
            db.execute('INSERT INTO tasks VALUES(?,?,?,?,?,NULL,?,?)',(task,project,key,encoded(spec),'queued',now,now))
            self.event(db,task,None,'queued')
            return dict(db.execute('SELECT * FROM tasks WHERE id=?',(task,)).fetchone())
    def tasks(self):
        with self.transaction() as db:return [dict(x) for x in db.execute('SELECT id,project,key,state,active_attempt,created,updated FROM tasks ORDER BY created')]
    def attempts(self,active=True):
        with self.transaction() as db:
            query='SELECT * FROM attempts'+(" WHERE state IN ('launching','running','unknown')" if active else '')
            return [dict(x,request=json.loads(x['spec'])) for x in db.execute(query)]
    def claim(self,config,probes,excluded=()):
        with self.transaction() as db:
            active=[dict(x,request=json.loads(x['spec'])) for x in db.execute("SELECT * FROM attempts WHERE state IN ('launching','running','unknown')")]
            # Least-recently dispatched project first, then FIFO. One active job/project.
            jobs=db.execute('''SELECT t.* FROM tasks t WHERE state='queued'
                ORDER BY COALESCE((SELECT MAX(a.created) FROM attempts a JOIN tasks p ON a.task=p.id WHERE p.project=t.project),0),t.created''').fetchall()
            for job in jobs:
                if any(a['request']['project']==job['project'] for a in active):continue
                spec=json.loads(job['spec']);policy=config['projects'].get(job['project'])
                if not policy or spec['recipe']['kind'] not in policy['allowed_kinds']:continue
                for node_id,node in config['nodes'].items():
                    probe=probes.get(node_id)
                    if node_id in excluded or not node.get('enabled') or node_id not in policy['nodes'] or job['project'] not in node['projects'] or not probe:continue
                    if probe.get('schema')!='research.node.v1' or probe.get('node')!=node_id:continue
                    used=[a for a in active if a['node']==node_id]
                    r=spec['recipe']['resources'];cap=node['capacity']
                    if any(sum(a['request']['recipe']['resources'][k] for a in used)+r[k]>cap[k] for k in ('cpu','ram_mb')):continue
                    if r['ram_mb']>probe['available_ram_mb'] or r['disk_mb']>probe['available_disk_mb']:continue
                    busy={gpu for a in used for gpu in a['request']['gpu_ids']}
                    gpus=[g['uuid'] for g in probe['gpus'] if g['uuid'] not in busy and g['free_mb']>=r['gpu_mb']]
                    if len(gpus)<r['gpus']:continue
                    attempt=uuid.uuid4().hex;token=uuid.uuid4().hex;now=time.time()
                    request=dict(spec,task=job['id'],attempt=attempt,token=token,node=node_id,
                        boot_id=probe['boot_id'],gpu_ids=gpus[:r['gpus']])
                    db.execute('INSERT INTO attempts VALUES(?,?,?,?,?,?,NULL,?,?)',(attempt,job['id'],node_id,token,'launching',encoded(request),now,now))
                    db.execute("UPDATE tasks SET state='launching',active_attempt=?,updated=? WHERE id=?",(attempt,now,job['id']))
                    self.event(db,job['id'],attempt,'launching')
                    return request
        return None
    def observe(self,request,evidence):
        with self.transaction() as db:
            row=db.execute('SELECT * FROM attempts WHERE id=?',(request['attempt'],)).fetchone()
            task=db.execute('SELECT * FROM tasks WHERE id=?',(request['task'],)).fetchone()
            if not row or row['token']!=request['token'] or not task or task['active_attempt']!=request['attempt']:raise ValueError('stale attempt/token')
            if row['state'] not in ACTIVE:return
            state=evidence.get('state','unknown')
            if state in ('running','launching','unknown'):
                if evidence.get('token')!=request['token']:state='unknown'
                receipt=None
            else:
                if (evidence.get('schema')!='research.receipt.v1' or evidence.get('attempt')!=request['attempt']
                    or evidence.get('task')!=request['task'] or evidence.get('token')!=request['token']
                    or evidence.get('node')!=request['node'] or not evidence.get('quiescent') or state not in ('complete','failed','timed_out','cancelled','lease_expired','rejected','node_rebooted')):
                    raise ValueError('invalid terminal evidence; reservation retained')
                if state=='complete':
                    if evidence.get('revision')!=request['revision'] or evidence.get('returncode')!=0 or set(evidence.get('outputs',{}))!=set(request['recipe']['outputs']):raise ValueError('incomplete output contract')
                    if any(not re.fullmatch('[0-9a-f]{64}',v.get('sha256','')) or type(v.get('bytes')) is not int or v['bytes']<0 for v in evidence['outputs'].values()):raise ValueError('invalid artifact identity')
                receipt=encoded(evidence)
            now=time.time()
            db.execute('UPDATE attempts SET state=?,receipt=?,updated=? WHERE id=?',(state,receipt,now,request['attempt']))
            next_state=state
            count=db.execute('SELECT COUNT(*) FROM attempts WHERE task=?',(request['task'],)).fetchone()[0]
            if state in ('lease_expired','node_rebooted') and request['recipe']['replay_safe'] and count<request['recipe']['max_attempts']:next_state='queued'
            elif state in ('lease_expired','rejected','node_rebooted'):next_state='failed'
            if task['state']=='cancel_requested':next_state='cancelled' if state not in ACTIVE else 'cancel_requested'
            db.execute('UPDATE tasks SET state=?,updated=? WHERE id=?',(next_state,now,request['task']))
            self.event(db,request['task'],request['attempt'],next_state)
    def cancel(self,task):
        with self.transaction() as db:
            row=db.execute('SELECT * FROM tasks WHERE id=?',(task,)).fetchone()
            if not row:raise ValueError('unknown task')
            state='cancelled' if row['state']=='queued' else ('cancel_requested' if row['state'] in ACTIVE else row['state'])
            db.execute('UPDATE tasks SET state=?,updated=? WHERE id=?',(state,time.time(),task))
            self.event(db,task,row['active_attempt'],state)
    def heartbeat(self,pid):
        with self.transaction() as db:db.execute('INSERT OR REPLACE INTO control VALUES(1,?,?)',(time.time(),pid))
    def backup(self,target):
        target=Path(target)
        if target.exists():raise ValueError('backup target must be new')
        with sqlite3.connect(self.path) as source,sqlite3.connect(target) as dest:source.backup(dest)
        with sqlite3.connect(target) as db:
            if db.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('backup verification failed')
