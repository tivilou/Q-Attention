#!/usr/bin/env python3
"""Shared research queue: status, registered recipe submission and controller."""
import argparse
import json
import os
from pathlib import Path
import sqlite3
import time
import uuid

from controller import serve
from queue_store import Store

def config_path():
    return Path(os.environ.get('RESEARCH_SCHEDULER_CONFIG',str(Path(os.environ.get('LOCALAPPDATA',Path.home()/'.local/share'))/'research-infrastructure/scheduler/config.json')))
def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def project_id(config,value):
    if value in config['projects']:return value
    path=Path(value).resolve()
    matches=[key for key,entry in config['projects'].items() if Path(entry['path']).resolve()==path]
    if len(matches)!=1:raise ValueError('project is not registered with shared research controller')
    return matches[0]
def status(store,config):
    with sqlite3.connect(store.path) as db:row=db.execute('SELECT heartbeat,pid FROM control WHERE id=1').fetchone()
    age=time.time()-row[0] if row else None
    return dict(schema='research.queue-status.v1',controller_online=age is not None and age<60,
        heartbeat_age_seconds=round(age,1) if age is not None else None,
        projects=[dict(id=key,recipes=list(value['recipes']),authorized_nodes=value['nodes']) for key,value in config['projects'].items()],
        nodes=[dict(id=key,enabled=value['enabled'],capacity=value['capacity']) for key,value in config['nodes'].items()],tasks=store.tasks())

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(config_path()))
    sub=parser.add_subparsers(dest='action',required=True)
    sub.add_parser('status')
    submit=sub.add_parser('submit');submit.add_argument('--project',default=str(Path.cwd()));submit.add_argument('--recipe',required=True)
    submit.add_argument('--key',default=None);submit.add_argument('--wait',action='store_true');submit.add_argument('--wait-timeout',type=int,default=900)
    control=sub.add_parser('controller');control.add_argument('--once',action='store_true')
    sub.add_parser('stop')
    cancel=sub.add_parser('cancel');cancel.add_argument('task')
    backup=sub.add_parser('backup');backup.add_argument('--output',required=True)
    args=parser.parse_args(argv);config=load(args.config);store=Store(config['database'])
    if args.action=='controller':serve(config,args.once);return 0
    if args.action=='cancel':store.cancel(args.task);return 0
    if args.action=='stop':(store.path.parent/'stop').touch();return 0
    if args.action=='backup':store.backup(args.output);print(json.dumps(dict(consistent_backup=str(Path(args.output).resolve()),cos_restore_certified=False)));return 0
    if args.action=='status':print(json.dumps(status(store,config),ensure_ascii=False));return 0
    row=store.submit(config,project_id(config,args.project),args.recipe,args.key or uuid.uuid4().hex)
    print(json.dumps(dict(task=row['id'],state=row['state'],controller_online=status(store,config)['controller_online'])),flush=True)
    if args.wait:
        until=time.monotonic()+args.wait_timeout
        while time.monotonic()<until:
            row=next(x for x in store.tasks() if x['id']==row['id'])
            if row['state'] not in ('queued','launching','running','unknown','cancel_requested'):
                print(json.dumps(row));return 0 if row['state']=='complete' else 2
            time.sleep(.5)
        print(json.dumps(dict(task=row['id'],state=row['state'],wait_expired=True)));return 3
    return 0
if __name__=='__main__':raise SystemExit(main())
