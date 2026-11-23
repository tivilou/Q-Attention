"""Check the32 selected source labels against the official public TRAIN patch."""
from pathlib import Path
import hashlib,json,sys,urllib.request

r=Path(__file__).resolve().parent
sys.path.insert(0,'C:/Users/14630/AppData/Local/research-infrastructure/qattn-2080ti2-38434-20261122');import ops
code="from pathlib import Path;print(Path('/hy-tmp/Q-Attention/control/semantic-problem-audit-20261123/selection.private.json').read_text())"
selected=json.loads(ops.remote([ops.PY,'-B','-c',code],timeout=30))
sys.path.insert(0,'C:/Users/14630/AppData/Local/research-infrastructure/qattn-l5-credit-pairing-20261121');import owner_ops as own
base=own.argv
def argv(command):
    a=base(command);a[1:1]=['-o','ConnectionAttempts=2','-o','ConnectTimeout=12'];return a
own.argv=argv
code=r'''
from pathlib import Path
import hashlib,json,sys
chosen=json.load(sys.stdin);wanted={x['input_sha256']:x['case_id'] for x in chosen};out=[]
p=Path('/home/Q-Attention/Q-Attention-public/data/relation/retacred/train.jsonl');raw=p.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='f1051d574a44df3ecbd036c1de4fbbe0a372c768f6ec990330e2cea4a657269f'
for line in raw.splitlines():
 w=json.loads(line);key=hashlib.sha256(json.dumps([w['tokens'],w['subject'],w['object']],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
 if key in wanted:out.append(dict(case_id=wanted[key],original_id=w['metadata']['id'],source_label=w['label']))
assert {x['case_id'] for x in out}=={x['case_id'] for x in chosen}
print(json.dumps(out))
'''
records=json.loads(own.remote([own.PY,'-B','-c',code],data=json.dumps(selected).encode(),timeout=35))
url='https://raw.githubusercontent.com/gstoica27/Re-TACRED/master/Re-TACRED/train_id2label.json'
with urllib.request.urlopen(url,timeout=30) as response:raw=response.read()
patch=json.loads(raw);results=[]
for row in records:
    original_id=row.pop('original_id');label=patch.get(original_id)
    results.append(dict(**row,official_label=label,matches=label==row['source_label'],id_sha256=hashlib.sha256(str(original_id).encode()).hexdigest()))
report=dict(url=url,patch_sha256=hashlib.sha256(raw).hexdigest(),patch_entries=len(patch),selected_cases=len(selected),source_records=len(records),
    matching=sum(x['matches'] for x in results),mismatches=[x for x in results if not x['matches']],all_selected_present=all(x['official_label'] is not None for x in results),
    scope='Only32 frozen selected TRAIN inputs. No relabeling or annotation correctness claim; current public patch is a versioned comparison, not a substitute benchmark.')
(r/'official-patch-check.json').write_bytes((json.dumps(report,indent=2)+'\n').encode())
print(json.dumps(report))
