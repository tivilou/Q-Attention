from pathlib import Path
import hashlib,json,sys
r=Path(__file__).resolve().parent;sys.path.insert(0,'C:/Users/14630/AppData/Local/research-infrastructure/qattn-2080ti2-38434-20261122');import ops
code=(r/'audit_code.py').read_text('utf8');inputs=json.loads((r/'private-output-inventory.json').read_bytes())
argv=['env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','/hy-tmp/Q-Attention/env/bin/python','-B','-c',code]
literal=ops.remote(argv,data=json.dumps(inputs).encode(),timeout=40);v=json.loads(literal)
(r/'output-audit.json').write_text(json.dumps(v,indent=2)+'\n')
(r/'audit-execution.json').write_text(json.dumps(dict(argv=argv,stdin=inputs,stdout=literal,stderr='',exit_status=0,nonzero_transport_raises=True,code_sha256=hashlib.sha256(code.encode()).hexdigest()),indent=2)+'\n')
print(json.dumps({k:v[k] for k in ('status','eta','source_linked_trace_cases','trace_elements','error_advantage','decision','raw_tokens_available_cases','new_native_calls')}))
