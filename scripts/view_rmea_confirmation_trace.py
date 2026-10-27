"""Local private trace reader. Exposes selected value coordinates on loopback only."""
import argparse
import json
from pathlib import Path
import sys
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlparse,parse_qs
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baselines.retacred import rmea_confirmation_protocol as p
from baselines.retacred.rmea_confirmation_values import coordinates

PAGE='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Q-RMEA 私有样本追踪</title>
<style>body{font:16px system-ui;margin:40px;max-width:1100px;background:#f6f4ed;color:#182c36}select,input,button{font:inherit;padding:8px;margin:6px}article{padding:18px;border:1px solid #aaa;margin:14px 0}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #aaa}pre{white-space:pre-wrap}</style>
<h1>Q-RMEA 样本与中间值</h1><p>私有数据仅在本机读取。选择样本和语义阶段，再查询完整张量中的任意坐标；输入轴顺序显示在下方。</p>
<label>样本 <select id="sample" title="固定抽取的同一样本、条件和检查点"></select></label><article id="source"></article>
<label>阶段 <select id="group" title="按实际数据流组织，每个阶段只有一个入口"></select></label>
<label>中间值 <select id="representation" title="实际保存的完整数值表示"></select></label><p id="meaning"></p>
<label>坐标 <input id="coordinate" title="按显示的轴顺序输入从0开始的坐标，以逗号分隔"></label><button id="read">读取实际数值</button>
<table><thead><tr><th>坐标</th><th>实际值</th></tr></thead><tbody id="values"></tbody></table><p id="state" role="status"></p>
<script>
let data,sample,rep;const $=id=>document.getElementById(id);
function option(s,v,t){const o=document.createElement('option');o.value=v;o.textContent=t;s.append(o)}
function chooseSample(){sample=data.samples.find(s=>s.sample_id===$('sample').value);$('source').textContent=sample.roles.source_sample.tokens.join(' ')+' | subject '+sample.roles.task_objects.subject+' | object '+sample.roles.task_objects.object+' | target '+sample.roles.target_or_gold.value+' | prediction '+sample.roles.prediction.value;
 $('group').replaceChildren();sample.stages.filter(s=>s.stage_id!=='evaluation').forEach(s=>option($('group'),s.stage_id,s.purpose));chooseGroup()}
function chooseGroup(){$('representation').replaceChildren();sample.representations.filter(r=>r.stage_id===$('group').value).forEach(r=>option($('representation'),r.id,r.meaning));chooseRep()}
function chooseRep(){rep=sample.representations.find(r=>r.id===$('representation').value);$('meaning').textContent=rep.meaning+'；轴：'+rep.axis_semantics.join(' × ')+'；形状：'+rep.shape.join(' × ');$('coordinate').value=rep.shape.map(s=>s-1).join(',');read()}
async function read(){try{const coord=$('coordinate').value.split(',').map(Number);const q=new URLSearchParams({sample:sample.sample_id,representation:rep.id,coordinates:JSON.stringify([coord])});const r=await fetch('/api/value?'+q);const v=await r.json();if(!r.ok)throw Error(v.error);$('values').replaceChildren();const tr=document.createElement('tr');for(const x of [coord.join(','),JSON.stringify(v.values[0])]){const td=document.createElement('td');td.textContent=x;tr.append(td)}$('values').append(tr);$('state').textContent='已核对完整源文件 SHA-256；读取任意坐标。'}catch(e){$('state').textContent=e.message}}
fetch('/api/trace').then(r=>r.json()).then(v=>{data=v;v.samples.forEach(s=>option($('sample'),s.sample_id,s.sample_id));chooseSample()});
$('sample').onchange=chooseSample;$('group').onchange=chooseGroup;$('representation').onchange=chooseRep;$('read').onclick=read;
</script></html>'''

def server(root,port):
    root=Path(root).resolve();trace=p.read(root/'sample-trace.json');samples={s['sample_id']:s for s in trace['samples']}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            if self.headers.get('Host') not in (f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}') or self.headers.get('Origin') not in (None,f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}'):
                self.send_error(403);return
            url=urlparse(self.path)
            try:
                if url.path=='/':body=PAGE.encode();kind='text/html; charset=utf-8'
                elif url.path=='/api/trace':body=json.dumps(trace).encode();kind='application/json'
                elif url.path=='/api/value':
                    q=parse_qs(url.query)
                    if len(url.query)>16000:raise ValueError('Request too large')
                    obj=coordinates(root,samples[q['sample'][0]],q['representation'][0],json.loads(q['coordinates'][0]));body=json.dumps(obj).encode();kind='application/json'
                else:self.send_error(404);return
                self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(body)
            except (ValueError,KeyError,StopIteration) as e:
                body=json.dumps(dict(error='Invalid coordinate or source integrity check failed')).encode();self.send_response(400);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(body)
    return ThreadingHTTPServer(('127.0.0.1',port),Handler)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run-dir',required=True);ap.add_argument('--port',type=int,default=4191);args=ap.parse_args()
    print(f'Private reader http://127.0.0.1:{args.port}/',flush=True);server(args.run_dir,args.port).serve_forever()
