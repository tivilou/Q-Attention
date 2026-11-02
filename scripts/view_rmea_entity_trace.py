"""Private fixed-sample JSON reader, loopback only; arbitrary bounded coordinates."""
import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import sys
from urllib.parse import urlparse,parse_qs
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baselines.retacred.rmea_entity_trace import read_value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--port',type=int,default=4192)
    args=parser.parse_args();root=args.run_dir.resolve()
    trace=json.loads((root/'sample-trace.json').read_text())
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            try:
                parsed=urlparse(self.path);q=parse_qs(parsed.query)
                if parsed.path=='/api/value':
                    sample=next(s for s in trace['samples'] if s['sample_id']==q['sample'][0])
                    value=dict(values=read_value(root,sample,q['representation'][0],json.loads(q['coords'][0])))
                elif parsed.path in ('/','/api/trace'):value=trace
                else:raise ValueError('Unknown endpoint')
                raw=json.dumps(value).encode();self.send_response(200)
            except Exception:
                raw=b'{"error":"invalid private value request"}';self.send_response(400)
            self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(raw)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(raw)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'Private reader http://127.0.0.1:{server.server_address[1]}',flush=True)
    server.serve_forever()


if __name__=='__main__':main()
