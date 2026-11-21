"""Open private complete numeric values; never upload its output by default."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--sample',required=True);parser.add_argument('--representation',required=True)
    parser.add_argument('--coordinates',required=True,help='JSON list of complete integer coordinates')
    a=parser.parse_args()
    from baselines.retacred.rmea_token_transport_protocol import read
    from baselines.retacred.rmea_token_transport_trace import read_value
    samples=read(a.run_dir/'sample-trace.json')['samples']
    sample=next(s for s in samples if s['sample_id']==a.sample)
    print(json.dumps(read_value(a.run_dir,sample,a.representation,json.loads(a.coordinates)),allow_nan=False))


if __name__=='__main__':
    main()
