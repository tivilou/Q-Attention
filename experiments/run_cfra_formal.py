#!/usr/bin/env python3
"""Q-CFRA collaborator parent and its single declared worker target."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

def main():
    if '--worker' in sys.argv:
        cli=argparse.ArgumentParser(description='Q-CFRA worker')
        cli.add_argument('--worker',required=True);cli.add_argument('--run-dir',type=Path,required=True)
        cli.add_argument('--device',choices=('cpu','cuda'),default='cuda');cli.add_argument('--stop-after',type=int)
        cli.add_argument('--recover-previous',action='store_true')
        args=cli.parse_args()
        from q_attention.experiments.cfra_worker import main as worker
        return worker(args.run_dir,args.worker,device=args.device,stop_after=args.stop_after,recover_previous=args.recover_previous)
    from q_attention.experiments.cfra_runner import main as parent
    return parent()

if __name__=='__main__':raise SystemExit(main())
