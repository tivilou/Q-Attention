#!/usr/bin/env python3
"""Portable Q-ECPA parent and allowlisted child entry point."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))


def main():
    if "--worker" in sys.argv:
        parser = argparse.ArgumentParser(description="One Q-ECPA worker")
        parser.add_argument("--worker",required=True)
        parser.add_argument("--run-dir",required=True,type=Path)
        parser.add_argument("--device",choices=["cpu","cuda"],default="cuda")
        parser.add_argument("--inject-oom-once",action="store_true")
        parser.add_argument("--stop-after",type=int)
        args = parser.parse_args()
        from q_attention.experiments.ecpa_worker import main as worker
        return worker(args.run_dir,args.worker,device=args.device,inject_oom=args.inject_oom_once,stop_after=args.stop_after)
    from q_attention.experiments.ecpa_runner import main as parent
    return parent()


if __name__ == "__main__":
    raise SystemExit(main())
