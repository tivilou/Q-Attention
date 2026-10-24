"""Bounded Q-SRM screen and lifecycle canary; full-data runs are unsupported."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from q_attention.experiments.srm_runner import main

if __name__=='__main__':raise SystemExit(main())
