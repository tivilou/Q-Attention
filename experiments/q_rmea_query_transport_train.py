"""Registered shared-unitary query study entry."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baselines.retacred.rmea_experiments import build
if __name__=='__main__':build('query-transport').main(sys.argv[1:])
