"""Compatibility entry for the registered entity-query study."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baselines.retacred.rmea_experiments import main
if __name__=='__main__':main(['--experiment','entity-query']+sys.argv[1:])
