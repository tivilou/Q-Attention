"""Compatibility entry for shared parent-assigned condition workers."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baselines.retacred.rmea_experiments import build
if __name__=='__main__':build('fair-reference').main()
