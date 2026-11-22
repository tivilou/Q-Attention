"""Compatibility CLI for the fixed entity-scattering study."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baselines.retacred.rmea_frozen_training import FrozenFeatureTraining
from baselines.retacred import rmea_entity_scattering_protocol as protocol
from baselines.retacred import rmea_entity_scattering_runtime as runtime

if __name__=='__main__':
    FrozenFeatureTraining(protocol,runtime).main()
