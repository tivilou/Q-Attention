"""Allowlisted study adapters for the shared frozen-feature runner.

A new hypothesis supplies model/scientific callbacks in its adapter; lifecycle,
metrics and recovery remain in the existing core. Imports are lazy for --help.
"""
from dataclasses import dataclass
from importlib import import_module
from types import MappingProxyType


@dataclass(frozen=True)
class Experiment:
    protocol: str
    runtime: str


EXPERIMENTS = MappingProxyType({
    'entity-scattering': Experiment('rmea_entity_scattering_protocol', 'rmea_entity_scattering_runtime'),
    'protected-context': Experiment('rmea_protected_context_protocol', 'rmea_protected_context_runtime'),
    'relative-return': Experiment('rmea_relative_return_protocol', 'rmea_relative_return_runtime'),
    'entity-query': Experiment('rmea_entity_query_protocol', 'rmea_entity_query_runtime'),
    'fair-reference': Experiment('rmea_fair_reference_protocol', 'rmea_fair_reference_runtime'),
})


def build(name):
    try:
        spec = EXPERIMENTS[name]
    except KeyError as exc:
        raise ValueError('Unknown registered experiment: ' + name) from exc
    protocol = import_module('.' + spec.protocol, __package__)
    runtime = import_module('.' + spec.runtime, __package__)
    from .rmea_frozen_training import FrozenFeatureTraining
    return FrozenFeatureTraining(protocol, runtime)


def main(argv=None):
    import argparse
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--experiment', required=True, choices=tuple(EXPERIMENTS))
    if argv in (['--help'], ['-h']):
        parser.print_help()
        return
    args, remaining = parser.parse_known_args(argv)
    return build(args.experiment).main(remaining)
