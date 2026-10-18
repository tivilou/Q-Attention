"""The paired result must preserve seed order, uncertainty and attribution limits."""
import importlib.util
from pathlib import Path
import math

import pytest

SPEC = importlib.util.spec_from_file_location("ecpa_replication_stats", Path(__file__).resolve().parents[1] / "experiments/ecpa_replication_stats.py")
stats = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stats)


def test_paired_interval_and_sample_sd():
    result = stats.paired([.12, .23, .34], [.1, .2, .3])
    assert result["mean"] == pytest.approx(.03)
    assert result["std"] == pytest.approx(.01)
    half = stats.T95_DF2 * .01 / math.sqrt(3)
    assert result["ci95"] == pytest.approx([.03-half, .03+half])
    with pytest.raises(ValueError):
        stats.paired([1, 2], [1, 2, 3])
    with pytest.raises(ValueError):
        stats.distribution([0, 1, float("nan")])


def fixture():
    modes = ["disabled", "quantum", "classical", "dephased", "random_phase", "entity_permuted", "original_qk"]
    summaries = {}
    for seed, delta in zip(stats.SEEDS, [.04, -.02, .01]):
        summaries[seed] = dict(primary="macro_f1", test_used_for_training_or_selection=False,
            rows={mode: dict(reports=dict(test=dict(metrics=dict(macro_f1=.2+(delta if mode == "quantum" else .003 if mode == "classical" else 0),
                accuracy=.6, loss=1.2)))) for mode in modes})
    return summaries, modes


def test_unstable_positive_mean_does_not_become_l2_or_quantum_attribution():
    summaries, modes = fixture()
    result = stats.aggregate(summaries, modes)
    assert result["rows"]["quantum"]["by_seed"]["29"]["macro_f1"] == pytest.approx(.18)
    assert result["paired_comparisons"]["quantum"]["mean"] == pytest.approx(.01)
    assert result["paired_comparisons"]["quantum"]["ci95"][0] < 0
    assert result["gates"] == dict(quantum_L2_utility=False, classical_QI_candidate=True, quantum_specific_advantage_claimed=False)
    assert result["rows"]["quantum"]["metrics"]["accuracy"]["mean"] == .6
    del summaries[53]
    with pytest.raises(ValueError):
        stats.aggregate(summaries, modes)


def test_metric_coverage_and_selection_flags_fail_closed():
    summaries, modes = fixture()
    del summaries[29]["rows"]["quantum"]["reports"]["test"]["metrics"]["accuracy"]
    with pytest.raises(ValueError):
        stats.aggregate(summaries, modes)
    summaries, modes = fixture()
    summaries[53]["test_used_for_training_or_selection"] = True
    with pytest.raises(ValueError):
        stats.aggregate(summaries, modes)
