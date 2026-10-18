"""Predeclared three-seed descriptive and paired statistics; no model imports."""
import math
import statistics

SEEDS = (13, 29, 53)
T95_DF2 = 4.302652729696142


def distribution(values):
    values = list(values)
    if len(values) != 3 or any(not math.isfinite(v) for v in values):
        raise ValueError("statistics require three finite seed values")
    return dict(values=values, mean=statistics.mean(values), std=statistics.stdev(values))


def paired(left, right):
    if len(left) != 3 or len(right) != 3:
        raise ValueError("paired comparison requires three seed-aligned values")
    values = [a-b for a, b in zip(left, right)]
    result = distribution(values)
    half = T95_DF2 * result["std"] / math.sqrt(3)
    result["ci95"] = [result["mean"]-half, result["mean"]+half]
    result["method"] = "paired Student t, two-sided95%, df2; three independent seed differences"
    return result


def aggregate(summaries, modes):
    if set(summaries) != set(SEEDS):
        raise ValueError("aggregate requires exactly seeds13/29/53")
    if any(s["primary"] != "macro_f1" or s["test_used_for_training_or_selection"] is not False for s in summaries.values()):
        raise ValueError("primary/selection contract differs")
    rows = {}
    for mode in modes:
        by_seed = {str(seed): summaries[seed]["rows"][mode]["reports"]["test"]["metrics"] for seed in SEEDS}
        names = set(by_seed["13"])
        if any(set(values) != names for values in by_seed.values()):
            raise ValueError("metric coverage differs between seeds")
        rows[mode] = dict(by_seed=by_seed, metrics={name: distribution(by_seed[str(s)][name] for s in SEEDS) for name in sorted(names)})
    primary = lambda mode: [rows[mode]["by_seed"][str(s)]["macro_f1"] for s in SEEDS]
    comparisons = {mode: paired(primary(mode), primary("disabled")) for mode in modes}
    comparisons["quantum_minus_classical"] = paired(primary("quantum"), primary("classical"))
    comparisons["quantum_minus_dephased"] = paired(primary("quantum"), primary("dephased"))
    quantum = comparisons["quantum"]
    classical_mean = rows["classical"]["metrics"]["macro_f1"]["mean"]
    baseline_mean = rows["disabled"]["metrics"]["macro_f1"]["mean"]
    relative = (classical_mean-baseline_mean) / abs(baseline_mean) if baseline_mean else None
    return dict(schema_version="ecpa-replication-summary.v1", seeds=list(SEEDS), primary="macro_f1", split="test",
        rows=rows, paired_comparisons=comparisons, classical_relative_primary_gain=relative,
        gates=dict(quantum_L2_utility=quantum["mean"] > 0 and quantum["ci95"][0] > 0,
            classical_QI_candidate=relative is not None and relative > .01, quantum_specific_advantage_claimed=False),
        test_used_for_training_or_selection=False,
        limitations=["Three-seed t interval has substantial uncertainty and an approximate sampling assumption.",
            "L2 utility is independent of quantum attribution; adverse accuracy/loss/benchmark-F1 must remain visible."])
