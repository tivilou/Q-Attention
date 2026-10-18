"""Frozen replication identities; existing scientific worker stays unchanged."""
from __future__ import annotations

import copy
import math
from pathlib import Path
import subprocess
import sys

from q_attention.experiments import ecpa_protocol as p

EXPERIMENT = "retacred_q_ecpa_formal_multi_seed"
SOURCE_FILES = (
    "experiments/run_ecpa_replication.py", "experiments/ecpa_replication_contract.py",
    "experiments/ecpa_replication_stats.py", "experiments/ecpa_replication_export.py",
    "experiments/ecpa_replication_queue.py", "scripts/run_retacred_q_ecpa_formal_multi_seed.sh",
    "configs/retacred_q_ecpa_replication.json",
)
PUBLIC_SEED_FILES = {"run_summary.json", "run_summary.data", "run_summary.md", "run_config.json",
    "run_manifest.json", "data.sha256", "data_counts.txt", "case_study.json", "exchange_manifest.json"}


def protocol():
    value = p.read(p.ROOT / "configs/retacred_q_ecpa_replication.json")
    if value["seeds"] != [13, 29, 53] or value["new_seeds"] != [29, 53]:
        raise ValueError("replication seeds must remain frozen at13/29/53, new29/53")
    if p.identity(p.source_contract()) != value["frozen_runtime_sha256"]:
        raise ValueError("scientific source differs from audited seed13 runtime")
    return value


def source_contract():
    return {name: p.sha(p.ROOT / name) for name in SOURCE_FILES}


def validate_summary(summary, config, fingerprint):
    if (summary["primary"] != "macro_f1" or set(summary["rows"]) != set(p.MODES)
            or summary["fingerprint"] != fingerprint or summary["test_used_for_training_or_selection"] is not False):
        raise ValueError("incomplete or invalid seed summary")
    expected = math.ceil(config["expected_records"]["train"] / config["kernel"]["batch_size"]) * config["kernel"]["epochs"]
    for mode, row in summary["rows"].items():
        if (row["fingerprint"] != fingerprint or row["mode"] != mode
                or row["test_used_for_training_or_selection"] is not False
                or row["updates"] != (0 if mode in ("disabled", "random_phase") else expected)):
            raise ValueError(f"incomplete seed budget or identity: {mode}")
        for split in ("valid", "test"):
            report = row["reports"][split]
            if any(not math.isfinite(v) for v in report["metrics"].values()):
                raise ValueError("nonfinite returned metric")
            classes = list(report["per_class"].values())
            count = sum(c["support"] for c in classes)
            if count != config["expected_records"][split]:
                raise ValueError("incomplete evaluation population")
            macro = sum(c["f1"] for c in classes) / len(classes)
            accuracy = sum(c["true_positive"] for c in classes) / count
            if not (math.isclose(macro, report["metrics"]["macro_f1"], abs_tol=1e-12)
                    and math.isclose(accuracy, report["metrics"]["accuracy"], abs_tol=1e-12)):
                raise ValueError("returned metric contradicts per-class evidence")


def seed13_import(spec, *, canary_run=None):
    import json
    if canary_run is not None:
        run = Path(canary_run).resolve()
        marker, manifest, summary = [p.read(run / name) for name in ("RUN_COMPLETE.json", "run_manifest.json", "run_summary.json")]
        config = p.read(run / "run_config.json")
        if config != p.load_config(True) or marker["summary_sha256"] != p.sha(run / "run_summary.json"):
            raise ValueError("seed13 canary fixture is not immutable and complete")
        if manifest["source_sha256"] != spec["frozen_runtime_sha256"] or marker["fingerprint"] != manifest["fingerprint"]:
            raise ValueError("canary fixture source/identity mismatch")
        case_index = {mode: dict(validation=p.read(run / "jobs" / mode / "trace_validation.json"),
            trace_sha256=p.sha(run / "jobs" / mode / "sample_trace.json")) for mode in ("baseline", *p.MODES)}
        provenance = dict(seed=13, kind="synthetic_fixture", summary_sha256=p.sha(run / "run_summary.json"),
            fingerprint=manifest["fingerprint"], data=manifest["data"], case_study=dict(jobs=case_index))
        validate_summary(summary, config, manifest["fingerprint"])
        return summary, provenance, manifest["data_dir"]
    frozen = spec["seed13"]
    def blob(name):
        return subprocess.check_output(["git", "show", f"{frozen['commit']}:{frozen['report']}/{name}"], cwd=p.ROOT)
    marker = json.loads(blob("REPORT_COMPLETE.json"))
    if set(marker["files"]) != PUBLIC_SEED_FILES or marker["fingerprint"] != frozen["fingerprint"]:
        raise ValueError("seed13 report whitelist/identity mismatch")
    import hashlib
    values = {}
    for name, digest in marker["files"].items():
        raw = blob(name)
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("corrupt immutable seed13 report")
        if name.endswith(".json"):
            values[name] = json.loads(raw)
    if marker["files"]["run_summary.json"] != frozen["summary_sha256"]:
        raise ValueError("seed13 is not the audited historical result")
    manifest, config, summary = (values[n] for n in ("run_manifest.json", "run_config.json", "run_summary.json"))
    if config != p.load_config(False) or manifest["source_sha256"] != spec["frozen_runtime_sha256"]:
        raise ValueError("seed13 protocol mismatch")
    validate_summary(summary, config, frozen["fingerprint"])
    provenance = dict(seed=13, kind="audited_formal_report", **frozen, data=manifest["data"],
        case_study=values["case_study.json"], exchange_manifest=values["exchange_manifest.json"])
    return summary, provenance, None


def seed_config(seed, canary):
    config = copy.deepcopy(p.load_config(canary))
    config["seed"] = seed
    # profile formal_seed13 identifies the unchanged legacy recipe, not the RNG seed.
    return config


def worker_spec(run, mode, device, *, inject_oom=False, stop_after=None):
    from q_attention.experiments.ecpa_runner import child_spec
    return child_spec(run, mode, device, inject_oom=inject_oom, stop_after=stop_after)


def summarize_seed(run):
    manifest, config = p.read(run / "run_manifest.json"), p.read(run / "run_config.json")
    rows = {mode: p.read(run / "jobs" / mode / "metrics.json") for mode in p.MODES}
    baseline = rows["disabled"]["reports"]["test"]["metrics"]["macro_f1"]
    summary = dict(schema_version="ecpa-summary.v1", experiment=config["name"], profile=config["profile"],
        fingerprint=manifest["fingerprint"], primary="macro_f1", rows=rows,
        delta_vs_disabled={mode: row["reports"]["test"]["metrics"]["macro_f1"]-baseline for mode, row in rows.items()},
        test_used_for_training_or_selection=False)
    validate_summary(summary, config, manifest["fingerprint"])
    p.atomic_json(run / "run_summary.json", summary)
    p.atomic_json(run / "run_summary.data", summary)
    p.atomic_json(run / "RUN_COMPLETE.json", dict(fingerprint=manifest["fingerprint"], summary_sha256=p.sha(run / "run_summary.json")))
    return summary
