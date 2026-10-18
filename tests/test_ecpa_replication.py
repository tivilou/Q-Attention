"""Recovery and dependency boundaries for the frozen scientific worker."""
import copy
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))
from q_attention.experiments import ecpa_protocol as p
import ecpa_replication_contract as c
from ecpa_replication_queue import ready, build_jobs
from run_ecpa_replication import validate_run


def test_selector_depends_on_its_own_seed_only():
    jobs = {"29:baseline": dict(seed=29, mode="baseline", status="complete"),
        "53:baseline": dict(seed=53, mode="baseline", status="running"),
        "29:quantum": dict(seed=29, mode="quantum", status="pending"),
        "53:quantum": dict(seed=53, mode="quantum", status="pending")}
    assert ready(jobs) == ["29:quantum"]
    jobs["53:baseline"]["status"] = "complete"
    assert ready(jobs) == ["29:quantum", "53:quantum"]


def test_restore_memory_tier_without_restarting_the_search(tmp_path):
    runs = {seed: tmp_path / str(seed) for seed in (29, 53)}
    for seed, run in runs.items():
        p.atomic_json(run / "run_manifest.json", dict(fingerprint=str(seed)))
        p.atomic_json(run / "jobs/quantum/memory_state.json", dict(tier=2, retries=2, events=[]))
    jobs = build_jobs(runs)
    assert list(jobs)[:2] == ["29:baseline", "53:baseline"]
    assert len(jobs) == 16 and all(jobs[f"{seed}:quantum"]["tier"] == 2 for seed in runs)
    assert all(jobs[f"{seed}:baseline"]["tier"] == 0 for seed in runs)


def test_seed_changes_rng_only():
    old = p.load_config(False)
    for seed in (29, 53):
        config = c.seed_config(seed, False)
        assert config["seed"] == seed
        config["seed"] = 13
        assert config == old
    assert p.load_config(False) == old


def test_manifest_duplicate_field_tampering_rejected(tmp_path, monkeypatch):
    spec, source = c.protocol(), c.source_contract()
    imported = dict(rows={})
    p.atomic_json(tmp_path / "imported_seed13_summary.json", imported)
    provenance = dict(summary_sha256=p.sha(tmp_path / "imported_seed13_summary.json"), data={})
    contract = dict(protocol=spec, imported_summary_sha256=provenance["summary_sha256"], data={},
        execution_source=source, runtime_sha256=spec["frozen_runtime_sha256"], canary=True)
    manifest = dict(protocol=spec, execution_source=source, runtime_sha256=contract["runtime_sha256"],
        imported_seed13=provenance, contract=contract, fingerprint=p.identity(contract), canary=True, data={})
    p.atomic_json(tmp_path / "run_config.json", spec)
    p.atomic_json(tmp_path / "run_manifest.json", manifest)
    assert validate_run(tmp_path) == manifest
    for field, value in (("canary", False), ("data", {"train": {"records": 1}})):
        damaged = copy.deepcopy(manifest)
        damaged[field] = value
        p.atomic_json(tmp_path / "run_manifest.json", damaged)
        with pytest.raises(ValueError):
            validate_run(tmp_path)
