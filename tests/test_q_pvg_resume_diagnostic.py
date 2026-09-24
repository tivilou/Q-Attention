from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_retacred_q_pvg_resume.py"
SPEC = importlib.util.spec_from_file_location("q_pvg_resume_diagnostic", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


def test_contract_diff_reports_paths_without_contract_values() -> None:
    previous = {
        "training_semantics": {"selector_gpu_ids": [0], "seed": 13},
        "source": {"files": {"runner": {"sha256": "old-secret-hash"}}},
    }
    current = {
        "training_semantics": {"selector_gpu_ids": [0, 1], "seed": 13},
        "source": {"files": {"runner": {"sha256": "new-secret-hash"}}},
    }

    paths = diagnostic._leaf_difference_paths(previous, current)
    classified = [(path, diagnostic._difference_class(path)) for path in paths]

    assert classified == [
        ("source.files.runner.sha256", "code"),
        ("training_semantics.selector_gpu_ids", "gpu_topology"),
    ]
    assert "old-secret-hash" not in repr(classified)
    assert "new-secret-hash" not in repr(classified)


def test_missing_contract_keys_are_reported_at_their_field_path() -> None:
    paths = diagnostic._leaf_difference_paths(
        {"config": {"sha256": "a", "bytes": 12}},
        {"config": {"sha256": "a"}},
    )

    assert paths == ["config.bytes"]


def test_invalid_json_error_does_not_echo_file_contents(tmp_path: Path) -> None:
    path = tmp_path / "run_manifest.json"
    path.write_text('{"private": "must not be echoed"', encoding="utf-8")

    _, error = diagnostic._read_json(path)

    assert error == "invalid JSON (JSONDecodeError)"
    assert "must not be echoed" not in error
