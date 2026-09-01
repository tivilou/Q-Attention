#!/usr/bin/env python3
"""Diagnose a Q-RPEC batch-resume directory without changing it.

This is an operator-facing, read-only check.  It reconstructs the current
resume contract using the checked-out runner, compares it with the persisted
``run_manifest.json``, and reports the smallest explicit migration flag that
the runner accepts.  It never writes to the run directory.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any


RUNNER_NAME = "run_retacred_qrpec_formal_single_seed.py"
DEFAULT_CONFIG_NAME = "retacred_qrpec_formal_single_seed.json"


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, "missing"
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"invalid: {exc}"
    if not isinstance(value, dict):
        return None, "invalid: expected a JSON object"
    return value, None


def _load_runner(repo_root: Path) -> tuple[Any | None, str | None]:
    runner_path = repo_root / "experiments" / RUNNER_NAME
    if not runner_path.is_file():
        return None, f"missing runner: {runner_path}"
    src = repo_root / "src"
    experiments = repo_root / "experiments"
    for path in (src, experiments):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location("qrpec_resume_runner", runner_path)
    if spec is None or spec.loader is None:
        return None, f"cannot load runner: {runner_path}"
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # pragma: no cover - depends on the operator env
        return None, f"runner import failed: {type(exc).__name__}: {exc}"
    return module, None


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _leaf_diffs(left: Any, right: Any, prefix: str = "") -> list[dict[str, Any]]:
    if isinstance(left, dict) and isinstance(right, dict):
        result: list[dict[str, Any]] = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left:
                result.append({"path": path, "old": "<missing>", "new": right[key]})
            elif key not in right:
                result.append({"path": path, "old": left[key], "new": "<missing>"})
            else:
                result.extend(_leaf_diffs(left[key], right[key], path))
        return result
    if left != right:
        return [{"path": prefix or "$", "old": left, "new": right}]
    return []


def _diff_class(path: str) -> str:
    if path.startswith("source"):
        return "code"
    if path == "training_semantics.selector_gpu_ids":
        return "gpu_topology"
    if path.startswith("training_semantics"):
        return "training_contract"
    if path.startswith("config"):
        return "config"
    if path.startswith("data") or path.startswith("materialization"):
        return "data"
    if path.startswith("baseline"):
        return "baseline"
    return "immutable_contract"


def _short(value: Any, limit: int = 180) -> str:
    rendered = json.dumps(value, ensure_ascii=True, sort_keys=True)
    if len(rendered) <= limit:
        return rendered
    return rendered[: limit - 3] + "..."


def _build_current_contract(
    runner: Any,
    *,
    repo_root: Path,
    run_dir: Path,
    config_path: Path,
    gpus: str | None,
    device: str,
    hardware_profile_name: str,
    model_parallel_gpus: str | None,
) -> tuple[dict[str, Any] | None, dict[str, Any], str | None]:
    config, config_error = _read_json(config_path)
    if config is None:
        return None, {}, f"config {config_path}: {config_error}"
    try:
        model_parallel_ids = runner.parse_model_parallel_gpu_ids(model_parallel_gpus)
        if model_parallel_ids:
            profile_gpu_ids = model_parallel_ids
            inventory = runner.query_gpu_inventory()
        else:
            inventory = (
                runner.query_gpu_inventory()
                if gpus and gpus.strip().lower() == "auto"
                else []
            )
            gpu_ids = runner.resolve_gpu_ids(gpus, device, inventory)
            if gpu_ids and not inventory:
                inventory = runner.query_gpu_inventory()
            profile_gpu_ids = gpu_ids
        profile_request = (
            "auto"
            if gpus
            and gpus.strip().lower() == "auto"
            and hardware_profile_name == "config"
            else hardware_profile_name
        )
        hardware_profile = runner.choose_hardware_profile(
            profile_request, config, profile_gpu_ids, inventory
        )
        hardware_profile.update(
            {
                "requested_gpu_spec": model_parallel_gpus or gpus or "default",
                "selected_gpu_ids": profile_gpu_ids,
                "gpu_inventory": inventory,
            }
        )
        data_dir = run_dir / "data"
        contract = runner._run_resume_contract(
            config_path=config_path,
            config=config,
            seed=int(config["seed"]),
            data_dir=data_dir,
            hardware_profile=hardware_profile,
            model_parallel_gpu_ids=model_parallel_ids,
        )
        return contract, {
            "config": config,
            "hardware_profile": hardware_profile,
            "gpu_inventory": inventory,
            "selected_gpu_ids": profile_gpu_ids,
            "model_parallel_gpu_ids": model_parallel_ids,
        }, None
    except Exception as exc:  # pragma: no cover - depends on run/env state
        return None, {}, f"current contract could not be built: {type(exc).__name__}: {exc}"


def _artifact_status(run_dir: Path, selectors: list[str]) -> dict[str, Any]:
    baseline = run_dir / "baseline"
    baseline_files = {
        name: (baseline / name).is_file()
        for name in ("model.pt", "vocab.json", "labels.json", "metrics.json")
    }
    selector_rows: list[dict[str, Any]] = []
    for selector in selectors:
        selector_dir = run_dir / "selectors" / selector
        metrics_path = selector_dir / "metrics.json"
        metrics, metrics_error = _read_json(metrics_path)
        valid_metrics = bool(
            metrics
            and metrics.get("selector") == selector
            and isinstance(metrics.get("valid"), dict)
            and isinstance(metrics.get("test"), dict)
        )
        heartbeat, heartbeat_error = _read_json(selector_dir / "heartbeat.json")
        selector_rows.append(
            {
                "selector": selector,
                "directory": str(selector_dir),
                "metrics": "complete" if valid_metrics else (metrics_error or "partial"),
                "checkpoint": (selector_dir / "checkpoints" / "latest.pt").is_file(),
                "heartbeat": heartbeat or heartbeat_error or "missing",
                "status": "complete"
                if valid_metrics
                else ("checkpointed" if (selector_dir / "checkpoints" / "latest.pt").is_file() else "not_complete"),
            }
        )
    return {
        "markers": {
            name: (run_dir / name).is_file()
            for name in ("RUN_COMPLETE", "RUN_PAUSED", "RUN_FAILED")
        },
        "manifest": (run_dir / "run_manifest.json").is_file(),
        "data_manifest": (run_dir / "data" / "data_manifest.json").is_file(),
        "baseline": {
            "files": baseline_files,
            "complete": all(baseline_files.values()),
            "checkpoint": (baseline / "checkpoints" / "latest.pt").is_file(),
        },
        "selectors": selector_rows,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, metavar="RUN_DIR")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--gpus", default="auto", help="current planned GPU spec: auto or N[,N...]")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--hardware-profile",
        choices=("config", "auto", "adaptive", "low_memory", "balanced", "high_memory"),
        default="adaptive",
    )
    parser.add_argument("--model-parallel-gpus", default=None)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON only")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = (args.repo_root or Path(__file__).resolve().parents[1]).resolve()
    run_dir = _resolve_path(Path(args.run_dir), repo_root).resolve()
    config_path = _resolve_path(
        args.config or Path("configs") / DEFAULT_CONFIG_NAME, repo_root
    ).resolve()
    manifest_path = run_dir / "run_manifest.json"
    persisted, manifest_error = _read_json(manifest_path)
    result: dict[str, Any] = {
        "read_only": True,
        "repo_root": str(repo_root),
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "manifest": {"path": str(manifest_path), "error": manifest_error},
        "artifacts": {},
        "current": {},
        "compatibility": {},
        "differences": [],
        "errors": [],
    }
    if persisted is None:
        result["errors"].append(f"run_manifest.json: {manifest_error}")
        return _emit(result, args.json, failure=True)

    persisted_contract = persisted.get("contract")
    if not isinstance(persisted_contract, dict):
        result["errors"].append("run_manifest.json has no object-valued contract")
    runner, runner_error = _load_runner(repo_root)
    if runner is None:
        result["errors"].append(runner_error)
    selectors = []
    if isinstance(persisted_contract, dict):
        semantics = persisted_contract.get("training_semantics")
        if isinstance(semantics, dict) and isinstance(semantics.get("selectors"), list):
            selectors = [str(item) for item in semantics["selectors"]]
    if not selectors:
        config, _ = _read_json(config_path)
        if config and isinstance(config.get("selectors"), list):
            selectors = [str(item) for item in config["selectors"]]
    result["artifacts"] = _artifact_status(run_dir, selectors)
    if runner is not None and isinstance(persisted_contract, dict):
        current, details, current_error = _build_current_contract(
            runner,
            repo_root=repo_root,
            run_dir=run_dir,
            config_path=config_path,
            gpus=args.gpus,
            device=args.device,
            hardware_profile_name=args.hardware_profile,
            model_parallel_gpus=args.model_parallel_gpus,
        )
        if current is None:
            result["errors"].append(current_error)
        else:
            result["current"] = {
                "contract_fingerprint": runner.fingerprint(current),
                "selected_gpu_ids": details["selected_gpu_ids"],
                "model_parallel_gpu_ids": details["model_parallel_gpu_ids"],
                "hardware_profile": details["hardware_profile"],
                "gpu_inventory": details["gpu_inventory"],
            }
            persisted_fingerprint = persisted.get("contract_fingerprint")
            strict = persisted_fingerprint == runner.fingerprint(current)
            code_check = getattr(runner, "_code_update_contract_compatible", None)
            topology_check = getattr(runner, "_elastic_run_contract_compatible", None)
            code_update = bool(code_check and code_check(persisted_contract, current))
            topology = bool(topology_check and topology_check(persisted_contract, current))
            result["compatibility"] = {
                "strict": strict,
                "allow_code_update": code_update,
                "allow_gpu_topology_change": topology,
                "persisted_contract_fingerprint": persisted_fingerprint,
                "current_contract_fingerprint": runner.fingerprint(current),
                "recommended": (
                    "resume without extra flag"
                    if strict
                    else "--allow-gpu-topology-change"
                    if topology
                    else "--allow-code-update"
                    if code_update
                    else "no compatible resume flag; immutable contract differs"
                ),
            }
            differences = _leaf_diffs(persisted_contract, current)
            for difference in differences:
                difference["class"] = _diff_class(str(difference["path"]))
            result["differences"] = differences
    return _emit(result, args.json, failure=bool(result["errors"]))


def _emit(result: dict[str, Any], json_only: bool, *, failure: bool) -> int:
    if json_only:
        print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    else:
        print("Q-RPEC resume diagnostic (READ ONLY)")
        print(f"run:  {result['run_dir']}")
        print(f"repo: {result['repo_root']}")
        artifacts = result.get("artifacts", {})
        print(f"markers: {artifacts.get('markers', {})}")
        print(f"baseline: {artifacts.get('baseline', {})}")
        for row in artifacts.get("selectors", []):
            heartbeat = row.get("heartbeat")
            progress = ""
            if isinstance(heartbeat, dict):
                progress = (
                    f" batches={heartbeat.get('completed_batches', '?')}/{heartbeat.get('total_batches', '?')}"
                    f" epoch={heartbeat.get('epoch', '?')} rate={heartbeat.get('batches_per_second', '?')}"
                )
            print(
                f"selector {row['selector']}: {row['status']}"
                f" checkpoint={row['checkpoint']}{progress}"
            )
        compatibility = result.get("compatibility", {})
        if compatibility:
            print(
                "compatibility: "
                f"strict={compatibility.get('strict')} "
                f"code-update={compatibility.get('allow_code_update')} "
                f"gpu-topology={compatibility.get('allow_gpu_topology_change')}"
            )
            print(f"recommended: {compatibility.get('recommended')}")
        differences = result.get("differences", [])
        print(f"contract differences: {len(differences)}")
        for difference in differences[:40]:
            print(
                f"  [{str(difference.get('class', 'unknown')).upper()}] {difference['path']}: "
                f"{_short(difference.get('old'))} -> {_short(difference.get('new'))}"
            )
        for error in result.get("errors", []):
            print(f"ERROR: {error}", file=sys.stderr)
    return 1 if failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
