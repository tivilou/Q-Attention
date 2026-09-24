#!/usr/bin/env python3
"""Read-only diagnosis for a Q-PVG batch-resume directory.

The diagnostic rebuilds the run contract from the current checkout and the
private run's already-materialized data. It prints changed field paths only;
it never prints data rows, checkpoint contents, or contract values.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


RUNNER_NAME = "run_q_pvg_formal_single_seed.py"
DEFAULT_CONFIG_NAME = "retacred_q_pvg_formal_single_seed.json"


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, "missing"
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"invalid JSON ({type(exc).__name__})"
    if not isinstance(value, dict):
        return None, "invalid (expected object)"
    return value, None


def _load_runner(repo_root: Path) -> tuple[Any | None, str | None]:
    runner_path = repo_root / "experiments" / RUNNER_NAME
    if not runner_path.is_file():
        return None, "current Q-PVG runner is missing"
    for path in (repo_root / "src", repo_root / "experiments"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location("q_pvg_resume_runner", runner_path)
    if spec is None or spec.loader is None:
        return None, "current Q-PVG runner cannot be loaded"
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # environment-specific imports can fail on the collaborator host
        return None, f"current Q-PVG runner import failed ({type(exc).__name__})"
    return module, None


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _leaf_difference_paths(
    previous: Any, current: Any, prefix: str = ""
) -> list[str]:
    if isinstance(previous, dict) and isinstance(current, dict):
        paths: list[str] = []
        for key in sorted(set(previous) | set(current)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in previous or key not in current:
                paths.append(path)
            else:
                paths.extend(_leaf_difference_paths(previous[key], current[key], path))
        return paths
    if previous != current:
        return [prefix or "$"]
    return []


def _difference_class(path: str) -> str:
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
    return "immutable_contract"


def _recommended_action(
    *,
    strict: bool,
    code_update: bool,
    topology_change: bool,
    combined_migration: bool,
    difference_paths: list[str],
) -> str:
    if strict:
        return "resume without extra flag"
    has_code_difference = any(path.startswith("source") for path in difference_paths)
    has_topology_difference = "training_semantics.selector_gpu_ids" in difference_paths
    if has_code_difference and has_topology_difference:
        if combined_migration:
            return "resume with --allow-code-update and --allow-gpu-topology-change"
        return "stop; combined code and GPU topology migration is not compatible"
    if topology_change:
        return "resume with --allow-gpu-topology-change"
    if code_update:
        return "resume with --allow-code-update"
    return "stop; immutable config, data, selector, or training contract differs"


def _build_current_contract(
    runner: Any,
    *,
    repo_root: Path,
    run_dir: Path,
    config_path: Path,
    gpus: str | None,
    device: str,
    hardware_profile_name: str,
) -> tuple[dict[str, Any] | None, dict[str, Any], str | None]:
    config, config_error = _read_json(config_path)
    if config is None:
        return None, {}, f"Q-PVG config is {config_error}"
    scheduler = runner._base
    try:
        inventory = (
            scheduler.query_gpu_inventory()
            if gpus and gpus.strip().lower() == "auto"
            else []
        )
        gpu_ids = scheduler.resolve_gpu_ids(gpus, device, inventory)
        if gpu_ids and not inventory:
            inventory = scheduler.query_gpu_inventory()
        profile_request = (
            "auto"
            if gpus
            and gpus.strip().lower() == "auto"
            and hardware_profile_name == "config"
            else hardware_profile_name
        )
        hardware_profile = scheduler.choose_hardware_profile(
            profile_request, config, gpu_ids, inventory
        )
        hardware_profile.update(
            {
                "requested_gpu_spec": gpus or "default",
                "selected_gpu_ids": gpu_ids,
                "gpu_inventory": inventory,
            }
        )
        data_dir = run_dir / "data"
        contract = scheduler._run_resume_contract(
            config_path=config_path,
            config=config,
            seed=int(config["seed"]),
            data_dir=data_dir,
            hardware_profile=hardware_profile,
            model_parallel_gpu_ids=[],
        )
        return contract, {
            "config": config,
            "selected_gpu_ids": gpu_ids,
            "hardware_profile": hardware_profile,
            "gpu_inventory": inventory,
        }, None
    except Exception as exc:  # report a concise diagnostic instead of a traceback
        return None, {}, f"current resume contract could not be rebuilt ({type(exc).__name__})"


def _artifact_status(run_dir: Path, selectors: list[str]) -> dict[str, Any]:
    baseline = run_dir / "baseline"
    baseline_files = {
        name: (baseline / name).is_file()
        for name in ("model.pt", "vocab.json", "labels.json", "metrics.json")
    }
    selector_status = []
    for selector in selectors:
        selector_dir = run_dir / "selectors" / selector
        metrics, _ = _read_json(selector_dir / "metrics.json")
        complete_metrics = bool(
            metrics
            and metrics.get("selector") == selector
            and isinstance(metrics.get("valid"), dict)
            and isinstance(metrics.get("test"), dict)
        )
        selector_status.append(
            {
                "selector": selector,
                "complete": complete_metrics,
                "checkpoint": (selector_dir / "checkpoints" / "latest.pt").is_file(),
            }
        )
    return {
        "markers": {
            name: (run_dir / name).is_file()
            for name in ("RUN_COMPLETE", "RUN_PAUSED", "RUN_FAILED")
        },
        "baseline_complete": all(baseline_files.values()),
        "baseline_files": baseline_files,
        "data_manifest": (run_dir / "data" / "data_manifest.json").is_file(),
        "selectors": selector_status,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, metavar="RUN_DIR")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--gpus", default="auto", help="current GPU selection: auto or N[,N...]")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--hardware-profile",
        choices=("config", "auto", "adaptive", "low_memory", "balanced", "high_memory"),
        default="adaptive",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON only")
    return parser.parse_args()


def _emit(result: dict[str, Any], json_only: bool) -> int:
    if json_only:
        print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    else:
        print("Q-PVG resume diagnostic (read only)")
        print(f"run directory: {result['run_dir']}")
        artifacts = result["artifacts"]
        print(f"run markers: {artifacts['markers']}")
        print(f"baseline complete: {artifacts['baseline_complete']}")
        print(f"materialized data manifest present: {artifacts['data_manifest']}")
        for row in artifacts["selectors"]:
            state = "complete" if row["complete"] else "partial"
            if row["checkpoint"] and not row["complete"]:
                state = "checkpointed"
            print(f"selector {row['selector']}: {state}; checkpoint={row['checkpoint']}")
        compatibility = result.get("compatibility", {})
        if compatibility:
            print(
                "compatibility: "
                f"strict={compatibility['strict']}, "
                f"code-update={compatibility['allow_code_update']}, "
                f"GPU-topology-change={compatibility['allow_gpu_topology_change']}"
            )
            print(f"recommended action: {compatibility['recommended']}")
        print(f"contract differences: {len(result['differences'])}")
        for difference in result["differences"][:60]:
            print(f"  [{difference['class'].upper()}] {difference['path']}")
        for error in result["errors"]:
            print(f"ERROR: {error}", file=sys.stderr)
    return 1 if result["errors"] else 0


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
        "run_dir": str(run_dir),
        "artifacts": {},
        "compatibility": {},
        "differences": [],
        "errors": [],
    }
    if persisted is None:
        result["errors"].append(f"run_manifest.json is {manifest_error}")
        return _emit(result, args.json)
    persisted_contract = persisted.get("contract")
    if not isinstance(persisted_contract, dict):
        result["errors"].append("run manifest has no contract object")
        return _emit(result, args.json)

    selectors = []
    semantics = persisted_contract.get("training_semantics")
    if isinstance(semantics, dict) and isinstance(semantics.get("selectors"), list):
        selectors = [str(item) for item in semantics["selectors"]]
    if not selectors:
        config, _ = _read_json(config_path)
        if config and isinstance(config.get("selectors"), list):
            selectors = [str(item) for item in config["selectors"]]
    result["artifacts"] = _artifact_status(run_dir, selectors)

    runner, runner_error = _load_runner(repo_root)
    if runner is None:
        result["errors"].append(runner_error or "Q-PVG runner could not be loaded")
        return _emit(result, args.json)
    current, details, current_error = _build_current_contract(
        runner,
        repo_root=repo_root,
        run_dir=run_dir,
        config_path=config_path,
        gpus=args.gpus,
        device=args.device,
        hardware_profile_name=args.hardware_profile,
    )
    if current is None:
        result["errors"].append(current_error or "current contract is unavailable")
        return _emit(result, args.json)

    scheduler = runner._base
    persisted_fingerprint = persisted.get("contract_fingerprint")
    current_fingerprint = scheduler.fingerprint(current)
    strict = persisted_fingerprint == current_fingerprint
    code_update = bool(
        scheduler._code_update_contract_compatible(persisted_contract, current)
    )
    topology = bool(
        scheduler._elastic_run_contract_compatible(persisted_contract, current)
    )
    combined = bool(
        scheduler._combined_code_and_topology_contract_compatible(
            persisted_contract, current
        )
    )
    differences = _leaf_difference_paths(persisted_contract, current)
    result["differences"] = [
        {"path": path, "class": _difference_class(path)} for path in differences
    ]
    result["current"] = {
        "selected_gpu_ids": details["selected_gpu_ids"],
        "hardware_profile": details["hardware_profile"].get("name"),
        "contract_fingerprint": current_fingerprint,
    }
    result["compatibility"] = {
        "strict": strict,
        "allow_code_update": code_update,
        "allow_gpu_topology_change": topology,
        "allow_combined_code_and_gpu_topology_change": combined,
        "recommended": _recommended_action(
            strict=strict,
            code_update=code_update,
            topology_change=topology,
            combined_migration=combined,
            difference_paths=differences,
        ),
    }
    return _emit(result, args.json)


if __name__ == "__main__":
    raise SystemExit(main())
