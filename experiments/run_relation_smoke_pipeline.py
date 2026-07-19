from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q_attention.tasks.relation import load_relation_jsonl, sample_relation_records, write_relation_jsonl  # noqa: E402
from q_attention.tasks.relation_formats import relation_record_summary  # noqa: E402

STAGE_CHOICES = (
    "baseline",
    "classical_projector",
    "classical_steering",
    "quantum_projector",
    "quantum_steering",
    "supervised_quantum_projector",
    "supervised_quantum_steering",
    "spectral_sweep",
    "routing",
)

DEFAULT_STAGES = tuple(
    stage for stage in STAGE_CHOICES if stage not in {"supervised_quantum_projector", "supervised_quantum_steering"}
)

DEFAULT_STAGE_OPTIONS: dict[str, dict[str, Any]] = {
    "baseline": {
        "epochs": 2,
        "batch_size": 8,
        "lr": 0.001,
        "dim": 64,
        "num_layers": 2,
        "num_heads": 4,
        "ff_dim": 128,
        "dropout": 0.1,
        "seed": 13,
    },
    "classical_projector": {"batch_size": 16, "rank": 4, "max_vectors": 512, "seed": 13},
    "classical_steering": {"batch_size": 16, "gain": 0.25},
    "quantum_projector": {
        "batch_size": 16,
        "rank": 4,
        "num_qubits": 4,
        "angle_scale": 1.25,
        "max_vectors": 512,
        "seed": 13,
        "feature_seed": 17,
    },
    "quantum_steering": {"batch_size": 16, "gain": 0.25},
    "supervised_quantum_projector": {
        "batch_size": 16,
        "rank": 4,
        "num_qubits": 4,
        "depth": 2,
        "angle_scale": 1.0,
        "max_vectors": 512,
        "max_train_samples": 256,
        "learning_rate": 0.05,
        "training_steps": 80,
        "seed": 13,
        "feature_seed": 17,
        "center": True,
    },
    "supervised_quantum_steering": {"batch_size": 16, "gain": 0.25},
    "spectral_sweep": {
        "batch_size": 16,
        "families": "classical,quantum",
        "modes": "hard_topk,soft_energy",
        "ranks": "2,4",
        "thresholds": "0.5",
        "sharpnesses": "8.0",
        "gains": "0.25",
        "num_qubits": 4,
        "angle_scale": 1.25,
        "max_vectors": 512,
        "seed": 13,
        "feature_seed": 17,
    },
    "routing": {
        "batch_size": 16,
        "gain": 0.25,
        "temperature": 0.5,
        "rank": 2,
        "num_qubits": 4,
        "angle_scale": 1.25,
        "max_vectors": 512,
        "seed": 13,
        "feature_seed": 17,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small canonical-real-data relation smoke pipeline.")
    parser.add_argument("--config", required=True, help="data_config.json or smoke config with train_path/valid_path")
    parser.add_argument("--train_path", default=None, help="Override canonical train JSONL path")
    parser.add_argument("--valid_path", default=None, help="Override canonical validation JSONL path")
    parser.add_argument("--test_path", default=None, help="Override canonical final-test JSONL path")
    parser.add_argument("--output_dir", default=None, help="Override run directory")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--stages", default=",".join(DEFAULT_STAGES), help="Comma-separated stage names")
    parser.add_argument("--max_train_records", type=int, default=None, help="Optional train subset for smoke runs")
    parser.add_argument("--max_valid_records", type=int, default=None, help="Optional validation subset for smoke runs")
    parser.add_argument("--max_test_records", type=int, default=None, help="Optional test subset for smoke runs")
    parser.add_argument("--seed", type=int, default=None, help="Global seed override for sampling and seeded stages")
    parser.add_argument("--dry_run", action="store_true", help="Print commands and write summary without executing")
    return parser.parse_args()


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def read_config(path: str | Path) -> dict[str, Any]:
    return json.loads(resolve_project_path(path).read_text(encoding="utf-8"))


def split_path_from_config(config: Mapping[str, Any], split_name: str) -> str | None:
    direct = config.get(f"{split_name}_path")
    if direct is not None:
        return str(direct)
    splits = config.get("splits", {})
    if isinstance(splits, Mapping):
        split = splits.get(split_name, {})
        if isinstance(split, Mapping) and split.get("path") is not None:
            return str(split["path"])
    return None


def parse_stage_list(value: str) -> list[str]:
    stages = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(stages) - set(STAGE_CHOICES))
    if invalid:
        raise ValueError(f"unknown stages: {invalid}; expected one of {STAGE_CHOICES}")
    return stages


def merged_stage_options(config: Mapping[str, Any], stage: str, *, seed_override: int | None = None) -> dict[str, Any]:
    options = dict(DEFAULT_STAGE_OPTIONS.get(stage, {}))
    configured = config.get(stage, {})
    if isinstance(configured, Mapping):
        options.update(configured)
    if seed_override is not None:
        for key in ("seed", "feature_seed"):
            if key in options:
                options[key] = seed_override
    return options


def add_cli_options(cmd: list[str], options: Mapping[str, Any], *, skip: Iterable[str] = ()) -> list[str]:
    skip_set = set(skip)
    for key, value in options.items():
        if key in skip_set or value is None:
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        elif isinstance(value, (list, tuple)):
            cmd.extend([flag, ",".join(str(item) for item in value)])
        else:
            cmd.extend([flag, str(value)])
    return cmd


def materialize_split(
    *,
    name: str,
    source_path: Path,
    output_dir: Path,
    limit: int | None,
    seed: int,
) -> tuple[Path, dict[str, Any]]:
    records = load_relation_jsonl(source_path)
    if limit is None or limit <= 0 or len(records) <= limit:
        return source_path, {
            "path": str(source_path),
            "sha256": file_sha256(source_path),
            "summary": relation_record_summary(records),
            "subset": False,
        }
    selected = sample_relation_records(records, limit, seed=seed, stratified=True)
    output_path = output_dir / f"{name}.jsonl"
    write_relation_jsonl(selected, output_path)
    return output_path, {
        "path": str(output_path),
        "sha256": file_sha256(output_path),
        "source_path": str(source_path),
        "summary": relation_record_summary(selected),
        "source_summary": relation_record_summary(records),
        "subset": True,
    }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata() -> dict[str, Any]:
    def git_output(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return None
        return result.stdout.strip()

    status = git_output("status", "--porcelain")
    return {
        "commit": git_output("rev-parse", "HEAD"),
        "branch": git_output("branch", "--show-current"),
        "dirty": None if status is None else bool(status),
    }


def runtime_metadata(global_seed: int) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "global_seed": global_seed,
        "git": git_metadata(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
    }


def run_command(cmd: list[str], *, dry_run: bool, records: list[dict[str, Any]]) -> None:
    printable = " ".join(cmd)
    print(json.dumps({"command": printable, "dry_run": dry_run}, sort_keys=True))
    record: dict[str, Any] = {
        "command": cmd,
        "dry_run": dry_run,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    started = time.perf_counter()
    if dry_run:
        record["returncode"] = None
    else:
        try:
            result = subprocess.run(cmd, cwd=ROOT, check=True)
            record["returncode"] = int(result.returncode)
        except subprocess.CalledProcessError as exc:
            record["returncode"] = int(exc.returncode)
            raise
        finally:
            record["duration_seconds"] = time.perf_counter() - started
            records.append(record)
        return
    record["duration_seconds"] = time.perf_counter() - started
    records.append(record)


def script(name: str) -> str:
    return str(ROOT / "experiments" / name)


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    stages = parse_stage_list(args.stages)
    output_dir = resolve_project_path(args.output_dir or config.get("output_dir", "runs/relation_real_smoke"))
    output_dir.mkdir(parents=True, exist_ok=True)

    train_value = args.train_path or split_path_from_config(config, "train")
    valid_value = args.valid_path or split_path_from_config(config, "valid")
    test_value = args.test_path or split_path_from_config(config, "test")
    if train_value is None or valid_value is None:
        raise ValueError("provide train/valid paths in config or via --train_path/--valid_path")

    subset_seed = args.seed if args.seed is not None else int(config.get("seed", 13))
    stage_seed_override = args.seed
    max_train = args.max_train_records if args.max_train_records is not None else config.get("max_train_records")
    max_valid = args.max_valid_records if args.max_valid_records is not None else config.get("max_valid_records")
    max_test = args.max_test_records if args.max_test_records is not None else config.get("max_test_records")
    data_dir = output_dir / "smoke_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_path, train_info = materialize_split(
        name="train", source_path=resolve_project_path(train_value), output_dir=data_dir, limit=max_train, seed=subset_seed
    )
    valid_path, valid_info = materialize_split(
        name="valid", source_path=resolve_project_path(valid_value), output_dir=data_dir, limit=max_valid, seed=subset_seed
    )
    test_path: Path | None = None
    test_info: dict[str, Any] | None = None
    if test_value is not None:
        test_path, test_info = materialize_split(
            name="test", source_path=resolve_project_path(test_value), output_dir=data_dir, limit=max_test, seed=subset_seed
        )
    final_eval_path = test_path or valid_path

    model_dir = output_dir / "baseline"
    commands: list[dict[str, Any]] = []

    if "baseline" in stages:
        cmd = [
            sys.executable,
            script("train_relation_baseline.py"),
            "--train_path",
            str(train_path),
            "--valid_path",
            str(valid_path),
            "--output_dir",
            str(model_dir),
            "--device",
            args.device,
        ]
        run_command(
            add_cli_options(cmd, merged_stage_options(config, "baseline", seed_override=stage_seed_override), skip={"device"}),
            dry_run=args.dry_run,
            records=commands,
        )

    if "classical_projector" in stages:
        cmd = [
            sys.executable,
            script("build_relation_projector.py"),
            "--model_dir",
            str(model_dir),
            "--data_path",
            str(train_path),
            "--output_path",
            str(model_dir / "relation_projector.pt"),
            "--device",
            args.device,
        ]
        run_command(
            add_cli_options(cmd, merged_stage_options(config, "classical_projector", seed_override=stage_seed_override), skip={"device"}),
            dry_run=args.dry_run,
            records=commands,
        )

    if "classical_steering" in stages:
        cmd = [
            sys.executable,
            script("eval_relation_steering.py"),
            "--model_dir",
            str(model_dir),
            "--projector_path",
            str(model_dir / "relation_projector.pt"),
            "--data_path",
            str(final_eval_path),
            "--output_dir",
            str(output_dir / "classical_steering_eval"),
            "--device",
            args.device,
        ]
        run_command(
            add_cli_options(cmd, merged_stage_options(config, "classical_steering", seed_override=stage_seed_override), skip={"device"}),
            dry_run=args.dry_run,
            records=commands,
        )

    if "quantum_projector" in stages:
        cmd = [
            sys.executable,
            script("build_relation_quantum_projector.py"),
            "--model_dir",
            str(model_dir),
            "--data_path",
            str(train_path),
            "--output_path",
            str(model_dir / "relation_quantum_projector.pt"),
            "--device",
            args.device,
        ]
        run_command(
            add_cli_options(cmd, merged_stage_options(config, "quantum_projector", seed_override=stage_seed_override), skip={"device"}),
            dry_run=args.dry_run,
            records=commands,
        )

    if "quantum_steering" in stages:
        cmd = [
            sys.executable,
            script("eval_relation_steering.py"),
            "--model_dir",
            str(model_dir),
            "--projector_path",
            str(model_dir / "relation_quantum_projector.pt"),
            "--data_path",
            str(final_eval_path),
            "--output_dir",
            str(output_dir / "quantum_steering_eval"),
            "--device",
            args.device,
        ]
        run_command(
            add_cli_options(cmd, merged_stage_options(config, "quantum_steering", seed_override=stage_seed_override), skip={"device"}),
            dry_run=args.dry_run,
            records=commands,
        )

    if "supervised_quantum_projector" in stages:
        cmd = [
            sys.executable,
            script("build_relation_supervised_quantum_projector.py"),
            "--model_dir",
            str(model_dir),
            "--data_path",
            str(train_path),
            "--output_path",
            str(model_dir / "relation_supervised_quantum_projector.pt"),
            "--device",
            args.device,
        ]
        run_command(
            add_cli_options(
                cmd,
                merged_stage_options(config, "supervised_quantum_projector", seed_override=stage_seed_override),
                skip={"device"},
            ),
            dry_run=args.dry_run,
            records=commands,
        )

    if "supervised_quantum_steering" in stages:
        cmd = [
            sys.executable,
            script("eval_relation_steering.py"),
            "--model_dir",
            str(model_dir),
            "--projector_path",
            str(model_dir / "relation_supervised_quantum_projector.pt"),
            "--data_path",
            str(final_eval_path),
            "--output_dir",
            str(output_dir / "supervised_quantum_steering_eval"),
            "--device",
            args.device,
        ]
        run_command(
            add_cli_options(
                cmd,
                merged_stage_options(config, "supervised_quantum_steering", seed_override=stage_seed_override),
                skip={"device"},
            ),
            dry_run=args.dry_run,
            records=commands,
        )

    if "spectral_sweep" in stages:
        cmd = [
            sys.executable,
            script("sweep_relation_spectral_filters.py"),
            "--model_dir",
            str(model_dir),
            "--projector_data_path",
            str(train_path),
            "--eval_path",
            str(valid_path),
            "--output_dir",
            str(output_dir / "spectral_filter_sweep"),
            "--device",
            args.device,
        ]
        if test_path is not None:
            cmd.extend(["--test_path", str(test_path)])
        run_command(
            add_cli_options(cmd, merged_stage_options(config, "spectral_sweep", seed_override=stage_seed_override), skip={"device"}),
            dry_run=args.dry_run,
            records=commands,
        )

    if "routing" in stages:
        cmd = [
            sys.executable,
            script("eval_relation_routing.py"),
            "--model_dir",
            str(model_dir),
            "--projector_data_path",
            str(train_path),
            "--eval_path",
            str(final_eval_path),
            "--output_dir",
            str(output_dir / "relation_routing_eval"),
            "--device",
            args.device,
        ]
        run_command(
            add_cli_options(cmd, merged_stage_options(config, "routing", seed_override=stage_seed_override), skip={"device"}),
            dry_run=args.dry_run,
            records=commands,
        )

    summary = {
        "config": str(resolve_project_path(args.config)),
        "output_dir": str(output_dir),
        "device": args.device,
        "stages": stages,
        "train": train_info,
        "valid": valid_info,
        "test": test_info,
        "evaluation_protocol": {
            "selection_split": "valid",
            "final_split": "test" if test_path is not None else "valid",
            "final_path": str(final_eval_path),
            "test_isolated": test_path is not None,
        },
        "reproducibility": {
            **runtime_metadata(subset_seed),
            "config_sha256": file_sha256(resolve_project_path(args.config)),
        },
        "commands": commands,
    }
    summary_path = output_dir / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if not args.dry_run:
        subprocess.run(
            [sys.executable, script("summarize_relation_run.py"), "--run_dir", str(output_dir)],
            cwd=ROOT,
            check=True,
        )
    print(json.dumps({"summary_path": str(summary_path), "output_dir": str(output_dir), "num_commands": len(commands)}, sort_keys=True))


if __name__ == "__main__":
    main()
