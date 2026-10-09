"""Dependency-light identity, metrics, source pinning and atomic artifacts."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def cohort() -> dict:
    return json.loads((PACKAGE / "cohort.json").read_text(encoding="utf-8"))


def select_models(raw: str) -> list[dict]:
    registry = cohort()
    names = registry["default_models"] if raw == "ready" else [x.strip() for x in raw.split(",")]
    if raw == "all":
        names = [row["id"] for row in registry["models"] if row.get("run_eligible", True)]
    index = {row["id"]: row for row in registry["models"]}
    if not names or len(set(names)) != len(names):
        raise ValueError("model list is empty or contains duplicates")
    unknown = [name for name in names if name not in index]
    if unknown:
        raise ValueError(f"unknown models: {unknown}")
    blocked = [f"{name}: {index[name].get('reason', index[name]['status'])}" for name in names
               if index[name]["status"] != "implemented_runtime_gate_required"]
    if blocked:
        raise ValueError("Requested cohort is not runnable; no jobs were launched:\n" + "\n".join(blocked))
    return [index[name] for name in names]


def pin_source(key: str, *, fetch: bool = True) -> Path:
    spec = cohort()["sources"][key]
    relative = Path(spec.get("cache_dir", ".external/retacred/" + key))
    if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != ".external":
        raise ValueError("source cache must be a project-relative .external path")
    path = ROOT / relative
    from .assets import reject_symlinks
    reject_symlinks(path)
    if not path.exists():
        if not fetch:
            raise FileNotFoundError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", spec["url"], str(path)], check=True)
        subprocess.run(["git", "-C", str(path), "checkout", "--detach", spec["commit"]], check=True)
    untracked = git("ls-files", "--others", "--exclude-standard", cwd=path).splitlines()
    # Python caches from earlier imports are harmless here: load_module compiles
    # pinned .py bytes directly and never executes cached bytecode.
    foreign = [name for name in untracked if not ("__pycache__/" in name and name.endswith(".pyc"))]
    if path.is_symlink() or git("status", "--porcelain", "--untracked-files=no", cwd=path) or foreign:
        raise ValueError(f"unsafe or dirty upstream source: {path}")
    if git("rev-parse", "HEAD", cwd=path) != spec["commit"]:
        raise ValueError(f"upstream commit mismatch: {path}")
    if git("remote", "get-url", "origin", cwd=path).removesuffix(".git") != spec["url"].removesuffix(".git"):
        raise ValueError("upstream origin mismatch")
    return path


def metrics(gold: list[int], predicted: list[int], labels: list[str], loss: float) -> dict:
    if not gold or len(gold) != len(predicted) or not 0 <= loss < float("inf"):
        raise ValueError("invalid metric inputs")
    n = len(labels)
    confusion = [[0] * n for _ in range(n)]
    for true, pred in zip(gold, predicted):
        if not 0 <= true < n or not 0 <= pred < n:
            raise ValueError("label index out of bounds")
        confusion[true][pred] += 1
    per_class = []
    for i, name in enumerate(labels):
        tp = confusion[i][i]
        support = sum(confusion[i])
        guesses = sum(row[i] for row in confusion)
        p, r = (tp / guesses if guesses else 0), (tp / support if support else 0)
        per_class.append({"label": name, "support": support, "precision": p, "recall": r,
                          "f1": 2*p*r/(p+r) if p+r else 0})
    null = labels.index("no_relation")
    correct = sum(confusion[i][i] for i in range(n) if i != null)
    positives = sum(sum(row) for i, row in enumerate(confusion) if i != null)
    guesses = sum(sum(row[i] for row in confusion) for i in range(n) if i != null)
    p, r = correct / guesses if guesses else 0, correct / positives if positives else 0
    accuracy = sum(confusion[i][i] for i in range(n)) / len(gold)
    return {"items": len(gold), "loss": loss, "accuracy": accuracy,
            "micro_precision": p, "micro_recall": r, "micro_f1": 2*p*r/(p+r) if p+r else 0,
            "micro_convention": "benchmark_excluding_no_relation",
            "all_label_micro_precision": accuracy, "all_label_micro_recall": accuracy,
            "all_label_micro_f1": accuracy,
            **{f"macro_{key}": sum(row[key] for row in per_class)/n for key in ("precision", "recall", "f1")},
            "macro_convention": "all_40_labels_zero_division_0", "per_class": per_class,
            "confusion_matrix": confusion}
