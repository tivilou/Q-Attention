"""Recover hash-locked PTR metadata from the official repository's history.

Only the verbalizer and label mapping are fetched. Never download fork datasets
or treat resource recovery as a completed model implementation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from .assets import reject_symlinks, stream_download
from .common import ROOT, atomic_json, digest

REVISION = "420926b83025f313f27d77dbb5c74d7089121b89"
SOURCE = "https://raw.githubusercontent.com/thunlp/PTR/" + REVISION + "/datasets/retacred/"
FILES = {
    "temp.txt": {"sha256": "70f57788b93a83ea968c9ede19b79391a138b230e954ed01433deeb10f9799f6", "size_bytes": 2040},
    "rel2id.json": {"sha256": "bedf6ac6216a4098a0d174c610dec086d59ae455b5fb81dbd48a8d16ce83ce89", "size_bytes": 1009},
}


def validate_resources(directory: Path) -> dict:
    for name, expected in FILES.items():
        path = directory/name
        reject_symlinks(path)
        if path.stat().st_size != expected["size_bytes"] or digest(path) != expected["sha256"]:
            raise ValueError("PTR resource identity mismatch: " + name)
    labels = json.loads((directory/"rel2id.json").read_text())
    rows = [line.split("\t") for line in (directory/"temp.txt").read_text().splitlines() if line.strip()]
    if (len(labels) != 40 or labels.get("no_relation") != 0
            or sorted(labels.values()) != list(range(40))
            or len(rows) != 40 or any(len(row) != 7 for row in rows)
            or len({row[1] for row in rows}) != 40 or {row[1] for row in rows} != set(labels)):
        raise ValueError("PTR verbalizer / canonical 40-label interface mismatch")
    return {"status": "metadata_verified_adapter_pending", "classes": 40, "masked_objectives": 5,
            "labels": labels, "verbalizers": {row[1]: row[2:] for row in rows}}


def fetch_resources() -> Path:
    directory = ROOT/".external/retacred/ptr_metadata"/REVISION
    reject_symlinks(directory)
    for name, expected in FILES.items():
        stream_download(SOURCE + name, directory/name,
                        expected=expected["sha256"], size=expected["size_bytes"])
    result = validate_resources(directory)
    atomic_json(directory/"recovery_manifest.json", {
        "schema": "ptr-resource-recovery.v1", "status": result["status"],
        "source_url": "https://github.com/thunlp/PTR", "revision": REVISION,
        "time_utc": datetime.now(timezone.utc).isoformat(), "access": "public_https_pinned_commit",
        "files": {name: {**record, "url": SOURCE + name} for name,record in FILES.items()},
        "no_raw_dataset_downloaded": True, "reversed_recipe_verified": False})
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    directory = ROOT/".external/retacred/ptr_metadata"/REVISION if args.verify_only else fetch_resources()
    result = validate_resources(directory)
    print(json.dumps({"directory": str(directory), "status": result["status"],
                      "classes": result["classes"], "masked_objectives": result["masked_objectives"]}, indent=2))


if __name__ == "__main__":
    main()
