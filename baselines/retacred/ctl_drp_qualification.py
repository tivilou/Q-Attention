"""Bind an already-qualified physical dataset to CTL's native task boundary.

No download, automatic benchmark qualification, label repair or row filtering.
The optional suite bridge inherits its external qualification and records the
dev rename and stable IDs as explicit interface adaptations.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile

from .common import atomic_json, digest, fingerprint
from .ctl_drp_runner import safe, sync_dir
from .ctl_drp_task import COMPLETE_COUNTS, LABELS, PhysicalSplits, regular


def materialize(receipt, *, prepared=None, output=None, allow_synthetic=False):
    receipt = regular(receipt)
    spec = json.loads(receipt.read_text(encoding="utf-8"))
    if spec.get("schema") == "ctl-drp-physical-splits.v1":
        if prepared is not None:
            raise ValueError("native physical receipt does not take prepared-data")
        data = PhysicalSplits(receipt)
        if data.identity["synthetic"] and not allow_synthetic:
            raise ValueError("formal qualification requires non-synthetic complete splits")
        return receipt
    if prepared is None or output is None:
        raise ValueError("suite qualification requires explicit prepared-data and bridge output")
    prepared, output = safe(prepared), safe(output)
    manifest_path = regular(prepared / "data_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    synthetic = spec.get("synthetic", False)
    if (spec.get("status") != "qualified"
            or spec.get("task") != "supervised_sentence_entity_pair_relation_classification"
            or spec.get("labels") != 40 or spec.get("null_relation") != "no_relation"
            or type(synthetic) is not bool or synthetic and not allow_synthetic
            or spec.get("verification") != "converter_validates_labels_entity_spans_types_and_full_split_counts"
            or manifest.get("schema") != "q-attention.retacred-reference-improved-baseline-data.v1"
            or manifest.get("dataset") != "Re-TACRED" or manifest.get("format") != "upstream_tacred_json"
            or set(manifest.get("upstream_label_to_id", {})) != set(LABELS)
            or set(manifest.get("upstream_label_to_id", {}).values()) != set(range(40))
            or set(spec.get("splits", {})) != {"train", "valid", "test"}
            or spec["splits"] != manifest.get("splits")):
        raise ValueError("external qualified suite receipt/manifest boundary mismatch")
    parent_identity = {"qualification_sha256": digest(receipt), "manifest_sha256": digest(manifest_path)}
    native = {"schema": "ctl-drp-physical-splits.v1", "status": "qualified",
              "task": "retacred_sentence_relation_40", "canonical_labels": LABELS,
              "synthetic": synthetic, "splits": {}, "parent_qualification": parent_identity,
              "adaptations": {"valid_to_dev": True, "missing_ids": "split_row_ordinal_and_row_content_hash",
                              "source_path_dereferenced": False, "generated_ids": {}}}
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output.name + ".stage-", dir=output.parent))
    try:
        for split, source in (("train", "train"), ("dev", "valid"), ("test", "test")):
            entry = spec["splits"][source]
            # Never follow stale absolute paths from another machine's receipt.
            path = regular(prepared / ("dev.json" if source == "valid" else source + ".json"))
            if digest(path) != entry.get("output_sha256"):
                raise ValueError("qualified physical prepared split hash drift")
            rows = json.loads(path.read_text(encoding="utf-8"))
            if (not isinstance(rows, list) or not rows or type(entry.get("records")) is not int
                    or len(rows) != entry["records"] or not synthetic and len(rows) != COMPLETE_COUNTS[split]):
                raise ValueError("qualified physical prepared split count drift")
            generated = 0
            for i, row in enumerate(rows):
                if not isinstance(row, dict):
                    raise ValueError("qualified row must be a mapping")
                if "id" not in row:
                    row["id"] = f"ctl:{split}:{i}:{fingerprint(row)}"
                    generated += 1
                # Existing malformed/duplicate IDs are errors, not repaired.
            destination = stage / (split + ".json")
            atomic_json(destination, rows)
            native["splits"][split] = {"file": destination.name, "format": "json_array",
                                      "sha256": digest(destination), "count": len(rows),
                                      "qualified_prepared_sha256": entry["output_sha256"],
                                      "declared_canonical_source_sha256": entry["source_sha256"]}
            native["adaptations"]["generated_ids"][split] = generated
        atomic_json(stage / "qualification.json", native)
        PhysicalSplits(stage / "qualification.json")  # every row/span/label/ID and train-only H
        if output.exists():
            old = regular(output / "qualification.json")
            if json.loads(old.read_text(encoding="utf-8")) != native:
                raise ValueError("existing qualification bridge identity drift")
            PhysicalSplits(old)
        else:
            sync_dir(stage)
            os.rename(stage, output)
            sync_dir(output.parent)
        return output / "qualification.json"
    finally:
        if stage.exists():
            if safe(stage).parent != output.parent:
                raise ValueError("qualification staging cleanup boundary drift")
            shutil.rmtree(stage)
