"""CTL-DRP task/recipe adapter. Callable, but not admitted to the formal queue.

Physical split identities come from an external qualification receipt, never
from a split name supplied to H fitting. Synthetic receipts are diagnostic only.
This module does not launch training, download assets or publish results.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import random

from .common import digest, fingerprint, metrics
from .ctl_drp_candidates import candidate_labels, coverage_diagnostic, fit_train_pairs
from .ctl_drp_neural import (collate_groups, descriptions_from_native, encode_groups,
                             joint_model, marked_sentence, tokenizer_from_cache)
from .ctl_drp_paper import load_lock as paper_lock

LABEL_IDS = paper_lock()["canonical_labels"]
LABELS = sorted(LABEL_IDS, key=LABEL_IDS.get)
RECIPE = {
    "effective_batch_sentences": 512, "epochs": 10, "learning_rate": 1e-5,
    "optimizer": "torch.optim.AdamW", "betas": [.9, .999], "eps": 1e-8,
    "weight_decay": 0., "scheduler": "constant", "warmup": 0,
    "gradient_clipping": None, "precision": "FP32_no_autocast",
    "foreach": False, "fused": False, "selection": "final_epoch",
    "seeds": [13, 29, 53], "shuffle": "local_python_Random_seed_plus_epoch",
    "micro_batch": "immutable_sentences_not_expanded_candidate_pairs",
    "overflow": "error_no_truncation_or_dropped_rows",
    "unknown_pair": "preflight_error_no_heldout_label_repair",
    "heldout_excluded_gold": "retain_in_all_classification_metrics_zero_probability_null_infinite_NLL",
}
COMPLETE_COUNTS = {"train": 58465, "dev": 19584, "test": 13418}


def regular(path):
    path = Path(path).absolute()
    if not path.is_file() or path.resolve() != path or any(p.is_symlink() for p in path.parents):
        raise ValueError("regular physical file and parents required")
    return path


class PhysicalSplits:
    """Explicit physical JSON-array/JSONL boundary, train/dev/test only.

    The receipt schema is ctl-drp-physical-splits.v1, status qualified, task
    retacred_sentence_relation_40, canonical_labels in ID order, and synthetic
    bool. splits names train/dev/test each bind a manifest-relative regular file,
    format (json_array or jsonl), sha256 and positive count. No auto-qualification.
    """
    def __init__(self, receipt):
        receipt = regular(receipt)
        spec = json.loads(receipt.read_text(encoding="utf-8"))
        if (spec.get("schema") != "ctl-drp-physical-splits.v1" or spec.get("status") != "qualified"
                or spec.get("task") != "retacred_sentence_relation_40"
                or spec.get("canonical_labels") != LABELS or type(spec.get("synthetic")) is not bool
                or set(spec.get("splits", {})) != {"train", "dev", "test"}):
            raise ValueError("qualified explicit canonical physical split receipt required")
        self.records, self.identity = {}, {"qualification_sha256": digest(receipt),
                                          "task": spec["task"], "synthetic": spec["synthetic"], "splits": {}}
        if "parent_qualification" in spec:
            self.identity["interface_qualification"] = {"parent": copy.deepcopy(spec["parent_qualification"]),
                                                        "adaptations": copy.deepcopy(spec["adaptations"])}
        paths, all_ids = set(), set()
        for split in ("train", "dev", "test"):
            entry = spec["splits"][split]
            relative = Path(entry["file"])
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise ValueError("split must remain inside receipt directory")
            path = regular(receipt.parent / relative)
            raw = path.read_bytes()
            if path in paths or hashlib.sha256(raw).hexdigest() != entry.get("sha256"):
                raise ValueError("split identity drift or physical file reuse")
            paths.add(path)
            text = raw.decode("utf-8")
            if entry.get("format") == "json_array":
                rows = json.loads(text)
            elif entry.get("format") == "jsonl":
                rows = [json.loads(line) for line in text.splitlines() if line.strip()]
            else:
                raise ValueError("explicit json_array/jsonl format required")
            if (not isinstance(rows, list) or not rows or type(entry.get("count")) is not int
                    or len(rows) != entry["count"]):
                raise ValueError("physical split count mismatch")
            if not spec["synthetic"] and len(rows) != COMPLETE_COUNTS[split]:
                raise ValueError("real reference requires qualified complete Re-TACRED split counts")
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError("physical sample must be a mapping")
                name = row.get("id")
                if not isinstance(name, str) or not name or name in all_ids:
                    raise ValueError("unique nonempty sample IDs across splits required")
                all_ids.add(name)
                if row.get("relation") not in LABEL_IDS:
                    raise ValueError("canonical relation required; no label alias repair")
                marked_sentence(row)  # validate roles/spans before any model load
            self.records[split] = rows
            self.identity["splits"][split] = {"sha256": entry["sha256"], "count": len(rows), "format": entry["format"]}
            for key in ("qualified_prepared_sha256", "declared_canonical_source_sha256"):
                if key in entry:
                    self.identity["splits"][split][key] = entry[key]
        self.table = fit_train_pairs(self.records["train"], source_split="train")
        # Do not mutate the historical primitive's table flags; bind physical
        # identity in this owner-side wrapper and freeze its semantic fingerprint.
        self.binding = {"schema": "ctl-drp-physical-H-binding.v1", "data": self.identity,
                        "table_sha256": self.table["table_sha256"], "source": "physical_train_only",
                        "physical_split_identity_verified": True, "author_exact_H": False}
        self.coverage = {s: coverage_diagnostic(self.table, rows) for s, rows in self.records.items()}
        if any(report["unknown_pair_indices"] for report in self.coverage.values()):
            raise ValueError("unseen physical training type pair; preflight H policy error: " + json.dumps(self.coverage))

    def assert_unchanged(self):
        current = fit_train_pairs(self.records["train"], source_split="train")
        if current != self.table:
            raise ValueError("in-memory train support or frozen H drift")


def audit_encodings(data, tokenizer, descriptions):
    """Full receipt-bound overflow/marker coverage before loading any weights.

    Rows are encoded lazily, without keeping a full expanded feature dataset.
    Evaluation labels are removed before encoding and do not affect the hash.
    """
    audits = {}
    for split, rows in data.records.items():
        value, pairs, longest = hashlib.sha256(), 0, 0
        for row in rows:
            record = copy.deepcopy(row)
            if split != "train":
                record.pop("relation")
            group = encode_groups([record], data.table, tokenizer, descriptions, split=split)[0]
            value.update(bytes.fromhex(fingerprint(group)))
            pairs += len(group["pairs"])
            longest = max(longest, *(len(p["input_ids"]) for p in group["pairs"]))
        audits[split] = {"sentences": len(rows), "candidate_pairs": pairs, "max_pair_length": longest,
                         "encoded_sequence_sha256": value.hexdigest(), "rows_dropped": 0}
    return audits


class UpdatePlan:
    """Random-access post-update cursor without global RNG consumption.

    Cursor is the number of *completed* logical updates, not sentence/pair count.
    The fixed seed, epochs, train count and batch width reproduce every next index.
    Full checkpoint lifecycle remains a runner responsibility.
    """
    def __init__(self, count, seed, *, batch_size=512, epochs=10):
        if any(type(n) is not int or n < 1 for n in (count, batch_size, epochs)) or seed not in RECIPE["seeds"]:
            raise ValueError("positive plan counts and frozen seed required")
        self.count, self.seed, self.batch_size, self.epochs = count, seed, batch_size, epochs
        self.per_epoch = math.ceil(count / batch_size)
        self.total = self.per_epoch * epochs

    def at(self, cursor):
        if type(cursor) is not int or not 0 <= cursor <= self.total:
            raise ValueError("invalid completed-update cursor")
        if cursor == self.total:
            return None
        epoch, batch = divmod(cursor, self.per_epoch)
        order = list(range(self.count))
        random.Random(self.seed + epoch).shuffle(order)
        return {"epoch": epoch, "batch": batch, "update": cursor,
                "indices": order[batch*self.batch_size:(batch+1)*self.batch_size]}


class CTLDRPTask:
    """Ragged candidates remain private to each sentence's Eq2 denominator."""
    def __init__(self, data, tokenizer, descriptions, model, *, seed=13, micro_batch=4,
                 device="cpu", initialization=None, diagnostic=True, encoding_audit=None):
        if seed not in RECIPE["seeds"] or type(micro_batch) is not int or not 1 <= micro_batch <= 512:
            raise ValueError("frozen seed and physical sentence micro-batch in 1..512 required")
        if set(descriptions) != set(LABELS) or any(not isinstance(s, str) or not s for s in descriptions.values()):
            raise ValueError("complete canonical descriptions required")
        if type(diagnostic) is not bool or not diagnostic and (data.identity["synthetic"] or not initialization):
            raise ValueError("non-synthetic qualified data and verified pretrained receipt required")
        data.assert_unchanged()
        self.data, self.tokenizer, self.descriptions = data, tokenizer, copy.deepcopy(descriptions)
        self.model, self.seed, self.micro_batch, self.device = model.to(device), seed, micro_batch, device
        self.initialization, self.diagnostic = copy.deepcopy(initialization), diagnostic
        self.plan = UpdatePlan(len(data.records["train"]), seed)
        self.contract = {
            "schema": "ctl-drp-task-contract.v1", "data_H": copy.deepcopy(data.binding),
            "description_sha256": fingerprint(descriptions), "recipe": copy.deepcopy(RECIPE),
            "seed": seed, "physical_micro_batch_sentences": micro_batch, "initialization": self.initialization,
            "model_configuration": copy.deepcopy(self.model.encoder.config.to_dict()),
            "execution_memory": copy.deepcopy(self.model.execution_memory),
            "diagnostic": diagnostic, "formal_comparison_admitted": False,
            "encoding_audit": copy.deepcopy(encoding_audit),
            "project_code": {name: digest(Path(__file__).with_name(name)) for name in (
                "ctl_drp_task.py", "ctl_drp_neural.py", "ctl_drp_candidates.py", "common.py")},
        }
        self.contract_sha256 = fingerprint(self.contract)

    @classmethod
    def from_assets(cls, receipt, cache, paper_source, weight, *, variant="sup", seed=13, micro_batch=4, device="cuda:0"):
        if seed not in RECIPE["seeds"] or type(micro_batch) is not int or not 1 <= micro_batch <= 512:
            raise ValueError("frozen seed and supported micro-batch required before weight loading")
        data = PhysicalSplits(receipt)
        if data.identity["synthetic"]:
            raise ValueError("pretrained task factory requires real qualified data")
        tokenizer = tokenizer_from_cache(cache, variant=variant)
        descriptions = descriptions_from_native(paper_source)
        audit = audit_encodings(data, tokenizer, descriptions)
        from .ctl_drp_pretrained import load_encoder
        encoder, initialization = load_encoder(weight, cache, variant=variant, marker_seed=seed)
        return cls(data, tokenizer, descriptions, joint_model(encoder, head_seed=seed), seed=seed,
                   micro_batch=micro_batch, device=device, initialization=initialization, diagnostic=False, encoding_audit=audit)

    def encode(self, split, indices):
        if split not in self.data.records or not indices or len(indices) != len(set(indices)):
            raise ValueError("explicit nonempty unique split indices required")
        rows = self.data.records[split]
        if any(type(i) is not int or not 0 <= i < len(rows) for i in indices):
            raise ValueError("sample index out of bounds")
        # Eval gold is omitted from the producer input, not just ignored later.
        selected = [copy.deepcopy(rows[i]) for i in indices]
        if split != "train":
            for row in selected:
                row.pop("relation")
        return encode_groups(selected, self.data.table, self.tokenizer, self.descriptions, split=split)

    def make_optimizer(self):
        import torch
        return torch.optim.AdamW(self.model.parameters(), lr=RECIPE["learning_rate"], betas=tuple(RECIPE["betas"]),
                                 eps=RECIPE["eps"], weight_decay=RECIPE["weight_decay"], foreach=False, fused=False)

    def train_update(self, indices, optimizer, *, observer=None):
        """One complete logical batch; tail weights use its actual sentence count.

        Micro-batch partitions are immutable for resume because dropout changes.
        No OOM retry or mid-update checkpoint is implemented here. A runner must
        recover a failed update from its last complete durable checkpoint.
        """
        import torch
        expected_params = list(self.model.parameters())
        if (type(optimizer) is not torch.optim.AdamW or not indices or len(indices) > 512
                or len(optimizer.param_groups) != 1
                or [id(p) for p in optimizer.param_groups[0]["params"]] != [id(p) for p in expected_params]
                or any(optimizer.param_groups[0].get(k) != v for k, v in {
                    "lr": RECIPE["learning_rate"], "betas": tuple(RECIPE["betas"]), "eps": RECIPE["eps"],
                    "weight_decay": RECIPE["weight_decay"], "foreach": False, "fused": False,
                    "amsgrad": False, "maximize": False, "capturable": False, "differentiable": False}.items())):
            raise ValueError("frozen AdamW and at most 512 logical sentences required")
        # Validate the entire logical input before the first gradient mutation.
        groups = self.encode("train", indices)
        self.model.train()
        optimizer.zero_grad(set_to_none=True)
        value = 0.
        for start in range(0, len(groups), self.micro_batch):
            part = groups[start:start+self.micro_batch]
            batch = collate_groups(part, device=self.device)
            scale = len(part) / len(groups)
            context = observer.context("train", part, batch, scale) if observer else nullcontext()
            with context:
                result = self.model(**batch, gold=[g["gold"] for g in part])
                if observer: observer.forward(result)
                scaled = result["loss"] * scale
                if not bool(torch.isfinite(scaled)):
                    raise ValueError("nonfinite CTL loss; optimizer update not performed")
            # Stop forward observation before activation recomputation. Tensor
            # gradient hooks remain attached to the original forward graph, so
            # actual gradients are still recorded without duplicate forwards.
            scaled.backward()
            value += float(scaled.detach())
            del result, scaled
        optimizer.step()
        if observer: observer.post_update()
        return {"loss": value, "sentences": len(groups), "pairs": sum(len(g["pairs"]) for g in groups),
                "micro_batches": math.ceil(len(groups)/self.micro_batch), "optimizer_updates": 1}

    def evaluate(self, split, *, observer=None):
        """Complete split classification metrics; test access is runner-gated.

        For heldout gold outside H, probability is exactly zero. Such samples
        remain in all classification metrics; NLL is explicitly infinite/null,
        not an epsilon-floor repair, covered-subset loss or silently skipped row.
        """
        import torch
        if split not in ("dev", "test"):
            raise ValueError("heldout evaluation split required")
        was_training = self.model.training
        gold, predicted, nll, excluded, predictions = [], [], [], [], []
        self.model.eval()
        try:
            with torch.no_grad():
                for start in range(0, len(self.data.records[split]), self.micro_batch):
                    indices = list(range(start, min(start+self.micro_batch, len(self.data.records[split]))))
                    groups = self.encode(split, indices)
                    batch = collate_groups(groups, device=self.device)
                    context = observer.context(split, groups, batch, 1.) if observer else nullcontext()
                    with context:
                        result = self.model(**batch)
                        if observer: observer.forward(result)
                    if observer: observer.post_evaluation()
                    offset = 0
                    for index, group, prob, name in zip(indices, groups, result["probabilities"], result["predictions"]):
                        true = self.data.records[split][index]["relation"]
                        if not bool(torch.isfinite(prob).all()):
                            raise ValueError("nonfinite heldout probabilities")
                        gold.append(LABEL_IDS[true]); predicted.append(LABEL_IDS[name])
                        if true not in group["candidates"]:
                            excluded.append(index)
                        else:
                            # Recompute stable logp from actual scores, not log(exp(logp)).
                            scores = result["scores"][offset:offset+len(group["candidates"])]
                            nll.append(float(-torch.log_softmax(scores, 0)[group["candidates"].index(true)]))
                        dense = [0.] * len(LABELS)
                        for label, p in zip(group["candidates"], prob.tolist()):
                            dense[LABEL_IDS[label]] = p
                        predictions.append({"sample_id": group["sample_id"], "predicted": name,
                                            "probabilities_canonical40": dense})
                        offset += len(group["candidates"])
        finally:
            self.model.train(was_training)
        loss = None if excluded else sum(nll) / len(gold)
        report = metrics(gold, predicted, LABELS, loss)
        report.update(split=split, loss_status="infinite_zero_gold_support" if excluded else "finite_mean_all_sentences",
                      excluded_gold_indices=excluded, excluded_gold_count=len(excluded),
                      candidate_support="physical_train_only", gold_used_for_encoding=False, samples_dropped=0)
        return report, predictions


def describe():
    return {"schema": "ctl-drp-task-adapter.v1", "status": "implemented_bounded_gates_only",
            "recipe": RECIPE, "formal_comparison_admitted": False, "model_jobs": 0,
            "pending": ["real_qualified_split_receipt", "formal_runner_checkpoint_and_trace_lifecycle",
                        "independent_mechanism_review", "unsupervised_pretrained_runtime"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(describe(), indent=2))


if __name__ == "__main__":
    main()
