"""Frozen Q-ECPA seed-13 protocol and shared, non-model lifecycle primitives."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import tempfile
from contextlib import contextmanager

import numpy as np
import torch

from q_attention.metrics import classification_report
from q_attention.tasks.relation import (RelationDataset, build_label_map, build_vocab,
    collate_relation_batch, load_relation_jsonl)

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = "retacred_q_ecpa_formal_single_seed"
MODES = ("disabled", "quantum", "dephased", "classical", "random_phase", "entity_permuted", "original_qk")
OOM_EXIT, PAUSED_EXIT = 86, 85


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_state(path, value):
    """Only latest and best survive; a failed atomic write preserves old latest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix="checkpoint-", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        torch.load(temporary, map_location="cpu", weights_only=False)
        os.replace(temporary, path)
        if os.name == "posix":
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def exclusive(path):
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another process owns {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def source_contract():
    paths = sorted((ROOT / "src/q_attention").rglob("*.py"))
    paths += [ROOT / "experiments/run_ecpa_formal.py", ROOT / "scripts/run_retacred_q_ecpa_formal_single_seed.sh"]
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def revision():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def load_config(canary=False):
    config = read(ROOT / "configs/retacred_q_ecpa_formal_single_seed.json")
    if canary:
        config.update(profile="synthetic_canary", expected_records={"train":12,"valid":6,"test":6})
        config["baseline"].update(epochs=2, batch_size=4)
        config["kernel"].update(epochs=2, batch_size=4)
    return config


def load_data(directory, config):
    directory = Path(directory).resolve()
    data, manifest = {}, {}
    for split, count in config["expected_records"].items():
        path = directory / f"{split}.jsonl"
        rows = load_relation_jsonl(path)
        if len(rows) != count:
            raise ValueError(f"{split}: expected {count} complete records, got {len(rows)}")
        data[split] = rows
        manifest[split] = dict(file=path.name, records=count, sha256=sha(path))
    vocab, labels = build_vocab(data["train"]), build_label_map(data["train"])
    if any(row.label not in labels for rows in data.values() for row in rows):
        raise ValueError("held-out label absent from training label map")
    # Position capacity, not a learned statistic. No records are truncated/dropped.
    maximum = max(len(row.tokens) for rows in data.values() for row in rows)
    datasets = {split: RelationDataset(rows, vocab, labels) for split,rows in data.items()}
    return data, datasets, vocab, labels, maximum, manifest


def synthetic_data(directory):
    from q_attention.tasks.ecpa_toy_data import datasets, LABELS
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    data = datasets(13)
    for split, count in (("train",12),("valid",6),("test",6)):
        path = directory / f"{split}.jsonl"
        rows = [dict(tokens=r["tokens"],subject=[r["subject"],r["subject"]+1],
                     object=[r["object"],r["object"]+1],label=LABELS[r["gold"]]) for r in data[split][:count]]
        text = "".join(json.dumps(r)+"\n" for r in rows)
        if path.exists() and path.read_text() != text:
            raise ValueError("canary data identity changed")
        path.write_text(text)


def batch(dataset, indexes):
    return collate_relation_batch([dataset[int(i)] for i in indexes], pad_id=0)


def inputs(value):
    return {key:value[key] for key in ("input_ids","attention_mask","subject_mask","object_mask")}


def epoch_order(count, seed, epoch):
    order = torch.randperm(count, generator=torch.Generator().manual_seed(seed + epoch)).tolist()
    # Freeze training trace sample 0 without changing the epoch's population.
    if epoch == 0:
        where = order.index(0)
        order[0], order[where] = order[where], order[0]
    return order


def update_indexes(count, logical, update, seed):
    batches = math.ceil(count / logical)
    epoch, position = divmod(update, batches)
    order = epoch_order(count, seed, epoch)
    return order[position*logical:(position+1)*logical]


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(value):
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])
    if value["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(value["cuda"])


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def memory_tiers(logical):
    # Owner policy: introduce physical microbatch only after actual OOM; linear
    # reductions, then exact recomputation. Never truncate matrix/token axes.
    micros = list(range(logical, 0, -8)) if logical > 16 else list(range(logical,0,-1))
    micros = sorted(set(micros + list(range(min(16,logical),0,-1))), reverse=True)
    return [dict(micro=n,recompute=False) for n in micros] + [dict(micro=1,recompute=True)]


def metrics(predicted, gold, labels, loss):
    report = classification_report(predicted,gold,len(labels),label_names={v:k for k,v in labels.items()})
    report["metrics"]["loss"] = loss / len(gold)
    null = labels.get("no_relation")
    tp = sum(p==g and g!=null for p,g in zip(predicted,gold))
    proposed, actual = sum(p!=null for p in predicted), sum(g!=null for g in gold)
    precision, recall = tp/max(1,proposed), tp/max(1,actual)
    report["metrics"].update(relation_micro_precision=precision,relation_micro_recall=recall,
        relation_micro_f1=2*precision*recall/max(1e-12,precision+recall))
    return report
