"""Offline hash-locked official SimCSE candidate; not exact author identity."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

from .ctl_drp_simcse import load_lock


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def regular(path):
    p = Path(path).absolute()
    if not p.is_file() or any(x.is_symlink() for x in (p, *p.parents)):
        raise ValueError("regular asset and parents required")
    return p


def load_encoder(weight, cache, *, variant="sup", marker_seed=13):
    import torch
    from transformers import RobertaConfig, RobertaModel
    lock = load_lock()  # Historical metadata audit stays immutable.
    if variant not in ("sup", "unsup") or type(marker_seed) is not int:
        raise ValueError("explicit locked variant and integer marker seed required")
    identity = lock["facts"]["models"][variant]
    p = regular(weight)
    if p.stat().st_size != identity["weight"]["bytes"] or sha(p) != identity["weight"]["sha256"]:
        raise ValueError("pretrained weight hash/size drift")
    url = f'https://huggingface.co/{identity["id"]}/resolve/{identity["revision"]}/config.json'
    row = next(r for r in lock["artifacts"] if r["url"] == url)
    config_path = regular(Path(cache) / row["path"])
    if sha(config_path) != row["sha256"]:
        raise ValueError("pretrained configuration drift")
    raw_config = json.loads(config_path.read_text())
    if raw_config != identity["config"]:
        raise ValueError("configuration and historical identity differ")
    config = RobertaConfig.from_dict(raw_config)
    config._attn_implementation = "eager"
    # Public converted checkpoint, safe data loading only; no downloaded code.
    with p.open("rb") as f:
        zipped = f.read(4) == b"PK\x03\x04"
    state = torch.load(p, map_location="cpu", weights_only=True, mmap=zipped)
    removed = []
    # Transformers 4.6 used a persistent position_ids buffer; 4.44 does not.
    if "embeddings.position_ids" in state:
        value = state["embeddings.position_ids"]
        if value.dtype != torch.int64 or not torch.equal(value, torch.arange(514).reshape(1, 514)):
            raise ValueError("legacy position_ids differs from exact constant buffer")
        del state["embeddings.position_ids"]
        removed.append("embeddings.position_ids: verified constant version-boundary buffer")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(marker_seed)
        encoder = RobertaModel(config, add_pooling_layer=True)
    expected = encoder.state_dict()
    if (set(state) != set(expected) or any(not torch.is_tensor(v) or v.shape != expected[k].shape
            or v.dtype != expected[k].dtype or not torch.isfinite(v).all() for k, v in state.items())):
        raise ValueError("strict pretrained state key/shape/dtype/finiteness gate failed")
    encoder.load_state_dict(state, strict=True)
    if any(not torch.equal(v, encoder.state_dict()[k]) for k, v in state.items()):
        raise ValueError("loaded parameter bytes differ")
    encoder.eval()
    literal = torch.tensor([[0, 31414, 232, 2, 2, 31414, 2]])
    with torch.no_grad():
        before = encoder(literal).last_hidden_state.clone()
    original_rows = encoder.embeddings.word_embeddings.weight.detach().clone()
    # CTL raw first-token readout explicitly bypasses the SimCSE pooler.
    encoder.pooler = None
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(marker_seed)
        encoder.resize_token_embeddings(50269)
        torch.nn.init.normal_(encoder.embeddings.word_embeddings.weight.data[50265:], std=.02)
    if not torch.equal(original_rows, encoder.embeddings.word_embeddings.weight[:50265]):
        raise ValueError("original vocabulary rows changed during marker extension")
    for name, tensor in encoder.state_dict().items():
        if name != "embeddings.word_embeddings.weight" and not torch.equal(tensor, state[name]):
            raise ValueError("backbone changed during marker extension: " + name)
    with torch.no_grad():
        after = encoder(literal).last_hidden_state
    if not torch.equal(before, after):
        raise ValueError("old-vocabulary forward changed at adapter boundary")
    receipt = {"id": identity["id"], "revision": identity["revision"], "weight_sha256": sha(p),
               "weight_bytes": p.stat().st_size, "safe_weights_only": True,
               "strict_parameter_equality": True, "removed_constant_buffers": removed,
               "removed_pooler": "verified official pooler; intentionally unused by raw CLS CTL readout",
               "marker_seed": marker_seed, "marker_std": .02, "marker_ids": list(range(50265, 50269)),
               "original_embedding_rows_unchanged": True, "old_vocabulary_forward_exact": True,
               "paper_exact_checkpoint_identity": False, "intended_grade": "paper_interpretive"}
    return encoder, receipt
