"""Explicit paper-interpretive CTL-DRP neural core; no formal-run dispatch.

The encoder is injected. Synthetic random tiny RoBERTa is not a pretrained
SimCSE checkpoint and never produces a benchmark reproduction claim.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from .ctl_drp_candidates import candidate_labels
from .ctl_drp_paper import ALIASES, load_lock as paper_lock, paper_facts
from .ctl_drp_simcse import load_lock as simcse_lock

MARKERS = ("[u1]", "[u2]", "[u3]", "[u4]")
RENDERING = {
    "id": "ctl_drp_iem_drp_pair_v1",
    "markers": list(MARKERS),
    "annotation_location": "inside_closing_marker_after_entity",
    "type_text": "lowercase_and_underscore_to_space_explicit_interpretation",
    "joining": "single_spaces_and_Roberta_sentence_pair",
    "max_length": 512,
    "overflow": "diagnostic_error_no_implicit_truncation",
    "candidate_policy": "frozen_training_observed_ordered_pairs_plus_NA",
    "paper_exact_identity_claim": False,
}


def marked_sentence(record):
    tokens = record.get("token")
    if (not isinstance(tokens, list) or not tokens
            or any(not isinstance(t, str) or not t for t in tokens)
            or any(marker in token for token in tokens for marker in MARKERS)):
        raise ValueError("nonempty tokens without reserved-marker collision required")
    entities = []
    for role, prefix, opening, closing in (("subject", "subj", MARKERS[0], MARKERS[1]),
                                           ("object", "obj", MARKERS[2], MARKERS[3])):
        start, end, type_ = (record.get(prefix+"_"+field) for field in ("start", "end", "type"))
        if (type(start) is not int or type(end) is not int or not 0 <= start <= end < len(tokens)
                or not isinstance(type_, str) or not type_ or type_.strip() != type_):
            raise ValueError("invalid entity span/type: " + role)
        entities.append(dict(role=role, start=start, end=end, type=type_,
                             text=" ".join(tokens[start:end+1]), opening=opening, closing=closing))
    subject, object_ = entities
    if max(subject["start"], object_["start"]) <= min(subject["end"], object_["end"]):
        raise ValueError("overlapping spans require an explicit different rendering")
    by_start = {entity["start"]: entity for entity in entities}
    result = []
    i = 0
    while i < len(tokens):
        if i in by_start:
            entity = by_start[i]
            type_text = entity["type"].lower().replace("_", " ")
            result.extend([entity["opening"], *tokens[i:entity["end"]+1],
                           "("+entity["role"]+" "+type_text+")", entity["closing"]])
            i = entity["end"]+1
        else:
            result.append(tokens[i]); i += 1
    return {"text": " ".join(result), "entities": entities, "raw_tokens": list(tokens),
            "rendering": dict(RENDERING)}


def descriptions_from_native(path):
    data = Path(path).read_bytes()
    lock = paper_lock()
    if hashlib.sha256(data).hexdigest() != lock["native_tex"]["sha256"]:
        raise ValueError("native paper bytes differ from locked evidence")
    descriptions = {}
    for name, text in paper_facts(data.decode())["descriptions"].items():
        label = ALIASES.get(name, name.replace("-", "_"))
        # The appendix entries are prose, not arbitrary TeX programs.
        text = text.replace(r"\&", "&").replace(r"\_", "_")
        if "\\" in text or "{" in text or "}" in text:
            raise ValueError("unsupported native description markup requires explicit handling")
        descriptions[label] = re.sub(r"\s+", " ", text).strip()
    labels = lock["canonical_labels"]
    return {label: descriptions[label] for label in labels}


def tokenizer_from_cache(cache, *, variant="sup"):
    """Verify small files actually consumed; no weights, network or auto-cache."""
    from tokenizers import AddedToken, processors
    from tokenizers.implementations import ByteLevelBPETokenizer
    lock = simcse_lock()
    model = lock["facts"]["models"][variant]
    prefix = "https://huggingface.co/"+model["id"]+"/resolve/"+model["revision"]+"/"
    paths = {}
    for name in ("vocab.json", "merges.txt"):
        row = next(row for row in lock["artifacts"] if row["url"] == prefix+name)
        path = Path(cache)/row["path"]
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("BPE file identity differs")
        paths[name] = path
    tok = ByteLevelBPETokenizer(str(paths["vocab.json"]), str(paths["merges.txt"]), add_prefix_space=False)
    tok.add_special_tokens(["<s>", "<pad>", "</s>", "<unk>",
                            AddedToken("<mask>", lstrip=True, normalized=False), *MARKERS])
    tok._tokenizer.post_processor = processors.RobertaProcessing(("</s>", 2), ("<s>", 0), add_prefix_space=False)
    if [tok.token_to_id(m) for m in MARKERS] != list(range(50265,50269)):
        raise ValueError("four new marker IDs changed")
    return tok


def encode_groups(records, table, tokenizer, descriptions, *, split, max_length=512):
    if split not in ("train", "dev", "test") or type(max_length) is not int or not 8 <= max_length <= 512:
        raise ValueError("explicit split and supported capacity required")
    result = []
    for index, record in enumerate(records):
        marked = marked_sentence(record)
        names = candidate_labels(table, record["subj_type"], record["obj_type"])
        pairs = []
        for name in names:
            prompt = ("There is no relation between subject and object" if name == "no_relation" else
                      "There is a relation between subject and object: " + descriptions[name])
            encoded = tokenizer.encode(marked["text"], prompt)
            if len(encoded.ids) > max_length:
                raise ValueError(f"sample {index} candidate {name}: overflow {len(encoded.ids)} > {max_length}; no sample dropped")
            marker_positions = {m: encoded.ids.index(tokenizer.token_to_id(m)) for m in MARKERS}
            if any(encoded.ids.count(tokenizer.token_to_id(m)) != 1 for m in MARKERS):
                raise ValueError("entity marker lost or duplicated")
            pairs.append({"input_ids": encoded.ids, "attention_mask": encoded.attention_mask,
                          "tokens": encoded.tokens, "offsets": encoded.offsets,
                          "marker_positions": marker_positions, "prompt": prompt})
        group = {"sample_id": record.get("id", "synthetic-"+str(index)), "split": split,
                 "marked": marked, "candidates": names, "pairs": pairs}
        if split == "train":
            gold = record.get("relation")
            if gold not in names:
                raise ValueError("gold excluded; no insertion/sample skipping")
            group["gold"] = gold
        result.append(group)
    if not result:
        raise ValueError("empty sentence group batch")
    return result


def collate_groups(groups, *, device="cpu"):
    import torch
    if not groups:
        raise ValueError("empty groups")
    pairs = [pair for group in groups for pair in group["pairs"]]
    width = max(len(pair["input_ids"]) for pair in pairs)
    ids, masks = [], []
    for pair in pairs:
        n = len(pair["input_ids"])
        ids.append(pair["input_ids"] + [1]*(width-n))
        masks.append(pair["attention_mask"] + [0]*(width-n))
    return {"input_ids": torch.tensor(ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(masks, dtype=torch.long, device=device),
            "candidate_groups": [list(group["candidates"]) for group in groups]}


def joint_model(encoder, *, head_seed=13, memory_efficient=True):
    """One injected encoder and one shared scalar head; not a 40-output head."""
    import torch
    from torch import nn
    canonical = paper_lock()["canonical_labels"]
    if memory_efficient:
        # Keep every candidate and FP32 operation. Recompute layer activations
        # during backward with the original dropout RNG instead of retaining
        # all 24 layers on the GPU. Actual case values are copied by hooks.
        encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={
            "use_reentrant": False, "preserve_rng_state": True})

    class JointModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.execution_memory = {"activation_checkpointing": bool(memory_efficient),
                                     "checkpoint_use_reentrant": False,
                                     "checkpoint_preserve_rng_state": True,
                                     "returned_layer_collections": False}
            # Initialize on CPU without consuming the caller's dropout RNG.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(head_seed)
                self.head = nn.Linear(encoder.config.hidden_size, 1)
                nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
                nn.init.zeros_(self.head.bias)

        def forward(self, input_ids, attention_mask, candidate_groups, *, gold=None):
            if (input_ids.ndim != 2 or 0 in input_ids.shape or attention_mask.shape != input_ids.shape
                    or input_ids.dtype != torch.long or input_ids.shape[1] > 512
                    or bool((input_ids < 0).any())
                    or bool((input_ids >= self.encoder.config.vocab_size).any())
                    or bool(((attention_mask != 0)&(attention_mask != 1)).any())
                    or not bool((attention_mask[:,0] == 1).all())
                    or not bool((input_ids[:,0] == 0).all())):
                raise ValueError("invalid joint pair tensors")
            if (not candidate_groups or sum(map(len, candidate_groups)) != input_ids.shape[0]
                    or gold is not None and len(gold) != len(candidate_groups)):
                raise ValueError("candidate sentence grouping differs from encoded pairs")
            for i, names in enumerate(candidate_groups):
                if (not names or "no_relation" not in names or len(set(names)) != len(names)
                        or any(n not in canonical for n in names)
                        or names != sorted(names,key=canonical.get)):
                    raise ValueError("unique canonical-order candidates including NA required")
                if gold is not None and gold[i] not in names:
                    raise ValueError("gold excluded; no repair")
            encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                                   output_hidden_states=False,
                                   output_attentions=False, return_dict=True)
            first = encoded.last_hidden_state[:,0,:]
            scores = self.head(first).squeeze(-1)
            probabilities, losses, choices = [], [], []
            cursor = 0
            for i, names in enumerate(candidate_groups):
                logp = torch.log_softmax(scores[cursor:cursor+len(names)],dim=0)
                probabilities.append(logp.exp())
                choices.append(logp.argmax())
                if gold is not None:
                    losses.append(-logp[names.index(gold[i])])
                cursor += len(names)
            predictions=[names[j] for names,j in zip(candidate_groups,torch.stack(choices).tolist())]
            return {"scores":scores, "probabilities":probabilities, "predictions":predictions,
                    "first_token":first, "hidden_states":encoded.hidden_states,
                    "attentions":encoded.attentions,
                    "per_sentence_loss":torch.stack(losses) if losses else None,
                    "loss":torch.stack(losses).mean() if losses else None}
    return JointModel()


def tiny_model(*, seed=13, dropout=0.0):
    import torch
    from transformers import RobertaConfig, RobertaModel
    torch.manual_seed(seed)
    config = RobertaConfig(vocab_size=50269, hidden_size=32, num_hidden_layers=2,
                           num_attention_heads=4, intermediate_size=64, max_position_embeddings=514,
                           hidden_dropout_prob=dropout, attention_probs_dropout_prob=dropout,
                           pad_token_id=1, bos_token_id=0, eos_token_id=2)
    config._attn_implementation = "eager"
    return joint_model(RobertaModel(config, add_pooling_layer=False),head_seed=seed)


def save_checkpoint(path, model, optimizer, *, contract, cursor):
    import torch
    import random
    if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("this diagnostic resume scope is CPU FP32; CUDA requires its own RNG/precision gate")
    if type(cursor) is not int or cursor < 1:
        raise ValueError("checkpoint only after a completed optimizer update")
    torch.save({"schema":"ctl-drp-post-update.v1", "phase":"post_optimizer_update",
                "contract":contract, "cursor":cursor, "model":model.state_dict(),
                "optimizer":optimizer.state_dict(), "training":model.training, "torch_rng":torch.get_rng_state(),
                "python_rng":random.getstate()},path)


def load_checkpoint(path, model, optimizer, *, contract):
    import torch
    import random
    value = torch.load(path,map_location="cpu",weights_only=True)
    if (value.get("schema") != "ctl-drp-post-update.v1" or value.get("phase") != "post_optimizer_update"
            or value.get("contract") != contract or type(value.get("cursor")) is not int or value["cursor"] < 1):
        raise ValueError("resume identity/phase drift rejected before mutation")
    expected = model.state_dict()
    if (set(value["model"]) != set(expected)
            or any(value["model"][k].shape != expected[k].shape or value["model"][k].dtype != expected[k].dtype for k in expected)):
        raise ValueError("resume architecture drift rejected before mutation")
    old_groups, new_groups = optimizer.state_dict()["param_groups"], value["optimizer"]["param_groups"]
    if len(old_groups) != len(new_groups) or any(len(a["params"]) != len(b["params"]) for a,b in zip(old_groups,new_groups)):
        raise ValueError("resume optimizer parameter topology differs")
    if (type(value.get("training")) is not bool or not torch.is_tensor(value.get("torch_rng"))
            or value["torch_rng"].dtype != torch.uint8 or value["torch_rng"].shape != torch.get_rng_state().shape):
        raise ValueError("resume RNG/mode drift rejected before mutation")
    random.Random().setstate(value["python_rng"])
    parameters=[p for group in optimizer.param_groups for p in group["params"]]
    indices=[i for group in new_groups for i in group["params"]]
    if len(indices)!=len(set(indices)):
        raise ValueError("resume optimizer indices are duplicated")
    for index,param in zip(indices,parameters):
        for name,state in value["optimizer"]["state"].get(index,{}).items():
            if torch.is_tensor(state) and (name!="step" and state.shape!=param.shape
                                         or name=="step" and state.numel()!=1):
                raise ValueError("resume optimizer tensor shape drift rejected before mutation")
    model.load_state_dict(value["model"],strict=True)
    optimizer.load_state_dict(value["optimizer"])
    torch.set_rng_state(value["torch_rng"])
    random.setstate(value["python_rng"])
    model.train(value["training"])
    return value["cursor"]
