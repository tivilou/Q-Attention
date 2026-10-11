"""Offline MNLI identity, sentence-pair encoding and explicit budget contracts.

This module does not load model weights or admit a formal MetaEntail worker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .common import PACKAGE, fingerprint
from .metaentail import nli_order_checked

REPO = "FacebookAI/roberta-large-mnli"
REVISION = "2a8f12d27941090092df78e4ba6f0928eb5eac98"
LOCK = PACKAGE/"metaentail_checkpoint.lock.json"


def checkpoint_spec():
    spec = json.loads(LOCK.read_text(encoding="utf-8"))
    if (spec.get("schema") != "metaentail-checkpoint-contract.v1"
            or spec.get("provenance_mode") != "paper"
            or spec.get("fidelity_status") != "not_verifiable"
            or spec.get("strict_comparison_eligible") is not False
            or spec.get("neural_runtime_verified") is not False
            or spec["checkpoint"]["repository"] != REPO
            or spec["checkpoint"]["revision"] != REVISION):
        raise ValueError("MNLI identity/admission contract drift")
    return spec


def config_contract(config, tokenizer_config):
    """Read NLI IDs from the actual config, rejecting generic relation heads."""
    expected = checkpoint_spec()["config"]
    for key, value in expected.items():
        if key not in ("id2label", "label2id") and config.get(key) != value:
            raise ValueError("MNLI architecture/head mismatch: "+key)
    labels = config.get("id2label", {})
    if set(labels) != {"0", "1", "2"} or any(not isinstance(v,str) for v in labels.values()):
        raise ValueError("explicit three-class NLI id2label required")
    order = nli_order_checked(labels[str(i)].lower() for i in range(3))
    if config.get("label2id") != {labels[str(i)]:i for i in range(3)}:
        raise ValueError("NLI label2id must invert id2label")
    # RoBERTa positions start at pad_token_id + 1 (HF create_position_ids...).
    capacity = config["max_position_embeddings"] - config["pad_token_id"] - 1
    tokenizer_limit = tokenizer_config.get("model_max_length")
    if type(tokenizer_limit) is not int or tokenizer_limit != capacity or capacity != 512:
        raise ValueError("checkpoint/tokenizer length mismatch")
    return {"nli_order":list(order), "nli_label_ids":{name:i for i,name in enumerate(order)},
            "max_pair_tokens_including_specials":capacity, "pair_special_tokens":4,
            "content_token_budget":capacity-4, "relation_head_replaced":False}


def verify_metadata(directory):
    """Check six small official files only; never open a weight file."""
    directory = Path(directory).absolute()
    if directory.resolve() != directory:
        raise ValueError("symlinked MNLI metadata directory")
    spec = checkpoint_spec()
    files = {}
    for name, row in spec["checkpoint"]["files"].items():
        path = directory/name
        if path.resolve() != path.absolute() or not path.is_file() or path.stat().st_size != row["size_bytes"]:
            raise ValueError("MNLI small-file size/path mismatch: "+name)
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        blob = hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()
        if sha != row["sha256"] or blob != row["git_blob"]:
            raise ValueError("MNLI small-file identity mismatch: "+name)
        files[name] = {"size_bytes":len(raw),"sha256":sha}
    config = json.loads((directory/"config.json").read_text())
    token_config = json.loads((directory/"tokenizer_config.json").read_text())
    binding = config_contract(config, token_config)
    if config != dict(config, **spec["config"]):
        raise ValueError("pinned NLI configuration changed")
    return {"status":"checkpoint_metadata_verified", "repository":REPO, "revision":REVISION,
            "files":files, **binding, "weight_hash_evidence":"provider_metadata_only",
            "weights_locally_verified":False, "neural_runtime_verified":False,
            "formal_admission":False, "network_requests":0, "model_jobs":0}


def encode_pair(premise, hypothesis, tokenizer, *, max_length):
    """No silent truncation, entity substitution, weight/model creation or targets."""
    limit = checkpoint_spec()["input"]["max_pair_tokens_including_specials"]
    if type(max_length) is not int or not 4 < max_length <= limit:
        raise ValueError("explicit pair length must fit checkpoint capacity512")
    if any(not isinstance(text,str) or not text.strip() for text in (premise,hypothesis)):
        raise ValueError("nonempty premise and hypothesis required")
    if (getattr(tokenizer,"model_max_length",None) != limit
            or tokenizer.num_special_tokens_to_add(pair=True) != 4
            or getattr(tokenizer,"pad_token_id",None) != 1
            or getattr(tokenizer,"bos_token_id",None) != 0
            or getattr(tokenizer,"eos_token_id",None) != 2
            or getattr(tokenizer,"add_prefix_space",None) is not False):
        raise ValueError("tokenizer/checkpoint pair interface mismatch")
    encoded = tokenizer(premise,hypothesis,add_special_tokens=True,truncation=False,
                        padding=False,return_attention_mask=True,return_special_tokens_mask=True,
                        return_offsets_mapping=True,return_token_type_ids=False)
    ids = encoded["input_ids"]
    if len(ids)>max_length:
        raise ValueError(f"pair exceeds declared length: actual={len(ids)} limit={max_length}; preserved input, no truncation")
    sequence_ids = encoded.sequence_ids()
    special = encoded["special_tokens_mask"]
    offsets = encoded["offset_mapping"]
    if (not ids or ids[0]!=0 or ids[-1]!=2 or sum(special)!=4
            or any(len(encoded[key])!=len(ids) for key in ("attention_mask","special_tokens_mask","offset_mapping"))
            or len(sequence_ids)!=len(ids) or set(sequence_ids)-{None,0,1}
            or any(value!=1 for value in encoded["attention_mask"])):
        raise ValueError("invalid unpadded pair encoding")
    return {"premise":premise,"hypothesis":hypothesis,"input_ids":list(ids),
            "attention_mask":list(encoded["attention_mask"]),
            "tokens":tokenizer.convert_ids_to_tokens(ids),"special_tokens_mask":list(special),
            "sequence_ids":sequence_ids,"offset_mapping":[list(pair) for pair in offsets],
            "pair_tokens":len(ids),"max_length":max_length,"truncated":False,"target_access":"none"}


def expanded_pair_budget(candidate_counts, *, epochs, pair_batch_size, accumulation_steps):
    """Proposed flat-pair schedule with complete tails and epoch-end flush.

    Every parameter is explicit. This is planning arithmetic, not a paper recipe.
    """
    for name, value in (("epochs",epochs),("pair_batch_size",pair_batch_size),
                        ("accumulation_steps",accumulation_steps)):
        if type(value) is not int or value<=0:
            raise ValueError("positive integer required: "+name)
    groups, pairs = 0, 0
    projection = hashlib.sha256()
    for count in candidate_counts:
        if type(count) is not int or not 1 <= count <= 40:
            raise ValueError("each training group must contain1..40 feasible pairs")
        groups += 1; pairs += count
        projection.update((str(count)+"\n").encode())
    if not groups:
        raise ValueError("nonempty training candidate counts required")
    mini = (pairs+pair_batch_size-1)//pair_batch_size
    updates = (mini+accumulation_steps-1)//accumulation_steps
    last_mini_pairs = pairs-(mini-1)*pair_batch_size
    last_window_mini = mini-(updates-1)*accumulation_steps
    last_window_pairs = pairs-(updates-1)*accumulation_steps*pair_batch_size
    result = {"unit":"expanded_feasible_pairs", "original_examples":groups,"pairs_per_epoch":pairs,
              "candidate_count_projection_sha256":projection.hexdigest(),
              "epochs":epochs,"pair_batch_size":pair_batch_size,"accumulation_steps":accumulation_steps,
              "mini_batches_per_epoch":mini,"optimizer_updates_per_epoch":updates,
              "total_pair_presentations":pairs*epochs,"total_optimizer_updates":updates*epochs,
              "last_mini_batch_pairs":last_mini_pairs,"last_update_mini_batches":last_window_mini,
              "last_update_pairs":last_window_pairs,
              "loss_denominator":"actual_pairs_in_optimizer_window", "drop_last":False,
              "flush_accumulation_at_epoch_end":True,"policy":"proposed_adapter_not_paper_recipe",
              "formal_admission":False,"model_jobs":0}
    result["contract_sha256"] = fingerprint(result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir",type=Path,required=True)
    args = parser.parse_args(argv)
    print(json.dumps(verify_metadata(args.metadata_dir),indent=2))


if __name__ == "__main__":
    main()
