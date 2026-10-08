"""Offline official SimCSE candidate audit, not a CTL-DRP encoder or admission."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path

from .ctl_drp_paper import regular, sha

PACKAGE = Path(__file__).resolve().parent
SOURCE = "https://github.com/princeton-nlp/SimCSE"
COMMIT = "49fa580a853752ede55b8c76d9debf748e214d3f"
REVISIONS = {
    "sup": "96d164d9950b72f4ce179cb1eb3414de0910953f",
    "unsup": "d3f863b476c59b0673264042f159cea15842e265",
}
TOKEN_FILES = ("tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt")

def git_blob(content: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()

def load_lock(path: Path = PACKAGE / "ctl_drp_simcse.lock.json") -> dict:
    lock = json.loads(regular(path).read_text(encoding="utf-8"))
    if (lock["schema"] != "retacred-ctl-drp-simcse-candidates.v1"
            or lock["source_commit"] != COMMIT or lock["source_url"] != SOURCE
            or lock["paper_checkpoint_identity_bound"] is not False
            or lock["formal_comparison_admitted"] is not False
            or lock["weights_downloaded"] is not False
            or lock["weights_numerically_verified"] is not False
            or lock["paper_literals"] != ["sup-smi-roberta-large", "unsup-smi-roberta-large"]
            or lock["identity_status"] != "official_candidates_paper_exact_mapping_unconfirmed"
            or lock["ctl_readout"] != "last_hidden_state[:,0,:]_without_pooler_or_l2_normalization"
            or len(lock["artifacts"]) != 24
            or len({x["url"] for x in lock["artifacts"]}) != 24
            or len({x["path"] for x in lock["artifacts"]}) != 24
            or any(Path(x["path"]).name != x["path"] for x in lock["artifacts"])):
        raise ValueError("candidate provenance or admission boundary differs")
    for variant, revision in REVISIONS.items():
        model = lock["facts"]["models"][variant]
        if (model["id"] != "princeton-nlp/" + variant + "-simcse-roberta-large"
                or model["revision"] != revision
                or model["weight"]["hash_status"] != "provider_metadata_only_not_local_bytes"):
            raise ValueError("official candidate identity differs")
    return lock

def derive_facts(contents: dict[str, bytes]) -> dict:
    """Read JSON/AST only. Never import/execute downloaded model or converter code."""
    def data(url):
        return json.loads(contents[url])
    raw = "https://raw.githubusercontent.com/princeton-nlp/SimCSE/" + COMMIT + "/"
    repo = data("https://api.github.com/repos/princeton-nlp/SimCSE")
    commit = data("https://api.github.com/repos/princeton-nlp/SimCSE/commits/main")
    tree = data("https://api.github.com/repos/princeton-nlp/SimCSE/git/trees/main?recursive=1")
    if (repo["full_name"] != "princeton-nlp/SimCSE" or commit["sha"] != COMMIT
            or tree["truncated"] is not False):
        raise ValueError("official source identity differs")
    paths = {x["path"]: x for x in tree["tree"] if x["type"] == "blob"}
    for url, content in contents.items():
        if url.startswith(raw):
            name = url[len(raw):]
            if paths[name]["sha"] != git_blob(content) or paths[name]["size"] != len(content):
                raise ValueError("source blob identity differs: " + name)
    readme = contents[raw + "README.md"].decode()
    models_ast = ast.parse(contents[raw + "simcse/models.py"])
    classes = {x.name: x for x in models_ast.body if isinstance(x, ast.ClassDef)}
    pooler = ast.unparse(classes["Pooler"])
    mlp = ast.unparse(classes["MLPLayer"])
    evaluation = ast.unparse(ast.parse(contents[raw + "evaluation.py"]))
    tool = ast.unparse(ast.parse(contents[raw + "simcse/tool.py"]))
    converter = ast.unparse(ast.parse(contents[raw + "simcse_to_huggingface.py"]))
    if ("return last_hidden[:, 0]" not in pooler
            or "nn.Linear(config.hidden_size, config.hidden_size)" not in mlp
            or "nn.Tanh()" not in mlp
            or "return pooler_output.cpu()" not in evaluation
            or "return last_hidden[:, 0].cpu()" not in evaluation
            or "outputs.pooler_output" not in tool or "outputs.last_hidden_state[:, 0]" not in tool
            or "normalize_to_unit:bool=True" not in tool.replace(" ", "")
            or "key.replace('mlp', 'pooler')" not in converter
            or "key.replace('roberta.', '')" not in converter):
        raise ValueError("pooling/conversion source witness differs")
    models = {}
    token_bytes = {}
    special_ids = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, "<mask>": 50264}
    for variant, revision in REVISIONS.items():
        name = "princeton-nlp/" + variant + "-simcse-roberta-large"
        api = data("https://huggingface.co/api/models/" + name + "?blobs=true")
        prefix = "https://huggingface.co/" + name + "/resolve/" + revision + "/"
        if api["id"] != name or api["sha"] != revision or api["private"] is not False or api["gated"] is not False:
            raise ValueError("candidate revision/access differs")
        siblings = {x["rfilename"]: x for x in api["siblings"]}
        for filename in ("config.json", *TOKEN_FILES):
            blob = contents[prefix + filename]
            if git_blob(blob) != siblings[filename]["blobId"] or len(blob) != siblings[filename]["size"]:
                raise ValueError("checkpoint metadata/file blob differs")
        config = data(prefix + "config.json")
        tokenizer = data(prefix + "tokenizer_config.json")
        vocab = data(prefix + "vocab.json")
        if (config["architectures"] != ["RobertaModel"] or config["model_type"] != "roberta"
                or config["hidden_size"] != 1024 or config["num_hidden_layers"] != 24
                or config["vocab_size"] != 50265 or config["max_position_embeddings"] != 514
                or len(vocab) != 50265 or set(vocab.values()) != set(range(50265))
                or any(vocab[k] != v for k, v in special_ids.items())
                or tokenizer["add_prefix_space"] is not False or tokenizer["model_max_length"] != 512):
            raise ValueError("tokenizer/backbone contract differs")
        if name not in readme:
            raise ValueError("official README candidate link missing")
        weight = siblings["pytorch_model.bin"]
        models[variant] = {
            "id": name, "revision": revision,
            "weight": {"file": "pytorch_model.bin", "bytes": weight["size"],
                       "sha256": weight["lfs"]["sha256"], "blob_id": weight["blobId"],
                       "hash_status": "provider_metadata_only_not_local_bytes"},
            "config": config, "tokenizer_config": tokenizer,
            "special_ids": special_ids,
            "upstream_sentence_embedding_pooling": "cls_with_linear_tanh" if variant == "sup" else "cls_before_pooler",
        }
        token_bytes[variant] = {f: contents[prefix + f] for f in TOKEN_FILES}
    if token_bytes["sup"] != token_bytes["unsup"]:
        raise ValueError("candidate tokenizer bytes differ")
    return {"models": models, "tokenizer_bytes_equal": True,
            "source_tree_for_commit": commit["commit"]["tree"]["sha"],
            "converted_mlp_mapping": "mlp_to_pooler_and_remove_roberta_prefix",
            "upstream_tool_l2_normalization_default": True,
            "source_imported_or_executed": False}

def verify_cache(cache: Path, lock: dict) -> dict:
    cache = cache.absolute()
    if not cache.is_dir() or any(p.is_symlink() for p in (cache, *cache.parents)):
        raise ValueError("regular source cache required")
    receipt = regular(cache / "collection.json")
    if sha(receipt) != lock["collection_sha256"]:
        raise ValueError("collection receipt mismatch")
    records = json.loads(receipt.read_text())
    if len(records) != 24:
        raise ValueError("collection scope differs")
    contents = {}
    for actual, expected in zip(records, lock["artifacts"]):
        if (any(actual.get(k) != v for k, v in expected.items()) or actual["status"] != 200
                or actual["method"] != "GET" or actual["retry_count"] != 0
                or actual["access"] != "public_https"):
            raise ValueError("collection provenance differs")
        p = regular(cache / expected["path"])
        if p.stat().st_size != expected["bytes"] or sha(p) != expected["sha256"]:
            raise ValueError("original evidence hash/size differs")
        contents[expected["url"]] = p.read_bytes()
    if derive_facts(contents) != lock["facts"]:
        raise ValueError("source and candidate facts differ")
    return {"originals_verified": 24, "source_executed": False, "weights_loaded": False}

def tokenizer_fixture(cache: Path, lock: dict) -> dict:
    """Real byte BPE on synthetic strings, not Transformers/neural equivalence."""
    verify_cache(cache, lock)
    from tokenizers import AddedToken, processors
    from tokenizers.implementations import ByteLevelBPETokenizer
    paths = {x["url"]: cache / x["path"] for x in lock["artifacts"]}
    outputs = {}
    for variant, revision in REVISIONS.items():
        prefix = "https://huggingface.co/princeton-nlp/" + variant + "-simcse-roberta-large/resolve/" + revision + "/"
        tokenizer = ByteLevelBPETokenizer(str(paths[prefix + "vocab.json"]), str(paths[prefix + "merges.txt"]),
                                         add_prefix_space=False)
        tokenizer.add_special_tokens(["<s>", "<pad>", "</s>", "<unk>",
                                      AddedToken("<mask>", lstrip=True, normalized=False)])
        tokenizer._tokenizer.post_processor = processors.RobertaProcessing(("</s>", 2), ("<s>", 0),
                                                                          add_prefix_space=False)
        single = tokenizer.encode("Hello world")
        pair = tokenizer.encode("Hello world", "Hello")
        if single.ids != [0, 31414, 232, 2] or pair.ids != [0, 31414, 232, 2, 2, 31414, 2]:
            raise ValueError("BPE single/pair fixture differs")
        if tokenizer.encode(" Hello").ids == tokenizer.encode("Hello").ids:
            raise ValueError("leading-space BPE distinction lost")
        unicode_case = tokenizer.encode("北京 — naïve café")
        if 3 in unicode_case.ids or tokenizer.decode(unicode_case.ids, skip_special_tokens=True) != "北京 — naïve café":
            raise ValueError("byte-level unicode fixture differs")
        boundary = tokenizer.encode(" ".join(["a"] * 507), "a")
        overflow = tokenizer.encode(" ".join(["a"] * 508), "a")
        if len(boundary.ids) != 512 or len(overflow.ids) != 513 or tokenizer.get_vocab_size() != 50265:
            raise ValueError("sequence/vocabulary boundary differs")
        outputs[variant] = {"single_ids": single.ids, "pair_ids": pair.ids,
                            "unicode_ids": unicode_case.ids, "boundary_length": 512, "overflow_length": 513}
    if outputs["sup"] != outputs["unsup"]:
        raise ValueError("candidate BPE behavior differs")
    return {"kind": "real_tokenizers_byte_bpe_synthetic_fixture_only", "variants": outputs,
            "added_entity_markers": False, "truncation_chosen": False, "neural_model_executed": False}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-cache", type=Path)
    parser.add_argument("--tokenizer-fixture", action="store_true")
    args = parser.parse_args()
    try:
        lock = load_lock()
        evidence = verify_cache(args.verify_cache, lock) if args.verify_cache else {"originals_verified": 0}
        if args.tokenizer_fixture and not args.verify_cache:
            raise ValueError("tokenizer fixture requires verified originals")
        tokens = tokenizer_fixture(args.verify_cache, lock) if args.tokenizer_fixture else None
        result = {"status": "official_candidates_bound_paper_mapping_unconfirmed", "facts": lock["facts"],
                  "cache_evidence": evidence, "tokenizer_fixture": tokens,
                  "paper_checkpoint_identity_bound": False, "formal_comparison_admitted": False,
                  "model_jobs": 0, "weights_loaded": False, "source_executed": False, "network_requests": 0}
    except (ValueError, KeyError, OSError, SyntaxError, ImportError) as error:
        print(json.dumps({"status": "invalid_evidence", "error": str(error), "formal_comparison_admitted": False}))
        return 1
    print(json.dumps(result, indent=2))
    return 2

if __name__ == "__main__":
    raise SystemExit(main())
