"""Hash-locked, sharded pretrained assets; authenticated HTTPS only."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import ssl
import tempfile
import time
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener, urlopen

from .common import ROOT, PACKAGE, atomic_json, digest

REPO = "FacebookAI/roberta-large"
REVISION = "722cf37b1afa9454edce342e7895e588b6ff1d59"
WEIGHT_SHA256 = "36a10a8b694fadf9bf4f9049d14e257e88be45313ae02d882af9e60f39b8b2e8"
FILES = ("config.json", "merges.txt", "vocab.json", "tokenizer.json", "pytorch_model.bin")
SHARD_BYTES = 512 * 1024 * 1024
CA = ROOT / "scripts/collab/certs/qattn-exchange-ca.crt"


def safe_name(name: str) -> str:
    if not name or Path(name).name != name or name in (".", "..") or name.startswith("."):
        raise ValueError("unsafe asset filename")
    return name


def reject_symlinks(path: Path) -> None:
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError("asset cache symlink rejected")


class AuthenticatedRedirectGuard(HTTPRedirectHandler):
    """An exchange must not forward a bearer credential to another origin."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        before, after = urlsplit(req.full_url), urlsplit(newurl)
        def origin(value):
            return value.scheme, value.hostname, value.port or 443
        if after.scheme != "https" or origin(before) != origin(after):
            raise ValueError("authenticated asset redirect outside exchange origin")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def stream_download(url: str, target: Path, *, expected: str | None = None,
                    size: int | None = None, token: str | None = None, context=None) -> None:
    """Retry a complete shard, never append an unverified partial file."""
    if urlsplit(url).scheme != "https":
        raise ValueError("asset downloads require HTTPS")
    reject_symlinks(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and expected and digest(target) == expected and (size is None or target.stat().st_size == size):
        return
    for attempt in range(3):
        fd, temporary = tempfile.mkstemp(prefix=target.name + ".", dir=target.parent)
        try:
            headers = {"User-Agent": "q-attention-baseline-assets/1"}
            if token:
                headers["Authorization"] = "Bearer " + token
            with os.fdopen(fd, "wb") as output:
                request = Request(url, headers=headers)
                open_response = (build_opener(HTTPSHandler(context=context), AuthenticatedRedirectGuard()).open(request, timeout=120)
                                 if token else urlopen(request, timeout=120, context=context))
                with open_response as response:
                    if response.status != 200:
                        raise ValueError("asset response must be HTTP 200")
                    shutil.copyfileobj(response, output, 1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
            staged = Path(temporary)
            if (expected and digest(staged) != expected) or (size is not None and staged.stat().st_size != size):
                raise ValueError("asset size/SHA-256 mismatch")
            os.replace(staged, target)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
        finally:
            Path(temporary).unlink(missing_ok=True)


def uploader():
    source = ROOT / "scripts/collab/lib/exchange_upload.py"
    spec = importlib.util.spec_from_file_location("retacred_exchange_upload", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def shard_file(path: Path, directory: Path, *, shard_bytes: int = SHARD_BYTES) -> list[dict]:
    directory.mkdir(parents=True, exist_ok=True)
    parts = []
    with path.open("rb") as source:
        number = 0
        while True:
            block = source.read(min(shard_bytes, 1024 * 1024))
            if not block:
                break
            target = directory / f"{path.name}.part{number:04d}"
            with target.open("wb") as output:
                written = len(block)
                output.write(block)
                while written < shard_bytes:
                    block = source.read(min(shard_bytes - written, 1024 * 1024))
                    if not block:
                        break
                    output.write(block)
                    written += len(block)
            parts.append({"name": target.name, "sha256": digest(target), "size_bytes": written})
            number += 1
    return parts


def assemble(parts: list[Path], destination: Path, sha256: str, size: int) -> None:
    reject_symlinks(destination)
    for part in parts:
        reject_symlinks(part)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            for part in parts:
                with part.open("rb") as source:
                    shutil.copyfileobj(source, output, 1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        staged = Path(temporary)
        if digest(staged) != sha256 or staged.stat().st_size != size:
            raise ValueError("assembled asset identity mismatch")
        os.replace(staged, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def prepare(*, publish: bool, token_file: Path | None = None) -> Path:
    cache = ROOT / ".external/assets" / REVISION
    model = cache / "model"
    reject_symlinks(model)
    model.mkdir(parents=True, exist_ok=True)
    spec = {"schema": "retacred-assets.v1", "repo": REPO, "revision": REVISION,
            "license": "MIT", "files": [], "exchange_namespace": f"q-attention/reference-assets/{REVISION}"}
    for name in FILES:
        target = model / name
        print(f"[assets] download pinned {name}", flush=True)
        stream_download(f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}", target,
                        expected=WEIGHT_SHA256 if name == "pytorch_model.bin" else None)
        parts = shard_file(target, cache / "parts")
        spec["files"].append({"name": name, "sha256": digest(target), "size_bytes": target.stat().st_size, "parts": parts})
    if publish:
        url = os.environ.get("PROJECT_EXCHANGE_URL", "https://117.50.198.37:18084")
        token = token_file.read_text().strip() if token_file else os.environ.get("PROJECT_EXCHANGE_TOKEN", "")
        if not token:
            raise ValueError("PROJECT_EXCHANGE_TOKEN is required")
        files = [(cache / "parts" / part["name"], part["name"]) for row in spec["files"] for part in row["parts"]]
        receipt = uploader().upload_declared_files(base_url=url, token=token,
                   target_dir=spec["exchange_namespace"], files=files, ca_file=CA)
        spec["exchange_url"] = url
        atomic_json(cache / "upload_receipt.json", receipt)
    lock = cache / "assets.lock.json"
    atomic_json(lock, spec)
    print(f"[assets] lock={lock}", flush=True)
    return lock


def fetch(lock_path: Path = PACKAGE / "assets.lock.json") -> Path:
    spec = json.loads(lock_path.read_text(encoding="utf-8"))
    if spec.get("schema") != "retacred-assets.v1" or spec["revision"] != REVISION:
        raise ValueError("asset lock revision/schema mismatch")
    token = os.environ.get("PROJECT_EXCHANGE_TOKEN", "")
    url = os.environ.get("PROJECT_EXCHANGE_URL", spec.get("exchange_url", ""))
    if not token or urlsplit(url).scheme != "https":
        raise ValueError("configure HTTPS PROJECT_EXCHANGE_URL and PROJECT_EXCHANGE_TOKEN")
    namespace = uploader().safe_relative(spec["exchange_namespace"], label="asset namespace")
    context = ssl.create_default_context(cafile=CA)
    cache = ROOT / ".external/assets" / spec["revision"]
    model = cache / "model"
    reject_symlinks(model)
    model.mkdir(parents=True, exist_ok=True)
    for row in spec["files"]:
        target = model / safe_name(row["name"])
        if target.is_file() and not target.is_symlink() and digest(target) == row["sha256"] and target.stat().st_size == row["size_bytes"]:
            from .reference_storage import clean_verified_parts
            clean_verified_parts(target,row,cache/'parts')
            continue
        parts = []
        for part in row["parts"]:
            name = safe_name(part["name"])
            path = cache / "parts" / name
            stream_download(f"{url.rstrip('/')}/files/{namespace}/{quote(name)}", path,
                            expected=part["sha256"], size=part["size_bytes"], token=token, context=context)
            parts.append(path)
        assemble(parts, target, row["sha256"], row["size_bytes"])
        from .reference_storage import clean_verified_parts
        clean_verified_parts(target,row,cache/'parts')
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "fetch"))
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--token-file", type=Path, help="external secret file, never persisted")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(publish=args.publish, token_file=args.token_file)
    else:
        print(fetch())


if __name__ == "__main__":
    main()
