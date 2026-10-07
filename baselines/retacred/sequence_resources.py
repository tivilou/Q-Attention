"""Recover one hash-locked historical source file, separately from active source.

This neither executes the file nor certifies/adopts Sequence Routing. No weights,
datasets, token, dependency installation, source patching or training are involved.
"""
from __future__ import annotations

import argparse
import ast
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request

from .assets import reject_symlinks
from .common import PACKAGE, ROOT, atomic_json, digest


def resource_lock() -> dict:
    return json.loads((PACKAGE / "sequence_routing_resources.lock.json").read_text())


def resource_directory() -> Path:
    return ROOT / ".external/retacred/sequence_routing_resources" / resource_lock()["historical_commit"]


def _check_bytes(body: bytes) -> None:
    expected = resource_lock()["artifact"]
    blob = hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()
    if (len(body) != expected["size_bytes"] or blob != expected["git_blob"]
            or hashlib.sha256(body).hexdigest() != expected["sha256"]):
        raise ValueError("historical Abstraction source identity mismatch")
    ast.parse(body.decode("utf-8"), filename="Abstraction.py")  # Parse, never execute.


def validate_resources(directory: Path) -> dict:
    path = directory / resource_lock()["artifact"]["name"]
    reject_symlinks(path)
    if path.stat().st_size != resource_lock()["artifact"]["size_bytes"]:
        raise ValueError("historical Abstraction source identity mismatch")
    _check_bytes(path.read_bytes())
    return {"status": "historical_source_verified", "sha256": digest(path),
            "active_source_modified": False, "strict_comparison_eligible": False}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("historical resource redirect rejected")


def fetch_resources() -> Path:
    lock = resource_lock()
    directory = resource_directory()
    destination = directory / lock["artifact"]["name"]
    reject_symlinks(destination)
    if destination.exists():
        validate_resources(directory)  # Reject corrupt cache, never silently overwrite it.
        return directory
    request = urllib.request.Request(lock["artifact"]["url"], headers={
        "User-Agent": "Q-Attention-source-recovery/1", "Accept": "application/vnd.github+json"})
    with urllib.request.build_opener(_NoRedirect()).open(request, timeout=30) as response:
        raw = response.read(128 * 1024 + 1)
        if response.status != 200 or len(raw) > 128 * 1024:
            raise ValueError("historical resource response status/size rejected")
        payload = json.loads(raw)
    if (payload.get("encoding") != "base64" or payload.get("sha") != lock["artifact"]["git_blob"]
            or payload.get("size") != lock["artifact"]["size_bytes"]):
        raise ValueError("historical resource API identity mismatch")
    body = base64.b64decode("".join(payload["content"].split()), validate=True)
    _check_bytes(body)
    directory.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="Abstraction.", dir=directory)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(body)
            output.flush()
            os.fsync(output.fileno())
        # Never install this into the active upstream checkout or import it.
        reject_symlinks(destination)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    validate_resources(directory)
    atomic_json(directory / "recovery_manifest.json", {
        "schema": "sequence-routing-resource-recovery.v1", "artifact": lock["artifact"],
        "repository": lock["repository"], "historical_commit": lock["historical_commit"],
        "retrieved_at": datetime.now(timezone.utc).isoformat(), "access": "public_https",
        "max_requests": 1, "max_retries": 0, "file_executed": False,
        "active_source_modified": False, "strict_comparison_eligible": False})
    return directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="No network access or writes")
    args = parser.parse_args()
    directory = resource_directory() if args.verify_only else fetch_resources()
    print(json.dumps({"directory": str(directory), **validate_resources(directory)}, indent=2))


if __name__ == "__main__":
    main()
