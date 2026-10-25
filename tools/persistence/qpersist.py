#!/usr/bin/env python3
"""Immutable private artifact snapshots. Python 3.10+, standard library only.

No training launch, token refresh, deletion, mount or bidirectional sync.
The caller must select closed artifacts; this is not a live-checkpoint adapter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid

BLOCK = 4 * 1024 * 1024
CHUNK = 64 * 1024 * 1024
HEX = re.compile(r"^[0-9a-f]{64}$")
SNAP = re.compile(r"^[0-9a-f]{32}$")
SECRET_NAMES = {".git", ".ssh", ".env", "credentials", "credentials.json", "token.json", "tokens.json", "id_rsa", "id_ed25519"}


class Error(Exception):
    pass


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode()


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(BLOCK), b""):
            h.update(block)
    return h.hexdigest()


def safe_relative(name):
    if not isinstance(name, str) or not name or "\\" in name or ":" in name or "\x00" in name:
        raise Error("Invalid relative path")
    p = PurePosixPath(name)
    if not p.parts or p.is_absolute() or str(p) != name or any(x in {".", ".."} for x in p.parts):
        raise Error("Unsafe relative path")
    # Portable Windows/Linux restore; no ambiguous aliases or device names.
    for part in p.parts:
        if part.endswith((" ", ".")) or re.search(r'[<>"|?*\x00-\x1f]', part):
            raise Error("Nonportable filename")
        if part.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)], *[f"LPT{i}" for i in range(1, 10)]}:
            raise Error("Reserved filename")
        if part.lower() in SECRET_NAMES or part.lower().endswith((".pem", ".key")):
            raise Error("Credential-like path excluded")
    return p


def regular(root, name):
    rel = safe_relative(name)
    cursor = root
    for part in rel.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise Error("Symlinks are not snapshot inputs")
    if not stat.S_ISREG(cursor.stat().st_mode):
        raise Error("Only regular files are supported")
    return cursor


def write_new(path, data):
    with open(path, "xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    sync_directory(path.parent)


def sync_directory(path):
    # Windows directory handles cannot be fsynced via the stdlib CRT.
    # Windows is a verified copy target, not certified for power-loss durability.
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def json_file(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_metadata(metadata):
    if not isinstance(metadata, dict) or set(metadata) != {"kind", "code_commit", "environment_sha256", "dataset_sha256"}:
        raise Error("Metadata needs exactly kind/code_commit/environment_sha256/dataset_sha256")
    if metadata["kind"] not in {"dataset", "closed_run", "code", "environment", "canary"}:
        raise Error("Unsupported artifact kind")
    if not re.fullmatch(r"[0-9a-f]{40}", metadata["code_commit"]) or not HEX.fullmatch(metadata["environment_sha256"]):
        raise Error("Full code commit and environment hash required")
    if not isinstance(metadata["dataset_sha256"], dict):
        raise Error("Dataset identity must be a hash map")
    for name, value in metadata["dataset_sha256"].items():
        safe_relative(name)
        if not HEX.fullmatch(value):
            raise Error("Invalid dataset hash")


def check_declared_datasets(metadata, rows):
    actual = {row["path"]: row["sha256"] for row in rows}
    for name, value in metadata["dataset_sha256"].items():
        if name in actual and actual[name] != value:
            raise Error("Declared dataset hash differs from included file")
        if metadata["kind"] == "dataset" and name not in actual:
            raise Error("Dataset snapshot must include each declared dataset file")
    if metadata["kind"] == "dataset" and not metadata["dataset_sha256"]:
        raise Error("Dataset snapshot requires declared file hashes")


def inventory(source, includes, metadata, out, closed):
    validate_metadata(metadata)
    if not closed:
        raise Error("Stop writers / select immutable checkpoint generation, then attest --closed")
    source = Path(source).resolve(strict=True)
    out = Path(out).absolute()
    names = set()
    for entry in includes:
        safe_relative(entry)
        target = source / entry
        # Refuse symlinks in parent components as well as leaves.
        for part in [target, *target.parents]:
            if part == source:
                break
            if part.is_symlink():
                raise Error("Symlink include")
        if target.is_dir():
            for directory, dirs, files in os.walk(target, followlinks=False):
                if any((Path(directory) / d).is_symlink() for d in dirs):
                    raise Error("Symlink directory")
                for name in files:
                    names.add((Path(directory) / name).relative_to(source).as_posix())
        else:
            names.add(entry)
    if not names:
        raise Error("Empty snapshot")
    rows = []
    for name in sorted(names):
        p = regular(source, name)
        if p.resolve() == out.resolve():
            raise Error("Inventory output must not be included")
        rows.append({"path": name, "size": p.stat().st_size, "sha256": digest(p), "executable": bool(p.stat().st_mode & 0o111)})
    spec = {"schema": "qpersist.spec.v1", "closed_attestation": True, "metadata": metadata, "files": rows}
    validate_files(rows)
    check_declared_datasets(metadata, rows)
    write_new(out, encoded(spec))
    return {"files": len(rows), "bytes": sum(r["size"] for r in rows), "spec_sha256": digest(out)}


def validate_files(rows):
    if not isinstance(rows, list) or not rows:
        raise Error("Empty file manifest")
    names = set()
    for row in rows:
        name = str(safe_relative(row["path"]))
        folded = name.casefold()
        if folded in names or not HEX.fullmatch(row["sha256"]) or type(row["size"]) is not int or row["size"] < 0 or type(row["executable"]) is not bool:
            raise Error("Invalid/duplicate file manifest")
        names.add(folded)
    for name in names:
        if any(str(p) in names for p in PurePosixPath(name).parents if str(p) != "."):
            raise Error("File/directory collision")


def pack(source, spec_path, output):
    spec = json_file(spec_path)
    if spec.get("schema") != "qpersist.spec.v1" or spec.get("closed_attestation") is not True:
        raise Error("Invalid or unsealed specification")
    validate_files(spec["files"])
    validate_metadata(spec["metadata"])
    check_declared_datasets(spec["metadata"], spec["files"])
    source = Path(source).resolve(strict=True)
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise Error("Package destination already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".qpersist-pack-", dir=output.parent))
    chunks = []
    current = None
    used = 0
    try:
        for row in spec["files"]:
            p = regular(source, row["path"])
            before = p.stat()
            h = hashlib.sha256()
            size = 0
            with open(p, "rb") as src:
                while True:
                    data = src.read(min(BLOCK, CHUNK - used))
                    if not data:
                        break
                    if current is None:
                        name = f"chunk-{len(chunks):06d}.bin"
                        current = open(stage / name, "xb")
                    current.write(data)
                    h.update(data)
                    size += len(data)
                    used += len(data)
                    if used == CHUNK:
                        current.flush()
                        os.fsync(current.fileno())
                        current.close()
                        current = None
                        chunks.append({"name": name, "size": used, "sha256": digest(stage / name)})
                        used = 0
            after = p.stat()
            if h.hexdigest() != row["sha256"] or size != row["size"] or (before.st_mtime_ns, before.st_size, before.st_ino) != (after.st_mtime_ns, after.st_size, after.st_ino):
                raise Error("Source changed or does not match frozen inventory")
        if current is not None:
            current.flush()
            os.fsync(current.fileno())
            current.close()
            current = None
            chunks.append({"name": name, "size": used, "sha256": digest(stage / name)})
        manifest = {"schema": "qpersist.snapshot.v1", "snapshot": uuid.uuid4().hex, "spec_sha256": digest(spec_path), "metadata": spec["metadata"], "files": spec["files"], "chunks": chunks}
        write_new(stage / "manifest.json", encoded(manifest))
        os.rename(stage, output)
        sync_directory(output.parent)
        return {"snapshot": manifest["snapshot"], "manifest_sha256": digest(output / "manifest.json"), "status": "local_package_only"}
    finally:
        if current is not None:
            current.close()
        if stage.exists():
            shutil.rmtree(stage)


class LocalStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, name, destination):
        safe_relative(name)
        path = regular(self.root, name)
        shutil.copyfile(path, destination)

    def put(self, name, source):
        safe_relative(name)
        target = self.root / name
        if target.exists() or target.is_symlink():
            if digest(regular(self.root, name)) != digest(source):
                raise Error("Immutable object conflict")
            return
        temp = self.root / (".upload-" + uuid.uuid4().hex)
        try:
            shutil.copyfile(source, temp)
            with open(temp, "rb+") as f:
                os.fsync(f.fileno())
            # Hard link is atomic and refuses replacement of existing targets.
            os.link(temp, target)
            sync_directory(self.root)
        finally:
            temp.unlink(missing_ok=True)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise Error("HTTP redirects rejected")


class AlipanStore:
    """Official API only; short-lived access token injected by owner.

    No automatic refresh/credential persistence. Failed uploads remain uncommitted;
    rerun reuses completed verified objects. Individual chunks are <=64 MiB.
    """
    API = "https://openapi.alipan.com"

    def __init__(self, drive_id, folder_id):
        self.drive = drive_id
        self.folder = folder_id
        self.token = os.environ.get("ALIPAN_ACCESS_TOKEN")
        if not self.token or not drive_id or not folder_id or folder_id == "root":
            raise Error("Need owner-authorized token, drive_id and dedicated project folder_id")
        self.http = urllib.request.build_opener(NoRedirect())

    def api(self, endpoint, payload):
        for attempt in range(5):
            req = urllib.request.Request(self.API + "/adrive/v1.0/" + endpoint, data=encoded(payload), headers={"Authorization": "Bearer " + self.token, "Content-Type": "application/json"})
            try:
                with self.http.open(req, timeout=60) as response:
                    return json.load(response)
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < 4:
                    wait = float(e.headers.get("x-retry-after", "1000")) / 1000
                    time.sleep(min(max(wait, 1), 60))
                    continue
                # Do not echo response bodies, signed URLs or request headers.
                raise Error(f"Alipan API HTTP {e.code}; 401/403 require owner reauthorization") from None
            except (urllib.error.URLError, TimeoutError):
                raise Error("Alipan network failure; retry command after checking connection") from None
        raise Error("Alipan rate limit")

    def find(self, name):
        marker = ""
        matches = []
        seen = set()
        for _ in range(1000):
            page = self.api("openFile/list", {"drive_id": self.drive, "parent_file_id": self.folder, "limit": 100, "marker": marker})
            matches.extend(x for x in page.get("items", []) if x.get("name") == name)
            marker = page.get("next_marker", "")
            if not marker:
                break
            if marker in seen:
                raise Error("Repeated listing cursor")
            seen.add(marker)
        else:
            raise Error("Cloud listing exceeded 1000 pages; use a smaller project folder")
        if len(matches) > 1:
            raise Error("Ambiguous cloud object; owner intervention required")
        return matches[0] if matches else None

    def get(self, name, destination):
        obj = self.find(name)
        if not obj or obj.get("type") != "file":
            raise Error("Cloud object missing")
        size = obj["size"]
        if type(size) is not int or size < 0 or size > CHUNK:
            raise Error("Cloud object exceeds size bound")
        with open(destination, "wb") as out:
            for start in range(0, size, BLOCK):
                end = min(size, start + BLOCK) - 1
                info = self.api("openFile/getDownloadUrl", {"drive_id": self.drive, "file_id": obj["file_id"]})
                url = info["url"]
                if not url.startswith("https://"):
                    raise Error("Download requires HTTPS")
                req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
                try:
                    with self.http.open(req, timeout=90) as r:
                        expected_range = f"bytes {start}-{end}/{size}"
                        if r.status != 206 or r.headers.get("Content-Range") != expected_range:
                            raise Error("Invalid ranged response")
                        data = r.read(end - start + 2)
                        if len(data) != end - start + 1:
                            raise Error("Truncated/oversized download")
                        out.write(data)
                except (urllib.error.URLError, TimeoutError):
                    raise Error("Cloud download failed; retry obtains fresh download URLs") from None

    def put(self, name, source):
        if self.find(name):
            with tempfile.TemporaryDirectory(prefix="qpersist-check-") as tmp:
                copy = Path(tmp) / "object"
                self.get(name, copy)
                if digest(copy) != digest(source):
                    raise Error("Cloud immutable object conflict")
            return
        size = Path(source).stat().st_size
        if size > CHUNK:
            raise Error("Cloud upload exceeds chunk bound")
        info = self.api("openFile/create", {"drive_id": self.drive, "parent_file_id": self.folder, "name": name, "type": "file", "check_name_mode": "refuse", "size": size, "part_info_list": [{"part_number": 1}]})
        url = info["part_info_list"][0]["upload_url"]
        if not url.startswith("https://"):
            raise Error("Upload requires HTTPS")
        # Bounded memory, one <=64 MiB storage object per upload.
        req = urllib.request.Request(url, data=Path(source).read_bytes(), method="PUT")
        try:
            with self.http.open(req, timeout=180) as response:
                if response.status not in (200, 201):
                    raise Error("Cloud upload rejected")
        except (urllib.error.URLError, TimeoutError):
            raise Error("Cloud upload interrupted; snapshot remains uncommitted") from None
        self.api("openFile/complete", {"drive_id": self.drive, "file_id": info["file_id"], "upload_id": info["upload_id"]})


def validate_manifest(manifest):
    if manifest.get("schema") != "qpersist.snapshot.v1" or not SNAP.fullmatch(manifest.get("snapshot", "")):
        raise Error("Invalid snapshot manifest")
    validate_files(manifest["files"])
    validate_metadata(manifest["metadata"])
    check_declared_datasets(manifest["metadata"], manifest["files"])
    chunks = manifest["chunks"]
    for i, c in enumerate(chunks):
        if c["name"] != f"chunk-{i:06d}.bin" or not HEX.fullmatch(c["sha256"]) or type(c["size"]) is not int or not 0 < c["size"] <= CHUNK:
            raise Error("Invalid chunk manifest")
    if sum(c["size"] for c in chunks) != sum(f["size"] for f in manifest["files"]):
        raise Error("Manifest byte count mismatch")


def checked_get(store, name, target, expected_hash, expected_size=None):
    store.get(name, target)
    if (expected_size is not None and target.stat().st_size != expected_size) or digest(target) != expected_hash:
        raise Error("Remote object checksum mismatch")


def publish(package, store, receipt):
    package = Path(package)
    manifest = json_file(package / "manifest.json")
    validate_manifest(manifest)
    sid = manifest["snapshot"]
    mh = digest(package / "manifest.json")
    if Path(receipt).exists():
        raise Error("Receipt already exists; inspect status rather than overwrite")
    with tempfile.TemporaryDirectory(prefix="qpersist-verify-") as tmp:
        for c in manifest["chunks"]:
            source = regular(package, c["name"])
            if source.stat().st_size != c["size"] or digest(source) != c["sha256"]:
                raise Error("Local package corrupt")
            name = sid + "--" + c["name"]
            store.put(name, source)
            checked_get(store, name, Path(tmp) / "verify", c["sha256"], c["size"])
        # Publishing the manifest is the commit point, only after chunk readback.
        store.put(sid + "--manifest.json", package / "manifest.json")
        checked_get(store, sid + "--manifest.json", Path(tmp) / "manifest", mh)
    result = {"schema": "qpersist.receipt.v1", "snapshot": sid, "manifest_sha256": mh, "status": "remote_readback_verified", "restore_drill": False, "backend": type(store).__name__}
    write_new(Path(receipt), encoded(result))
    return result


def restore(store, snapshot, manifest_hash, destination):
    if not SNAP.fullmatch(snapshot) or not HEX.fullmatch(manifest_hash):
        raise Error("Trusted snapshot id and manifest SHA256 required")
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise Error("Restore target must not exist")
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".qpersist-restore-", dir=destination.parent))
    try:
        payload = stage / "payload"
        payload.mkdir()
        with tempfile.TemporaryDirectory(prefix="qpersist-download-") as tmp:
            tmp = Path(tmp)
            checked_get(store, snapshot + "--manifest.json", tmp / "manifest", manifest_hash)
            manifest = json_file(tmp / "manifest")
            validate_manifest(manifest)
            if manifest["snapshot"] != snapshot:
                raise Error("Snapshot identity mismatch")
            def blocks():
                for c in manifest["chunks"]:
                    checked_get(store, snapshot + "--" + c["name"], tmp / "chunk", c["sha256"], c["size"])
                    with open(tmp / "chunk", "rb") as f:
                        for block in iter(lambda: f.read(BLOCK), b""):
                            yield block
            stream = iter(blocks())
            pending = b""
            for row in manifest["files"]:
                target = payload / row["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                h = hashlib.sha256()
                remaining = row["size"]
                with open(target, "xb") as out:
                    while remaining:
                        if not pending:
                            pending = next(stream)
                        data, pending = pending[:remaining], pending[remaining:]
                        out.write(data)
                        h.update(data)
                        remaining -= len(data)
                    out.flush()
                    os.fsync(out.fileno())
                if h.hexdigest() != row["sha256"]:
                    raise Error("Restored file hash mismatch")
                target.chmod(0o700 if row["executable"] else 0o600)
            if pending or next(stream, None) is not None:
                raise Error("Unexpected trailing payload")
        # Never use replace(); final publication occurs only after full validation.
        if destination.exists() or destination.is_symlink():
            raise Error("Restore target appeared concurrently")
        for directory, _, _ in os.walk(payload, topdown=False):
            sync_directory(directory)
        os.rename(payload, destination)
        sync_directory(destination.parent)
        return {"snapshot": snapshot, "manifest_sha256": manifest_hash, "status": "restored_and_verified", "training_resume": "requires_original_runner_validation"}
    except StopIteration:
        raise Error("Truncated snapshot payload") from None
    finally:
        shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inv = sub.add_parser("inventory")
    inv.add_argument("--source", required=True)
    inv.add_argument("--include", action="append", required=True, help="Explicit relative file/directory; never entire home")
    inv.add_argument("--metadata", required=True, help="Private JSON: code commit, data identity, environment definition, kind")
    inv.add_argument("--out", required=True)
    inv.add_argument("--closed", action="store_true", help="Attest all selected files are immutable and logically consistent")
    pk = sub.add_parser("pack")
    pk.add_argument("--source", required=True)
    pk.add_argument("--spec", required=True)
    pk.add_argument("--out", required=True)
    for cmd in ("publish", "restore", "status"):
        p = sub.add_parser(cmd)
        p.add_argument("--backend", choices=("local", "alipan"), required=True)
        p.add_argument("--store", help="Independent filesystem storage (local backend only)")
        p.add_argument("--drive-id")
        p.add_argument("--folder-id")
        if cmd == "publish":
            p.add_argument("--package", required=True)
            p.add_argument("--receipt", required=True)
        else:
            p.add_argument("--snapshot", required=True)
            p.add_argument("--manifest-sha256", required=True)
            if cmd == "restore":
                p.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        if args.command == "inventory":
            result = inventory(args.source, args.include, json_file(args.metadata), args.out, args.closed)
        elif args.command == "pack":
            result = pack(args.source, args.spec, args.out)
        else:
            if args.backend == "local" and not args.store:
                raise Error("Local backend needs --store")
            store = LocalStore(args.store) if args.backend == "local" else AlipanStore(args.drive_id, args.folder_id)
            if args.command == "publish":
                result = publish(args.package, store, args.receipt)
            elif args.command == "restore":
                result = restore(store, args.snapshot, args.manifest_sha256, args.out)
            else:
                with tempfile.TemporaryDirectory(prefix="qpersist-status-") as tmp:
                    if not SNAP.fullmatch(args.snapshot) or not HEX.fullmatch(args.manifest_sha256):
                        raise Error("Invalid snapshot/hash")
                    checked_get(store, args.snapshot + "--manifest.json", Path(tmp) / "manifest", args.manifest_sha256)
                    manifest = json_file(Path(tmp) / "manifest")
                    validate_manifest(manifest)
                    if manifest["snapshot"] != args.snapshot:
                        raise Error("Snapshot identity mismatch")
                result = {"status": "committed_manifest_verified", "payload_rechecked": False, "snapshot": args.snapshot}
        print(json.dumps(result, ensure_ascii=False))
    except (Error, OSError, ValueError, KeyError, TypeError):
        # Paths and URL-bearing exception messages may contain private material.
        print("qpersist: operation failed; no success receipt issued. Check arguments, immutable inputs, credentials and available space.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
