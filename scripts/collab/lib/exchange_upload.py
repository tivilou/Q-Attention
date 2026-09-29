#!/usr/bin/env python3
"""Upload explicitly declared files to an authenticated artifact exchange."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import ssl
import sys
import tempfile
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen


SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
CHUNK_SIZE = 1024 * 1024
DEFAULT_CA_FILE = Path(__file__).resolve().parents[1] / "certs" / "qattn-exchange-ca.crt"


class ExchangeError(RuntimeError):
    """A transport, path, or receipt validation failure."""


def safe_relative(value: str, *, label: str) -> str:
    normalised = value.replace("\\", "/")
    if not normalised or normalised.startswith("/"):
        raise ExchangeError(f"{label} must be a relative path")
    segments = normalised.split("/")
    if any(
        not segment
        or segment in {".", ".."}
        or segment.startswith(".")
        or not SAFE_SEGMENT.fullmatch(segment)
        for segment in segments
    ):
        raise ExchangeError(f"unsafe {label}: {value!r}")
    return "/".join(segments)


def parse_base_url(raw: str) -> tuple[str, str, int | None, str]:
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ExchangeError("exchange URL must be an absolute http(s) URL")
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise ExchangeError("exchange URL must not contain credentials, query, or fragment")
    base_path = parsed.path.rstrip("/")
    return parsed.scheme, parsed.hostname or "", parsed.port, base_path


def ssl_context_for(scheme: str, ca_file: Path) -> ssl.SSLContext | None:
    if scheme != "https":
        return None
    if not ca_file.is_file() or ca_file.is_symlink():
        raise ExchangeError(f"HTTPS trust certificate is missing: {ca_file}")
    return ssl.create_default_context(cafile=str(ca_file))


def _auth_headers(token: str) -> dict[str, str]:
    if not token:
        raise ExchangeError("PROJECT_EXCHANGE_TOKEN is not configured")
    return {"Authorization": f"Bearer {token}"}


def _read_json_response(response: Any, *, action: str) -> dict[str, Any]:
    try:
        payload = json.loads(response.read().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExchangeError(f"{action} returned a non-JSON response") from exc
    if not isinstance(payload, dict):
        raise ExchangeError(f"{action} returned an invalid JSON response")
    return payload


def create_directory(
    base_url: str,
    token: str,
    target_dir: str,
    *,
    context: ssl.SSLContext | None,
) -> dict[str, Any]:
    body = json.dumps({"path": target_dir}).encode("utf-8")
    request = Request(
        f"{base_url.rstrip('/')}/api/directories",
        data=body,
        method="POST",
        headers={
            **_auth_headers(token),
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        },
    )
    try:
        with urlopen(request, timeout=30, context=context) as response:
            payload = _read_json_response(response, action="directory creation")
            if response.status not in {200, 201}:
                raise ExchangeError(f"directory creation returned HTTP {response.status}: {payload}")
    except HTTPError as exc:
        detail = exc.read(2048).decode("utf-8", errors="replace")
        raise ExchangeError(f"directory creation returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise ExchangeError(f"directory creation failed: {exc.reason}") from exc
    if payload.get("path") != target_dir:
        raise ExchangeError(f"directory creation receipt path mismatch: {payload}")
    return payload


def _local_file(raw_path: Path) -> Path:
    candidate = raw_path.expanduser()
    absolute = candidate if candidate.is_absolute() else Path.cwd() / candidate
    absolute = absolute.absolute()
    for part in (absolute, *absolute.parents):
        if part.exists() and part.is_symlink():
            raise ExchangeError(f"symlink is not allowed in local path: {part}")
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as exc:
        raise ExchangeError(f"local file cannot be resolved: {raw_path}") from exc
    if not resolved.is_file():
        raise ExchangeError(f"declared local path is not a regular file: {raw_path}")
    return resolved


def _digest_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for block in iter(lambda: source.read(CHUNK_SIZE), b""):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def _upload_file(
    base_url: str,
    token: str,
    local_path: Path,
    remote_path: str,
    size: int,
    digest: str,
    *,
    context: ssl.SSLContext | None,
) -> dict[str, Any]:
    scheme, host, port, base_path = parse_base_url(base_url)
    encoded = "/".join(quote(segment, safe="._-") for segment in remote_path.split("/"))
    route = f"{base_path}/upload/{encoded}" if base_path else f"/upload/{encoded}"
    connection_class = (
        http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
    )
    options: dict[str, Any] = {"timeout": 300}
    if scheme == "https":
        options["context"] = context
    connection = connection_class(host, port=port, **options)
    try:
        connection.putrequest("PUT", route)
        for name, value in {
            **_auth_headers(token),
            "Content-Type": "application/octet-stream",
            "Content-Length": str(size),
        }.items():
            connection.putheader(name, value)
        connection.endheaders()
        with local_path.open("rb") as source:
            for block in iter(lambda: source.read(CHUNK_SIZE), b""):
                connection.send(block)
        response = connection.getresponse()
        raw_body = response.read()
    except OSError as exc:
        raise ExchangeError(f"upload failed for declared file {local_path.name}: {exc}") from exc
    finally:
        connection.close()

    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExchangeError(f"upload returned a non-JSON response for {local_path.name}") from exc
    if not isinstance(payload, dict):
        raise ExchangeError(f"upload returned an invalid JSON response for {local_path.name}")
    if response.status not in {200, 201}:
        raise ExchangeError(f"upload returned HTTP {response.status}: {payload}")
    if (
        payload.get("path") != remote_path
        or payload.get("size_bytes") != size
        or payload.get("sha256") != digest
    ):
        raise ExchangeError(f"upload receipt mismatch for {local_path.name}: {payload}")
    return payload


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def upload_declared_files(
    *,
    base_url: str,
    token: str,
    target_dir: str,
    files: Iterable[tuple[Path, str]],
    ca_file: Path = DEFAULT_CA_FILE,
) -> dict[str, Any]:
    target_dir = safe_relative(target_dir, label="target directory")
    if len(target_dir.split("/")) < 3:
        raise ExchangeError("target directory must be project/experiment/run-id")
    scheme, _, _, _ = parse_base_url(base_url)
    context = ssl_context_for(scheme, ca_file)
    if scheme == "http":
        print("warning: exchange is using HTTP; bearer authentication is not encrypted", file=sys.stderr)

    prepared: list[dict[str, Any]] = []
    seen_remote: set[str] = set()
    for raw_local, raw_remote in files:
        local = _local_file(raw_local)
        remote = safe_relative(raw_remote, label="remote file")
        if remote in seen_remote:
            raise ExchangeError(f"duplicate remote file path: {remote}")
        seen_remote.add(remote)
        size, digest = _digest_file(local)
        prepared.append(
            {
                "local_path": local,
                "path": remote,
                "size_bytes": size,
                "sha256": digest,
                "uploaded": False,
            }
        )
    if not prepared:
        raise ExchangeError("no declared files were provided")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest_fd, manifest_name = tempfile.mkstemp(
        prefix=f"qattn-exchange-pending-{stamp}-",
        suffix=".json",
    )
    os.close(manifest_fd)
    manifest_path = Path(manifest_name)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "project": target_dir.split("/")[0],
        "experiment": target_dir.split("/")[1],
        "run_id": target_dir.split("/")[-1],
        "exchange_url": base_url.rstrip("/"),
        "target_dir": target_dir,
        "files": [
            {key: value for key, value in item.items() if key != "local_path"}
            for item in prepared
        ],
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_manifest(manifest_path, manifest)

    try:
        create_directory(base_url, token, target_dir, context=context)
        for index, item in enumerate(prepared):
            receipt = _upload_file(
                base_url,
                token,
                item["local_path"],
                f"{target_dir}/{item['path']}",
                item["size_bytes"],
                item["sha256"],
                context=context,
            )
            item["uploaded"] = True
            manifest["files"][index]["uploaded"] = True
            manifest["files"][index]["receipt_status"] = receipt.get("status")
            _write_manifest(manifest_path, manifest)
    except ExchangeError as exc:
        manifest["error"] = str(exc)
        manifest["pending_manifest"] = str(manifest_path)
        _write_manifest(manifest_path, manifest)
        raise ExchangeError(f"{exc}; pending manifest: {manifest_path}") from exc

    manifest["status"] = "uploaded"
    manifest["uploaded_at"] = datetime.now(timezone.utc).isoformat()
    manifest["pending_manifest"] = str(manifest_path)
    _write_manifest(manifest_path, manifest)
    return manifest


__all__ = [
    "DEFAULT_CA_FILE",
    "ExchangeError",
    "safe_relative",
    "upload_declared_files",
]
