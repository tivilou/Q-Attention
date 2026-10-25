import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import qpersist as q


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "data").mkdir()
        (self.source / "data/a.bin").write_bytes(b"sample-data" * 37)
        (self.source / "data/empty").write_bytes(b"")
        self.metadata = {"kind": "canary", "code_commit": "1" * 40, "environment_sha256": "2" * 64, "dataset_sha256": {"fixture": "3" * 64}}
        self.spec = self.root / "spec.json"
        self.package = self.root / "package"
        self.store = q.LocalStore(self.root / "store")

    def tearDown(self):
        self.tmp.cleanup()

    def package_it(self):
        q.inventory(self.source, ["data"], self.metadata, self.spec, True)
        with patch.object(q, "CHUNK", 100):
            return q.pack(self.source, self.spec, self.package)

    def test_multichunk_roundtrip_and_empty_file(self):
        identity = self.package_it()
        receipt = q.publish(self.package, self.store, self.root / "receipt.json")
        q.restore(self.store, identity["snapshot"], identity["manifest_sha256"], self.root / "restored")
        self.assertEqual((self.root / "restored/data/a.bin").read_bytes(), (self.source / "data/a.bin").read_bytes())
        self.assertEqual((self.root / "restored/data/empty").stat().st_size, 0)
        self.assertFalse(receipt["restore_drill"])

    def test_changed_source_rejected_and_stage_cleaned(self):
        q.inventory(self.source, ["data"], self.metadata, self.spec, True)
        (self.source / "data/a.bin").write_bytes(b"new")
        with self.assertRaises(q.Error):
            q.pack(self.source, self.spec, self.package)
        self.assertFalse(self.package.exists())
        self.assertFalse(list(self.root.glob(".qpersist-pack-*")))

    def test_publish_failure_no_commit_then_retry(self):
        identity = self.package_it()
        original = self.store.put
        count = 0
        def fail(name, source):
            nonlocal count
            count += 1
            if count == 2:
                raise OSError("injected transport failure")
            original(name, source)
        with patch.object(self.store, "put", fail), self.assertRaises(OSError):
            q.publish(self.package, self.store, self.root / "receipt.json")
        self.assertFalse((self.store.root / (identity["snapshot"] + "--manifest.json")).exists())
        self.assertFalse((self.root / "receipt.json").exists())
        q.publish(self.package, self.store, self.root / "receipt.json")
        self.assertEqual(q.digest(self.package / "manifest.json"), identity["manifest_sha256"])

    def test_corruption_restore_failure_cleanup_then_retry(self):
        identity = self.package_it()
        q.publish(self.package, self.store, self.root / "receipt.json")
        chunk = self.store.root / (identity["snapshot"] + "--chunk-000000.bin")
        original = chunk.read_bytes()
        chunk.write_bytes(b"corrupt")
        dest = self.root / "restored"
        with self.assertRaises(q.Error):
            q.restore(self.store, identity["snapshot"], identity["manifest_sha256"], dest)
        self.assertFalse(dest.exists())
        self.assertFalse(list(self.root.glob(".qpersist-restore-*")))
        chunk.write_bytes(original)
        q.restore(self.store, identity["snapshot"], identity["manifest_sha256"], dest)

    def test_manifest_hash_required_and_existing_destination_preserved(self):
        identity = self.package_it()
        q.publish(self.package, self.store, self.root / "receipt.json")
        with self.assertRaises(q.Error):
            q.restore(self.store, identity["snapshot"], "0" * 64, self.root / "restored")
        dest = self.root / "existing"
        dest.mkdir()
        (dest / "keep").write_text("unchanged")
        with self.assertRaises(q.Error):
            q.restore(self.store, identity["snapshot"], identity["manifest_sha256"], dest)
        self.assertEqual((dest / "keep").read_text(), "unchanged")

    def test_paths_collisions_and_secrets(self):
        for name in ("../x", "/x", "a/../b", "a//b", "C:/x", "a\\b", ".", ".env", "foo/id_rsa", "CON", "x.", "a/.git/config"):
            with self.subTest(name=name), self.assertRaises(q.Error):
                q.safe_relative(name)
        row = {"path": "A", "size": 0, "sha256": hashlib.sha256(b"").hexdigest(), "executable": False}
        for other in ("a", "a/child"):
            with self.assertRaises(q.Error):
                q.validate_files([row, {**row, "path": other}])

    def test_symlink_rejected(self):
        try:
            (self.source / "data/link").symlink_to(self.source / "data/a.bin")
        except OSError:
            self.skipTest("Symlink privilege unavailable; run on Linux")
        with self.assertRaises(q.Error):
            q.inventory(self.source, ["data"], self.metadata, self.spec, True)

    def test_closed_attestation_required(self):
        with self.assertRaises(q.Error):
            q.inventory(self.source, ["data"], self.metadata, self.spec, False)

    def test_immutable_store_conflict(self):
        src = self.source / "data/a.bin"
        self.store.put("blob", src)
        src.write_bytes(b"different")
        with self.assertRaises(q.Error):
            self.store.put("blob", src)

    def test_cloud_missing_token_fails_without_network(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(q.Error):
            q.AlipanStore("drive", "folder")

    def test_manifest_chunks_cannot_traverse(self):
        self.package_it()
        m = q.json_file(self.package / "manifest.json")
        m["chunks"][0]["name"] = "../outside"
        with self.assertRaises(q.Error):
            q.validate_manifest(m)

    def test_metadata_disallows_credentials(self):
        with self.assertRaises(q.Error):
            q.validate_metadata({**self.metadata, "token": "do-not-store"})

    def test_dataset_identity_checked_against_actual_file(self):
        meta = {**self.metadata, "kind": "dataset", "dataset_sha256": {"data/a.bin": "0" * 64}}
        with self.assertRaises(q.Error):
            q.inventory(self.source, ["data"], meta, self.spec, True)
        meta["dataset_sha256"] = {"missing.bin": "0" * 64}
        with self.assertRaises(q.Error):
            q.inventory(self.source, ["data"], meta, self.spec, True)
        meta["dataset_sha256"] = {"data/a.bin": q.digest(self.source / "data/a.bin")}
        q.inventory(self.source, ["data"], meta, self.spec, True)

    def test_cloud_pagination_bounded(self):
        with patch.dict(os.environ, {"ALIPAN_ACCESS_TOKEN": "synthetic-not-real"}):
            store = q.AlipanStore("fixture", "fixture")
        with patch.object(store, "api", side_effect=[{"items": [], "next_marker": str(i)} for i in range(1000)]) as api:
            with self.assertRaises(q.Error):
                store.find("missing")
            self.assertEqual(api.call_count, 1000)


class LifecycleCLI(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("QPERSIST_LIFECYCLE_CANARY") == "1", "Requires lifecycle review first")
    def test_real_cli_failure_retry_restore(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            source = root / "source"
            source.mkdir()
            (source / "fixture.bin").write_bytes(b"immutable-canary" * 1000)
            meta = {"kind": "canary", "code_commit": "1" * 40, "environment_sha256": "2" * 64, "dataset_sha256": {}}
            (root / "meta.json").write_text(json.dumps(meta))
            tool = str(Path(q.__file__).resolve())
            def call(*args, success=True):
                proc = subprocess.run([sys.executable, tool, *map(str, args)], capture_output=True, text=True, cwd=root, timeout=60)
                self.assertEqual(proc.returncode == 0, success, proc.stderr)
                return json.loads(proc.stdout) if success else None
            call("inventory", "--source", source, "--include", "fixture.bin", "--metadata", root / "meta.json", "--out", root / "spec.json", "--closed")
            identity = call("pack", "--source", source, "--spec", root / "spec.json", "--out", root / "package")
            call("publish", "--backend", "local", "--store", root / "store", "--package", root / "package", "--receipt", root / "receipt.json")
            chunk = root / "store" / (identity["snapshot"] + "--chunk-000000.bin")
            original = chunk.read_bytes()
            chunk.write_bytes(b"injected corruption")
            args = ["restore", "--backend", "local", "--store", root / "store", "--snapshot", identity["snapshot"], "--manifest-sha256", identity["manifest_sha256"], "--out", root / "restored"]
            call(*args, success=False)
            self.assertFalse((root / "restored").exists())
            self.assertFalse(list(root.glob(".qpersist-restore-*")))
            chunk.write_bytes(original)
            call(*args)
            self.assertEqual(q.digest(root / "restored/fixture.bin"), q.digest(source / "fixture.bin"))
            self.assertEqual(q.digest(root / "package/manifest.json"), identity["manifest_sha256"])


if __name__ == "__main__":
    unittest.main()
