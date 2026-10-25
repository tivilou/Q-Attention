import io
import json
from pathlib import Path
import tempfile
import unittest

import qpersist as q
from cos_store import CosStore, CosError


class Missing(Exception):
    def get_status_code(self):
        return 404


class Body:
    def __init__(self, data):
        self.data = data
    def get_raw_stream(self):
        return io.BytesIO(self.data)


class VersionedClient:
    def __init__(self):
        self.objects = {}
        self.reads = []
        self.puts = 0
        self.unversioned = False
        self.forbidden = False
        self.corrupt = False
    def get_object(self, Bucket, Key, **kwargs):
        if self.forbidden:
            raise RuntimeError('sensitive-token-must-not-escape')
        self.reads.append(kwargs)
        versions = self.objects.get(Key)
        if not versions:
            raise Missing()
        version = kwargs.get('VersionId', str(len(versions)))
        if version not in versions:
            raise Missing()
        data = b'corrupt' if self.corrupt else versions[version]
        return {'x-cos-version-id': version, 'Body': Body(data)}
    def put_object(self, Bucket, Key, Body, **kwargs):
        self.puts += 1
        versions = self.objects.setdefault(Key, {})
        version = str(len(versions) + 1)
        versions[version] = Body.read()
        return {'x-cos-version-id': 'null' if self.unversioned else version}


class CosTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.client = VersionedClient()
        self.store = self.make()
        self.name = '1' * 32 + '--chunk-000000.bin'
        self.source = self.root / 'source'
        self.source.write_bytes(b'original')
    def tearDown(self):
        self.tmp.cleanup()
    def make(self, receipt=None):
        return CosStore('research-test-123456', 'ap-beijing', 'projects/example/snapshots/', receipt, self.client)

    def test_pinned_restore_survives_later_version(self):
        self.store.put(self.name, self.source)
        receipt = self.store.receipt_metadata()
        self.client.objects[self.store.key(self.name)]['2'] = b'changed'
        restore = self.make(receipt)
        out = self.root / 'out'
        restore.get(self.name, out)
        self.assertEqual(out.read_bytes(), b'original')
        self.assertEqual(self.client.reads[-1]['VersionId'], '1')

    def test_identical_retry_reuses_and_conflict_refuses(self):
        self.store.put(self.name, self.source)
        self.make().put(self.name, self.source)
        self.assertEqual(self.client.puts, 1)
        self.source.write_bytes(b'different')
        with self.assertRaises(CosError):
            self.make().put(self.name, self.source)
        self.assertEqual(self.client.puts, 1)

    def test_missing_version_does_not_fall_back_to_latest(self):
        self.store.put(self.name, self.source)
        restore = self.make(self.store.receipt_metadata())
        self.client.objects[self.store.key(self.name)] = {'2': b'changed'}
        with self.assertRaises(CosError):
            restore.get(self.name, self.root / 'out')

    def test_reject_missing_pin_unversioned_and_wrong_scope(self):
        with self.assertRaises(CosError):
            self.store.get(self.name, self.root / 'out')
        self.client.unversioned = True
        with self.assertRaises(CosError):
            self.store.put(self.name, self.source)
        with self.assertRaises(CosError):
            CosStore(self.store.bucket, 'ap-beijing', 'projects/../', client=self.client)
        self.client.unversioned = False
        self.store.put(self.name, self.source)
        receipt = self.store.receipt_metadata()
        receipt['cos']['prefix'] = 'projects/other/snapshots/'
        with self.assertRaises(CosError):
            self.make(receipt)

    def test_permission_failure_is_not_missing_and_is_sanitized(self):
        self.client.forbidden = True
        with self.assertRaises(CosError) as result:
            self.store.put(self.name, self.source)
        self.assertNotIn('sensitive-token', str(result.exception))
        self.assertEqual(self.client.puts, 0)

    def test_publish_restore_and_corruption_gate(self):
        data = self.root / 'data'
        data.mkdir()
        (data / 'a').write_bytes(b'fixture')
        (data / 'empty').touch()
        metadata = {'kind': 'canary', 'code_commit': '1'*40,
                    'environment_sha256': '2'*64, 'dataset_sha256': {}}
        spec = self.root / 'spec.json'
        package = self.root / 'package'
        q.inventory(data, ['a','empty'], metadata, spec, True)
        identity = q.pack(data, spec, package)
        receipt = q.publish(package, self.store, self.root / 'receipt.json')
        self.assertEqual(len(receipt['cos']['object_versions']), 2)
        restore = self.make(receipt)
        q.restore(restore, identity['snapshot'], identity['manifest_sha256'], self.root / 'out')
        self.assertEqual((self.root/'out/a').read_bytes(), b'fixture')
        self.client.corrupt = True
        with self.assertRaises(q.Error):
            q.restore(restore, identity['snapshot'], identity['manifest_sha256'], self.root/'bad')
        self.assertFalse((self.root/'bad').exists())


if __name__ == '__main__':
    unittest.main()
