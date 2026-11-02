import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from experiments.q_rmea_entity_dev import load_cache_archive
from tests.test_rmea_entity import fixture


class CacheTests(unittest.TestCase):
    def archive(self, directory, profile='canary'):
        path = Path(directory)/'owner-canary.tar.gz'
        with tarfile.open(path, 'w:gz') as archive:
            def add(name, raw):
                info = tarfile.TarInfo('canary/'+name)
                info.size = len(raw)
                archive.addfile(info, io.BytesIO(raw))
            add('manifest.json', json.dumps(dict(config=dict(profile=profile,counts=dict(train=16, valid=8, test=8)),
                                                 source_sha256='a'*64)).encode())
            for split, n in (('train', 16), ('valid', 8)):
                h, roles, mask = fixture(n, dtype=torch.float32)
                arrays = dict(h=h.numpy().reshape(-1, 1024), roles=roles.numpy().reshape(-1, 4),
                              offsets=np.arange(n+1, dtype=np.int64)*6, targets=np.arange(n, dtype=np.int64)%4)
                for key, a in arrays.items():
                    data=io.BytesIO();np.save(data,a);add(f'cache/{split}-{key}.npy',data.getvalue())
            add('cache/test-h.npy', b'INVALID: must never load a test member')
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def test_only_pinned_train_valid_members_are_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            path, sha = self.archive(directory)
            original = tarfile.TarFile.extractfile
            opened = []
            def extract(archive, member):
                opened.append(member.name if hasattr(member,'name') else member)
                return original(archive, member)
            with mock.patch.object(tarfile.TarFile, 'extractfile', extract):
                batches, identity = load_cache_archive(path, sha)
            self.assertEqual(len(batches['train'][1]), 16)
            self.assertEqual(len(batches['valid'][1]), 8)
            self.assertEqual(len(identity['selected_member_sha256']), 8)
            self.assertTrue(all('/test-' not in name for name in opened))

    def test_wrong_archive_or_full_profile_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, sha = self.archive(directory)
            with self.assertRaises(ValueError):
                load_cache_archive(path, 'b'*64)
            path, sha = self.archive(directory, profile='discovery')
            with self.assertRaises(ValueError):
                load_cache_archive(path, sha)
