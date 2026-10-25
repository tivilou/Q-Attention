import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import dev_archive as d
import dev_archive_owner as owner
import qpersist as q
from cos_store import CosStore, CosError
from test_cos_store import VersionedClient


def command(*args, cwd=None):
    result = subprocess.run(list(args), cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    if result.returncode:
        raise AssertionError(result.stderr.decode(errors='replace'))
    return result.stdout.decode().strip()


def repository(path):
    command('git', 'init', '-b', 'main', str(path))
    command('git', '-C', str(path), 'config', 'user.name', 'Archive Fixture')
    command('git', '-C', str(path), 'config', 'user.email', 'fixture@example.invalid')


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root / 'code'
        repository(self.repo)
        (self.repo / 'env.lock').write_text('stdlib-only\n', encoding='utf-8')
        (self.repo / 'test.py').write_text('print("fixture passed")\n', encoding='utf-8')
        command('git', '-C', str(self.repo), 'add', '.')
        command('git', '-C', str(self.repo), 'commit', '-m', 'fixture')
        self.recipe = dict(schema='research.dev-test.recipe.v1', project_id='example', kind='unit',
                           timeout_seconds=3, command=['{python}', 'test.py'],
                           include=['env.lock', 'test.py'], outputs=[], environment_file='env.lock')
        self.recipe_path = self.root / 'recipe.json'
        self.save_recipe()
        self.client = VersionedClient()
        self.stores = []

    def tearDown(self):
        self.temp.cleanup()

    def save_recipe(self):
        self.recipe_path.write_bytes(q.encoded(self.recipe))

    def capture(self):
        return d.capture(self.repo, self.recipe_path, self.root / 'attempt')

    def make_store(self, bucket, region, prefix, receipt=None):
        result = CosStore(bucket, region, prefix, receipt, self.client)
        self.stores.append(result)
        return result

    def make_index(self):
        index = self.root / 'index'
        repository(index)
        project = index / 'projects/example'
        (project / 'snapshots').mkdir(parents=True)
        (index / 'storage.json').write_bytes(q.encoded(dict(bucket='research-fixture-123456', region='ap-beijing')))
        (project / 'deployment.json').write_bytes(q.encoded(dict(project_id='example', storage_namespace='projects/example',
            cos_bucket='research-fixture-123456', cos_region='ap-beijing')))
        command('git', '-C', str(index), 'add', '.')
        command('git', '-C', str(index), 'commit', '-m', 'index fixture')
        self.origin = self.root / 'origin.git'
        command('git', 'clone', '--bare', str(index), str(self.origin))
        command('git', '-C', str(index), 'remote', 'add', 'origin', str(self.origin))
        command('git', '-C', str(index), 'fetch', 'origin', 'main')
        self.index = index
        return index

    def fixture_seal(self, status='passed'):
        # Exercise publication on Windows too; capture itself is Linux-only.
        attempt = self.root / 'attempt'
        source = attempt / 'payload'
        source.mkdir(parents=True)
        info = dict(schema='research.dev-test.capture.v1', attempt='a'*32, project_id='example', kind='unit',
            code_commit='1'*40, environment_sha256='2'*64, recipe_sha256='3'*64,
            test_status=status, returncode=0 if status == 'passed' else 7,
            started_utc='2026-10-25T00:00:00+00:00', duration_seconds=0.1,
            python_version='3.10.22', cpu_only=True)
        (source / 'capture.json').write_bytes(q.encoded(info))
        (source / 'stdout.txt').write_bytes(b'fixture')
        meta = dict(kind='closed_run', code_commit='1'*40, environment_sha256='2'*64, dataset_sha256={})
        q.inventory(source, ['capture.json', 'stdout.txt'], meta, attempt / 'inventory.json', True)
        identity = q.pack(source, attempt / 'inventory.json', attempt / 'package')
        (attempt / 'SEALED.json').write_bytes(q.encoded(dict(schema='research.dev-test.sealed.v1', capture=info, **identity)))
        return attempt

    @unittest.skipUnless(os.name == 'posix', 'Linux capture process groups')
    def test_capture_runs_real_command_sanitizes_credentials_and_restores_exact_files(self):
        self.recipe['command'] = ['{python}', '-c', 'import os; print(os.environ.get("TENCENTCLOUD_SECRET_ID", "absent")); print(os.environ["CUDA_VISIBLE_DEVICES"])']
        self.save_recipe()
        with patch.dict(os.environ, TENCENTCLOUD_SECRET_ID='fixture-do-not-leak'):
            sealed = self.capture()
        self.assertEqual(sealed['capture']['test_status'], 'passed')
        self.assertEqual((self.root / 'attempt/payload/test/stdout.txt').read_text(), 'absent\n\n')
        store = q.LocalStore(self.root / 'objects')
        receipt = q.publish(self.root / 'attempt/package', store, self.root / 'receipt.json')
        q.restore(store, receipt['snapshot'], receipt['manifest_sha256'], self.root / 'restored')
        self.assertEqual(q.digest(self.root / 'restored/code/test.py'), q.digest(self.repo / 'test.py'))
        self.assertFalse((self.root / 'restored/.git').exists())

    @unittest.skipUnless(os.name == 'posix', 'Linux capture process groups')
    def test_failed_test_is_sealed_with_failure_status(self):
        self.recipe['command'] = ['{python}', '-c', 'raise SystemExit(7)']
        self.save_recipe()
        sealed = self.capture()
        self.assertEqual(sealed['capture']['test_status'], 'failed')
        self.assertEqual(sealed['capture']['returncode'], 7)
        self.assertTrue((self.root / 'attempt/SEALED.json').exists())

    @unittest.skipUnless(os.name == 'posix', 'Linux capture process groups')
    def test_timeout_closes_process_and_can_archive_failure(self):
        self.recipe.update(command=['{python}', '-c', 'import time; time.sleep(20)'], timeout_seconds=1)
        self.save_recipe()
        sealed = self.capture()
        self.assertEqual(sealed['capture']['test_status'], 'timed_out')
        self.assertLess(sealed['capture']['duration_seconds'], 10)

    @unittest.skipUnless(os.name == 'posix', 'Linux capture process groups')
    def test_dirty_source_or_test_mutation_cannot_be_sealed(self):
        (self.repo / 'env.lock').write_bytes(b'changed')
        with self.assertRaises(d.Error):
            self.capture()
        self.assertFalse((self.root / 'attempt').exists())
        command('git', '-C', str(self.repo), 'checkout', '--', 'env.lock')
        self.recipe['command'] = ['{python}', '-c', 'open("test.py", "w").write("changed")']
        self.save_recipe()
        with self.assertRaises(d.Error):
            self.capture()
        self.assertFalse((self.root / 'attempt/SEALED.json').exists())

    def test_recipe_rejects_training_kind_traversal_and_unbounded_timeout(self):
        for key, value in [('kind', 'formal'), ('include', ['../secret']), ('timeout_seconds', 901), ('command', ['sh', '-c', 'true'])]:
            original = self.recipe[key]
            self.recipe[key] = value
            self.save_recipe()
            with self.assertRaises((d.Error, q.Error)):
                d.recipe_load(self.recipe_path)
            self.recipe[key] = original

    def test_publish_restores_and_commits_only_compact_metadata_and_retry_is_idempotent(self):
        attempt = self.fixture_seal()
        index = self.make_index()
        complete = d.publish(attempt, index, self.make_store)
        self.assertEqual(complete['status'], 'archived')
        count = self.client.puts
        original = q.json_file(attempt / 'ARCHIVED.json')
        self.assertEqual(d.publish(attempt, index, self.make_store), original)
        self.assertEqual(self.client.puts, count)
        self.assertEqual(command('git', '-C', str(index), 'diff-tree', '--no-commit-id', '--name-only', '-r', 'HEAD'), complete['index_path'])
        record = q.json_file(index / complete['index_path'])
        self.assertEqual(record['independent_restore']['status'], 'restored_and_verified')
        self.assertEqual(command('git', '--git-dir', str(self.origin), 'rev-parse', 'refs/heads/main'), complete['index_commit'])
        self.assertTrue(all('VersionId' in read for read in self.client.reads[-2:]))

    def test_upload_interruption_no_complete_retry_reuses_objects_without_test_execution(self):
        attempt = self.fixture_seal()
        index = self.make_index()
        normal = self.make_store
        def fail(bucket, region, prefix, receipt=None):
            store = normal(bucket, region, prefix, receipt)
            put = store.put
            def interrupted(name, source):
                if name.endswith('--manifest.json'):
                    raise OSError('injected')
                return put(name, source)
            store.put = interrupted
            return store
        with self.assertRaises(OSError):
            d.publish(attempt, index, fail)
        self.assertFalse((attempt / 'ARCHIVED.json').exists())
        self.assertFalse((attempt / 'publish-receipt.json').exists())
        self.assertEqual(self.client.puts, 1)
        d.publish(attempt, index, normal)
        self.assertEqual(self.client.puts, 2)

    def test_push_failure_after_commit_retries_without_upload_or_test_rerun(self):
        attempt = self.fixture_seal()
        index = self.make_index()
        original = d.git
        def fail_push(repo, *args):
            if args[0] == 'push':
                raise d.Error('injected offline')
            return original(repo, *args)
        with patch.object(d, 'git', fail_push), self.assertRaises(d.Error):
            d.publish(attempt, index, self.make_store)
        self.assertFalse((attempt / 'ARCHIVED.json').exists())
        self.assertTrue((attempt / 'publish-receipt.json').exists())
        count = self.client.puts
        d.publish(attempt, index, self.make_store)
        self.assertEqual(self.client.puts, count)

    def test_commit_failure_after_stage_can_retry(self):
        attempt = self.fixture_seal()
        index = self.make_index()
        original = d.git
        def fail_commit(repo, *args):
            if args[0] == 'commit':
                raise d.Error('injected identity failure')
            return original(repo, *args)
        with patch.object(d, 'git', fail_commit), self.assertRaises(d.Error):
            d.publish(attempt, index, self.make_store)
        self.assertFalse((attempt / 'ARCHIVED.json').exists())
        d.publish(attempt, index, self.make_store)
        self.assertTrue((attempt / 'ARCHIVED.json').exists())

    def test_corrupt_download_or_package_never_emits_complete(self):
        attempt = self.fixture_seal()
        index = self.make_index()
        self.client.corrupt = True
        with self.assertRaises(q.Error):
            d.publish(attempt, index, self.make_store)
        self.assertFalse((attempt / 'ARCHIVED.json').exists())
        chunk = next((attempt / 'package').glob('chunk-*.bin'))
        chunk.write_bytes(b'corrupt')
        count = self.client.puts
        with self.assertRaises(d.Error):
            d.publish(attempt, index, self.make_store)
        self.assertEqual(self.client.puts, count)

    def test_failed_test_archives_as_failed_not_passed(self):
        attempt = self.fixture_seal('failed')
        complete = d.publish(attempt, self.make_index(), self.make_store)
        self.assertEqual(complete['status'], 'archived')
        self.assertEqual(complete['test_status'], 'failed')

    def test_unrelated_dirty_index_and_receipt_mismatch_fail_closed(self):
        attempt = self.fixture_seal()
        index = self.make_index()
        (index / 'unrelated.txt').write_bytes(b'user content')
        with self.assertRaises(d.Error):
            d.publish(attempt, index, self.make_store)
        self.assertEqual(self.client.puts, 0)
        (index / 'unrelated.txt').unlink()
        (attempt / 'publish-receipt.json').write_bytes(q.encoded(dict(snapshot='b'*32)))
        with self.assertRaises(d.Error):
            d.publish(attempt, index, self.make_store)
        self.assertFalse((attempt / 'ARCHIVED.json').exists())

    def test_private_index_record_conflict_does_not_get_overwritten(self):
        attempt = self.fixture_seal()
        index = self.make_index()
        relative = 'projects/example/snapshots/dev-' + 'a'*32 + '.json'
        target = index / relative
        target.write_bytes(b'{"owner":"other-attempt"}\n')
        with self.assertRaises(d.Error):
            d.publish(attempt, index, self.make_store)
        self.assertEqual(target.read_bytes(), b'{"owner":"other-attempt"}\n')
        self.assertFalse((attempt / 'ARCHIVED.json').exists())

    def test_project_namespace_mismatch_blocks_before_cloud_write(self):
        attempt = self.fixture_seal()
        index = self.make_index()
        deployment = index / 'projects/example/deployment.json'
        config = q.json_file(deployment)
        config['storage_namespace'] = 'projects/another'
        deployment.write_bytes(q.encoded(config))
        command('git', '-C', str(index), 'add', '.')
        command('git', '-C', str(index), 'commit', '-m', 'wrong namespace fixture')
        command('git', '-C', str(index), 'push', 'origin', 'main')
        with self.assertRaises(d.Error):
            d.publish(attempt, index, self.make_store)
        self.assertEqual(self.client.puts, 0)

    def test_tampered_seal_does_not_match_restored_capture(self):
        attempt = self.fixture_seal()
        index = self.make_index()
        seal = q.json_file(attempt / 'SEALED.json')
        seal['capture']['recipe_sha256'] = '4'*64
        (attempt / 'SEALED.json').write_bytes(q.encoded(seal))
        with self.assertRaises(d.Error):
            d.publish(attempt, index, self.make_store)
        self.assertFalse((attempt / 'ARCHIVED.json').exists())

    def test_concurrent_publisher_lock_rejects_second_writer(self):
        with d.lock(self.root / 'lock'):
            with self.assertRaises(d.Error):
                with d.lock(self.root / 'lock'):
                    self.fail('second publisher obtained lock')

    def test_lock_excludes_an_independent_publisher_process(self):
        script = 'import sys; from pathlib import Path; import dev_archive as d; '\
                 'p=Path(sys.argv[1]); '\
                 '\ntry:\n with d.lock(p): pass\nexcept d.Error: sys.exit(7)\n'
        with d.lock(self.root / 'process-lock'):
            env = dict(os.environ, PYTHONPATH=str(Path(d.__file__).parent))
            proc = subprocess.run([sys.executable, '-c', script, str(self.root / 'process-lock')], env=env, capture_output=True, timeout=10)
            self.assertEqual(proc.returncode, 7)

    def test_credential_filter_is_case_insensitive_for_common_names(self):
        names = ['GITHUB_TOKEN', 'github_token', 'npm_token', 'session_id', 'AUTHORIZATION', 'TENCENTCLOUD_SECRET_ID']
        with patch.dict(os.environ, {name: 'fixture-private' for name in names}):
            env = d.clean_env()
        self.assertTrue(all(name not in env for name in names))
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')

    def test_owner_retry_never_contacts_worker_and_restores_credential_environment(self):
        attempt = self.fixture_seal()
        secret = self.root / 'owner-secret.json'
        secret.write_bytes(q.encoded(dict(SecretId='fixture-id', SecretKey='fixture-key')))
        profile = dict(credential_file=str(secret), index='unused-fixture')
        def publish(local, index):
            self.assertEqual(os.environ['TENCENTCLOUD_SECRET_ID'], 'fixture-id')
            self.assertNotIn('TENCENTCLOUD_SESSION_TOKEN', os.environ)
            return {'test_status': 'passed'}
        with patch.object(owner, 'resolve', side_effect=AssertionError('retry contacted worker')), patch.object(d, 'publish', publish), patch.dict(os.environ, TENCENTCLOUD_SECRET_ID='previous', TENCENTCLOUD_SESSION_TOKEN='previous-token'):
            self.assertEqual(owner.flow(profile, attempt)['test_status'], 'passed')
            self.assertEqual(os.environ['TENCENTCLOUD_SECRET_ID'], 'previous')
            self.assertEqual(os.environ['TENCENTCLOUD_SESSION_TOKEN'], 'previous-token')


if __name__ == '__main__':
    unittest.main()
