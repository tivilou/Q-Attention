"""Stdlib-only checks for shared wiring; no model, CUDA or cloud access."""
import contextlib
import os
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from baselines.retacred import rmea_experiments as registry
from baselines.retacred import rmea_task_runtime as task


class RuntimeTests(unittest.TestCase):
    def protocol(self, root):
        def digest(path):
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()
        def atomic(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, sort_keys=True), encoding='utf8')
        return types.SimpleNamespace(ROOT=root, MODES=('a', 'b'),
            sources=lambda: {'core': 'fixed'}, identity=lambda value: json.dumps(value),
            git=lambda *a: 'revision', atomic_json=atomic,
            read=lambda path: json.loads(Path(path).read_text(encoding='utf8')), digest=digest)

    def test_registry_is_allowlisted_and_immutable(self):
        self.assertEqual(set(registry.EXPERIMENTS), {'entity-scattering', 'protected-context'})
        with self.assertRaises(TypeError):
            registry.EXPERIMENTS['dynamic'] = None
        with self.assertRaises(ValueError):
            registry.build('os')

    def test_cli_forwards_existing_arguments(self):
        runner = unittest.mock.Mock()
        with patch.object(registry, 'build', return_value=runner) as build:
            registry.main(['--experiment', 'protected-context', 'unit', '--output', 'x', '--resume'])
        build.assert_called_once_with('protected-context')
        runner.main.assert_called_once_with(['unit', '--output', 'x', '--resume'])

    def test_cli_passes_action_help_without_loading_another_study(self):
        runner = unittest.mock.Mock()
        with patch.object(registry, 'build', return_value=runner):
            registry.main(['--experiment', 'entity-scattering', 'worker', '--help'])
        runner.main.assert_called_once_with(['worker', '--help'])

    def test_cli_preserves_options_before_action(self):
        runner = unittest.mock.Mock()
        with patch.object(registry, 'build', return_value=runner):
            registry.main(['--experiment=protected-context', '--output', 'x', 'unit'])
        runner.main.assert_called_once_with(['--output', 'x', 'unit'])

    def test_disabled_preparation_has_no_side_effect(self):
        task.prepare(None, types.SimpleNamespace(enabled=False), None, None)

    def test_worker_argv_and_assignment(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); p = self.protocol(root)
            with patch('uuid.uuid4', return_value=types.SimpleNamespace(hex='fixed-token')):
                spec = task.launch_spec(p, 'experiments/legacy.py', root/'run', 'a', '2', resume=True)
            assignment = p.read(root/'run/.workers/a/assignment.json')
            self.assertEqual(spec['physical_gpu'], '2')
            self.assertEqual(spec['cwd'], str(root))
            self.assertEqual(assignment['condition'], 'a')
            self.assertEqual(assignment['token'], 'fixed-token')
            self.assertEqual(assignment['source'], p.identity(p.sources()))
            self.assertEqual(spec['argv'][2:4], [str(root/'experiments/legacy.py'), 'worker'])
            self.assertEqual(spec['argv'][-1], '--resume')
            self.assertEqual(spec['argv'][spec['argv'].index('--output')+1], str((root/'run').resolve()))

    def test_invalid_worker_does_not_create_assignment(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.protocol(Path(d))
            for kind, device in [('unknown', '0'), ('a', ' ')]:
                with self.assertRaises(ValueError):
                    task.launch_spec(p, 'legacy.py', Path(d)/'run', kind, device, resume=False)
            self.assertFalse((Path(d)/'run').exists())

    def fixture(self, root):
        p = self.protocol(root); output = root/'run'; output.mkdir()
        p.EXPECTED = {'train': (1, 'a'*64), 'valid': (1, 'b'*64), 'test': (1, 'c'*64)}
        p.EXPERIMENT = 'fixture'
        p.REPORT_FILES = {'summary.json', 'manifest.json', 'analysis-readiness.json',
                          'data.sha256', 'data_counts.txt', 'run_summary.json', 'receipt.json'}
        p.atomic_json(output/'manifest.json', dict(config=dict(counts={'train': 1, 'valid': 1, 'test': 1}, profile='canary'),
            canary_input_manifest='inputs', cache={}, source_sha256='source', git_revision='revision'))
        p.atomic_json(output/'summary.json', dict(results={}, reference={}))
        p.atomic_json(output/'analysis-readiness.json', dict(status='fixture'))
        p.atomic_json(output/'RUN_COMPLETE', dict(artifacts={name:dict(sha256=p.digest(output/name), bytes=(output/name).stat().st_size)
            for name in ('summary.json', 'manifest.json', 'analysis-readiness.json')}))
        return p, output

    def test_export_failure_retry_and_existing_immutability(self):
        # This is the moved export function on tiny JSON, not a live restore canary.
        recovery = types.ModuleType('baselines.retacred.rmea_token_transport_recovery')
        recovery.sync_directory = lambda path: None
        # Windows rejects fsync on a read-only descriptor; POSIX durability is unchanged.
        fsync = patch.object(task.os, 'fsync') if os.name == 'nt' else contextlib.nullcontext()
        with tempfile.TemporaryDirectory() as d, patch.dict('sys.modules', {recovery.__name__: recovery}), fsync:
            root = Path(d); p, output = self.fixture(root); target = root/'reports/result'
            original = {f.name: f.read_bytes() for f in output.iterdir()}
            with self.assertRaises(OSError):
                task.export(p, lambda _: None, output, target, inject_failure=True)
            self.assertFalse(target.exists())
            self.assertEqual(list(target.parent.iterdir()), [])
            first = task.export(p, lambda _: None, output, target)
            second = task.export(p, lambda _: None, output, target)
            self.assertEqual(first, second)
            self.assertEqual({f.name for f in target.iterdir()}, p.REPORT_FILES)
            self.assertEqual({f.name: f.read_bytes() for f in output.iterdir()}, original)
            (target/'summary.json').write_text('{}')
            with self.assertRaises(ValueError):
                task.export(p, lambda _: None, output, target)
            self.assertEqual({x.name for x in target.parent.iterdir()}, {'result'})

    def test_export_rejects_changed_source(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); p, output = self.fixture(root)
            (output/'summary.json').write_text('{}')
            with self.assertRaises(ValueError):
                task.export(p, lambda _: None, output, root/'report')
            self.assertFalse((root/'report').exists())


if __name__ == '__main__':
    unittest.main()
