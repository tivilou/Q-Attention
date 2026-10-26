"""Focused trust-boundary and recovery tests; no formal training."""
import os
from pathlib import Path
import tempfile
import subprocess
import unittest
from unittest.mock import patch
from baselines.retacred import rmea_protocol as p
from baselines.retacred import rmea_handoff as h

class Lifecycle(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name)
        self.checkout=self.base/'checkout';self.checkout.mkdir()
        self.patch=patch.object(p,'ROOT',self.checkout);self.patch.start()
        self.sources=patch.object(p,'sources',return_value={'test':'frozen'});self.sources.start()
        self.root=self.base/'20261026T000000Z_seed13';self.root.mkdir()
        self.phase=self.root/'canary';self.phase.mkdir()
        manifest=dict(source_sha256=p.identity(p.sources()),config=p.config('canary'),git_revision='abc',environment={'same':True},
            data={s:dict(source_sha256=p.EXPECTED[s][1],source_count=p.EXPECTED[s][0],selected_count=n) for s,n in p.config('canary')['counts'].items()})
        p.atomic_json(self.root/'handoff.json',dict(canary_only=True,git_revision='abc'))
        p.atomic_json(self.phase/'manifest.json',manifest)
        p.atomic_json(self.phase/'summary.json',dict(results={m:{} for m in p.MODES},common_initialization_equal=True))
        p.atomic_json(self.phase/'trace_validation.json',dict(semantic='pass'))
        p.seal(self.phase);self.target=self.base/'safe-report'
    def tearDown(self):self.sources.stop();self.patch.stop();self.temp.cleanup()
    def test_injected_copy_cleanup_and_export_retry_immutable(self):
        before=p.inventory(self.phase);times={n:(self.phase/n).stat().st_mtime_ns for n in before}
        with self.assertRaisesRegex(RuntimeError,'Injected'):h.export(self.root,self.target,True)
        self.assertFalse(self.target.exists());self.assertEqual(list(self.base.glob('.safe-report-*')),[])
        h.export(self.root,self.target);first=h.validate_report(self.target)
        h.export(self.root,self.target);self.assertEqual(first,h.validate_report(self.target))
        self.assertEqual(before,p.inventory(self.phase));self.assertEqual(times,{n:(self.phase/n).stat().st_mtime_ns for n in before})
    def test_closed_artifact_mutation_rejected(self):
        (self.phase/'summary.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'inventory'):h.export(self.root,self.target)
    def test_extra_report_file_rejected(self):
        h.export(self.root,self.target);(self.target/'weights.pt').write_bytes(b'private')
        with self.assertRaisesRegex(ValueError,'whitelist'):h.validate_report(self.target)
    def test_symlink_artifact_rejected(self):
        try:(self.phase/'redirect').symlink_to(self.phase/'summary.json')
        except OSError:self.skipTest('symlinks unavailable')
        with self.assertRaises(ValueError):p.audit(self.phase)
    def test_profile_cannot_change_budget_or_seed(self):
        self.assertEqual(p.config('discovery')['seed'],13);self.assertEqual(p.config('discovery')['epochs'],20)
        self.assertEqual(p.config('discovery')['counts'],{s:v[0] for s,v in p.EXPECTED.items()})
        with self.assertRaises(ValueError):p.config('custom')
    def test_development_cannot_enable_discovery(self):
        import argparse
        with patch.object(p,'review_gate'):
            with self.assertRaisesRegex(ValueError,'canary-only'):h.probe(argparse.Namespace(development_canary=True,canary_only=False))
    def test_missing_completion_marker_rejected(self):
        (self.phase/'RUN_COMPLETE').unlink()
        with self.assertRaisesRegex(ValueError,'incomplete'):p.audit(self.phase)
    def test_git_publication_push_failure_retry_does_not_duplicate_commit(self):
        from baselines.retacred import rmea_exchange as exchange
        remote=self.base/'remote.git'
        subprocess.run(['git','init','--bare',str(remote)],check=True,capture_output=True)
        p.git('init','-b','main');p.git('config','user.name','Research Test Fixture');p.git('config','user.email','fixture@example.invalid')
        p.git('commit','--allow-empty','-m','fixture baseline');p.git('remote','add','origin',str(remote));p.git('push','origin','main')
        p.git('switch','-c','1.1');p.git('push','-u','origin','1.1');baseline=p.git('rev-parse','HEAD')
        for name in ('manifest.json',):
            obj=p.read(self.phase/name);obj['git_revision']=baseline;p.atomic_json(self.phase/name,obj)
        p.atomic_json(self.root/'handoff.json',dict(canary_only=True,git_revision=baseline));p.seal(self.phase)
        receipt=dict(status='uploaded',closed_sha256=p.digest(self.phase/'RUN_COMPLETE'),parts=[dict(verified=True)])
        original=p.git;injected=[False]
        def fail_push(*args):
            if args[0]=='push' and not injected[0]:injected[0]=True;raise RuntimeError('injected push failure')
            return original(*args)
        with patch.object(p,'review_gate'),patch.object(exchange,'upload',return_value=receipt):
            with patch.object(p,'git',side_effect=fail_push):
                with self.assertRaisesRegex(RuntimeError,'injected push'):h.publish(self.root)
            committed=p.git('rev-parse','HEAD');self.assertNotEqual(committed,baseline)
            h.publish(self.root);self.assertEqual(p.git('rev-parse','HEAD'),committed)
        self.assertEqual(p.git('rev-parse','origin/1.1'),committed);self.assertEqual(p.git('status','--porcelain'),'')

if __name__=='__main__':unittest.main()
