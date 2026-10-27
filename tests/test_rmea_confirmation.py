"""Confirmation trust boundaries and real bare-Git export recovery; no formal training."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from baselines.retacred import rmea_confirmation_protocol as p
from baselines.retacred import rmea_confirmation_handoff as h

class Confirmation(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name);self.checkout=self.base/'checkout';self.checkout.mkdir()
        self.patches=[patch.object(p,'ROOT',self.checkout),patch.object(p.legacy,'ROOT',self.checkout),patch.object(p,'sources',return_value={'fixture':'fixed'})]
        for q in self.patches:q.start()
        self.root=self.base/'20261027T000000Z_fresh5';self.root.mkdir();self.phase=self.root/'canary';self.phase.mkdir()
        self.manifest=dict(source_sha256=p.identity(p.sources()),config=p.config('canary',29),git_revision='abc',environment=dict(same=True),
          data={s:dict(source_count=p.EXPECTED[s][0],source_sha256=p.EXPECTED[s][1],selected_count=n) for s,n in p.config('canary',29)['counts'].items()})
        p.atomic_json(self.phase/'manifest.json',self.manifest)
        p.atomic_json(self.phase/'summary.json',dict(results={m:{} for m in p.MODES},common_initialization_equal=True,seed=29))
        p.atomic_json(self.phase/'trace_validation.json',dict(schema='q-rmea.trace-readiness.v2',structural='pass',semantic='pass',replay='pass',coordinate_reader='pass'))
        self.handoff=dict(canary_only=True,git_revision='abc',source_sha256=p.identity(p.sources()),contract_sha256=p.identity(p.contract()))
        p.atomic_json(self.root/'handoff.json',self.handoff);p.seal(self.phase);self.target=self.base/'safe-report'
    def tearDown(self):
        for q in reversed(self.patches):q.stop()
        self.temp.cleanup()
    def test_five_fresh_seeds_budget_and_legacy_unchanged(self):
        self.assertEqual(p.SEEDS,(29,53,71,97,113));self.assertEqual(p.legacy.config('discovery')['seed'],13)
        for seed in p.SEEDS:
            c=p.config('discovery',seed);self.assertEqual(c['epochs'],20);self.assertEqual(c['batch_size'],128);self.assertEqual(c['counts'],{k:v[0] for k,v in p.EXPECTED.items()})
        with self.assertRaises(ValueError):p.config('discovery',13)
    def test_stats_min_exact_p_and_no_false_claim(self):
        v=p.paired([1,2,3,4,5]);self.assertTrue(v['paired_t_gate']);self.assertEqual(v['exact_two_sided_sign_flip_p'],.0625)
        self.assertFalse(p.paired([.001,.001,.001,.001,-.1])['paired_t_gate'])
        with self.assertRaises(ValueError):p.paired([1,2,3,4])
    def test_copy_failure_cleanup_retry_same_closed_artifacts(self):
        before={f.name:(p.digest(f),f.stat().st_mtime_ns) for f in self.phase.iterdir()}
        with self.assertRaisesRegex(RuntimeError,'Injected'):h.export(self.root,self.target,True)
        self.assertFalse(self.target.exists());self.assertEqual(list(self.base.glob('.safe-report-*')),[])
        h.export(self.root,self.target);h.export(self.root,self.target);h.validate_report(self.target)
        self.assertEqual(before,{f.name:(p.digest(f),f.stat().st_mtime_ns) for f in self.phase.iterdir()})
    def test_closed_mutation_extra_report_and_v1_rejected(self):
        h.export(self.root,self.target);(self.target/'private.npz').write_bytes(b'not allowed')
        with self.assertRaisesRegex(ValueError,'whitelist'):h.validate_report(self.target)
        t=p.read(self.phase/'trace_validation.json');t['schema']='q-rmea.trace-readiness.v1';p.atomic_json(self.phase/'trace_validation.json',t);p.seal(self.phase)
        with self.assertRaisesRegex(ValueError,'v2'):h.audit_batch(self.root)
    def test_partial_batch_does_not_publish(self):
        self.handoff['canary_only']=False;p.atomic_json(self.root/'handoff.json',self.handoff)
        with self.assertRaisesRegex(ValueError,'incomplete'):h.export(self.root,self.target)
    def test_export_does_not_call_training(self):
        with patch.object(subprocess,'run',side_effect=AssertionError('No subprocess during export')):h.export(self.root,self.target)
    def test_development_cannot_enable_formal(self):
        from baselines.retacred import rmea_handoff as old
        with patch.object(p,'review_gate'),patch.object(p.legacy,'review_gate'):
            with self.assertRaisesRegex(ValueError,'canary-only'):h.probe(argparse.Namespace(development_canary=True,canary_only=False))
    def test_actual_git_push_failure_retry_one_commit(self):
        from baselines.retacred import rmea_confirmation_exchange as exchange
        bare=self.base/'remote.git';subprocess.run(['git','init','--bare',str(bare)],check=True,capture_output=True)
        p.git('init');p.git('symbolic-ref','HEAD','refs/heads/main');p.git('config','user.name','Research Test Fixture');p.git('config','user.email','fixture@example.invalid')
        p.git('commit','--allow-empty','-m','fixture');p.git('remote','add','origin',str(bare));p.git('push','origin','main');p.git('checkout','-b','1.1');p.git('push','-u','origin','1.1')
        baseline=p.git('rev-parse','HEAD');self.manifest['git_revision']=baseline;self.handoff['git_revision']=baseline
        p.atomic_json(self.phase/'manifest.json',self.manifest);p.atomic_json(self.root/'handoff.json',self.handoff);p.seal(self.phase)
        original=p.git;injected=[False]
        def fail_once(*args):
            if args[0]=='push' and not injected[0]:injected[0]=True;raise RuntimeError('injected push')
            return original(*args)
        with patch.object(p,'review_gate'),patch.object(exchange,'upload',return_value={'fixture':'canary'}):
            with patch.object(p,'git',side_effect=fail_once):
                with self.assertRaisesRegex(RuntimeError,'injected push'):h.publish(self.root)
            committed=p.git('rev-parse','HEAD');h.publish(self.root);self.assertEqual(committed,p.git('rev-parse','HEAD'))
        self.assertEqual(p.git('rev-parse','origin/1.1'),committed);self.assertEqual(p.git('status','--porcelain'),'')

if __name__=='__main__':unittest.main()
