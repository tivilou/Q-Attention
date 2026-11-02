"""Scientific isolation and report-only lifecycle contract regression."""
import unittest
from unittest.mock import patch
from baselines.retacred import rmea_entity_study_protocol as p
from baselines.retacred import rmea_entity_handoff as h


class ContractTests(unittest.TestCase):
    def test_formal_budget_and_controls(self):
        c=p.config('discovery')
        self.assertEqual(c['seed'],13);self.assertEqual(c['epochs'],20);self.assertEqual(c['batch_size'],128)
        self.assertEqual(c['counts'],dict(train=58465,valid=19584,test=13418))
        self.assertEqual(c['conditions'],['static','ordered','no_content','symmetric','classical_pair'])
        self.assertNotIn('test',p.config('pilot')['counts'])
        self.assertEqual(p.config('canary')['counts'],dict(train=16,valid=8,test=8))

    def test_full_scope_requires_pass(self):
        record=dict(review_scope='both',must_fix=[],source_sha256=p.identity(p.sources()),contract_sha256=p.identity(p.contract()),verdict='PASS_WITH_RISK')
        with patch.object(p,'read',return_value=record):
            self.assertEqual(p.review_gate()['verdict'],'PASS_WITH_RISK')
            with self.assertRaises(ValueError):p.review_gate(formal=True)

    def test_report_schema_has_no_transfer_dependency(self):
        import inspect
        self.assertEqual(len(h.REPORT_FILES),7)
        self.assertNotIn('rmea_exchange',inspect.getsource(h))
        self.assertIn('private_artifacts_policy',inspect.getsource(h.export))


if __name__=='__main__':unittest.main()
