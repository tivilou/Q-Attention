"""Bounded private trace contract tests. No task dataset, selection or training."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from baselines.retacred.rmea_joint import VARIANTS,make_head
from baselines.retacred import rmea_joint_trace as t


class JointTraceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);(self.root/'checkpoints').mkdir()
        torch.manual_seed(7001)
        self.h=torch.randn(2,3,1024,dtype=torch.float64)
        self.roles=torch.zeros(2,3,4,dtype=torch.float64);self.roles[:,0,2]=1;self.roles[:,1,3]=1
        self.mask=torch.tensor([[True,True,False],[True,True,True]])
        np.savez(self.root/'inputs.npz',h=self.h.numpy(),roles=self.roles.numpy(),mask=self.mask.numpy())
        models=[]
        for v in VARIANTS:
            torch.manual_seed(13);head=make_head(v).double()
            if v=='no_content':head.fit_role_means([(self.h[:1],self.roles[:1],self.mask[:1])],split='train')
            cp=self.root/'checkpoints'/f'{v}.pt';torch.save(head.state_dict(),cp)
            models.append(dict(variant=v,checkpoint=t.ref(self.root,cp),selection_rule='untrained fixture only'))
        self.m=dict(schema='q-rmea.joint.selected-replay.v1',lane='replay',source_revision='fixture-pinned-source',
            scope='synthetic fixture, no task evidence',sample_selection_rule='fixed synthetic examples',source_files=t.source_files(),
            samples=[dict(sample_id='train:0',split='train'),dict(sample_id='valid:0',split='valid')],
            models=models,inputs=t.ref(self.root,self.root/'inputs.npz'))

    def trace(self):return t.replay(self.root,self.m)[0]

    def test_all_variants_selected_only_and_complex_reader(self):
        payload=self.trace();r=t.audit(self.root,self.m,payload)
        self.assertEqual(r['samples'],10);self.assertGreater(r['arrays'],180)
        self.assertEqual({s['checkpoint'] for s in payload['samples']},{'selected'})
        s=next(s for s in payload['samples'] if s['sample_id'].startswith('joint:'))
        self.assertIn('imag',t.read_value(self.root,s,'conditional',[[0,0,2]])[0])

    def test_selected_checkpoint_and_dependencies_remain_immutable(self):
        refs=[self.m['inputs']]+[x['checkpoint'] for x in self.m['models']]
        before=[t.sha(self.root/r['path']) for r in refs]
        payload=self.trace();t.audit(self.root,self.m,payload)
        self.assertEqual(before,[t.sha(self.root/r['path']) for r in refs])
        with self.assertRaises(FileExistsError):t.replay(self.root,self.m)

    def test_tampered_checkpoint_and_input_hash_fail(self):
        payload=self.trace()
        for ref in (self.m['inputs'],self.m['models'][0]['checkpoint']):
            changed=copy.deepcopy(self.m)
            if ref is self.m['inputs']:changed['inputs']['sha256']='0'*64
            else:changed['models'][0]['checkpoint']['sha256']='0'*64
            with self.assertRaises(ValueError):t.audit(self.root,changed,payload)

    def test_private_coordinate_bounds_and_hash(self):
        payload=self.trace();s=payload['samples'][0]
        for coords in ([],[[True,0,0]],[[-1,0,0]],[[0,0,1024]],'0'):
            with self.assertRaises(ValueError):t.read_value(self.root,s,'h',coords)
        p=self.root/s['representations'][0]['artifact_refs'][0]['path'];p.write_bytes(b'corrupt')
        with self.assertRaises(ValueError):t.read_value(self.root,s,'h',[[0,0,0]])

    def test_applicable_key_inventory_cannot_be_omitted(self):
        payload=self.trace()
        s=next(s for s in payload['samples'] if s['sample_id'].startswith('joint:'))
        s['representations']=[r for r in s['representations'] if r['id']!='conditional']
        with self.assertRaises(ValueError):t.audit(self.root,self.m,payload)

    def test_axes_and_selected_model_binding_cannot_be_forged(self):
        payload=self.trace();bad=copy.deepcopy(payload)
        bad['samples'][0]['representations'][0]['axis_semantics']=['wrong']*3
        with self.assertRaises(ValueError):t.audit(self.root,self.m,bad)
        bad=copy.deepcopy(payload);bad['samples'][0]['roles']['provenance']['checkpoint']=self.m['models'][1]['checkpoint']
        with self.assertRaises(ValueError):t.audit(self.root,self.m,bad)

    def test_fixed_split_and_sample_coverage(self):
        payload=self.trace()
        for mutate in ('split','duplicate','missing'):
            bad=copy.deepcopy(payload)
            if mutate=='split':bad['samples'][0]['split']='test'
            if mutate=='duplicate':bad['samples'][-1]=copy.deepcopy(bad['samples'][0])
            if mutate=='missing':bad['samples'].pop()
            with self.assertRaises(ValueError):t.audit(self.root,self.m,bad)

    def test_path_escape_rejected(self):
        for name in ('../secret','/tmp/secret'):
            with self.assertRaises(ValueError):t.pinned(self.root,dict(path=name,sha256='0'*64))

    def test_metadata_lineage_and_prediction_are_bound(self):
        payload=self.trace()
        for which in ('revision','config','scope','selection','prediction','selection_rule','lineage'):
            bad=copy.deepcopy(payload)
            if which=='revision':bad['experiment']['code_revision']='other'
            if which=='config':bad['experiment']['config_sha256']='0'*64
            if which=='scope':bad['experiment']['dataset']='other'
            if which=='selection':bad['sample_selection']['rule']='cherry picked'
            if which=='prediction':bad['samples'][0]['roles']['prediction']['value']=99
            if which=='selection_rule':bad['samples'][0]['roles']['provenance']['selection_rule']='last epoch'
            if which=='lineage':bad['samples'][0]['stages'][-1]['input_refs']=['external:invented']
            with self.assertRaises(ValueError,msg=which):t.audit(self.root,self.m,bad)

    def test_auditor_revalidates_manifest_uniqueness_and_source(self):
        payload=self.trace()
        for which in ('model','sample','schema','lane','source'):
            bad=copy.deepcopy(self.m)
            if which=='model':bad['models'].append(copy.deepcopy(bad['models'][0]))
            if which=='sample':bad['samples'][1]=copy.deepcopy(bad['samples'][0])
            if which=='schema':bad['schema']='other'
            if which=='lane':bad['lane']='training'
            if which=='source':bad['source_files']={}
            with self.assertRaises(ValueError,msg=which):t.audit(self.root,bad,payload)


if __name__=='__main__':unittest.main()
