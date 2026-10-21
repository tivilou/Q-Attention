"""Contract boundaries and recovery failures, without full-data experiments."""
import copy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
from q_attention.experiments import cfra_protocol as p,cfra_checkpoint as cp,cfra_runner as runner


class ContractTests(unittest.TestCase):
    def test_frozen_population_and_cli(self):
        cfg=p.load_config()
        self.assertEqual(cfg['seed'],13);self.assertEqual(cfg['selectors'],list(p.MODES))
        self.assertEqual(cfg['expected_records'],dict(train=58465,valid=19584,test=13418))
        with self.assertRaises(SystemExit):runner.parser().parse_args(['--seed','29'])
        with self.assertRaises(ValueError):runner.main(['--stop-after','2'])

    def test_parent_import_is_torch_free(self):
        result=subprocess.run([sys.executable,'-c',"import sys;from q_attention.experiments import cfra_runner;assert 'torch' not in sys.modules"],
            cwd=p.ROOT,env=dict(os.environ,PYTHONPATH=str(p.ROOT/'src')),capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr.decode())

    def test_lazy_public_api_preserves_exports(self):
        import q_attention
        import q_attention.experiments
        for module in (q_attention,q_attention.experiments):
            for name in module.__all__:self.assertIsNotNone(getattr(module,name))

    def test_gpu_visibility_memory_and_uniqueness(self):
        inventory=[dict(id='0',free=8192),dict(id='2',free=26000),dict(id='5',free=8000)]
        self.assertEqual(runner.gpu_ids('auto',inventory=inventory),['0','2'])
        self.assertEqual(runner.execution_profile(['0','2'],inventory)['micro_batch'],1)
        self.assertEqual(runner.execution_profile(['2'],inventory)['micro_batch'],4)
        for spec in ('0,0','1','5','cpu',''):
            with self.assertRaises(ValueError):runner.gpu_ids(spec,inventory=inventory)
        self.assertEqual(runner.gpu_ids('cpu',canary=True,inventory=[]),['cpu'])

    def test_exact_child_target_understands_emitted_arguments(self):
        spec=runner.child_spec('/tmp/nonexistent-cfra-run','quantum','cpu',stop_after=2,recover_previous=True)
        self.assertEqual(Path(spec['argv'][1]),runner.WORKER_TARGET)
        result=subprocess.run(spec['argv']+['--help'],cwd=spec['cwd'],capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr.decode())

    def test_full_population_not_subset_or_changed_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            p.synthetic_data(tmp);cfg=p.load_config(True);manifest=p.inspect_data(tmp,cfg)
            self.assertEqual(manifest['train']['records'],12)
            cfg['expected_sha256']={k:v['sha256'] for k,v in manifest.items()}
            with open(Path(tmp)/'test.jsonl','a') as stream:stream.write(' \n')
            with self.assertRaises(ValueError):p.inspect_data(tmp,cfg)
            with self.assertRaises(ValueError):p.inspect_data(tmp,p.load_config())

    def test_direct_worker_rejects_modified_config(self):
        from q_attention.experiments.cfra_worker import Worker
        with tempfile.TemporaryDirectory() as tmp:
            cfg=p.load_config(True);cfg['kernel']['epochs']=1
            p.atomic_json(Path(tmp)/'run_config.json',cfg);p.atomic_json(Path(tmp)/'run_manifest.json',{})
            with self.assertRaisesRegex(ValueError,'frozen release'):Worker(tmp,'quantum',device='cpu')

    def test_direct_formal_worker_rejects_cpu_before_data_or_job_directory(self):
        from q_attention.experiments.cfra_worker import Worker
        with tempfile.TemporaryDirectory() as tmp:
            cfg=p.load_config();contract=dict(data={},source_sha256='unused')
            contract['fingerprint']=p.identity(dict(config=cfg,**contract))
            p.atomic_json(Path(tmp)/'run_config.json',cfg);p.atomic_json(Path(tmp)/'run_manifest.json',contract)
            with self.assertRaisesRegex(ValueError,'requires CUDA'):Worker(tmp,'baseline',device='cpu')
            self.assertFalse((Path(tmp)/'jobs').exists())

    def test_cursor_is_epoch_stable_and_record0_first(self):
        ids=[p.update_indexes(12,4,i,13) for i in range(6)]
        self.assertEqual(ids[0][0],0)
        for start in (0,3):self.assertEqual(sorted(sum(ids[start:start+3],[])),list(range(12)))
        self.assertNotEqual(ids[:3],ids[3:])


class CheckpointTests(unittest.TestCase):
    def test_completed_repair_refuses_predecessor_requiring_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);job=root/'jobs/quantum';job.mkdir(parents=True)
            for name in ('metrics.json','sample_trace.json','best.pt'):(job/name).write_bytes(b'fixture')
            for update in (5,6):cp.save(job,dict(next_update=update))
            pointer=p.read(job/'latest.json');manifest=dict(fingerprint='fixture',data=dict(train=dict(records=12)))
            marker=dict(fingerprint='fixture',metrics_sha256=p.sha(job/'metrics.json'),trace_sha256=p.sha(job/'sample_trace.json'),
                best_sha256=p.sha(job/'best.pt'),checkpoint_sha256=pointer['latest']['sha256'])
            p.atomic_json(job/'RUN_COMPLETE.json',marker)
            (job/'checkpoints'/pointer['latest']['file']).write_bytes(b'broken')
            with self.assertRaisesRegex(ValueError,'full optimizer budget'):runner.prepare_completed_recovery(root,'quantum',manifest,p.load_config(True))
            self.assertTrue((job/'RUN_COMPLETE.json').exists())

    def test_completed_checkpoint_integrity_and_explicit_invalidation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);job=root/'jobs/quantum';job.mkdir(parents=True)
            for name in ('metrics.json','sample_trace.json','best.pt'):(job/name).write_bytes(b'fixture')
            for _ in range(2):cp.save(job,dict(next_update=6))
            pointer=p.read(job/'latest.json');manifest=dict(fingerprint='fixture',data=dict(train=dict(records=12)))
            marker=dict(fingerprint='fixture',metrics_sha256=p.sha(job/'metrics.json'),trace_sha256=p.sha(job/'sample_trace.json'),
                best_sha256=p.sha(job/'best.pt'),checkpoint_sha256=pointer['latest']['sha256'])
            p.atomic_json(job/'RUN_COMPLETE.json',marker);p.atomic_json(root/'run_summary.json',{})
            self.assertTrue(runner.completed(root,'quantum',manifest))
            with self.assertRaises(ValueError):runner.prepare_completed_recovery(root,'quantum',manifest,p.load_config(True))
            (job/'checkpoints'/pointer['latest']['file']).write_bytes(b'broken')
            with self.assertRaises(ValueError):runner.completed(root,'quantum',manifest)
            unlink=Path.unlink
            def interrupt_invalidation(path,*args,**kwargs):
                if path==job/'RUN_COMPLETE.json':raise OSError('injected crash after repair receipt publication')
                return unlink(path,*args,**kwargs)
            with patch.object(Path,'unlink',interrupt_invalidation):
                with self.assertRaises(OSError):runner.prepare_completed_recovery(root,'quantum',manifest,p.load_config(True))
            self.assertTrue((job/'RUN_COMPLETE.json').exists())
            self.assertTrue((job/'completed_checkpoint_recovery.json').exists())
            runner.prepare_completed_recovery(root,'quantum',manifest,p.load_config(True))
            self.assertFalse((job/'RUN_COMPLETE.json').exists())
            self.assertEqual(p.read(job/'completed_checkpoint_recovery.json')['old_completion'],marker)

    def test_post_pointer_fsync_failure_retains_published_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp.save(tmp,dict(next_update=1))
            original=p.atomic_json
            def commit_then_fail(path,value):
                original(path,value)
                if Path(path).name=='latest.json':raise OSError('injected post-rename directory fsync failure')
            with patch.object(p,'atomic_json',side_effect=commit_then_fail):
                with self.assertRaises(OSError):cp.save(tmp,dict(next_update=2))
            self.assertEqual(cp.load(tmp)['next_update'],2)
            self.assertEqual(cp._entry(tmp,p.read(Path(tmp)/'latest.json')['previous'])['next_update'],1)

    def test_failed_publication_preserves_committed_pointer(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp.save(tmp,dict(next_update=1,value=torch.arange(3)))
            cp.save(tmp,dict(next_update=2,value=torch.arange(3)+1))
            before=(Path(tmp)/'latest.json').read_bytes()
            with self.assertRaises(RuntimeError):cp.save(tmp,dict(next_update=3),inject_failure=True)
            self.assertEqual((Path(tmp)/'latest.json').read_bytes(),before)
            self.assertEqual(cp.load(tmp)['next_update'],2)
            self.assertEqual(len(list((Path(tmp)/'checkpoints').glob('*.pt'))),2)

    def test_corrupt_latest_requires_explicit_valid_predecessor(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i in (1,2,3):cp.save(tmp,dict(next_update=i,value=torch.arange(3)))
            with self.assertRaises(ValueError):cp.load(tmp,recover_previous=True)
            pointer=p.read(Path(tmp)/'latest.json');latest=Path(tmp)/'checkpoints'/pointer['latest']['file'];latest.write_bytes(b'broken')
            with self.assertRaises(ValueError):cp.load(tmp)
            self.assertEqual(cp.load(tmp,recover_previous=True)['next_update'],2)
            cp.save(tmp,dict(next_update=3))
            pointer=p.read(Path(tmp)/'latest.json')
            self.assertEqual(cp._entry(tmp,pointer['previous'])['next_update'],2)
            self.assertTrue((Path(tmp)/'checkpoint_recovery.json').exists())


class MicrobatchTests(unittest.TestCase):
    def test_variable_length_logical_gradient_invariant(self):
        from q_attention.adapters.q_cfra import CFRAModel,objective
        from q_attention.models.relation_transformer import RelationExtractionModel,RelationTransformerConfig
        from q_attention.tasks import cfra_data as d
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            p.synthetic_data(tmp);cfg=p.load_config(True)
            rows,datasets,vocab,labels,maximum,_=p.load_data(tmp,cfg)
            p.seed_all(13)
            base=RelationExtractionModel(RelationTransformerConfig(vocab_size=len(vocab),num_labels=len(labels),max_length=maximum,**cfg['model']))
            d.initialize_neutral_embeddings(base,rows['train'],vocab)
            model=CFRAModel(base,'quantum',cfg['channel']).double()
            batch=d.batch(rows['train'],datasets['train'],vocab,[0,1,2,3]);query=batch['attention_mask'].sum()
            gradients=[]
            for micro in (1,2,4):
                model.zero_grad(set_to_none=True)
                for start in range(0,4,micro):
                    b={k:v[start:start+micro] for k,v in batch.items()}
                    logits=model(**{k:b[k] for k in ('input_ids','attention_mask','subject_mask','object_mask','neutral_ids')})
                    _,parts=objective(logits,model.attention,b['attention_mask'],b['labels'],**cfg['loss'])
                    loss=parts[0]*len(b['labels'])/4+cfg['loss']['prediction_js']*parts[1]*len(b['labels'])/4+cfg['loss']['attention_js']*parts[2]*b['attention_mask'].sum()/query
                    loss.backward()
                gradients.append(torch.cat([v.grad.flatten() for v in model.parameters() if v.requires_grad]))
            for g in gradients[1:]:self.assertTrue(torch.allclose(g,gradients[0],atol=1e-9,rtol=1e-8))

if __name__=='__main__':unittest.main()
