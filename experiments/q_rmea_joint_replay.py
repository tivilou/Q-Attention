"""Private selected-checkpoint replay: fixture, materialize, audit or coordinates."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['fixture','materialize','audit','value','_fixture','_materialize'])
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--sample');p.add_argument('--representation');p.add_argument('--coords')
    a=p.parse_args();root=a.run_dir.resolve()
    if a.action in ('fixture','materialize'):
        # Keep the control parent free of Torch. Producer exits before the
        # independent auditor starts; never hold two model runtimes concurrently.
        subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'_'+a.action,'--run-dir',str(root)],check=True)
        subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'audit','--run-dir',str(root)],check=True)
        return
    from baselines.retacred import rmea_joint_trace as t
    if a.action=='_fixture':
        import numpy as np
        import torch
        from baselines.retacred.rmea_joint import VARIANTS,make_head
        if root.exists():raise ValueError('Fresh private fixture path required')
        root.mkdir(parents=True);(root/'checkpoints').mkdir()
        torch.manual_seed(7301)
        h=torch.randn(2,4,1024);roles=torch.zeros(2,4,4);mask=torch.tensor([[True,True,True,False],[True,True,True,True]])
        roles[:,0,2]=1;roles[:,1,3]=1
        np.savez_compressed(root/'inputs.npz',h=h.numpy(),roles=roles.numpy(),mask=mask.numpy())
        models=[]
        for v in VARIANTS:
            torch.manual_seed(13);head=make_head(v)
            if v=='no_content':head.fit_role_means([(h[:1],roles[:1],mask[:1])],split='train')
            path=root/'checkpoints'/f'{v}.pt';torch.save(head.state_dict(),path)
            models.append(dict(variant=v,checkpoint=t.ref(root,path),selection_rule='bounded untrained fixture only; not a reported evaluation model'))
        manifest=dict(schema='q-rmea.joint.selected-replay.v1',lane='replay',scope='synthetic label-free head fixture only',
            source_revision=subprocess.check_output(['git','-C',str(Path(__file__).resolve().parents[1]),'rev-parse','HEAD'],text=True).strip(),source_files=t.source_files(),
            sample_selection_rule='fixed synthetic two examples; one train fixture and one valid fixture, no test',
            samples=[dict(sample_id='fixture:0',split='train'),dict(sample_id='fixture:1',split='valid')],
            inputs=t.ref(root,root/'inputs.npz'),models=models)
        (root/'replay-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    manifest=json.loads((root/'replay-manifest.json').read_text(encoding='utf-8'))
    repo=str(Path(__file__).resolve().parents[1])
    if subprocess.check_output(['git','-C',repo,'rev-parse','HEAD'],text=True).strip()!=manifest['source_revision'] or subprocess.check_output(['git','-C',repo,'status','--porcelain'],text=True).strip():
        raise ValueError('Replay requires its exact clean source revision')
    if a.action in ('_fixture','_materialize'):
        t.replay(root,manifest)
    elif a.action=='audit':
        trace=json.loads((root/'sample-trace.json').read_text(encoding='utf-8'))
        report=t.audit(root,manifest,trace)
        report.update(status='passed',manifest_sha256=t.sha(root/'replay-manifest.json'),trace_sha256=t.sha(root/'sample-trace.json'),source_files=t.source_files())
        (root/'trace-validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(report))
    else:
        trace=json.loads((root/'sample-trace.json').read_text(encoding='utf-8'))
        sample=next(s for s in trace['samples'] if s['sample_id']==a.sample)
        print(json.dumps(dict(values=t.read_value(root,sample,a.representation,json.loads(a.coords)))))


if __name__=='__main__':main()
