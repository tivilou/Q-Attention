"""Fixed selected-model validation trace intervention; no model fitting."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def circuit(angles, initial=0):
    shape=angles.shape[:-3];state=np.zeros(shape+(8,),dtype=np.complex128);state[...,initial]=1
    for layer in range(2):
        for wire in range(3):
            pair=state.reshape(shape+(2**wire,2,2**(2-wire)));a,b=pair[...,0,:],pair[...,1,:]
            y,z=angles[...,layer,wire,:].T if not shape else (angles[...,layer,wire,0],angles[...,layer,wire,1])
            c=np.cos(y/2)[...,None,None];s=np.sin(y/2)[...,None,None]
            minus=np.exp(-.5j*z)[...,None,None];plus=np.exp(.5j*z)[...,None,None]
            state=np.stack(((c*a-s*b)*minus,(s*a+c*b)*plus),axis=-2).reshape(shape+(8,))
        for control,target in ((0,1),(1,2),(2,0)):
            permutation=[i ^ ((1<<(2-target)) if (i>>(2-control))&1 else 0) for i in range(8)]
            state=state[...,permutation]
    return state


def intervention(saved):
    # This function accepts no labels, targets, sample text or outcome selection.
    psi=np.asarray(saved['psi'],np.complex128)
    phi=circuit(np.asarray(saved['effect_angles'],np.float64))
    chi=circuit(np.asarray(saved['effect_angles'],np.float64),1)
    np.testing.assert_allclose(phi,saved['phi'],atol=5e-6,rtol=3e-5)
    np.testing.assert_allclose(circuit(np.asarray(saved['angles'],np.float64)),psi,atol=5e-6,rtol=3e-5)
    a=np.einsum('rd,btd->brt',phi.conj(),psi);b=np.einsum('rd,btd->brt',chi.conj(),psi)
    mask=np.asarray(saved['mask'],bool)[:,None,:];T=mask.sum(-1,keepdims=True)
    assert (T>0).all()
    A=(np.abs(a)**2*mask).sum(-1,keepdims=True);B=(np.abs(b)**2*mask).sum(-1,keepdims=True)
    C=(b*a.conj()*mask).sum(-1,keepdims=True)
    pair=(B*np.abs(a)**2+A*np.abs(b)**2-2*(a*b.conj()*C).real)/T
    incoherent=(B*np.abs(a)**2+A*np.abs(b)**2-2*np.abs(a)**2*np.abs(b)**2)/T
    assert pair.min()>-1e-10 and incoherent.min()>-1e-10
    pair=np.maximum(pair,0)*mask;incoherent=np.maximum(incoherent,0)*mask
    u=np.asarray(saved['utilities'],np.float64);bias=np.asarray(saved['bias'],np.float64)
    weights={};logits={}
    for name,extra in [('static',0),('complement',pair),('dephased_pair',incoherent)]:
        raw=(np.abs(a)**2+extra+1e-6)*mask;weights[name]=raw/raw.sum(-1,keepdims=True)
        logits[name]=(weights[name]*u).sum(-1)+bias
        assert np.isfinite(logits[name]).all()
        np.testing.assert_allclose(weights[name].sum(-1),1,atol=1e-12)
    np.testing.assert_allclose(weights['static'],saved['alpha'],atol=5e-6,rtol=3e-4)
    np.testing.assert_allclose(logits['static'],saved['logits'],atol=5e-5,rtol=3e-4)
    pair_mass=pair.sum(-1)/(A[...,0]+pair.sum(-1)+T[...,0]*1e-6)
    assert pair_mass.max()<=2/3+1e-5
    return weights,logits,{'pair_mass_mean':float(pair_mass.mean()),'pair_mass_max':float(pair_mass.max()),
        'interference_l1':float(np.abs(pair-incoherent).sum()),'T':int(T[0,0,0])}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--inputs',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    receipt=json.loads((args.inputs/'input-contract.json').read_text())
    assert receipt['stage']=='seed29' and receipt['sample_count']==2
    for name,row in receipt['files'].items():
        p=args.inputs/name
        assert p.is_file() and not p.is_symlink() and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
    trace=json.loads((args.inputs/'sample-trace-legacy.json').read_text())
    trace_v2=json.loads((args.inputs/'sample-trace.json').read_text())
    assert trace['schema_version']=='sample-trace.v1' and trace_v2['schema_version']=='sample-trace.v2'
    manifest=json.loads((args.inputs/'manifest.json').read_text())
    assert manifest['git_revision']=='f3f22660b609811e6fb824f7d19e9cafd0c1e446'
    cases=[]
    for local in (0,1):
        name=f'trace/quantum-selected-valid-{local}.npz'
        matches=[s for s in trace['samples'] if s['condition']=='quantum' and s['checkpoint']=='selected' and s['split']=='valid' and
                 any(x.get('outputs',{}).get('numeric_file')==name for x in s['stages'])]
        assert len(matches)==1
        record=matches[0];stages={s['stage']:s.get('outputs',{}) for s in record['stages']}
        v2matches=[s for s in trace_v2['samples'] if s['sample_id']==record['sample_id']]
        assert len(v2matches)==1
        v2=v2matches[0];prov=v2['roles']['provenance']
        assert v2['split']=='valid' and v2['checkpoint']=='selected' and prov['condition']=='quantum'
        assert prov['cache_index']==record['cache_index'] and prov['checkpoint_file']=='checkpoints/quantum-selected.pt'
        assert prov['sample_id_hash']==stages['data']['sample_id_hash']
        assert v2['roles']['target_or_gold']['value']==stages['evaluation']['target']
        assert v2['roles']['prediction']['value']==stages['evaluation']['prediction']
        scoring=stages['scoring'];assert scoring['checkpoint_file']=='checkpoints/quantum-selected.pt'
        assert scoring['sha256']==receipt['files'][name]['sha256']
        with np.load(args.inputs/name,allow_pickle=False) as values:saved={k:values[k] for k in values.files}
        weights,logits,diagnostic=intervention(saved)
        # Post-generation labels are used only here, after all weights and logits are frozen.
        target=int(stages['evaluation']['target']);predictions={k:int(v[0].argmax()) for k,v in logits.items()}
        assert predictions['static']==int(stages['evaluation']['prediction'])
        utilities=np.asarray(saved['utilities'],np.float64)[0];mask=np.asarray(saved['mask'],bool)[0]
        bias=np.asarray(saved['bias'],np.float64)
        best_target=utilities[target,mask].max()+bias[target]
        low=np.where(mask[None,:],utilities,np.inf).min(-1)+bias;low[target]=-np.inf
        # Optimistic independent-simplex relaxation, not a realizability certificate for the candidate.
        relaxed_margin=float(best_target-low.max())
        cases.append(dict(case=local,cache_index=record['cache_index'],target=target,predictions=predictions,
            correct={k:v==target for k,v in predictions.items()},diagnostic=diagnostic,
            deltas={k:dict(logit_linf=float(np.abs(logits[k]-logits['static']).max()),
                          attention_l1=float(np.abs(weights[k]-weights['static']).sum()),
                          target_logit_delta=float(logits[k][0,target]-logits['static'][0,target])) for k in ('complement','dephased_pair')},
            independent_attention_relaxation_margin=relaxed_margin,
            relaxation_scope='Independent per-relation arbitrary simplex weights with fixed values/decoder. Positive is only feasibility in this relaxation, not candidate attainability or Macro headroom.'))
    result=dict(status='PASS',stage='seed29',cases=cases,sample_count=2,
        changed={k:sum(x['predictions'][k]!=x['predictions']['static'] for x in cases) for k in ('complement','dephased_pair')},
        corrected={k:sum(not x['correct']['static'] and x['correct'][k] for x in cases) for k in ('complement','dephased_pair')},
        harmed={k:sum(x['correct']['static'] and not x['correct'][k] for x in cases) for k in ('complement','dephased_pair')},
        claim_ceiling='Fixed-model case action/risk only. No independent training, population effect, significance, novelty or L3.',
        trace_reader='Archived producer-owned legacy NPZ plus legacy metadata cross-checked against same-case sample-trace.v2 provenance; no new full Case Study capture claim.',
        input_contract_sha256=hashlib.sha256((args.inputs/'input-contract.json').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases',)}))


if __name__=='__main__':main()
