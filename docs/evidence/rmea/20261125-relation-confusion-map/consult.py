"""One owner-selected mathematical consultation about the newly mapped task gap."""
from pathlib import Path
import hashlib
import json
import os
import time
import urllib.request

r=Path(__file__).resolve().parent
assert not (r/'advice.json').exists(),'Reuse completed advice'
load=lambda n:json.loads((r/n).read_bytes())
classes=load('class-analysis.json');mapping=load('confusion-map.json');fit=load('summary.json')
old=json.loads((r.parent/'staged-bottleneck-design-20261125/review-packet.private.json').read_bytes())
context=old['verified_context']
context['evidence_limits']='All data below are development TRAIN8192fit/4096assessment, same group-disjoint split/all40/seed13/1024updates, no independent confirmation. No newL3/T. Full studies previously also lost to fair classical controls; this small map does not erase them.'
context['closed_mechanisms']+='; common frozen utility256/768; equivalent folded linear readout; mean-state spectral unitary flow pi; connected value covariance; full-strength known BalancedSoftmax training. These exact profiles are closed, no coefficient/threshold/LR/seed/budget rescue. Semantic relation query sharing is not established as novel; old shared-coordinate/factorized scores already existed.'
context['new_verified_observation']='Q full40 fitMacro .806097 exceeds D .787402, but Q assessment .389301 <D .472501. G .875209/.484229 and Gcov .921118/.512054. On true positives forced39 sorting Q301 errors,D236,G240,Gcov218. Fit-support lower13classes (1..13examples each;59assessment) Q .642799/.203452,D .625771/.298662,G .805618/.345365. Not a generalization/variance/query-specific cause. Some classes underfit too.'
context['new_failed_intervention']='Known train-only CE(z+log FITcount,y) on nativeQ/D/G: assessment .369464/.385478/.405525 vs native .389301/.472501/.484229. Q positive-to-null450->93 but nullFP333->1182; Q forced-positive ranking correct1064->1011. Do not propose class weights/prior/threshold rescue.'
selection=dict(selected_model='gpt-6.2-astra',reasoning_effort='high',selection_basis='explicit_owner_mathematical_ideation_preference',
    benchmark_superiority_claim=False,scope='One bounded scientific mechanism proposal, no repository/tool authority')
packet=dict(task='Suggest ONE concrete next falsifiable mechanism or NONE for the mapped Q-RMEA relation-generalization gap, not another audit/portfolio.',
    verified_context=context,all40_class_fit_and_assessment=classes,
    directed_positive_confusions=mapping['all_observed_directed_errors'][:20],
    support_tertiles=load('support-analysis.json'),
    request=[
        'Distinguish observations from causes. Q trains better than D overall and in all3support groups but assesses worse; a generic need for more expressive queries is not established. Some rare relations train well yet transfer poorly; some underfit. Same-fit/eval gaps do not by themselves isolate model variance.',
        'Find one concrete task-linked problem and a minimal plausible mechanism; explanation must state what operation changes outputs/gradients and why a quantum implementation could beat a fair classical replacement. Generic semantic sharing, mixture/shrinkage/regularization may be a known tool, not theoretical novelty. Do not claim a new theorem from a familiar identity or invent citations.',
        'If proposing shared/compositional relation measurements, specify exact public label code/inputs and give those SAME inputs/structure/regularization to classical and dephased controls. Avoid handpicked rare-class rescue, adding independent classical classifier residual, or arbitrarily assuming interference helps. Known linear hypernetworks/PGM/covariance/Gram facts are not newT.',
        'Give exact math with explicit assumptions and one distinguishing prediction/counterexample. Mark any theorem as known, proved here, conjecture, or unverified. Do not promise a distribution-free generalization bound without assumptions. Name the closest collision; papers only as leads unless precise identity known.',
        'Keep stable training core/runner; propose at most4 new small8192fit/4096assessment/seed13/1024 arms with necessary disabled/key-ablation/fairclassical comparisons and no seed/coefficient sweep. Primaryall40Macro; conditional positive and tertiles are descriptive, not success metrics. Only a promising complete screen may advance to full design. If evidence insufficient, return NONE with one cheap discriminating test, not a long list.',
        'Answer compact JSON <=1700words: recommendation ONE|NONE, observation, problem, mechanism, exact_math, distinguishing_prediction, nearest_collision, fair_control, screen, stop_rule, limitations. You cannot run tools or approve experiments; primary agent decides.'
    ])
save=lambda n,v:(r/n).write_bytes((json.dumps(v,indent=2)+'\n').encode())
save('advice-selection.json',selection)
encoded=(json.dumps(packet,indent=2)+'\n').encode();(r/'advice-packet.private.json').write_bytes(encoded)
auth={'Authorization':'Bearer '+os.environ['A6API_API_KEY'],'Content-Type':'application/json'}
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(urllib.request.Request('https://api.a6api.vip/v1/models',headers=auth),timeout=30) as f:catalog=json.load(f)
assert any(v['id']=='gpt-6.2-astra' for v in catalog['data'])
save('model-preflight.json',dict(model='gpt-6.2-astra',present=True,observed_at=time.time()))
started=time.time();path=r/'advice-raw.private.json'
if path.exists():raw=json.loads(path.read_bytes())
else:
    body=dict(model='gpt-6.2-astra',reasoning=dict(effort='high'),max_output_tokens=2800,input=json.dumps(packet))
    with opener.open(urllib.request.Request('https://api.a6api.vip/v1/responses',data=json.dumps(body).encode(),headers=auth),timeout=360) as f:raw=json.load(f)
    save(path.name,raw)
assert raw['model']=='gpt-6.2-astra' and raw['reasoning']['effort']=='high'
answers=[]
for item in raw.get('output',[]):
    t=''.join(c.get('text','') for c in item.get('content',[]) if c.get('type')=='output_text').strip()
    if t.startswith('{'):answers.append(json.loads(t))
assert len(answers)==1
out=dict(model=raw['model'],effort='high',seconds=time.time()-started,packet_sha256=hashlib.sha256(encoded).hexdigest(),
    selection_basis='explicit_owner_preference',response=answers[0],status='ADVICE_NOT_ACCEPTANCE')
save('advice.json',out);print(json.dumps(out['response']))
