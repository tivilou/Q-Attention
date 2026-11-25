"""One bounded owner-selected mathematical consultation, without tool authority."""
from pathlib import Path
import hashlib,json,os,time,urllib.request
r=Path(__file__).resolve().parent
save=lambda n,v:(r/n).write_bytes((json.dumps(v,indent=2)+'\n').encode())
assert not (r/'theory-advice.json').exists()
selection=dict(selection_basis='owner_preference',task='One post-screen mathematical research suggestion',
    model='gpt-6.2-astra',reasoning_effort='high',benchmark_superiority_claim=False,
    alternate_available_profile='gpt-6.2-sol/high',authority='Advice only; no tools, implementation, experiment or grading authority')
packet=dict(selection=selection,
    task='Find ONE distinct task-linked falsifiable hypothesis worth considering next, or NONE. Do not propose extra audits. Reply JSON only.',
    goal='Same-core Q-RMEA attributed L3 followed by substantive novel theory, L4-T. A positive classical derivative is a separately graded candidate, not replacement of the quantum goal.',
    data='Frozen RoBERTa token features1024, entity roles4, sentence-group-disjoint originalTRAIN8192 fit/4096 assessment, all40 Macro, seed13,1024 AdamW updates. Already exposed development screen, no confirmation/novelty claim.',
    results=json.loads((r/'contrast-analysis.json').read_bytes()),
    source=(r/'plugin.py').read_text('utf8'),
    prior='Qnative.389301,Dnative.472501,Gnative.484229,GnativeCov.512054. Qshared.468022,Dshared.492804,Gshared.500862. Qnative fitMacro.806097>Dfit.787402 but assessment reverses. Only aggregate hypothesis, no cause isolated.',
    stop_constraints=[
      'Current exact covariance joint profile failed3/8 gates. Qmeanproduct.504046 also below strongGnativeCov.512054; it was a control, not a predeclared winning candidate. No retrospective success, no control deletion or parameter/seed/budget rescue.',
      'Closed profiles include shared low-rank name query alone, covariance alone, common fixed utility, BalancedSoftmax fullprior loss, flow readout, signed null differential, relation Sinkhorn competition, multi-return interference, optical source coherence, final/mid-layer learned dephasing, rank2measurement, finite-step attention Fisher natural gradient, ordered quantum memory and selected PGMs. They are exact profiles, not universal impossibility; do not rename/tune them as new.',
      'Keep stable runner/optimizer/core. New hypothesis can change a small scientific plugin. No GPU purchase, no brute-force sweep, no new heldout use. Known tools are not theory innovation; claims need real nearest-neighbor work later.',
      'Q/G/D all have fair available information. Parameter count does not establish effective-capacity equality. No synthetic quantum advantage assertion on NLP task.',
      'If facts do not support a distinct near-term hypothesis, return NONE with what concrete observation is missing, rather than force another module combination.'
    ],
    response_schema=dict(recommendation='ONE|NONE',problem='',hypothesis='',mechanism='',
      distinction_from_closed_profiles='',minimal_discriminating_test='',fair_controls=[],stop_rule='',
      theoretical_lead='',unverified_literature_leads=[],limitations=[]))
encoded=json.dumps(packet).encode();(r/'theory-packet.private.json').write_bytes(encoded)
save('theory-selection.json',selection)
auth={'Authorization':'Bearer '+os.environ['A6API_API_KEY'],'Content-Type':'application/json'}
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
assert json.loads((r/'model-preflight.json').read_bytes())['present']
start=time.time();path=r/'theory-raw.private.json'
if path.exists():raw=json.loads(path.read_bytes())
else:
    body=dict(model='gpt-6.2-astra',reasoning=dict(effort='high'),max_output_tokens=1700,input=encoded.decode())
    with opener.open(urllib.request.Request('https://api.a6api.vip/v1/responses',data=json.dumps(body).encode(),headers=auth),timeout=360) as f:raw=json.load(f)
    save('theory-raw.private.json',raw)
assert raw['model']=='gpt-6.2-astra' and raw['reasoning']['effort']=='high'
answers=[]
for item in raw.get('output',[]):
    text=''.join(c.get('text','') for c in item.get('content',[]) if c.get('type')=='output_text').strip()
    if text.startswith('{'):answers.append(json.loads(text))
assert len(answers)==1
record=dict(model=raw['model'],effort='high',seconds=time.time()-start,
    packet_sha256=hashlib.sha256(encoded).hexdigest(),profile_echo_verified=True,response=answers[0])
save('theory-advice.json',record);print(json.dumps(record))
