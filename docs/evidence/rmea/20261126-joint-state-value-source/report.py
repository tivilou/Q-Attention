"""Reopen pinned local sources and persist a scoped source-audit finding."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json

r=Path(__file__).resolve().parent
root=Path('E:/Claude.Code/Paper.Writing/Q-Attention')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
save=lambda n,x:(r/n).write_bytes((json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode())
previous=Path('D:/CodexHome/task-artifacts/qattn-context-pair-diagnostic-20261126')
pins=json.loads((previous/'next-direction.json').read_bytes())['source_pins']
for p in pins: assert sha(p['path'])==p['sha256'],p['path']
extra=[
 root/'.ai-progress/workstreams/quantum-projector-v2/decisions/2026-09-09-20260909-q-bam-candidate.md',
 root/'.codex-artifacts/semantic-connected-evidence-20261125/plugin.py',
 Path('D:/CodexHome/task-artifacts/qattn-state-value-preflight-20261126/source/gen92-plugin.py'),
 Path('D:/CodexHome/task-artifacts/qattn-relation-frame-20261125/source/baselines/retacred/rmea.py')]
pins += [dict(path=str(p),sha256=sha(p)) for p in extra]
receipt_path=r/'fulltexts.private/011f390a050bf4d3/receipt.json'
receipt=json.loads(receipt_path.read_bytes())
assert receipt['identity']=='verified' and receipt['version']=='arxiv_v2'
assert receipt['pages']==13 and receipt['bytes']==804181
assert receipt['sha256']=='95561d5444e91b185a60c39838ce7590c33779511803b461bf7b9e67ed2c51ad'
assert sha(receipt['path'])==receipt['sha256']
parsed=json.loads(Path(receipt['parsed_path']).read_bytes())
assert len(parsed['pages'])==13 and sha(receipt['parsed_path'])==receipt['parse_sha256']
for p in (3,4,5,7): assert (r/f'page{p}.private.png').is_file()
landing=json.loads((r/'source-resolution.json').read_bytes())
assert landing['status']==200 and landing['identity_match']
reading=dict(paper_id='arxiv:2205.05625v2',title=receipt['title'],
 authors=['Guangxi Li','Xuanqiang Zhao','Xin Wang'],
 source_url='https://arxiv.org/pdf/2205.05625v2',
 pdf_sha256=receipt['sha256'],pdf_bytes=receipt['bytes'],total_pages=13,
 identity='VERIFIED_TITLE_AUTHORS_ID_VERSION',version='arxiv_v2',
 version_notes='Official landing: revision 2023-09-28; PDF title date 2023-09-29. Both retained, not silently equated.',
 reading='READ_FOR_SCOPE',text_pages_read=list(range(1,8)),visually_inspected_pages=[3,4,5,7],
 unread_pages=list(range(8,14)),uninspected=['upstream implementation','supplements','published-version differences'],
 findings=[
  dict(locator='p3 Eq3; p4 Eq4-7',fact='Common encoded input state feeds query, key and value circuits. Value coordinates are Pauli-observable expectations after a unitary; the attention kernel is Gaussian on projected query/key expectations.'),
  dict(locator='p5 Eq8-10; p6 Eq11-18; p7 Algorithm1',fact='Task loss optimizes quantum ansatz and decoder parameters; previous-layer outputs feed later common encoded states. This is not a frozen-teacher value-regression protocol.'),
  dict(locator='p3-5',fact='The source does not implement the native 1028-to-12 preparer, forty rank-one Born class effects, EPS word normalization and arbitrary bounded class-Hermitian utility as one exact architecture.')],
 interpretation='Same-state bounded quantum values and task optimization have direct precedent. A specific native joint architecture remains untested here; that scope gap does not establish a new structural mechanism, efficacy or theory.',
 epistemic_limits=['No claim that this paper proves all native joint variants equivalent.','No empirical benefit transfers from its text-classification experiments.','No hardware speedup or task-independent impossibility is inferred.'],
 recorded_at=datetime.now(timezone.utc).isoformat())
save('reading-receipt.json',reading)
save('source-pins.json',pins)
comparison=[
 dict(name='gen102',same='Value-span question for identical native rho.',different='All teacher parameters were frozen and utility was imitated. Joint task learning was not run.',conclusion='Its STOP remains; no rank/ridge/teacher/seed rescue and no general joint-learning impossibility.'),
 dict(name='same-state value D',same='Exactly v_rt=Tr(V_r rho_t); original Born/EPS weighted mean.',different='A future joint-training protocol changes optimization, not this component formula.',conclusion='Do not present the existing first/second density-moment identity as new theory.'),
 dict(name='QSANN v2',same='Common-state bounded observable values, quantum attention and task optimization.',different='Projected-Gaussian token self-attention with Pauli value channels/residual layers, rather than native Born class routing.',conclusion='Direct component precedent; neither full equivalence nor novelty clearance.'),
 dict(name='Q-BAM',same='Uses quantum measurements.',different='Sequential signed disturbance/back-action; D reads preprojection expectation.',conclusion='Structurally different old family remains stopped; no instrument-family revival.'),
 dict(name='gen97/gen92',same='Joint task/router-value questions and pooling.',different='Independent h-linear value channels; covariance or frozen-value warmup, not same-rho expectation.',conclusion='Distinct formulas, but shared-state coadaptation alone is still a generic untested hypothesis.'),
 dict(name='conditional token POVM',same='Can exactly express existing normalized weights and signed pooling.',different='It does not force free utilities to be a Hermitian functional of rho.',conclusion='Reexpression is not a new intervention; tomography bounds do not transfer.')]
save('comparison.json',comparison)
next_direction=dict(status='NEXT_ONE_BOUNDED_MECHANISM_DESIGN_FROM_EXISTING_EVIDENCE',
 action='Produce at most one provisional quantum attention mechanism that changes an actual information path relative to the retained stopped families, using the existing full-study and gen98/102/104/105 ledger. A single bounded specialist design consultation may be used through the existing model-consultation protocol; the primary owns selection. Return NONE if no structural delta survives.',
 required=['Name the precise changed branch/field and task hypothesis, not only a different optimizer/training schedule.',
 'Compare the closest retained closed profiles and the newly read QSANN component. Do not rename generic same-state coadaptation as a new mechanism.',
 'Specify strongest appropriate Gaussian/real classical controls, same-function real16/Hermitian64 technical control when applicable, matched budget and key ablations.',
 'Give one falsifiable task prediction and a compact small-screen promotion/stop rule before task computation. Reuse established qualification and runner; no new evidence framework.',
 'Separate known mathematics from a potential nontrivial theorem; no novelty, benefit or L4-T promotion claim from advice.'],
 source_pins=[dict(path=str(r/'reading-receipt.json'),sha256=sha(r/'reading-receipt.json')),
  dict(path=str(r/'comparison.json'),sha256=sha(r/'comparison.json')),
  dict(path=str(previous/'summary.json'),sha256=sha(previous/'summary.json')),
  dict(path=str(root/'.ai-progress/workstreams/quantum-projector-v2/decisions/2026-11-24-close-fair-reference-full-seed13-20261124.md'),sha256=sha(root/'.ai-progress/workstreams/quantum-projector-v2/decisions/2026-11-24-close-fair-reference-full-seed13-20261124.md'))],
 candidate_selected=False,training_admitted=False,
 resources_next=dict(GPU=0,new_task_forward=0,fit=0,new_capacity=0),
 retained_stops=['gen98','gen102','gen104','gen105','all other exact stopped profiles'],
 goal='ACTIVE',newL3='NOT_ACCEPTED',newL4T='NOT_ACCEPTED')
save('next-direction.json',next_direction)
summary=dict(status='COMPLETE_SOURCE_SCOPE_CHECK',decision='NONE_NEW_STRUCTURAL_CANDIDATE',
 scope='No prior exact native joint task study identified in the pinned records; generic common-state observable values/task training have direct precedent, and D is the same retained component. No new independent task mechanism was established.',
 pinned_local_sources=len(pins),official_metadata_requests=1,official_pdf_requests=1,web_search_service='FAILED_404_API_ROUTE_NOT_FOUND_NO_SEARCH_EVIDENCE',
 paper_sha256=receipt['sha256'],source_pages_read=[1,2,3,4,5,6,7],source_visual_pages=[3,4,5,7],
 resources=dict(GPU=0,new_task_forward=0,native_circuit=0,fit=0,model_API=0,new_capacity=0),
 known_gradient='dz=sum_t alpha_t dv_t + sum_t alpha_t(v_t-z_without_bias) dp_t/(p_t+EPS) for fixed effect/observable/bias and state variation. This chain rule is known, not a new mechanism or T.',
 fairness='Any future utility comparison must retain strongest trained Gaussian/real references and separate dephasing route/value in a 2x2 design; exact Hermitian64 evaluation is technical parity, not an independent learned classical baseline.',
 candidate_selected=False,training_admitted=False,goal='ACTIVE',newL3='NOT_ACCEPTED',newL4T='NOT_ACCEPTED',
 original_L4_card_sha256='ba10c598e8f12b91668a339cff8d367647025124494afac0f37221c1f023d89e')
save('summary.json',summary)
print(json.dumps(dict(status=summary['status'],decision=summary['decision'],source_pins=len(pins),paper_sha256=receipt['sha256'],task_forward=0,fit=0)))
