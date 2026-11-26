from pathlib import Path
import hashlib,json
r=Path(__file__).resolve().parent;root=Path('E:/Claude.Code/Paper.Writing/Q-Attention')
sha=lambda b:hashlib.sha256(b).hexdigest()
pre=json.loads((r/'preflight.json').read_bytes());assert pre['model']=='gpt-6.2-astra'
paths=[
 Path('D:/CodexHome/task-artifacts/qattn-relation-frame-20261125/source/baselines/retacred/rmea.py'),
 root/'.codex-artifacts/conditional-coherence-20261125/plugin.py',
 root/'.codex-artifacts/query-unitary-transport-20261124/modified/baselines/retacred/rmea_query_transport.py',
 root/'.codex-artifacts/register-capacity-20261125/plugin.py',
 Path('D:/CodexHome/task-artifacts/qattn-joint-state-value-source-20261126/reading-receipt.json')]
pins=[dict(path=str(p),sha256=sha(p.read_bytes())) for p in paths]
critical=(root/'.ai-progress/workstreams/quantum-projector-v2/critical-invariants.md').read_text('utf8')
ledger=[x[:1150] for x in critical.splitlines() if x.startswith('- **2026-11-2') and any(k in x for k in ['gen','COMPLETE','STOP','CLOSED'])][:70]
packet=dict(
 task='ONE bounded independent mechanism/design review of the exact proposed code and equations. Do not propose additional candidates. No tools, repository authority, paid resources, task execution or external-search claims.',
 goal='Original Q-RMEA idea-maturity1 L4-T remains active; a pilot would only triage utility. New L3/L4-T not accepted. Classical simulability is not a veto and known building blocks are allowed, but relevant strong classical comparisons and a later nontrivial same-core theory are required.',
 hypothesis=(r/'hypothesis.md').read_text('utf8'),
 code=(r/'interleaved_math.py').read_text('utf8'),
 nearest_implemented_sources={str(p):p.read_text('utf8') for p in paths},
 retained_ledger=ledger,
 actual_complete_baseline='seed13/full TRAIN58465 VALID19584 TEST13418/20epochs: native Q .495323642 below dephased .504767888, softmax12 .534296111, real12 .546447526. Exact profile STOP, no seed rescue. Prior Fock full .542284741 below local-slope Gaussian .542537652/real12 .546447526; coherent optical .530943407 below mixed .542285124/normalized .542001062/Gaussian .551057980. All remain STOP.',
 current_screen_context='Existing TRAIN8192 FIT/4096 assessment group-disjoint/cohort8fb8393b is exposed development material, not independent. Strongest declared Gcov .512054050292; Qdistinct .500925142309/Gdistinct .510930539751. gen102 fixed-teacher observable replacement fails all3 gates; gen104 gold tangent advantage -.008097; gen105 aliasing AUC .509391 below length .541584. These are not causal evidence for this placement.',
 review_scope='mechanism',risk='R2 new mathematical branch. Only data-free fixed synthetic calculations next, not task model/plugin/training or lifecycle changes.',
 proposed_next_falsifier='32 deterministic finite-angle word/query tuples, direct dense complex chain vs exact real16 and normalized vec(Q) lift; compare explicit blockwise dephased Markov chain. A two-word all40 fixture with one signed utility row checks EPS-normalized output/softmax difference. Include one-word, padding, zero input and commuting identity cases. No task data, checkpoint, new seed fit or model-selected cases. Threshold all algebra comparisons <=1e-11, actual output difference reported but not a family/utility claim.',
 questions=[
 'Check p_mid versus p_out algebra, adjoint order, CNOT row convention, vec identity and D interpretation against the supplied original code. A known lift is not new T.',
 'Does a fixed reparameterization trivially reproduce p_mid within the same native 3q/2layer family for all word inputs? Distinguish unconstrained 64-dimensional process lift from that claim. Give a concrete identity or counterexample; mark unproved family separation.',
 'Is it an exact renamed gen76 channel, shared query transport,6q product kernel or other supplied stopped recipe? Compare equations rather than names. Ordinary data reupload/process kernels are known prior art; do not pretend literature novelty is settled.',
 'Is this a genuine new task information-path hypothesis eligible for a bounded mathematical feasibility check, or NONE? Flag effective-capacity changes and the strongest classical explanation. Do not require a solved new theorem before an ordinary pilot.',
 'Assess prospective Q/D/real8 and native real12/Gaussian/strongGcov controls and stop rule. Full task identities and a later implementation review remain necessary before training; no advice grants that admission.',
 'Return JSON only: verdict PASS_WITH_RISK or BLOCK for the next mathematical check, substantive_candidate PROVISIONAL or NONE, correctness/control_fairness/gradient_and_numerics, theory_mapping, fidelity_gaps, must_fix, counterexamples, useful_theory_route, next_action, confidence. BLOCK is substantive and never triggers a different-model retry.'
 ],source_pins=pins)
payload=json.dumps(packet,ensure_ascii=False,indent=2).encode()
(r/'review-packet.private.json').write_bytes(payload)
assert len(payload)<150000
selection=dict(status='selected',basis='owner_preference',selection_basis='Reused owner-selected R2 implementation mechanism review queue; not a benchmark ranking.',
 candidate_models=pre['exact_catalog_candidates'],selected_model_id='gpt-6.2-astra',reasoning_effort='high',protocol='openai-responses',
 candidate_universe=pre['exact_catalog_candidates'],ranked_candidates=[
 dict(rank=i+1,api_model_id=m,reasoning_effort='high',mapping_status='exact_catalog_id',transport_status='catalog_present_prior_profile_verified',dispatch_eligible=True)
 for i,m in enumerate(['gpt-6.2-astra','claude-fable-5-2','gpt-6.2-sol']) if m in pre['exact_catalog_candidates']],
 dispatch_budget=1,fallback='No silent model/effort substitution. One planned consultation; substantive BLOCK retained.',
 packet_sha256=sha(payload),capability_status='owner preference; no current exact-model benchmark-superiority claim',
 metadata_preflight=str(r/'preflight.json'))
(r/'selection.json').write_bytes((json.dumps(selection,indent=2)+'\n').encode())
(r/'source-pins.json').write_bytes((json.dumps(pins,indent=2)+'\n').encode())
print(json.dumps(dict(packet_bytes=len(payload),source_pins=len(pins),retained_ledger_rows=len(ledger),model=selection['selected_model_id'],effort='high')))
