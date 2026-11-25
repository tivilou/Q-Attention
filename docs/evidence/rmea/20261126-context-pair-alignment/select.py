from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json
r=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_bytes())
write=lambda n,x:(r/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
s=read(r/'scope.json')
for p in s['pins']:assert hashlib.sha256(Path(p['path']).read_bytes()).hexdigest()==p['sha256']
def source(i,anchor):
 p=s['pins'][i];lines=Path(p['path']).read_text('utf8').splitlines()
 matches=[j+1 for j,t in enumerate(lines) if anchor in t]
 assert matches,(i,anchor)
 return dict(**p,anchor=anchor,lines=matches)
routes=[
 dict(route='distinct-token evidence',source=source(0,'def statistic'),
  definition='Product of two alpha-weighted value means minus sum_t alpha_t^2*x_t*y_t. Collision-mass control multiplies the product by 1-sum alpha^2.',
  scope='Gen98 exact five-arm screen stopped. Same underlying states, changes readout of fixed value-channel pairs.',
  comparison='Input/word-state pair alignment is not this value readout. It authorizes neither changing pairs nor reviving the stopped screen.'),
 dict(route='context word frame',source=source(7,'def frame_attention'),
  definition='Whiten key states by inverse square root of sum_t rho_t+n*EPS*I; change query-conditioned word routing.',
  scope='Pinned original attempt exit1, mixed Float/Double; no completed efficacy outcome in this pinned execution. Do not convert technical failure into a valid negative.',
  comparison='The proposed statistic does not whiten states or alter any output. Frame spectrum alone does not identify its input-pair coupling.'),
 dict(route='attention Fisher query step',source=source(2,'def routing_metric'),
  definition='Per-example normalized-routing query derivatives form a block Fisher; damped inverse-square-root shapes AdamW query directions with row-step norm retained.',
  scope='Gen66 fixed profile stopped; no forward change, no new key encoder.',
  comparison='The proposed diagnostic uses no derivative, labels, Fisher, optimizer or coordinate update.'),
 dict(route='native shared-query learning diagnosis',source=source(10,'def query_gradient'),
  definition='All TRAIN CE query gradients at native initial/selected states, aggregated into within/between-class cancellation summaries.',
  scope='Zero-update diagnosis, not a trained intervention or a failed semantic-query method.',
  comparison='Pairwise contextual-input/state association differs from label-aware class query-gradient cancellation. No shared-query rescue is proposed.'),
 dict(route='semantic query sharing',source=source(4,'def mapped_query'),
  definition='Fixed 40x8 name-derived code times trainable 8x12 coordinates; native word encoder and pooling retained, permuted-code and D/G controls included.',
  scope='Gen96 exact rank/tokenization/initialization profile stopped.',
  comparison='The proposed diagnostic neither changes query rank nor uses label names/codes; no rank/tokenization retuning.'),
 dict(route='relation query frame',source=source(5,'def frame_probability'),
  definition='Whiten through all learned class effects sum_r E_r+40*EPS*I; score unchanged keys with transformed class effects.',
  scope='Stopped exact relation-frame profile; noncommuting frame and diagonal/real controls preserved.',
  comparison='Input-pair coupling does not alter the class frame or reinterpret its known square-root algebra as new theory.'),
]
# Read exact status fields, not a broad family name supplied by earlier prose.
failed=read(s['pins'][8]['path']);assert failed['exit_code']==1
obs=read(s['pins'][9]['path']);assert obs['progress']['phase']=='failed'
assert 'expected scalar type Double but found Float' in obs['stderr']
diag=read(s['pins'][12]['path']);assert diag['optimizer_updates']==0
out=dict(status='SIX_SOURCE_DEFINITIONS_READ_NO_OLD_PROFILE_REOPENED',routes=routes,
 corrections=['Pinned context-word-frame record is technical failure, not valid negative efficacy evidence.',
              'Native shared-query record is a zero-update diagnosis, not a stopped trained intervention.'],
 source_exclusion_scope='Only these six pinned definitions and directly read dependencies; not a complete literature-novelty exclusion.',
 decision='RETAIN_ONE_PROVISIONAL_PAIR_ALIGNMENT_DIAGNOSTIC_NOT_A_METHOD',method_selected=False,new_theory=False)
write('source-exclusion.json',out)
hyp=dict(status='FROZEN_PROVISIONAL_INPUT_STATE_PAIRING_PREDICTION',created_at=datetime.now(timezone.utc).isoformat(),
 formula=dict(input='For each valid word, h is the existing contextual 1024-vector; roles still feed original key encoders. Zero-norm h -> diagnostic undefined, report count; no chosen replacement.',
  distance='d_ij=1-<h_i,h_j>/(||h_i|| ||h_j||), over all unordered valid i<j, T>=2.',
  kernel_Q='KQ_ij=|psi_i^* psi_j|^2 from the existing Q head word states, no relation query.',
  kernel_D_sameQ='KD_ij=sum_a |psi_ia|^2 |psi_ja|^2, same Q states dephased only for the diagnostic.',
  kernel_G='KG_ij=exp(-||s*key_G(h_i,roles_i)-s*key_G(h_j,roles_j)||^2), exact frozen anisotropic Gaussian key projection/scale from strongest declared Gcov checkpoint.',
  kernel_real16='v_i=(Re psi_i,Im psi_i); K16_ij=(v_i dot v_j)^2+(v_i dot J v_j)^2, J=[0,-I;I,0]. Must match KQ, not a separately trained control.',
  association='A(K,d)=sum_{i<j} K_ij*d_ij / sum_{i<j} K_ij - mean_{i<j}(d_ij). Zero kernel mass -> undefined/report, no epsilon added.',
  one_primary_statistic='S=A(KQ,d)-A(KG,d). Higher S means Q similar-state pairs span more contextual-input distance than G similar-key pairs.',
  reference='Raw contextual cosine distance is a proxy, not a semantic truth, task-relevance certificate or graph. No role/length subset of pairs.'),
 empirical_prediction='On the same frozen gen98 exposed TRAIN assessment rows, higher label-blind S ranks Gcov-correct/Qdistinct-wrong above Qdistinct-correct/Gcov-wrong. Pair association must add information beyond scalar purity, length and Q-state dephasing.',
 population='Existing gen98 assessment4096/cohort8fb8393b; all40 classes including null, both length bands. Gen98 key encoder is native structure but learned inside semantic-query/distinct readout; diagnostic does not establish original full-Q task causality.',
 outcomes='Use sealed original predictions, never new decision thresholds or modified logits. Gcov is strongest previously declared reused control .512054050292, Qdistinct .500925142309; parameter/structure mismatch81924vs82308 retained.',
 controls=['Same-Q dephased score A(KD,d)-A(KG,d)',
  'Input word count (higher is predicted worse Q)',
  'Quantum mean off-diagonal fidelity (higher is predicted worse Q; scalar purity equivalent at fixed T)',
  'All retained trained gen98 Ddistinct/Gdistinct outputs reported descriptively; not same-parameter interference interventions',
  'Exact real16 representation, technical equality only; no quantum computational advantage claim'],
 diagnostic_gate=dict(primary='Ordinary pooled AUC on all Qdistinct/Gcov discordants, positive=Gcov correct only, higher S positive; ties count 1/2.',
  sample_minimum='At least100 discordants and at least30 in each direction. Report all missing/undefined counts. Any missing statistic in primary population stops this exact prediction; no dropping cases.',
  cutoff='AUC(S)>=0.55 and AUC(S)-max(AUC(each of the three prespecified scalar/dephased controls))>=0.02.',
  secondary='Report all80 gold-class x fixed length<=35/>35 strata counts and within-stratum concordance, including empty/unpaired strata. No subgroup overrides pooled gate.',
  failure='STOP_THIS_EXACT_ALIASING_PREDICTION; no distance/kernel/threshold/length/role/subgroup/checkpoint/seed/control rescue.',
  on_pass='Exploratory association only. Does not reopen gen98, admit training, establish cause, newT, L3 or L4-T.'),
 technical_contract='Before any task work, pin actual Q/D/Gdistinct/Gcov checkpoint identities, input/cache/metadata/prediction mapping, code and exact one-pass call/resource limits. Retain all four controls. Current packet performs no task forward or input-array read.',
 literature='This is a defined kernel/input association diagnostic, with no literature novelty claim. All displayed identities are elementary algebra, not a new theorem.',
 method_selected=False,training_admitted=False,goal='ACTIVE',newL3='NOT_ACCEPTED',newL4T='NOT_ACCEPTED')
write('hypothesis.json',hyp)
print(json.dumps(dict(status=out['status'],routes=len(routes),provisional_statistic='S=A_Q-A_G',training_admitted=False)))
