from pathlib import Path
import ast,hashlib,json
r=Path(__file__).resolve().parent
read=lambda n:json.loads((r/n).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
s=read('scope.json');v=read('verification.json');h=read('hypothesis.json')
assert v['status']=='PASS_REOPENED_SYNTHETIC_PAIRING_AND_SOURCE_EXCLUSION'
for p in s['pins']:assert sha(p['path'])==p['sha256']
out=dict(status='NEXT_EXISTING_IDENTITY_BINDING_BEFORE_ONE_READONLY_DIAGNOSTIC',
 action='Locate and hash existing final Qdistinct/Ddistinct/Gdistinct and strongest declared Gcov checkpoints plus exact source/plugin/metadata/cohort/sealed predictions. If consistent, freeze exact one-pass no-update diagnostic call/resource contract for the already frozen pairing statistic; otherwise record technical nonadmission without rerunning old fits.',
 prediction_sha256=sha(r/'hypothesis.json'),
 controls=['strongest declared trained Gcov','retained Ddistinct/Gdistinct reports','same-Q word-state dephasing','exact real16 identity','length and scalar purity'],
 checkpoint_pool='Only completed gen98 Qdistinct/Ddistinct/Gdistinct and originally declared reused Gcov. Do not substitute original full-data Q teacher8ec179f4 or an easier Gaussian baseline.',
 cohort_sha256='8fb8393bbe2241929748283f2bbaa219682696409857ec15678585bcd0b88ff0',
 exposure='Same4096 all40 assessment from already exposed TRAIN, not a new independent confirmation. Keep8192FIT mapping and no threshold search.',
 budget_now=s['resources'],future_task_compute='Not run or admitted by this packet. First verify existing identity/availability and freeze bounded shared-scheduler contract; no new capacity.',
 provisional_diagnostic_only=True,method_selected=False,training_admitted=False,
 gate=h['diagnostic_gate'],
 retained_stops=s['retained_stops'],
 remaining='Fair trained same-core L3 and ablations/appropriate independent confirmation, substantive neighbor distinction, nontrivial correct theory, discriminating prediction and implementation/independent audit remain incomplete.',
 original_L4_card_sha256='ba10c598e8f12b91668a339cff8d367647025124494afac0f37221c1f023d89e',
 goal='ACTIVE',newL3='NOT_ACCEPTED',newL4T='NOT_ACCEPTED')
(r/'next-direction.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(status=out['status'],task_efficacy_assessed=False,training_admitted=False)))
