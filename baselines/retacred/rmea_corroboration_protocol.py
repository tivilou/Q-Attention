"""Prospective full-data qualification of the unchanged screened corroboration plugin."""
from . import rmea_fair_reference_protocol as base
from .rmea_fair_reference_protocol import (BASE_MANIFEST,BASE_SUMMARY,BASE_INITIAL,LABELS_SHA,
    INPUT_REQUEST_SHA,INPUT_ACK_ID,REPORT_FILES,CANARY_MANIFEST,ROOT,EXPECTED,identity,read,git,atomic_json,digest)
EXPERIMENT='cross-side-corroboration'
SCHEMA_PREFIX='q-rmea.cross-side-corroboration'
SCOPED_ADMISSION_SCHEMA=SCHEMA_PREFIX+'.changed-plugin-admission.v1'
REFERENCE_METHOD=base.REFERENCE_METHOD
REFERENCE_DESCRIPTION=base.REFERENCE_DESCRIPTION
ANALYSIS_DIRECTORY='analysis'
MODES=('quantum_corroboration', 'dephased_corroboration', 'normalized_dephased_corroboration', 'gaussian_corroboration', 'gaussian_base')
EVIDENCE='docs/evidence/rmea/20261125-cross-side-corroboration/'
REVIEW=EVIDENCE+'review.json'
CORE_EVIDENCE={'dependency-verification.json':'923ec6836d0c32747327f521dffcfde39a83ed84ebca31fea88122df15234530',
    'canary-validation.json':'782fa6b3410c6d8ab5de0849826961eba88b8ccab05c6d67c1af13aa3ec32867',
    'RUN_COMPLETE':'9ed903c679d8d778550b47df1758bd01251c83cfb4dafb84ad62d7a90eb4b5a9'}
OPTICAL_REFERENCE_ROOT='/hy-tmp/Q-Attention/owner/attempts/049e9a8e73a4424b84050043b53cd924/output/number-coherent-optical-seed13-20261125-v1/discovery'
EXTRA=('baselines/retacred/rmea_corroboration.py','baselines/retacred/rmea_corroboration_protocol.py',
    'baselines/retacred/rmea_corroboration_runtime.py','baselines/retacred/rmea_corroboration_pairs.py','experiments/q_rmea_corroboration_train.py',
    'experiments/check_rmea_corroboration.py',EVIDENCE+'protocol.md',EVIDENCE+'native-summary.json',EVIDENCE+'native-manifest.json',
    EVIDENCE+'optical-summary.json',EVIDENCE+'optical-manifest.json',
    'baselines/retacred/rmea_number_optical.py','baselines/retacred/rmea_corroboration_factory.py')
def sources():
    rows=base.sources()
    for n in EXTRA:
        f=ROOT/n
        if not f.is_file() or f.is_symlink():raise ValueError('Missing cross-side-corroboration source: '+n)
        rows[n]=digest(f)
    return rows
def config(profile):
    if profile!='discovery':raise ValueError('Reuse accepted core; no repeated canary profile')
    cfg=base.config(profile);native=base.parameter_schemas()
    cfg.update(schema=SCHEMA_PREFIX+'.training.v1',conditions=list(MODES),
        parameter_schemas={k:{'native.'+n:v for n,v in native['real12'].items()} for k in MODES},
        primary='Selected TEST all40Macro: quantum_corroboration strictly greater than all thirteen new and reused controls',
        qualification='All thirteen selected TEST differences positive; otherwise close this exact profile; no maturity inherited',
        reference=dict(native_manifest=digest(ROOT/(EVIDENCE+'native-manifest.json')),
            native_summary=digest(ROOT/(EVIDENCE+'native-summary.json')),retrain_all_conditions=False),
        replication='Exposed seed13 qualification only; independent confirmation frozen prospectively only if all gates pass',
        corroboration=dict(plugin_sha256='6174cb5aba15c106ed432dbe08c6f750dae6b70499a32be34797e9d744f0b131',
            base='anisotropic Gaussian, weights (3-sqrt2,3+sqrt2)/24 repeated six times',
            gate='subject-side Tr(rho_i mean_object rho_j), object-side one; no object falls back to base',
            subject='absolute role0 <= absolute role1, ties subject',epsilon=1e-6,
            raw_dephasing='population dot mean object population',
            normalized_dephasing='unit-L2 populations before the same operation',
            classical_pair='mean object anisotropic Gaussian pair overlap',
            disabled_gate='Gaussian base, identical new training implementation'),
        optical_reference=dict(summary=digest(ROOT/(EVIDENCE+'optical-summary.json')),
            manifest=digest(ROOT/(EVIDENCE+'optical-manifest.json'))),
        classical_control='four new ablations/replacements and nine complete fixed references; registered count not capacity equality')
    return cfg
def contract():
    return dict(profiles={'discovery':config('discovery')},formal_owner='owner via existing allocation',
        launch='Unchanged frozen-feature core and dynamic multi-GPU queue; five new conditions only',
        feasibility='Reuse exact screened operator/gradient fixtures; full-factory integration check; actual first8 TRAIN one update each; no heldout read',
        recovery='Unchanged local best/latest and full-state10percent aggregate COS barriers; original pinned inputs reused',
        controls='Nine completed full native and number-optical controls authenticated by original summaries/manifests; no retraining',
        analysis='All fourteen VALID-selected TEST states replayed only after completion for thirteen repair/harm partitions; no reselection',
        comparator_dependency='Preserve native and old optical selected states and its existing pinned recovery until paired analysis completes',
        evaluation_timing='Existing worker evaluates selected TEST when each condition finishes; agent reads all effects only after every condition. No interim adaptation.',
        scope='No automatic L3/L4-T; known density/fidelity/Gaussian tools; OOD donor diagnostic does not establish semantic causality; no resource equality or hardness claim')
def review_gate(*,action):
    row=read(ROOT/REVIEW)
    if row.get('verdict')!='PASS' or row.get('source_sha256')!=identity(sources()) or row.get('contract_sha256')!=identity(contract()):
        raise ValueError('Changed-plugin review identity differs')
    if action not in ('run','worker','export'):raise ValueError('No repeated lifecycle matrix')
    return row
