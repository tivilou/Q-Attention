"""Prospective fixed number-coherent optical full-data qualification."""
from . import rmea_fair_reference_protocol as base
from .rmea_fair_reference_protocol import (BASE_MANIFEST,BASE_SUMMARY,BASE_INITIAL,LABELS_SHA,
    INPUT_REQUEST_SHA,INPUT_ACK_ID,REPORT_FILES,CANARY_MANIFEST,ROOT,EXPECTED,identity,read,git,atomic_json,digest)
EXPERIMENT='number-coherent-optical'
SCHEMA_PREFIX='q-rmea.number-coherent-optical'
SCOPED_ADMISSION_SCHEMA=SCHEMA_PREFIX+'.changed-plugin-admission.v1'
REFERENCE_METHOD=base.REFERENCE_METHOD
REFERENCE_DESCRIPTION=base.REFERENCE_DESCRIPTION
ANALYSIS_DIRECTORY='analysis'
MODES=('number_coherent02', 'number_mixed02', 'number_mixed02_purity_normalized', 'curvature_matched_gaussian02')
EVIDENCE='docs/evidence/rmea/20261125-number-coherent-optical/'
REVIEW=EVIDENCE+'review.json'
CORE_EVIDENCE={'dependency-verification.json':'923ec6836d0c32747327f521dffcfde39a83ed84ebca31fea88122df15234530',
    'canary-validation.json':'782fa6b3410c6d8ab5de0849826961eba88b8ccab05c6d67c1af13aa3ec32867',
    'RUN_COMPLETE':'9ed903c679d8d778550b47df1758bd01251c83cfb4dafb84ad62d7a90eb4b5a9'}
OPTICAL_REFERENCE_ROOT='/hy-tmp/Q-Attention/owner/attempts/6f6d06c07ff84cd9adc13a3cb3c44d7b/output/optical-fock-seed13-20261125-v1/discovery'
EXTRA=('baselines/retacred/rmea_number_optical.py','baselines/retacred/rmea_number_optical_protocol.py',
    'baselines/retacred/rmea_number_optical_runtime.py','baselines/retacred/rmea_number_optical_pairs.py','experiments/q_rmea_number_optical_train.py',
    'experiments/check_rmea_number_optical.py',EVIDENCE+'protocol.md',EVIDENCE+'native-summary.json',EVIDENCE+'native-manifest.json',
    EVIDENCE+'optical-summary.json',EVIDENCE+'optical-manifest.json',
    'baselines/retacred/rmea_optical_fock.py')
def sources():
    rows=base.sources()
    for n in EXTRA:
        f=ROOT/n
        if not f.is_file() or f.is_symlink():raise ValueError('Missing number-coherent-optical source: '+n)
        rows[n]=digest(f)
    return rows
def config(profile):
    if profile!='discovery':raise ValueError('Reuse accepted core; no repeated canary profile')
    cfg=base.config(profile);native=base.parameter_schemas()
    cfg.update(schema=SCHEMA_PREFIX+'.training.v1',conditions=list(MODES),
        parameter_schemas={k:native['real12'] for k in MODES},
        primary='Selected TEST all40Macro: number_coherent02 strictly greater than all thirteen new and reused controls',
        qualification='All thirteen differences positive; otherwise close this exact profile; no maturity inherited',
        reference=dict(native_manifest=digest(ROOT/(EVIDENCE+'native-manifest.json')),
            native_summary=digest(ROOT/(EVIDENCE+'native-summary.json')),retrain_all_conditions=False),
        replication='Exposed seed13 qualification only; independent confirmation frozen prospectively only if all gates pass',
        optical=dict(modes=6,mean_photons_per_mode=1,coordinate_scale_squared=24,epsilon=1e-6,
            resource='fixed (vacuum+two-photon)/sqrt2 per mode; different photon-number distribution from old Fock1',
            coherence_ablation='full base-number dephasing before displacement, retaining both populations',
            purity_control='separately2x mixed one-mode overlap; changes effect spectrum',
            gaussian_control='matches kernel curvature only, not a physical state covariance'),
        optical_reference=dict(summary=digest(ROOT/(EVIDENCE+'optical-summary.json')),
            manifest=digest(ROOT/(EVIDENCE+'optical-manifest.json'))),
        classical_control='curvature Gaussian and ten complete fixed references; registered count not capacity equality')
    return cfg
def contract():
    return dict(profiles={'discovery':config('discovery')},formal_owner='owner via existing allocation',
        launch='Unchanged frozen-feature core and dynamic multi-GPU queue; four new conditions only',
        feasibility='Independent number-state operator/gradient fixtures; actual first8 TRAIN one update each; no heldout read',
        recovery='Unchanged local best/latest and full-state10percent aggregate COS barriers; original pinned inputs reused',
        controls='Ten completed full native and optical controls authenticated by original summaries/manifests; no retraining',
        analysis='All fourteen VALID-selected TEST states replayed only after completion for thirteen repair/harm partitions; no reselection',
        comparator_dependency='Preserve native and old optical selected states and its existing pinned recovery until paired analysis completes',
        scope='No automatic L3/L4-T; known displacement formulas and number coherence; no resource equality or hardness claim')
def review_gate(*,action):
    row=read(ROOT/REVIEW)
    if row.get('verdict')!='PASS' or row.get('source_sha256')!=identity(sources()) or row.get('contract_sha256')!=identity(contract()):
        raise ValueError('Changed-plugin review identity differs')
    if action not in ('run','worker','export'):raise ValueError('No repeated lifecycle matrix')
    return row
