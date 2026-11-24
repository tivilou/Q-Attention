"""Prospective displaced-single-photon full-data qualification."""
from . import rmea_fair_reference_protocol as base
from .rmea_fair_reference_protocol import (BASE_MANIFEST,BASE_SUMMARY,BASE_INITIAL,LABELS_SHA,
    INPUT_REQUEST_SHA,INPUT_ACK_ID,REPORT_FILES,CANARY_MANIFEST,ROOT,EXPECTED,identity,read,git,atomic_json,digest)
EXPERIMENT='optical-fock'
SCHEMA_PREFIX='q-rmea.optical-fock'
SCOPED_ADMISSION_SCHEMA=SCHEMA_PREFIX+'.changed-plugin-admission.v1'
REFERENCE_METHOD=base.REFERENCE_METHOD
REFERENCE_DESCRIPTION=base.REFERENCE_DESCRIPTION
ANALYSIS_DIRECTORY='analysis'
MODES=('fock1','parity_fock1','vacuum_born','gaussian3_born','rbf12')
EVIDENCE='docs/evidence/rmea/20261125-optical-fock/'
REVIEW=EVIDENCE+'review.json'
CORE_EVIDENCE={'dependency-verification.json':'923ec6836d0c32747327f521dffcfde39a83ed84ebca31fea88122df15234530',
    'canary-validation.json':'782fa6b3410c6d8ab5de0849826961eba88b8ccab05c6d67c1af13aa3ec32867',
    'RUN_COMPLETE':'9ed903c679d8d778550b47df1758bd01251c83cfb4dafb84ad62d7a90eb4b5a9'}
EXTRA=('baselines/retacred/rmea_optical_fock.py','baselines/retacred/rmea_optical_fock_protocol.py',
    'baselines/retacred/rmea_optical_fock_runtime.py','baselines/retacred/rmea_optical_fock_pairs.py','experiments/q_rmea_optical_fock_train.py',
    'experiments/check_rmea_optical_fock.py',EVIDENCE+'protocol.md',EVIDENCE+'native-summary.json',EVIDENCE+'native-manifest.json')
def sources():
    rows=base.sources()
    for n in EXTRA:
        f=ROOT/n
        if not f.is_file() or f.is_symlink():raise ValueError('Missing optical-fock source: '+n)
        rows[n]=digest(f)
    return rows
def config(profile):
    if profile!='discovery':raise ValueError('Reuse accepted core; no repeated canary profile')
    cfg=base.config(profile);native=base.parameter_schemas()
    cfg.update(schema=SCHEMA_PREFIX+'.training.v1',conditions=list(MODES),
        parameter_schemas={k:native['real12'] for k in MODES},
        primary='Selected TEST all40Macro: fock1 strictly greater than all nine new and reused controls',
        qualification='All nine differences positive; otherwise close this exact profile; no maturity inherited',
        reference=dict(native_manifest=digest(ROOT/(EVIDENCE+'native-manifest.json')),
            native_summary=digest(ROOT/(EVIDENCE+'native-summary.json')),retrain_all_conditions=False),
        replication='Exposed seed13 qualification only; independent confirmation frozen prospectively only if all gates pass',
        optical=dict(modes=6,photons_per_mode=1,coordinate_scale_squared=24,epsilon=1e-6,
            probability='product exp(-d)*(1-d)^2; exact analytic, no numerical Fock cutoff',
            parity='independent parity-sector CPTP dephasing on both states; not full number dephasing'),
        classical_control='vacuum, slope-matched Gaussian3, strongest unchanged RBF12; identical real12 schema/init/readout')
    return cfg
def contract():
    return dict(profiles={'discovery':config('discovery')},formal_owner='owner via existing allocation',
        launch='Unchanged frozen-feature core and dynamic multi-GPU queue; five new conditions only',
        feasibility='Independent optical operator/parity/gradient fixtures; actual first8 TRAIN one update each; no heldout read',
        recovery='Unchanged local best/latest and full-state10percent aggregate COS barriers; original pinned inputs reused',
        controls='Five completed full native controls authenticated by original summary/manifest; no retraining',
        analysis='All ten VALID-selected TEST states replayed only after completion for nine repair/harm partitions; no reselection',
        comparator_dependency='Preserve native baseline selected states and its existing pinned recovery until paired analysis completes',
        scope='No automatic L3/L4-T; known Fock formulas and partial parity dephasing; no resource equality or hardness claim')
def review_gate(*,action):
    row=read(ROOT/REVIEW)
    if row.get('verdict')!='PASS' or row.get('source_sha256')!=identity(sources()) or row.get('contract_sha256')!=identity(contract()):
        raise ValueError('Changed-plugin review identity differs')
    if action not in ('run','worker','export'):raise ValueError('No repeated lifecycle matrix')
    return row
