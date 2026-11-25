from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
r=Path(__file__).resolve().parent
parent=Path('D:/CodexHome/task-artifacts/qattn-native-phase-construction-20261126/next-direction.json')
def sha(b):return hashlib.sha256(b).hexdigest()
x=json.loads(parent.read_bytes());assert x['collection']['max_ordinary_requests']==4
metadata=Path(x['metadata_pin']['path']);assert sha(metadata.read_bytes())==x['metadata_pin']['sha256']
assert not (r/'scope.json').exists()
scope=dict(status='FROZEN_BEFORE_ORCID_SOURCE_REQUEST',recorded_at=datetime.now(timezone.utc).isoformat(),
 parent=str(parent),parent_sha256=sha(parent.read_bytes()),
 exact_title='Toward the Optimal Preconditioned Eigensolver: Locally Optimal Block Preconditioned Conjugate Gradient Method',
 DOI='10.1137/s1064827500366124',author='Andrew V. Knyazev',year=2001,
 requested_version='publisher_version or explicitly identified author manuscript; unknown remains unresolved',
 first_request=x['first_request'],metadata_pin=x['metadata_pin'],collection=x['collection'],
 allowed_fields=['public author name','public researcher-URL links','exact title/DOI paper links','verified fulltext'],
 raw_personal_metadata_publication=False,resources=x['resources'],goal='ACTIVE',newL3='NOT_ACCEPTED',newL4T='NOT_ACCEPTED')
(r/'scope.json').write_bytes((json.dumps(scope,ensure_ascii=False,indent=2)+'\n').encode())
print(json.dumps(dict(status=scope['status'],max_requests=4,max_PDF_transfers=1,task_rows=0)))
