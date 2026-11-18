"""Check exact design closure; this does not test the proposed model or theorem."""
import hashlib,json
from pathlib import Path
def main():
 p=Path(__file__).resolve().parent;root=p.parents[3]
 def read(n):return json.loads((p/n).read_bytes())
 c=read('candidate.json');a=read('execution-and-review.json');h=read('historical-lessons.json');r=read('reading-receipts.json')
 assert c['id']=='P1-ROLE-PROTECTED-NUISANCE-FILTER' and c['status']=='EXPLORATORY_NOT_ACCEPTED'
 assert c['empirical_evidence']=='NOT_RUN' and c['L5']=='NOT_ACCEPTED' and not c['inherits_original_L4']
 assert c['mechanism']['dimension']==8 and c['mechanism']['epsilon']==1e-6 and c['mechanism']['new_learned_parameters']==0
 assert len(h['history'])==6 and all((root/n).is_file() for item in h['history'] for n in item['source'].split(';'))
 assert len(r['papers'])==2 and {x['paper_id'] for x in r['papers']}=={'arxiv:2409.06456v2','arxiv:2610.13261v1'}
 assert all(x['reading']=='read_for_scope' and x['scope'] and x['limits'] and len(x['sha256'])==64 for x in r['papers'])
 assert a['status']=='PASS_DESIGN_WITH_RISK' and a['review_scope']=='mechanism'
 assert a['mechanism_review']['freeze_sha256']==a['freeze_sha256'] and not a['mechanism_review']['must_fix']
 for name,digest in a['frozen_files'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
 for name,digest in a['source_dependencies'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
 assert len(a['consultations'])==2 and all(x['model']=='gpt-6.1-sol' and x['reasoning']=='medium' and x['automatic_retries']==0 for x in a['consultations'])
 assert not a['model_execution'] and a['scientific_tasks_started']==0 and not a['new_model_code'] and not a['training']
 assert a['COS_GETs']==a['COS_writes']==0 and not a['collaborator_server_access'] and not a['new_capacity']
 assert a['L5']=='NOT_ACCEPTED' and a['candidate_status']=='EXPLORATORY_NOT_ACCEPTED'
 print(json.dumps(dict(status='PASS_DESIGN_DOSSIER_CLOSURE',candidate=c['id'],empirical_evidence='NOT_RUN',L5='NOT_ACCEPTED',theorem_or_utility_test=False)))
if __name__=='__main__':main()
