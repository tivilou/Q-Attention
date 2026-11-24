"""Small known-frame illustration, not task data or a novelty/utility result."""
from pathlib import Path
import json,numpy as np
eps=1e-6;query=np.ones(2)/np.sqrt(2)
def evaluate(keys):
 q=np.array([abs(query@x)**2+eps for x in keys]);native=q/q.sum()
 c=sum(np.outer(x,x) for x in keys)+len(keys)*eps*np.eye(2)
 w,v=np.linalg.eigh(c);inverse=(v/np.sqrt(w))@v.T
 effects=[inverse@(np.outer(x,x)+eps*np.eye(2))@inverse for x in keys]
 frame=np.array([query@m@query for m in effects])
 assert np.max(abs(sum(effects)-np.eye(2)))<1e-12 and abs(frame.sum()-1)<1e-12
 return dict(native=native.tolist(),frame=frame.tolist(),povm_error=float(np.max(abs(sum(effects)-np.eye(2)))))
a=evaluate(np.eye(2));b=evaluate([np.array([1.,0]),np.array([0.,1]),np.array([1.,0])])
assert abs(b['native'][0]/b['native'][1]-1)<1e-12
assert b['frame'][0]/b['frame'][1]<.501
out=dict(EPS=eps,two_words=a,duplicate_first_word=b,
 known_property='Native fixed-key odds obey IIA; frame odds can change with another word. Orthogonal-mode multiplicity illustration, not semantic nuisance/novelty or quantum-exclusive separation. Real vectors reproduce it exactly.',
 native_state_port_embedding='Two orthogonal keys from last-layer first-wire RY +/-pi/2, all other angles0; query all0. Common finalCNOT preserves overlaps; raw angles finite +/-atanh(.5). This is a state-port fixture, not an encoder-image/text witness.',
 no_training=True,no_new_task_evidence=True,L3='NOT_ACCEPTED',L4_T='NOT_ACCEPTED')
r=Path(__file__).resolve().parent;(r/'frame-fixture.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
