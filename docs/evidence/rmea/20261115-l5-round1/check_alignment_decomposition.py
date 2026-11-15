"""Exact CE/Bregman decomposition plus a deterministic null-alignment witness."""
import hashlib,itertools,json,math
from pathlib import Path
P=Path(__file__).resolve().parent
def lse(z):
    m=max(z);return m+math.log(sum(math.exp(v-m) for v in z))
def prob(z):
    a=lse(z);return [math.exp(v-a) for v in z]
def ce(z,y):return lse(z)-z[y]
def split(z,zp,y):
    p=prob(z);q=prob(zp);delta=[b-a for a,b in zip(z,zp)]
    linear=sum((a-(i==y))*d for i,(a,d) in enumerate(zip(p,delta)))
    kl=sum(a*math.log(a/b) for a,b in zip(p,q))
    return ce(zp,y)-ce(z,y),linear,kl
def main():
    alpha=[.4,.2,.4];u=[-1,0,1];z=[0,0]
    permutation=[1,2,0];shuffled=[alpha[i] for i in permutation]
    s=sum(a*b for a,b in zip(shuffled,u));zp=[0,s]
    values=[split(z,zp,y) for y in [0,1]]
    loss=sum(v[0] for v in values)/2;linear=sum(v[1] for v in values)/2;kl=sum(v[2] for v in values)/2
    assert loss>0 and abs(linear)<1e-15 and abs(loss-kl)<1e-14
    # Different label values cannot change the centered permutation Jensen term.
    logits=[[0,sum(alpha[i]*v for i,v in zip(p,u))] for p in itertools.permutations(range(3))]
    mean=[sum(x[i] for x in logits)/len(logits) for i in [0,1]]
    j=[sum(ce(x,y) for x in logits)/len(logits)-ce(mean,y) for y in [0,1]]
    assert j[0]>0 and abs(j[0]-j[1])<1e-14
    result=dict(status='PASS',role='Known convex-loss identity and counterexample; not new task evidence or a novelty result',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),identity='CE_y(zprime)-CE_y(z)=(softmax(z)-e_y).(zprime-z)+KL(softmax(z)||softmax(zprime))',witness=dict(alpha=alpha,utilities=u,actual_logits=z,permutation=permutation,perturbed_logits=zp,labels=[0,1],per_label_splits=values,mean_ce_damage=loss,mean_directional_alignment=linear,mean_kl=kl,mean_prediction_accuracy_before=.5,mean_prediction_accuracy_after=.5,mean_actual_prediction_equals_mean_permutation_prediction=True),all_permutations=dict(count=len(logits),mean_logits=mean,jensen_penalty_per_label=j),scope='Fixed label-blind permutation on a balanced two-case algebraic cohort. Positive CE damage need not show a beneficial first-order label alignment. Does not rerate the existing narrow L4 functional-dependence claim or diagnose its actual decomposition.')
    (P/'alignment-decomposition-audit.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
    print(json.dumps({k:v for k,v in result.items() if k not in ['witness','all_permutations']}));print(json.dumps(dict(mean_ce_damage=loss,mean_directional_alignment=linear,mean_kl=kl,jensen_penalty=j)))
if __name__=='__main__':main()
