"""Conditional generic AdamW transfer counterexample; not a Q-RMEA task experiment."""
import json
from fractions import Fraction as F
from pathlib import Path
P=Path(__file__).resolve().parent
eps=F(1,10**8);wd=F(1,100);lr=F(1,1000);b1=F(9,10);b2=F(999,1000)
x=1/(2*wd);m=F(0);v=F(0);g=-eps;rows=[]
# L(x)=-eps*x, fullbatch deterministic, default bias-corrected AdamW formula.
for t in range(1,21):
 m=b1*m+(1-b1)*g;v=b2*v+(1-b2)*g*g
 mh=m/(1-b1**t);vh=v/(1-b2**t)
 assert mh==g and vh==eps*eps
 xnew=(1-lr*wd)*x-lr*mh/(eps+eps)
 assert xnew==x==50 and abs(g)<1
 rows.append(dict(step=t,x=str(x),corrected_first_moment=str(mh),corrected_sqrt_second_moment=str(eps)))
assert 0<x<1/wd and g!=0
without_epsilon=(1-lr*wd)*x-lr*g/eps;assert without_epsilon!=x
out=dict(schema='qattn.adamw-epsilon-transfer-counterexample.v1',status='PASS_EXACT_CONDITIONAL_FORMULA',
 objective='Generic scalar convex loss L(x)=-optimizer_epsilon*x, not the native complete40 CE or actual dataset.',
 optimizer=dict(lr=str(lr),weight_decay=str(wd),beta1=str(b1),beta2=str(b2),epsilon=str(eps),bias_correction=True,initial_moment_states_zero=True),
 exact_fixed_point=str(x),box_radius=str(1/wd),gradient=str(g),
 argument='At every positive step, bias-corrected m=-eps and sqrt(v)=eps. Adaptive direction=-1/2, while decay*50=1/2; updates cancel. Thus all iterates remain50. This is interior to[-100,100] with nonzero gradient and is not a KKT point of min L over that box; its minimizer is100.',
 original_algorithm_difference='2404.04454v1 Algorithm1 omits this positiveepsilon/bias-corrected implementation. The example does not refute its stated theorem; it blocks importing that theorem unchanged to another optimizer formula.',
 without_denominator_epsilon_first_x=str(without_epsilon),verified_steps=rows,
 limits=['Conditional exact-real formula, not nativefloat execution or verification of the installed PyTorch build.',
 'Generic convex counterexample, not a Q-RMEA empirical failure or full40 CE result.',
 'Does not disprove approximate/more general box constraints, every optimizer theorem, or produce a novel quantum mechanism.'],
 new_native_forwards=0,new_data_reads=0,new_training=0,l5_accepted=False)
(P/'epsilon-audit.json').write_bytes((json.dumps(out,ensure_ascii=False,indent=2)+'\n').encode())
print(json.dumps(dict(status=out['status'],fixed_x=50,box_radius=100,not_KKT_of_imported_box=True,native_forward=False,l5_accepted=False)))
