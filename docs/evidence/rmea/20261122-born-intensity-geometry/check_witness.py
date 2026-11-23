"""Small deterministic analytic witness; no model fitting or data access."""
from pathlib import Path
import cmath,json,math,time
r=Path(__file__).resolve().parent
inner=lambda a,b:sum(x.conjugate()*y for x,y in zip(a,b))
phi=[complex(1/math.sqrt(8))]*8
u=[1/math.sqrt(2),1j/math.sqrt(2)]+[0j]*6
lam=1/8;eps=1e-6
def w(t):return [cmath.exp(1j*t)/math.sqrt(8)]+phi[1:]
def power(t):return abs(inner(w(t),u))**2
def energy(t):return power(t)+lam*sum(abs(x-y)**2 for x,y in zip(w(t),phi))
errors=dict(norm=0.,mentions=0.,power_identity=0.,energy_identity=0.)
for i in range(257):
 t=2*math.pi*i/256;v=w(t)
 errors['norm']=max(errors['norm'],abs(inner(v,v)-1))
 errors['mentions']=max(errors['mentions'],max(abs(abs(x)**2-1/8) for x in v))
 errors['power_identity']=max(errors['power_identity'],abs(power(t)-(1-math.sin(t))/8))
 errors['energy_identity']=max(errors['energy_identity'],abs(energy(t)-energy(0)+math.sin(t)/8-lam*(1-math.cos(t))/4))
assert max(errors.values())<1e-14
h=1e-5;derivative=(energy(h)-energy(-h))/(2*h)
assert abs(derivative+1/8)<1e-10
assert abs(energy(math.pi/2)-energy(0)+3/32)<1e-15
native_alpha=(1/8+eps)/(1+1/8+9*eps)
moved_alpha=(1/8+eps)/(1+9*eps)
assert moved_alpha>native_alpha
out=dict(status='PASS',observed_at=time.time(),scope='Explicit full-span d8 state-space witness only; no data, trained model, novelty or utility claim',phase_points=257,max_errors=errors,energy_derivative_at0=derivative,energy_delta_pi_over2=energy(math.pi/2)-energy(0),native_context_power=power(0),moved_context_power=power(math.pi/2),native_mention_alpha=native_alpha,moved_mention_alpha=moved_alpha,optimizer_updates=0,cloud_requests=0,exit_status=0)
(r/'witness-verification.json').write_bytes((json.dumps(out,indent=2)+'\n').encode());print(json.dumps(out))
