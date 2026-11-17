"""Exact-real first AdamW step for two globally equivalent legal source heads."""
import hashlib,json
from fractions import Fraction as F
from pathlib import Path
P=Path(__file__).resolve().parent
import argparse
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-dir',type=Path,required=True)
S=parser.parse_args().source_dir.resolve()
def sh(p):return hashlib.sha256(p.read_bytes()).hexdigest()
source={n:(S/n).read_text(encoding='utf8') for n in ['rmea.py','frozen_feature_head.py','rmea_protocol.py','rmea_confirmation_protocol.py','rmea_confirmation_run.py']}
assert 'torch.optim.AdamW(head.parameters(),lr=cfg[\'lr\'],weight_decay=cfg[\'weight_decay\'])' in source['rmea_confirmation_run.py']
assert "clip_grad_norm_(head.parameters(),cfg['clip_norm']" in source['rmea_confirmation_run.py']
assert "lr=.001,weight_decay=.01,clip_norm=1." in source['rmea_protocol.py']
assert "cfg=legacy.config(profile)" in source['rmea_confirmation_protocol.py']
lr=F(1,1000);wd=F(1,100);opt_eps=F(1,10**8);g=F(-39,40);go=F(1,40);sumg2=F(39,40)
decay=1-lr*wd
def update(x,grad):return decay*x-lr*grad/(abs(grad)+opt_eps)
rows=[]
for a,b in [(F(1,20),F(1,20)),(F(1,10),F(1,40))]:
    assert a*b==F(1,400)
    # Mean CE on one example with all40initial logits0, y0; two identical valid token features.
    # Only h[0]=1, zero value biases, prepare/effects0 => alpha uniform, selector gradients0.
    # Active gradients: decoder[:,0]=g_vector*b; value.W[0,0]=g*a;
    # value.bias[0]=g*a; output.bias[:]=g_vector. All others zero.
    norm2=2*g*g*a*a+(1+b*b)*sumg2;assert norm2<1
    ap=update(a,g*b);bp=update(b,g*a);value_bias=update(F(0),g*a)
    other_decoder=update(F(0),go*b)
    bias0=update(-a*b,g);bias_other=update(F(0),go)
    z0=ap*(bp+value_bias)+bias0;zo=other_decoder*(bp+value_bias)+bias_other
    rows.append(dict(a=str(a),b=str(b),initial_B00=str(a*b),initial_logits='all40zero on fixture',
     gradient_norm_squared=str(norm2),gradient_clipping_active=False,
     post_B00=str(ap*bp),post_target_logit=str(z0),post_each_other_logit=str(zo),post_target_vs_other_margin=str(z0-zo)))
assert rows[0]['post_B00']!=rows[1]['post_B00']
margin_delta=F(rows[1]['post_target_vs_other_margin'])-F(rows[0]['post_target_vs_other_margin']);assert margin_delta>0
eps=F(1,10**6);T=40
max_alpha=(1+eps)/(1+T*eps);min_alpha=eps/(T-1+T*eps)
out=dict(schema='qattn.source-optimizer-boundary.v1',status='PASS_EXACT_REAL_SOURCE_FORMULA',
 source_hashes={n:sh(S/n) for n in source},
 scope='Engineered legal head parameters/H/R; not actual random initialization, encoder image, trained checkpoint or floating native execution.',
 first_step_assumptions=dict(optimizer='AdamW',lr=str(lr),weight_decay=str(wd),optimizer_epsilon=str(opt_eps),
  betas='Default first-step bias correction cancels beta1/beta2; epsilon default1e-8 is assumed explicitly, not read from pinned installed PyTorch.',amsgrad=False,
  gradients='Exact complete40 CE with uniform predicted probability1/40; all active coordinates accounted for; norm<1 so configured clipping is inactive.'),
 construction='Both selectors have all prepare/effect parameters0; both heads share B00=1/400, value biases0, class0bias=-1/400 and other parameters0. They realize the same function on every admitted H/R/mask before updating, not merely the same point prediction. On the fixture two valid entity tokens have identical h[0]=1, other hidden coordinates0.',
 factorizations=rows,post_margin_difference=str(margin_delta),post_margin_difference_float_illustration=float(margin_delta),
 prediction='All39non-target logits are equal within each updated head. Different target-versus-other margins imply different softmax probabilities and complete40 CE after the first source-formula AdamW step.',
 bounded_attention=dict(epsilon=str(eps),valid_tokens=T,max_any_weight=str(max_alpha),min_any_weight=str(min_alpha),
  min_l1_distance_to_one_hot=str(2*(1-max_alpha)),
  scope='Universal exact q in[0,1] bound; a finite epsilon blocks exact one-hot limits for T>1. Tiny gap is not a practical disadvantage or superiority claim.'),
 interpretation=['R14 equality of readout function classes does not imply the same optimizer trajectory.',
 'This one-step non-invariance does not disprove an eventual implicit-bias theorem, all possible norm characterizations or the original paper.',
 'No claim that AdamW decoupled weight decay is exactly optimizing a nuclear norm penalty.',
 'Known parameterization-sensitive optimization tools; no novelty or L5 claim.'],new_native_forwards=0,new_data_reads=0,new_training=0,l5_accepted=False)
(P/'optimizer-audit.json').write_bytes((json.dumps(out,ensure_ascii=False,indent=2)+'\n').encode())
print(json.dumps(dict(status=out['status'],full40_first_step_probability_changes=True,clipping_active=False,margin_difference_illustration=float(margin_delta),native_forward_executed=False,l5_accepted=False)))
