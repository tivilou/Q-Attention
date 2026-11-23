# Conditional source-level phase controllability with exactly fixed values

Status: a proposed theorem about the ORIGINAL input-feature interface. No claim
that its rank/nondegeneracy assumptions hold at the checkpoint, no execution,
natural-text reachability, independent task advantage, novelty or L5 acceptance.

The exact source has value(h)=Vh+c in R64 and preparation angles
alpha(h)=pi*tanh(Wh+R*roles+b) in R12. Roles are fixed. Let P be the Euclidean
orthogonal projector (P=P^T=P^2) onto
ker V and A=WP. If rank A=12, any sufficiently small angle displacement delta
is realized while preserving ALL values by

dh=P W^T (W P W^T)^-1 da,
da=atanh((alpha+delta)/pi)-atanh(alpha/pi).

Then Vdh=0 and Wdh=da exactly, hence the native angles are alpha+delta,
all token values/utilities remain exactly the same, and roles/mask stay fixed.
This is the minimum-Euclidean-norm value-preserving dh for the prescribed da;
||dh||<=||da||/sigma_min(A). It can be costly near tanh saturation or small
sigma_min. All statements are real-arithmetic; a numerical rank is not an
exact rank certificate.

Restrict delta to the last layer's three RZ coordinates (flat indices7,9,11)
at one or more valid tokens. The remaining nine angles are fixed. Last-layer
single-wire RZ rotations commute across wires with all other wires' rotations.
Writing C for the FINAL directed-CNOT ring, the changed output state is
psi'=C D(delta) C^-1 psi, where D is a product of diagonal RZ rotations.
Since C is a computational-basis permutation, C D C^-1 is diagonal. Therefore
all computational-basis state populations are unchanged EXACTLY.

For the frozen SAME-parameter dephased source branch, probabilities
sum_k |phi_rk|²|psi_tk|², Born+EPS normalizers, attention and all logits are
unchanged along this feature curve. This excludes only that fixed comparison,
NOT all separately trained dephased/classical/full-encoder models: another
model can encode the moved input features differently.

The coherent branch need not be constant. For the phase generator
G_j=C Z_j C^-1 at one token, a=<phi_r|psi_t>, its squared-overlap derivative is
dp_r/d(delta_j)=Im(conj(a)*<phi_r|G_j|psi_t>).
The corresponding SOURCE logit derivative, with N_r=sum_valid(p_rt+EPS), is

dz_r/d(delta_tj)=(u_rt-pooled_r)/N_r * dp_rt/d(delta_tj).

Here pooled_r=z_r-bias_r. Nonzero overlap response alone is insufficient: a
utilities equal across all valid tokens for relation r erase its task-visible
response. For a single changed token with dp_rt!=0, the exact cancellation
condition is u_rt=pooled_r.
Collect these derivatives into a row J_r over selected token-phase variables.
If J_r!=0 then the coherent source varies on this exact value/population-
preserving feature family, while the fixed dephased comparison remains constant.

Stronger conditional scalar protection: if J_0!=0 and J_r is not a scalar
multiple of J_0, choose v in ker J_0 with J_r v!=0. Smoothness (EPS>0) and the
implicit-function theorem give a local phase curve delta(s) with delta(0)=0,
delta'(0)=v and z_0(delta(s))=z_0(0) EXACTLY. The positive logit changes to first
order. Composing with the atanh/value-null feature realization preserves all
values and populations as above. This is a LOCAL existence statement with a
nondegeneracy assumption, not a globally defined algorithm, finite-error repair
or guarantee to preserve other positive logits/CE/Macro. If J_0=0, this proof
does not apply; do not infer an exactly protected curve from zero first order.

What might be useful: this closes a specific source-correspondence gap between
arbitrary state-port phases and the native preparation/value interface. It could
support a focused representation-level intervention if assumptions are checked.
What is NOT established: actual frozen-encoder/text reachability, any effect on
independent benchmark utility, a distinction against every classical counterpart,
or novel mathematics. Nullspace control, incoherent phase orbits and the
implicit-function theorem are known tools. Their application alone may remain
tool-level. Independent gpt-6.2-astra/high review judged the conditional argument
valid after the two wording clarifications above, with substantive novelty NOT
ESTABLISHED. It advised against measuring checkpoint rank/Jacobian assumptions
now: such a probe would instantiate known tools without resolving L5 novelty.
Retain this as a source-correspondence lemma; no GPU probe or L5 promotion.
