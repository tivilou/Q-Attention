# Input-conditioned interlayer dephasing: one fixed exploratory screen

Question: can input-conditioned suppression of interference *between* the two native circuit layers improve the complete classifier, rather than just increasing individual class scores? No task root cause or novelty is established. This is distinct from the closed fixed output-mixture training curriculum and closed keep/redraw profile; neither is reopened.

Native encoded angles theta=pi*tanh(prepare([h,roles])), split U=U2 U1. Let psi1=U1|0>, phi_r be unchanged native query state, and lambda_t=sigmoid(w dot (theta_t/pi)+b), w=0,b=0 initially. The shared word gate has13 new trainable scalars, sees the same12 projected inputs as the router, and never reads labels or values. Channel E_t(rho)=(1-lambda_t)rho+lambda_t diag(rho) is CPTP for each input. Scores are (1-lambda)|phi†U2 psi1|²+lambda sum_i |(U2†phi)_i|² |psi1_i|². Native EPS1e-6/token normalization, signed value/decoder/bias are unchanged.

Four new arms, all from the same native shared initialization:
1. quantum_midgate: above candidate,81041 parameters.
2. quantum_outgate: the same learned gate placed after U2; second endpoint is sum |phi_i|² |(U2 psi1)_i|²,81041. This is a location control, not a rescued curriculum target.
3. quantum_midfixed: exactly the same intermediate full dephasing with lambda=1,81028. Tests whether learned retention adds utility over full removal.
4. classical_gate: native real12 normalized squared overlap blended by a13-scalar sigmoid(tanh(key)) gate with its diagonal squared-overlap score,81041. Equal input fields and extra parameter count, not a claim of identical effective capacity.

Reuse stable screenv6/unchanged host wrapper and completed native/real12/softmax12/RBF12 compact CM/metrics from the identical-protocol heralded-token-prior study. Verify protocol/source/split/optimizer/order identities before using static rows. No old checkpoint/logit/CLI loader or baseline refit.

TRAIN4192 fit/4000 grouped assessment from existing8192 split; fixedseed13,512updates/b128,FP32/complex64/TF32off, AdamW lr.001/wd.01/betas.9,.999/eps1e-8/clip1. Two allocated GPUs; unchanged internal queue. All four fits finish before one final-update/all4000/all40 evaluation. Allseven comparisons frozen now: candidate Macro must strictly exceed allthree new controls and allfour pinned references, including stronger RBF12. CE descriptive; all CM and signed outcomes retained. Passing merely supports designing a full protocol, not L3/T. Fail closes this exact gate/location/initialization/512 profile, with no width/gate/seed/budget/metric/control rescue; not a whole-channel impossibility.

Scope of checks: changed layer/channel formula against independent NumPy density matrices, native composition parity, gate endpoints, masks/singleton/padding, gradient equivalence, and one cloned actual8-row update per arm. One exact Astra-high code/math review; no core/resume/COS lifecycle matrix. Short restartable screen has durable inputs/local best-latest, zero COS snapshots. Only compact results/code go to Windows/Git; no task arrays, models or datasets downloaded.

Paper0807.4902v1 concerns dissipative transport with a sink, not classification. Channel placement, CP algebra and the H-D-H witness are known tools, not a theory-innovation claim. Q-CFRA already had a learned channel. L3/L4-T remain NOT_ACCEPTED; historical scoped L4 remains unchanged.
