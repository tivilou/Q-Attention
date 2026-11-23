# Two-effect normalization-order feasibility, not a new mixture theorem

Keep the original native preparation, decoder, values and bias frozen. For two
unit pure circuit effects, p_lrt=|<phi_lr|psi_t>|^2. Candidate mixes equal raw
Born scores then adds the original EPS per valid token and normalizes. Control
normalizes each component with the same EPS then averages. Both have two
original twelve-angle effects (960rawparameters), identical initialization and
optimization. They contain native exactly at coincident effects.

Let Z_l=sum_valid(p_l+EPS), N_l=sum_valid u(p_l+EPS), z_l=N_l/Z_l.
Candidate excluding bias=(N_1+N_2)/(Z_1+Z_2); control=(z_1+z_2)/2.
Their difference is (Z_1-Z_2)(z_1-z_2)/(2(Z_1+Z_2)). This familiar mixture
identity predicts no label advantage and establishes neither novelty nor L5.
A classical mixed-state/multihead formulation is a direct nearest-method risk.

Moment compression is exact for the same unit circuit effects:
C_i=sum_valid(psi_t psi_t^*+EPS I),
B_ir=sum_valid u_irt(psi_t psi_t^*+EPS I).
Then Z_l=phi_l^* C_i phi_l and N_l=phi_l^* B_ir phi_l. This is a known Rayleigh
reduction used solely to accelerate the fixed-feature utility experiment.
No density decoder or extra scalar feature is learned. Derivatives propagate
through both original circuit effects, including their normalizers. No detach
surrogate or backbone gradient is introduced.

Before the frozen run, check actual eight TRAIN examples with varying lengths
and padding: coincident-effects native equality, direct token-versus-moment
logits AND gradients for both orders, the displayed identity, and one genuine
Adam step for each arm on discarded fixture parameters. Then start both actual
arms afresh at the shared prospective initialization. Reuse stable admission,
evaluator, resident lifecycle and source identities; no lifecycle matrix.

First8192TRAIN, all19584VALID, noTEST. Twoepochs/batch32/512updates perarm;
Adam(.001,.9,.999,1e-8,wd0), no clipping/scheduler/VALID selection. Fixed NumPy
seeds0 for direction and13 for shared epoch permutations. Float64/complex128.
All four frozen CE/Macro inequalities must pass. Any scientific miss closes this
profile with no rank, init, duration, regularization, cohort, threshold or seed
rescue. Numerical defect is separately invalid, not a negative scientific fact.
A positive screen would justify separately designed full-model testing only;
it is not independent confirmation, originalfresh5L4 inheritance or L5.
Cheap restartable heads use no periodicCOS. Windows receives compact evidence
only. Rebuildable moment caches stay private on GPU; active finalweights, if
needed after a pass, use the accepted minimal storage adapter, not all moments.
