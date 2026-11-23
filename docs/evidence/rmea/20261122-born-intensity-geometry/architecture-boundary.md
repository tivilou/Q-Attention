# State-port feasibility and the actual shallow circuit

Read from the pinned native `rmea.py`, SHA256
9f4bfc57a1a966c3383c1a2e82fe406ac3d115b653b062290fcf42d3c4683ee0.
This is a source-derived boundary, not a new numerical experiment.

The current three-qubit effect circuit has two layers, three wires and two
angles per wire: twelve real parameters per relation effect. The manifold of
pure eight-dimensional states modulo global phase has real dimension fourteen.
Consequently this smooth twelve-parameter family is not locally surjective
onto general pure states. A feasible vector returned by the Hilbert-space
retraction is not thereby reachable by the original circuit.

For a normalized circuit state phi(theta), let S be its real horizontal state
Jacobian, obtained by applying I-phi phi^H to each complex parameter derivative.
Let p=rank_R S, so p<=12. Let G(theta) collect the fixed mention intensities
and let r=rank_R DG<=p. On a branch where the relevant maps have constant rank,
the physically distinct first-order feasible dimension is p-r. The parameter
kernel dimension12-r also counts any redundant/gauge directions; it is not
automatically a count of useful state changes. Norm is already fixed by the
unitary circuit and contributes no independent parameter constraint.

Neither p=12 nor r=min(m,p) has been measured at the trained checkpoint. The
Hilbert-space count14-m therefore cannot be substituted for this circuit's
actual freedom, especially for the frozen mention-count8..13 domain. A later
circuit-specific diagnostic must measure its actual Jacobians and task-visible
feasible motion. It must retain all frozen cases and remain distinct from the
already reviewed state-port TRAIN32 probe; do not edit that probe's contract.

A more general state-preparation architecture would be a separately specified
method with its own fair controls. It is not an implementation certificate for
the existing shallow circuit. No new fit, circuit modification, measured rank,
utility or L5 acceptance follows from this dimension argument.
