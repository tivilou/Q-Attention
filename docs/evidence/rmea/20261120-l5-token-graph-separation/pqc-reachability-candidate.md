# Candidate: realize the stable witness with the actual two-layer circuit

Status: independently reviewed with risks and verified through the unchanged actual model head on the selected 2080 Ti. This is an implementation-reachability extension of the existing separation, not task-utility evidence or a novelty claim. See `pqc-reachability-review.json` and `pqc-reachability-numerical-check.json`.

The source circuit applies RY then RZ on each of three wires, followed by CNOTs 0→1,1→2,2→0, twice. With wire0 the most significant bit, the CNOT ring maps

    L(a,b,c)=(b xor c, a xor b, a xor b xor c),
    L²(a,b,c)=(c, a xor c, b).

Set all second-layer angles to zero. First-layer RY basis preparations followed by L² can therefore reach each computational basis state: output bits (x,y,z) require input bits (x xor y,z,x). Rotating only wire1 in the first layer instead maps a one-qubit superposition into span(e0,e1). Thus RY=pi/3 on wire1, with all other angles zero, gives the key v=sqrt(3)/2 e0+1/2 e1 exactly.

The implemented angle map pi*tanh(raw) excludes pi at finite raw. For basis preparations replace every required pi by pi-eta with eta=delta/2 and delta=1e-5. The product-unitary telescoping inequality and ||RY(pi-eta)-RY(pi)||<=eta/2 bound each key-state error by at most 3eta/2=3delta/4. Every raw angle is finite, atanh(angle/pi). All states remain normalized, so the already derived robust full-rank theorem applies to the nine implemented keys, including the small extra graph edges caused by approximation.

The same physical witness query is also reachable at finite effect angles. With tau=1/4, its unnormalized components are

    u0=cos(tau)*sqrt(3)/4 + i*sin(tau)*sqrt(21)/8,
    u1=cos(tau)/4 - i*sin(tau)/(8*sqrt(7)).

These equal M^(-1)w in the full-rank construction. Both components have positive real part, with opposite nonzero imaginary parts. After normalization set theta=2 atan2(|u1|,|u0|) and z=arg(u1)-arg(u0). Then 0<theta<pi and -pi<z<0. First-layer wire1 RY(theta),RZ(z), followed by L², prepares this query up to a global phase; both effect raw angles atanh(theta/pi),atanh(z/pi) are finite. Other angles can be zero.

A shared preparation layer can realize all nine key-angle targets under an explicit feature assumption. Let X contain the nine 1028-dimensional frozen feature/role rows augmented by a bias column. If rank(X)=9, the finite 9x12 raw-angle target Y admits the shared linear solution W^T=X^+Y. This is conditional on the supplied feature rows, not a measured property of our six samples or a claim about arbitrary encoder inputs.

Actual-head check: the nine-key fixture uses synthetic independent feature rows, one shared preparation layer, the implemented pi*tanh maps and the source circuit/forward path. All nine unit tests passed in 4.62647 seconds. Maximum key error was 4.330127e-6, phase-aligned query error 2.725007e-16, minimum frame eigenvalue 0.9999926400, and maximum preparation raw magnitude 7.02197446. Coherent probability 0.5032767384 exceeded native and dephased mixed-query ceilings 0.4999977087 and 0.4924363552, respectively; the simultaneous margin was 0.00327902967.

The derivation received PASS_WITH_RISK from the selected Astra 6.2 medium route. Its integer-bit dtype concern was resolved explicitly before execution. The synthetic feature rows do not establish rank nine for real encoder features. Near-saturated basis angles may have poor optimization conditioning; representability does not establish learnability, prevalence or generalization. Fixed-key comparator scope remains unchanged. Task utility still awaits the separately frozen collaborator experiment; L5 remains unaccepted.
