# Gram motion is not rotation-mediated credit

This note records a mathematical boundary supplied by independent A6API gpt-6.2-astra high advice and checked algebraically in the project. It is not a novelty or L5 claim. The short replay failed its joint gate and does not establish a causal training mechanism.

Let valid unit rows form X in C^(n×8), C=X†X, and R=exp(-iτC/c), where c=||XX†-I||op. At positive differentiable c, its value is determined by the spectrum of C and the fixed valid length n. The exact token transport differs from XR only by a common phase. Forward equality does not identify equivalent stop-gradient interventions.

For a fixed attainable real tangent space T and the real Frobenius metric, define A(v)=X†v+v†X. Its adjoint is A*(H)=2P_T(XH). The unique minimum-norm lift of an attainable δC is A*(AA*)†δC; the remainder lies in ker(A). For unit rows, P_T removes the real radial component; the actual circuit image needs its further tangent projection. This is standard pseudoinverse/orthogonal decomposition, and depends on the chosen tangent space and metric. AdamW in circuit parameters does not generally induce the Frobenius metric.

The Gram lift is not a measurement of the loss derivative mediated by R. Consider three real unit rows embedded in eight columns:

    x+ = (cos a, sin a), x- = (cos a, -sin a), x3 = (1,0), phi=e1.

Then C=diag(1+2cos²a,2sin²a,0,…). At a=π/6, the valid-token graph has eigenvalues 1.5,-.5,-1, so c=1.5 is positive with a unique maximizing branch. Write v=∂X/∂a and H=diag(-1,1,0,…). Row-tangent projection gives P_T(XH)=sin(2a)v. Hence v is orthogonal to ker(A) and is entirely the minimum-norm Gram lift. The same orthogonality holds for a restricted attainable tangent containing v; no assertion that every actual project input realizes this example is needed.

Nevertheless R(C)phi is only a phase times phi throughout this path. Its partial contribution through R to each Born weight is zero. The actual third attention weight is

    alpha3 = (1+epsilon)/(1+2cos²a+3epsilon).

Its total derivative at π/6 is positive:

    2(1+epsilon)sin(2a)/(1+2cos²a+3epsilon)² = 0.2771277412321715

for epsilon=1e-6. A central difference with spacing1e-6 agrees within2.3e-11. A utility selecting row3 observes a direct readout change while the rotation contributes zero. Thus 100% Gram-lift motion can coexist with zero rotation-mediated derivative.

The example rules out that attribution shortcut; it does not prove the learned model has no useful context dynamics. A prospective stronger test must preserve direct gradients and break the pairing of loss cotangents with context Jacobians, with a fixed scale/alignment-matching rule. Its claim would initially be a local directional advantage, not total-gain mediation or independent generalization.
