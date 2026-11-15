# Independent Q-RMEA theory pass

Binding: source `f36b85a50a29916fe52b78989593fb810dcda279ee62bda7dc5f49ad38a59087`; contract `fc4bd934fbbda3e0339a35292f8f68c48c6389f37bde635038f15e8cbff92abb`; scope `mechanism`.

**Verdict: BLOCK for advancement of these proposals.** Two conditional questions are identified below; neither is presently a substantive L5 contribution. Accepted scoped L4 remains unchanged. No primary candidate or other new round-two scientific files were read. No model, dataset, cache, remote, restoration, or training operation was performed.

## 1. Does permutation alignment extend to realizable measurement changes?

Write `rho_t=|psi_t><psi_t|`, `B_t=rho_t+epsilon I`. The executed weights are

`alpha_t=Tr(E B_t)/Tr(E sum_s B_s)`.

Allowing arbitrary mixed effects gives the optimistic feasible set

`C={beta: exists Q>=0, Tr(Q B_t)=beta_t for every valid t}`.

The actual rank-one depth-two circuit family is a subset. If `sum c_t B_t=0`, every feasible attention obeys `c^T beta=0`. Thus a violating shuffled vector is impossible even for a mixed effect. Feasibility in this relaxation does **not** certify circuit reachability or one globally shared effect across examples. These are known linear/semidefinite tools, not new measurement theory.

A six-token fixture directly connects the support question to the implemented CE endpoint. All first-layer angles and all RZ angles are zero. Set only second-layer wire-zero RY angles to

`theta/pi=[1/2,-1/2,1/4,-3/4,0,0]`.

The actual CNOT ring gives `psi_t=cos(theta_t/2)|000>+sin(theta_t/2)|011>`. These are finite, source-compatible preparation inputs: set the first hidden coordinate to `atanh(theta/pi)` and let preparation output coordinate six read it. The effect is `|000>`, obtainable with zero parameters. Its probabilities are

`p=[.5,.5,.8535533905932737,.1464466094067263,1,1]`.

Tokens 0–3 are context; tokens 4 and 5 are fixed subject/object anchors. The source's SHA256 ordering for synthetic ID `independent-support-2` is exactly `[0,2,1,3,4,5]`. Let true-class utilities be `[0,-1,1,0,0,0]`, its bias `log(39)`, and the other 39 class logits zero. A linear value head and decoder implement these utilities using a second hidden coordinate.

Since `rho_0+rho_1=rho_2+rho_3`, feasible weights satisfy `alpha_0+alpha_1-alpha_2-alpha_3=0`. The implemented shuffle violates this by **0.17677643013199168**, also a lower bound on its L1 distance to the relaxed feasible set. Nevertheless, the accepted-style CE damage is positive:

| Quantity | Synthetic value |
|---|---:|
| Normalizer | 4.000006 |
| Actual CE | 0.6499293148732039 |
| Shuffled CE | 0.7383175299391996 |
| CE damage | +0.0883882150659957 |

The swap fixes entities, padding, values, biases and total context mass. Negating the utilities reverses CE damage without changing the infeasibility. This was verified by elementary circuit reduction, numeric arithmetic and the actual hash rule, **not** by executing `RMEAHead` or using real data.

The counterexample refutes an implication from positive replay damage to an attainable changed measurement. It does not invalidate functional dependence under the defined replay. That replay deliberately intervenes on weights; off-family status alone is not a scientific defect.

**Possible contribution:** establish when measurement-realizable dependence differs materially from arbitrary weight-reassignment dependence, with a task consequence and an independent prediction. The packet contains no evidence of that difference. The nearest packet works already cover squared-overlap/density features and counterfactual attention (`1803.07128`, `2101.11020`, `N19-1357`, `D19-1002`); quantum probability-realization literature also needs an audit. This is a different question from the stopped Rayleigh certificate, but the new question is not automatically novel.

**Kill and next step:** first specify a meaningful, fairly matched attainable intervention, including global effect sharing and all changed invariants. If no such comparison exists, stop. Only after that premise is justified would a maximum-16-example descriptive check of existing detached arrays be useful. Infeasibility counts alone are not a success gate, and PSD feasibility is only a necessary condition. No restoration is justified by this fixture alone.

## 2. Can gates predict a meaningful context-redundancy robustness budget?

For one context token j, put `q_rj=p_rj+epsilon`, `Z_r=sum q_rt`, `a_r=z_r(0)` and `l_r=b_r+u_rj`. Duplicating its complete frozen feature and role tuple m additional times gives exactly

`z_r(m)=(Z_r a_r+m q_rj l_r)/(Z_r+m q_rj)`.

For true class y and rival k, its margin has the sign of `C0+C1 m+C2 m^2`, where

```text
C0 = Zy Zk (ay-ak)
C1 = Zy qk (ay-lk) + qy Zk (ly-ak)
C2 = qy qk (ly-lk)
```

All denominators are positive. Root finding must allow two-root/nonmonotone cases. Initial margin sensitivity is

`(qy/Zy)(ly-ay) - (qk/Zk)(lk-ak)`.

A low true-class gate alone does not establish robustness: the rival gate and both utility limits matter. Because epsilon is positive, every token eventually dominates each class logit under unlimited repetition; `z_r(m)->l_r`. Half the limiting shift occurs at `m=Z_r/q_rj <= T(1+epsilon)/epsilon`.

An arithmetic fixture with `(Z,q,a,l)_y=(2,.1,1,-1)` and `_k=(2,.2,0,1)` gives polynomial `4-.2m-.04m^2`. Its positive root is **7.807764064044152**; margins at 7 and 8 copies are **+.06971677559912848** and **-.015873015873015872**. This checks rational pooling algebra; these aggregate numbers were not certified as a jointly realizable circuit fixture.

The same formula holds for uniform, real-overlap, dephased and softmax per-token pooling with their respective q. Consequently the formula is a known aggregation property, not a quantum contribution. The bounded Born/kernel score and normalized quantum attention also have direct prior work in `1803.07128`, `2101.11020` and `2205.05625`. Normalized-pooling robustness literature needs a focused audit.

**Possible contribution:** make a nontrivial, independently validated prediction that the learned allocation responsible for useful alignment also supplies a task-relevant redundancy budget. The original Macro and permutation-CE aggregates do not measure it. The essential missing premise is an independently meaningful perturbation process. Copying hidden states does not equal inserting raw words: contextual RoBERTa features normally change, and even function-word states can carry entity information. Historical separately trained uniform and Q heads also cannot be replaced silently by a same-value uniformized replay.

**Kill and next step:** stop if the proposed result is only exact replay arithmetic, if label stability has no independent basis, or if existing general pooling theory already covers the whole claim. Specify the process and prediction before inspecting new outcomes. A tiny detached-array threshold check would then verify implementation only; substantive utility needs prospective independent evidence. This introduces no new architecture and reopens no stopped family.

The useful new result of this pass is the source-compatible, hash-specific counterexample in route 1. The missing scientific premise remains more consequential than additional algebra. Neither route warrants L5 promotion or a fresh experiment as currently specified.
