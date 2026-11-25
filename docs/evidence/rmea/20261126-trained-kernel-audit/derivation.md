# Original KTA objective versus native signed token routing

This is an assumption/gradient correspondence using known calculus. It is not a new theory result, an implemented training intervention or an experiment.

The visually checked source Eq27 uses `TA(K)=<K,T>_F/(||K||_F ||T||_F)` with binary `T=yy^T`; the imbalanced extension rescales each label by its within-class count. It does **not** insert centering matrices or a40-class one-hot target. If T is fixed, differentiation gives

`dTA = <dK, T/(||K||_F||T||_F) - TA*K/||K||_F^2>_F`.

Source pp15–16 AppendixA identifies the unnormalized rescaled numerator with the squared Hilbert-Schmidt distance between mean class density matrices. The denominator changes the optimization objective; do not silently replace normalized TA by polarity. Neither calculation is a new invention of this project.

For native Q-RMEA, restrict to valid tokens. Put `s_ct=p_ct+EPS`, `A_c=sum_t s_ct`, `alpha_ct=s_ct/A_c`, `f_c=sum_t alpha_ct*u_ct`, `z_c=f_c+b_c`. Differentiating at fixed utilities gives

`df_c/dp_ct=(u_ct-f_c)/A_c`.

This is already the source `rmea.py` sensitivity field. With `m_ct=log(s_ct)` and `q=softmax(z)`, chain rule for CE gives

`dCE/dm_ct=(q_c-1[y=c])*alpha_ct*(u_ct-f_c)`.

When utilities also depend on the parameter, their derivative adds a separate `sum_t alpha_ct*du_ct` term; bias derivatives likewise remain. No detach semantics or invariance of training gradients is inferred. The token-label shortcut does not follow: gold sentence labels do not label each contextual token, and the source does not construct native relation effects or decoder values.

The source compares embeddings of labelled examples pairwise and then fits an SVM. Native CE couples all40 class logits and local signed utility contrasts. Therefore the source's alignment heuristic is an available known tool, not a guarantee for the implemented native risk or Macro-F1. This distinction does not refute all kernel regularizers. No hyperparameter, class reweighting or prior stopped profile is reopened.
