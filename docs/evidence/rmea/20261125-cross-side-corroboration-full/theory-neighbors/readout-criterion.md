# 从印证强度到分类效用：含真实 EPS 的固定读出判据

对象仍是 plugin SHA6174cb5a 的当前五路全量研究。本页只分析已经执行的
`components` 和 affine value/decoder，不更改任何运行源码、门或预算。
这是固定输入、固定参数的代数解释，不是训练因果、自然文本定理或新 T 创新。

## 精确等式

只在有效 token 上讨论。对于 relation r，令

    a_ri = (b_ri + eps) / sum_j (b_rj + eps),
    c_ri = (b_ri * g_i + eps) / (b_ri + eps),
    u_ri = decoder_r dot value(h_i).

这里 a 是同一参数状态下关闭 gate 的 attention，不是另一个训练完成的
gaussian_base 模型。实际 gated attention 恰好为

    a'_ri = a_ri * c_ri / E_a[c_r].

令 ell_r = E_a[u_r] + bias_r，则

    ell'_r - ell_r = Cov_a(c_r, u_r) / E_a[c_r].

证明：将 E_a[c_r u_r] 分成 Cov_a(c_r,u_r)+E_a[c_r]E_a[u_r]，
再除以 E_a[c_r]；bias 在两者中相同而抵消。EPS 严格为正，因此分母为正。
mask 通过限制有效 token 处理；不可给 padding 分配概率。

这说明 gate 和正确 relation 的 decoded value 正相关才提高该 logit。
但提高正确 logit 本身仍不充分。对于 gold y 和任意 rival s，精确 margin 变化是

    delta_margin_ys = Cov_a_y(c_y,u_y)/E_a_y[c_y]
                    - Cov_a_s(c_s,u_s)/E_a_s[c_s].

各 relation 的 a、c 和 decoder 都可不同；即使 g 共享，EPS 也使 c 依赖 r。
因此不能只统计 gold attention 增大、g 与关键词一致，或 gold logit 增大，
就说完整分类变好。也不能把分别训练的 Q 与 Gaussian 参数塞进这个同参数等式。

## 可计算的固定参数保护界

取 C_r=max_i c_ri（仅有效 token），令 d_ri=c_ri/C_r，
lambda_r=1-E_a[d_r]。于是 0<d<=1 且 0<=lambda<1。
若 lambda>0，定义 q_i=a_i(1-d_i)/lambda，则

    a_r = (1-lambda_r) a'_r + lambda_r q_r,
    TV(a'_r,a_r) <= lambda_r,
    |delta ell_r| <= lambda_r * (max_i u_ri - min_i u_ri).

lambda=0 时 attention 相同。用 C 而不是直接假设 g<=1，也覆盖浮点 gate
稍超 1 的情况。若原预测 p 对每个 rival s 的 margin 大于对应两项上界之和，
该固定参数 gate 干预不改变预测。这个充分界可能宽松，不满足它不是可修复证明。

## 用途和边界

- 全量结束后若需要解释正负结果，可在同一 selected checkpoint 上做 gate-on/off
  回放，以所有预定单位的 margin/covariance 对照解释，而不是只展示成功样例。
- 目前不为此额外启动 GPU 任务，不读取部分完成条件的科学效果，也不新增消融。
- `check_readout_criterion.py` 只用随机代数数组检查恒等式/边界及反例；不代表
  所有指定 base、gate、value 可以在同一个 native 编码器参数下共同实现。
- 这些是已知重加权/协方差工具的实现对应，不构成非平凡新理论或 L4-T 接受。
