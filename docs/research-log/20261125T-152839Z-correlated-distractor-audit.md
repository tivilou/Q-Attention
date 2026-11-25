# 相关干扰项阈值原文审计：停止直接转移，原量子目标继续

已核对并范围阅读 [2610.34682v2](https://arxiv.org/abs/2610.34682v2)，PDF SHA `17d20d0f96e4c21ef05c4d95b420570850ac81f520fe9866f7fc5a0052355d50`，42页。其指数前平均机制没有给原生 Q-RMEA 带来相同保证。本轮没有选新插件，也没有启动训练；新 L3/L4-T 仍 NOT_ACCEPTED，原限定 L4、gen98 十三门及全部正负历史保持。

## 原文证据范围

官方 exact-ID Atom、PDF首页标题/Heehyun Park/版本栏一致；首页和第6页已渲染核看。全部42页可提取文本，但实际范围阅读仅 pp1–10、24–27、31、35–36：主模型、Theorems 1–5/Corollary 6 陈述、实验局限，Appendix D 的均值/Jensen、误差/value 证明，非Gaussian假设及复制规则。未完整审核 GREM、adaptive-temperature 或非Gaussian定理证明；未检查作者代码、附加网页和全部实验协议。解析覆盖不冒充全文理论认证。

源结果比较已给定干扰分组中 `mean(exp(βX))` 和 `exp(β mean(X))`；严格阈值建立在指定随机分层模型和单例目标上（pp2–7）。其一般化仍有尾部/独立增量假设，并明确排除有界饱和分布和任意 learned score；混合相关项/多个有用匹配时，平均也可能损失证据（pp8–10,31,35）。这不是原文错误，而是本任务的转移条件不成立。

## 原生边界一：Born 混合平均没有同一指数凸性增益

原生 `rmea.py` SHA `9f4bfc57a1a966c3383c1a2e82fe406ac3d115b653b062290fcf42d3c4683ee0`：每个 relation 独立二元效应，`p_rt=tr(E_r ρ_t)`，`w_rt=p_rt+ε`（仅valid token），`z_r=Σ_t w_rt u_rt/Σ_t w_rt+b_r`，ε=10^-6。不是40结果共同POVM，也不是对 score 取 exp。

对任意给定组 a，定义 `ρ̄_a=mean_t ρ_t`。则

`tr(E_r ρ̄_a)+ε = mean_t(p_rt+ε)`。

因此混合密度的 group score 与对 token Born 权重作 inverse-count 修正完全相等；不存在源 Theorem 5 中两种顺序的严格 Jensen 间隙。Dephased 同样线性。这个结论只比较这两个明确操作，并不否定所有分组、所有量子 attention 或其他读出。

原生电路8组固定随机角度、每组40query、组大小2/3/1，320端口核验：Q密度线性最大误差3.33e-16，D线性1.11e-16。没有访问任务行或执行训练head。

## 原生边界二：均值 value 会丢掉带符号配对

如再以组内均值 `ū_ra` 取代各 token utility，两种读出分母相等，但分子差为

`N_count,r − N_mean,r = Σ_a Cov_a(p_rt,u_rt)`。

ε部分抵消，Cov的符号没有固定方向；直接保留此Cov则回到已知带value矩阵/统计量记账，并非新理论。随机端口恒等式误差7.77e-16。具体给定一组 p=(.9,.1)、utility=(1,-1)，另有 p*=.5/utility*=0 的单例，count读出为+.3999992000016，均值value为0；只调换utility符号即为−.3999992000016。该合成例说明一般排序缺失，不声称实际编码器能共同产生这些p/u或解释gen98错误。

源 Proposition 14 的质量到value误差界要求明确目标value及统一范数界；这里尚无相应“正确token及其40类margin”前提。得到更多某token质量也不自动保证40类 CE/Macro 改善。

## 原生边界三：相干合并也没有单向保证

取原生三qubit、两层电路，除最后一层wire0的RY外角度全零。两个key角度±π/2均位于原生 `π tanh` 的内部范围，得到 `( |000> ± |011> )/√2`。其混合密度的query概率为1/2；归一化相干和为 |000>。

对最后RY为0的query，相干分组概率变为1；给定target key=query，target质量由count的.666666444445下降到.5。对RY为2π/3的query，相干概率变为1/4，target质量上升到.799999520001。因此不能从该源推出相干合并总能更好。这里只调用原生电路端口，没有证明任务h的可达性或提出已关闭 coherent-amplitude 路线的新试验。

## 原生边界四：固定 ε 与无限上下文检索极限不同

有N个其他valid token时，p*≤1且每个干扰项w≥ε，所以单目标质量

`α* ≤ (1+ε)/(1+ε+Nε)`。

固定ε、N→∞时上界趋零。这是有界权重的初等界；既不是新T，也不是对有限ReTACRED序列或所有EPS schedule的失败证明。本轮没有读取attention轨迹，因此不把ε或长度当成gen98失败原因。

## 决策及下一步

本轮机械核验 PASS，直接源机制转移 STOP；未形成Q/D/G可区分收益预测。上述线性性/Cov/凸性/有界性均为已知工具，不计作非平凡新T。不做聚类、temperature、长度gate、EPS或seed救援，不重开gen98–100；0任务数据/训练headforward/updates/GPU/COS，原生源未修改。

随后仅做两条预限定官方Atom top6检索，围绕conditional mean和operator-valued/value配对；两条均HTTP200，全部12条身份和失败覆盖已存档。下一阅读优先级为 `2605.25146v1`《Kernel Embedding for Operator-Valued Measures and Its Application to Quantum Tomography》，目前仅元数据/摘要：核对其所谓Quantum Covariance Embedding与当前C/A统计量是否同属已有对象、观测/误差前提能否转移。它不是已选插件，未读PDF，不把tomography定理改称任务或量子优势。`2403.10859v1` Neural-Kernel Conditional Mean Embeddings仅保留作经典近邻；不替换原量子目标。若没有真正改变配对读出且区分Q/D/G的预测，继续记录未找到候选。

本轮证据为 `scope.md`、`source-receipt.json`、`reading-record.json`、`check_native.py`、`native-checks.json`、`execution.json`、`verification.json` 和 `next-source-discovery-public.json`。原PDF/渲染留本地供原文复查，不发布原文副本。

[原生端口与精确公式核验](../evidence/rmea/20261125-correlated-distractor-audit/verification.json)。
