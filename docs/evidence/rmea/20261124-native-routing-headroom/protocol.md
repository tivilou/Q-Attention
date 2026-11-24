# 已完成模型的路由可达性：一次完整TRAIN诊断，前瞻协议

状态：仅冻结问题/数学/总体/预算，尚未实现或执行。本协议不改变三批已关闭的完整效用门，也不是另一个训练方案。

## 为什么测这件事

新共同酉查询完整TEST40类Macro .506314 < 上轮条件Q .521912 < 最强native/real12 .546448；几何守恒与更大的Q-D差距未带来进一步效用。继续选择查询变体前，先区分固定value/decoder的限制、固定key几何的限制，以及这些宽松上界没有解释的优化/泛化问题。不把其中任一限制预先认作已测原因。

只复用已完成原生五路的VALID-selected quantum和real12：其全部TRAIN重评估已存在，Macro分别.875590/.896750。精确既有身份来自../full-train-error-contrast-20261124/design.json，不换checkpoint：

- source owner/attempts/9a28320fd4174c8f95544c9ba96eb1ae/source；实际受管节点根/hy-tmp/Q-Attention/。
- completed owner/attempts/9a28320fd4174c8f95544c9ba96eb1ae/output/20261123T221843Z_seed13/fair-reference-seed13-20261123T221843Z/discovery。
- quantum-selected.pt SHA8ec179f4605a901ea1aeeeaefaacc20d78a0b202f236a2d41c1a37b06220cf3c，epoch17，stateSHAc8f29a9d6694fc1fc4c21a65ecb2cb4e237b820def31eb6d5fda92e6248ed7a7。
- real12-selected.pt SHAdb8be37b3e2ca2ecf7a122ff341671fe7a1ac1fd3b03a704191e3ad68544abf5，epoch17，stateSHA19781ca0c4e63e29d6c20845e4bd5d152f8274736f8fba79b439a1affbd1701d。
- 同一TRAIN缓存58465条/all40，h SHA b69b1bfb97f20932b26dcea8b13a788a12ec98ec43a68de1801960ad5c526635。其余缓存/labels/source pins复用上述design，禁止重建特征/重新训练。

## 固定数学

对每条样本、有效原词t、关系r，u_rt=decoder_r·value(h_t)，bias为b_r。模型logit是sum_t alpha_rt u_rt+b_r。

任意各关系独立单纯形权重的端点为L_r=min_t u_rt+b_r、U_r=max_t u_rt+b_r。它放宽了所有实际router约束，只是固定value/decoder的最宽路由范围。

原生Q固定词ket k_t的更窄放宽：允许每个关系自由选择任意单位复query q，保持同一key/value/bias和EPS=1e-6，score=|q*k_t|²+EPS。记

    B = sum_t k_t k_t* + EPS*T*I,
    A_r = sum_t u_rt k_t k_t* + EPS*(sum_t u_rt)*I.

该关系logit的上下端点是Hermitian generalized Rayleigh quotient q*A_r q / (q*Bq)的最小/最大广义特征值，再加b_r。B严格正定。原生real12用原F.normalize key及任意单位实query，矩阵为12维实对称，同一公式。EPS的I项与实际单位query约束对应，不得漏掉、放进value或忽略padding。

这些是标准凸包/广义Rayleigh工具，不申请T，不是可部署新模型。自由query集合大于实际固定/条件查询族；可达不代表同一共享模型能同时在所有输入做到，也不表明能泛化。无限制词权重与固定key自由query两层须分开报告，不能用score秩代替最终logit可达性。

## 一次执行及输出

- 完整原TRAIN58465条/all40，原存储顺序，全体不筛选标签、错误、长度或类别；labels仅计算事先定义的诊断分层。零VALID/TEST读取、零优化/选模/新seed、零COS。现有完整TEST结果不用于逐例选择。
- quantum/real12各一遍完整TRAIN前向，复用稳定模型加载/缓存/padding和批次读取，FP64数值计算端点。可在同一分配两GPU队列各跑一个模型；最多一次执行、各900秒（总1800秒），内存预算沿用既有全TRAIN重评估。不得为数值或科学负结果改总体/阈值/模型重试；机械错误单独标记invalid，先修实现。
- 重算两个原模型TRAIN40类混淆矩阵/CE，核对既有同权重完整TRAIN重评估；不将在线epoch train_ce当selected-state指标。
- 对gold y定义g_simplex=U_y-max_{r!=y}L_r、g_query用广义特征值端点同样定义。严格正差意味着自由query（或更宽词权重）存在可使gold唯一最高的选择；严格负差意味着该放宽亦不够。零边界保留，不偷换argmax tie。
- 浮点观察使用tol=1e-5*(1+max_{r,t}|u_rt|+max_r|b_r|)，|g|<=tol为unresolved。报告最小B特征值、condition、Hermitian/广义特征对残差和原logit在端点内的残差。没有经严格舍入包络时仅称numerical bound/diagnosis，不冒充形式不可行证书。
- 全体及原模型错例中分别报告：连simplex放宽也数值不可达；simplex可达但free-query不可达；free-query可达；未决。全部40类分项计数、gold支持、两个方向错误和剩余错误均保留，不用某一有利类代替总体。
- 不计算或宣传oracle部署收益/新的Macro上限，不输出逐样本文本/状态到Windows。紧凑报告Git，GPU上的私有逐例缓存可重建，不写COS；原生已固定输入/检查点依赖保留至本消费者结束。新共同酉full3和旧实体条件full5的负结果checkpoint均不是本诊断依赖。

## 最小实现验证与决策

只对新端点计算作影响范围内的检查：小型实/复/重复key/EPS例子、已知query读出落在端点内、固定少量真实行用独立NumPy重算；一次精确新增代码/数学审查。复用训练核心准入，不重复训练更新、暂停续训、COS恢复或生命周期矩阵。

本测量没有需要赢过阈值的科学晋级门：它是定位下一问题，全部正负结果有效。若两层自由度对多数错误都留有空间，就关闭“表达空间不足已被证明”的解释，回到可学习性/泛化假设；若出现明显更窄的固定value或key障碍，则据实际规模选择后续问题，但不把标签aware放宽推成可实现收益。不设事后比例门、挑类、改tol或寻找更有利checkpoint。任何后继插件另行写可证伪假设、控制与停止规则，仍须独立L3和实质T；本诊断本身不晋级L4-T。
