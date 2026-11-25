# 初始导数的几何分解：用于对照设计，不主张新理论

沿用有效词的单位纯态 psi_t、单位query phi、实数utility u_t 和原 EPS>0。只在有效词上求和，bias不进入 z0。记 a_t=<psi_t|phi>、p_t=|a_t|²、D=sum_t(p_t+EPS)、alpha_t=(p_t+EPS)/D、z0=sum alpha_t u_t、delta_t=u_t-z0。此前固定 C=A-z0B，初始分数导数 g=2||Cphi||²/D。

取 P=I-|phi><phi| 与 v_t=a_t P psi_t。由 sum delta_t(p_t+EPS)=0，

    Cphi=sum delta_t(rho_t+EPS I)phi=sum delta_t v_t.

因此，K_st=<v_s|v_t> 是半正定的词间Gram矩阵，

    g = (2/D) delta^T K delta
      = g_self + g_cross,
    g_self=(2/D) sum delta_t² p_t(1-p_t).

g_cross 是不同词之间的项，可为正或负。g_self只依赖原p/utility与D；这里“self/词间Gram对角”不是把量子density去相干的原生dephased模型，二者不可混称。

设 sigma²=sum alpha_t delta_t²（原scalar utility指数倾斜在零点的导数），Gamma=sum p_t(1-p_t)/(p_t+EPS)。令 W 的列为 v_t/sqrt(p_t+EPS)，x_t=delta_t sqrt(p_t+EPS)，则 Cphi=Wx、||x||²=D sigma²、||W||_F²=Gamma。由标准矩阵Cauchy–Schwarz/Frobenius界，

    0 <= g <= 2 Gamma sigma².

在上界非零时 eta=g/(2 Gamma sigma²)∈[0,1]；上界为零时eta未定义，原理上g=0，不能通过加另一个EPS来制造该比值。所有p为0或1时Gamma=0，即使utility有方差，这个更新的初始导数仍为0。该界是已知线性代数工具，不计新T。

旧原生plus/plus与plus/minus构造的p、alpha、u完全相同，sigma²=1、Gamma=1/(1+2EPS)。两者g_self=1/(1+2EPS)，但g_cross分别为负/正的同一量，所以g分别为0、2/(1+2EPS)，eta分别为0、1。第二例达到该界；此差别仍由real8算符控制精确复现，不支持量子必要性。

每个词的独立整体相位会使a_t与Ppsi_t获得相反相位，故v_t及上述量均不变。有效词同步排列不改变求和；无效词不参与D、A、B、Gram或方差。用概率求Gamma只是解析等式；数值实现亦计算||v_t||²，避免把舍入误差当作负几何能量。

这使未来的标签对齐诊断更具体：g_gold-g_rival须与原sigma²、仅依赖p/u的g_self、Gamma等量比较，并报告g_cross/eta。它们是诊断特征，不是新的可训练模型或公平训练对照的替代。还需保护原正确例、全部40类、强real-vector/operator与Gaussian控制、固定teacher障碍以及独立任务确认。Knyazev未读原文不能由本推导代替；source novelty、任务效用与L4-T仍未成立。
