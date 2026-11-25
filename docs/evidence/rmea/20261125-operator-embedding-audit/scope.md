# 算符值核嵌入源审计范围

精确源为2605.25146v1，PDF9f789d5cf2ed631e86ed12682597a3c6ea9694060d79698b32b6d0828b321949。只做原文/数学适用性检查，不执行任务数据、训练或测量方案优化。

检查三件具体内容：

1. 固定输入的native Born/token归一化能否精确写成token POVM的条件嵌入；保持带符号utility配对。该表示若不改变函数，不算插件、新T或frame重开。
2. 三qubit/40个独立binary relation effects是否能满足全8维混合态tomography完备性；维数结论不能当分类失败、原生纯态不可分或数据瓶颈结论。
3. 原文p21 Theorem4.25 对任意方向S的界，是否满足尺度一致性。p45 proof未给Z_ij≤1条件；用一个完全确定的Pauli/Dirac-kernel/r=1/ρ=diag(.999,.001)/S=-10σz/t=10精确有限例核对。只针对该版本该式，不评价整篇论文或其他定理。

采用原生circuit源SHA9f4bfc57；40effect/10key的固定随机角度仅是合法端口。utility为确定性sin((r+1)(t+1))。不触碰真实h或训练checkpoint。独立NumPy全矩阵RY/RZ+CNOT重建端口并重算，不把单一脚本自报PASS当完整证据。
