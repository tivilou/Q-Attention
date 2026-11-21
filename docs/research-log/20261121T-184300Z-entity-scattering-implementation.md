# 实体端点相干散射：实现与三卡真实缓存检查通过

## 结论与流程

[已测事实] 3060三卡节点恢复连接。共享调度器先执行CPU单元检查，再一次申请三张卡，分别检查complex、real_magnitude和linear_reference。三路各读取受保护缓存中固定的train前32条，逐条前向与反向，合计96次；没有优化器更新、验证/测试集读取或新checkpoint。三路阶段约7.025174秒完成；两次调度尝试均退出0且quiescent，网页显示completed、96/96、shared_stale=false。

关系分类输入仍是冻结文本表示、四维实体位置特征、token量子态与40个关系查询。先按原linear_coherent规则得到正的归一化q，再以tanh(原token关系utility)作为实在位势，构成H=√q(G−I)√q+diag(e)。固定能量0、阻尼0.1、双端引线强度1、log读出epsilon=1e−12，读出log(|R_subject,object|²+epsilon)+原bias；端点是输入两个实体跨度各自的首个有效词，缺失或重叠直接报错。所有参数和state_dict结构仍为原来的81,028个参数，不加入经典残差。

q在这里是上游路由分布，**不是新读出的最终池化注意力**。新头没有伪装成原weighted-utility接口，已有训练、trace和历史实验不受改写。正式训练尚待适配。实数对照仅把最终Gram hopping替换为|G|，同样保留q、e和所有上游梯度；它不是消除所有量子相位的全流程消融。

## 从上一方向到当前实现

[机制解释] 上一轮留下了有符号的实体端点散射预测，不再追求失败的“所有关系分数都应正向增加”的方差方向。当前用低秩求解实现该前向，并用相同q/e的实数hopping检验最后一层回路相位的定向容量。相干Green函数、Woodbury、Bargmann回路和手性方向性都是已知工具；本轮不把组合本身或数值正确性称为新颖理论。

## 检查结果与证据

源码 `06f42b28c342a4e93f28a2f66f8dd2381bf1aa7b`。独立A6API `gpt-6.2-astra` high审查最终为both作用域PASS、must_fix为空；此前审查指出的单元依赖、缓存身份和调度入口问题已在执行前修复，旧审查留在私有证据目录。

- 8维与直接求逆的振幅最大差：8.7770836714417531e-17；独立NumPy求逆最大差：8.8861199474166826e-17。
- 对准备态、q原始分数、在位势的完整梯度最大差：1.0658141036401503e-14；相位规范变换误差：9.7144514654701197e-17。
- 三token电路fixture交换固定输入下的读出端点：T_so−T_os=-0.34639106857542301，公式预测=-0.34639106857542307。共轭态重新计算q后方向符号翻转，差值=0.27814521206102472；不声称两者幅值相等。相同输入的实数hopping保持互易。
- 真实训练缓存三路均通过所有81,028个参数的有限梯度检查，输入manifest、checkpoint、训练缓存hash、单元身份和worker seal全部绑定并验证。训练fixture CE仅用于数值检查，不作为方法比较或选参依据。

合成fixture是已明确输入的电路容量见证；本次没有训练新模型，也没有完成Case Study或sample-trace.v2。真实样本只用于前后向兼容性，不编造文本解释或单例预测故事。受保护的原checkpoint没有变更；只发布代码与简短报告，无新COS副本，也没有向Windows下载模型、缓存或数据。

## 下一步

冻结一次从头训练的三路完整数据单种子协议，复用已验收的三卡runner，适配该读出和匹配对照，并验证相应best/latest与10% COS续训闭环后再启动。仍需检验真正任务收益、实质近邻差异和适当独立实证。当前L5 NOT_ACCEPTED，历史限定L4不变；不把本次正确实现直接升级为效果证据。

## 证据

- [结果](../../reports/q_rmea_entity_scattering/20261122-readiness/summary.json)、[单元检查](../../reports/q_rmea_entity_scattering/20261122-readiness/unit-validation.json)、[输入身份](../../reports/q_rmea_entity_scattering/20261122-readiness/inputs.json)、[审计](../../reports/q_rmea_entity_scattering/20261122-readiness/audit.json)。
- [精确源码审查](../evidence/rmea/20261122-entity-scattering/implementation-review.json)、[冻结实现规则](../evidence/rmea/20261122-entity-scattering/implementation-design.json)、[理论与边界](../evidence/rmea/20261122-entity-scattering/theory.md)。
