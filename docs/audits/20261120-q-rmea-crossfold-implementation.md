# Cross-fold：独立实现与合成检验准备

新增 cross_fold/self_fold 模块，复用已有 measured_attention 和原生参数；没有改动已冻结中心化实验的12个源码文件。两个条件均以有效token顺序交替分组，保留原生组质量和全部token读出，改变的是供给测量的组。一个token时回退原生；两组非空时都计算两个effect。三个token时cross-fold数学上仍为原生，self-fold可能变化。

八项固定合成检验覆盖原生等价、mask与40类、供给依赖、组内常数平移、局部响应公式、Jacobian符号结构、独立与相关噪声见证、梯度与重载。测试源码已写入，尚未执行。独立噪声为零的断言不意味着真实NLP误差独立；相关噪声的正负反例同样要求通过。

一次指定gpt-6.1-sol/medium联合源码/生命周期审查PASS_WITH_RISK，零must-fix，允许准入后的有限合成检验。审查指出Jacobian逐项解析核对、完整参数形状/训练标志核对和真实数值资格仍有限；完整value trace和reader未完成。所有风险保存在审查记录中。

实际父进程的stdlib替身子进程测试覆盖非零退出、超时、错误报告、成功输出，四项通过。成功仅是替身报告发布，不是合成科学检验通过。数值工作必须经共享调度，现有中心化六条件实验在首个符合原合同的空闲准入窗口优先。没有修改调度器或新建训练任务，也未运行全量实验。

后续顺序：保留现有冻结实验；在源码/合同精确绑定且获准后完成一次合成canary，再决定共同初始化任务筛选。尚无cross-fold任务收益、新颖性、独立确认或L5验收。原限定L4及所有停止路线保留。

[实现设计](../evidence/rmea/20261120-l5-crossfold/plugin-plan.json)
[合成合同](../evidence/rmea/20261120-l5-crossfold/canary-design.json)
[审查记录与全部限制](../evidence/rmea/20261120-l5-crossfold/implementation-review-record.json)
