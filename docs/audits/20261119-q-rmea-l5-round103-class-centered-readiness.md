# R103：去共模适配器已实现，数值验证待准入

类别去共模方向只有一个候选家族。每个有效token先减去全部40类（含null类）的平均效用，
再在测量构造和最终读出两处使用中心化效用。保留native、full、centered_native、
centered_full和construction_only五种显式适配条件，不增加可训练张量或隐藏detach。
simplex是后续联合训练对照，未加入该适配器allowlist。

源码cb4ab502c8fe0255456bfe432efad8afacc594d4已在开发分支发布。
原head.py、train.py及R98冻结科学源码未改。八项合成验证覆盖原生/完整对照精确重放、
40类与padding边界、公共解码器平移、端到端梯度、有限差分、抽象有利/有害标签见证、
零效用/单token以及确定性重载。静态检查与独立隔离逆向/重放通过，数值检查执行数仍为0。

指定A6API gpt-6.1-sol、medium，单次精确BOTH代码/生命周期审查PASS_WITH_RISK，零must-fix，
响应resp_05c17169fccd6f47016aff00aaf11087d2ab4cb980727ad7fb。
冻结哈希c4cb4dc088a603621d22d3575cf4094db6affbc956d52109696ef5e9e0e24f6f，
执行合同哈希555a277b41f2ffb8477553d6d5e13b6bda3944af0dd2aadb92d24e998df54719。
审查明确记录canary_execution_eligible=false，因为共享队列尚未注册此源码和合同；
该字段保持原值，后续需单独完成真实准入，不能把静态审查写成运行通过。

残余限制包括：同为81028存储参数不等于有效容量相等；中心化读出去掉64维公共解码器方向；
合成精确重放只覆盖固定输入；抽象见证不证明原电路可实现或真实数据收益；
报告和完成标记不是联合事务，实际成功必须同时满足调度终态、独立验收与哈希。
没有真实数据训练准入、正式Case Study、正式交付、任务收益或新颖性验收，L5 NOT_ACCEPTED。

后续复用稳定的调度和训练接口，集中投入最小区分实验。R98已完成，无需为此重跑历史baseline。

[完整代码审查](../evidence/rmea/20261119-l5-round103/implementation-review-record.json)

## 后续实际进展：合成验证已完成，联合训练待审

去共模合成检查现已通过共享队列实际运行并独立验收：任务84b91a20d7c949bba342debe2b957ebb，
attempt a30b1577852943f1aca7d25463aa9901，一次，退出0且进程静默，八项检查全部PASS。
有限差分误差约1.60e-10；移除构造路径梯度会产生约0.01119的梯度差异。
这些结果验证固定合成输入上的实现性质，没有真实数据收益或独立确认含义。
首次准入SSH预检查中止发生在配置写入前；恢复时保留其他项目并发更新，未重复提交科学任务。
上文静态审查时的eligible=false和当时的未运行状态保留为历史记录。

六条件执行代码已实现，复用同一训练循环与无Torch父进程，以显式factory区分条件。
真实父进程/替代子程序的非零、超时、坏checkpoint及成功四项夹具通过；
合成前置凭据九项隔离检查通过。数值训练仍未运行，尚待精确BOTH审查和新合同准入。
固定原train前缀fit3124/eval972、共同初始化、四epoch及2352次总更新；
候选须胜过native/full/simplex/centered_native的Macro，并满足冻结CE门槛。
中心化移除64个有效解码方向，centered_native是必要对照；结果不继承原限定L4。

[实际合成验收](../evidence/rmea/20261119-l5-round103/canary-owner-acceptance.json)
[可执行联合训练设计](../evidence/rmea/20261119-l5-round103/executable-training-design.json)
