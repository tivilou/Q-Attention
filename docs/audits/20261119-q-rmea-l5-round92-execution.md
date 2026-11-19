# Q-RMEA R92：严格回放差异定位完成

一次经精确 BOTH 审核、共享调度器准入的 CPU-only 诊断完成，退出0且进程静默。
只诊断 native_quantum，不训练、不新增权重、不读正式 valid/test、不调用 COS。

在同一旧972条开发 evaluation、同一 checkpoint、batch32/padding 下，ordinary/details
接口与 requires_grad true/false 四种组合的完整 972×40 logits 六组交叉比较全相同。
保存的34个 R91 旧 partial native logits 与四种当前结果也全部逐值相同。

对历史聚合报告的严格字段比较，四组均仅 CE 不同：历史
`1.2461352269835924`，当前 `1.246135219134421`，delta
`-7.849171312912517e-09`。混淆矩阵、Macro、Accuracy 等其他字段均一致。
这定位了本次复现的差异字段，不证明硬件、后端或其他某个因素是根因。
原 R91 的失败现场未保存实际比较值，因此本轮不能追认那个失败点的精确数值。

旧172个 partial 文件前后哈希不变；旧失败任务、严格门和缺失完成标记均保留。
本轮完成的是差异诊断，不是通过历史逐位一致门，不是完整 Case Study 验收。
API/冻结标志在当前检查中没有差异；无需继续围绕这些标志重放原任务。

精确 gpt-6.1-sol / medium 审核 PASS_WITH_RISK、must-fix=0。独立 observer 验证
源码、启动合同、输出闭包、NPZ头和哈希，未再次重算模型指标，边界保持。
初次父进程夹具遗漏了 worker 预建的空输出目录，已修正夹具并重验；没有修改科学源码。
八个零容差 comparator 夹具、真实子进程失败/超时夹具、四角色逆向/重放夹具通过。
主分支、师弟分支、旧限定L4均未变；L5 NOT_ACCEPTED。

- [紧凑结果](../evidence/rmea/20261119-l5-round92/diagnostic-result.json)
