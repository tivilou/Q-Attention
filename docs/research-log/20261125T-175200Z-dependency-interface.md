# 依存图输入资格：现有证据未过门

本轮完成上一检查点冻结的代码与来源审计，结论是 `NONE_CURRENT_GRAPH_INPUT_QUALIFICATION`。没有选图模型，没有读新任务行、重建依存分析或启动训练。这里否决的是当前路线的输入资格，不是所有图方法或量子方法。

已有完整 TRAIN 检查覆盖 58,465 条，源 SHA 为 `f1051d57…`，与参考清单的 **source** SHA 一致。记录的 metadata 字段只有 docid、id、obj_type、source_format、subj_type；没有 head/deprel。检查程序只汇总 metadata 内的键，未枚举所有顶层键，该汇总不足以断言原始数据没有依存图。参考清单另有基线转换后 output SHA，须与 canonical source SHA 区分。本轮仅复用该汇总，未再次扫描数据。

保留的历史 TACRED 转换代码确实只白名单保留若干字段，而且 `_compact_metadata` 只接受标量或 None：数组即使被列入白名单也会被丢弃。head/deprel 既不在白名单，列表也不被接受。历史 canonical loader/writer 只显式传 tokens、subject、object、label、metadata。不过，现有 canary 只有 12/6/6 条合成输入，不是完整 canonical 数据的生成回执；本地源码 SHA 可核，**它与当前 canonical 数据生产版本的绑定仍未核实**。

固定 native/extractor/inspection 源码的三个 SHA 与原回执相符。当前缓存按原词顺序平均 wordpiece，禁止截断，输出 h、roles、offsets、targets；批处理和原生 head 没有图张量接口。“原词特征可对齐”与“依存边已存在且可追溯”是两个独立条件。[逐项证据](../evidence/rmea/20261126-dependency-interface/interface-ledger.json) 和 [源码哈希](../evidence/rmea/20261126-dependency-interface/source-pins.json) 保留了各自范围。

历史 canary 提交 b0935d50 的一次 git 查询触发了 partial-clone 的远程取对象，返回 `upload-pack: not our ref b0935d50789df1dda21f08190ee389c22e85c5f5`，退出 128。未重复获取；该失败不产生新的 producer 证明。后续本地 git 读取设置 GIT_NO_LAZY_FETCH=1。此次包含这一次远程源码对象查询。

因此不猜 head 索引、不装 parser、不重算图、不将旧 entity-transition/ordered-memory/coherent-pool 加一个语法 mask 后重开。对这些已关闭配置的正负结果、gen98 十三门 STOP、原限定 L4 全部保持。新 L3/L4-T 仍未接受，主目标 ACTIVE。

下一步冻结一个范围很小的代数前置检查：同一词态产生 attention 和 value 时，秩一投影后读值是否退化为与输入无关的常数，弱值形式在原 EPS 归一化下是否有极点或等价已知矩表达。再与已关闭的 sequential back-action、shared-query covariance、shared-utility 以及已审 operator-valued 读出逐项排重。当前没有选这个方法，也没有以已知恒等式或经典可模拟性判定新 T/收益；检查决定是否值得形成独立、可区分的任务假设。[下一步及停止门](../evidence/rmea/20261126-dependency-interface/next-direction.json)。

机械核验通过，含固定代码 SHA、AST 字段、现有 TRAIN/source 清单身份和 literal exit 0。它不是外部科研审核、数据全字段普查或独立任务确认。0 新任务行、feature/checkpoint 读取、head/circuit forward、updates、GPU 申请、server writes、COS、model API。
