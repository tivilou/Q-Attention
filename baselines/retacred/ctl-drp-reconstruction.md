# CTL-DRP：解释性复现假设与验收矩阵

## 这一步交付什么

这是**设计记录，不是可运行 adapter 或准入证书**。当前 evidence grade 为
`not_verifiable`，后续实现首先标为 `paper_interpretive`。所有“暂定实现”仍为
`proposal_not_implemented`，写在这里不会变成作者事实或运行默认值。

依据是已锁定的 [CTL-DRP 论文](https://arxiv.org/abs/2304.04935v1)、
[官方 SimCSE](https://github.com/princeton-nlp/SimCSE/tree/49fa580a853752ede55b8c76d9debf748e214d3f)
及引用的 [RECENT 源码](https://github.com/Saintfe/RECENT/tree/4742c9e80517acbd1466a6fb3d5bcb4678e56c47)。
具体逐项证据在相邻三份 lock 中；机器记录同时绑定其 SHA256。
本文未新增来源采集或权重下载。

**保持不变：** ready cohort 为五个；两条 CTL 行仍 blocked；作者精确 checkpoint/H、
paper scorer 和神经/运行证据尚未确认。独立审查、敏感性和正式准入各自有门槛。
完成文档或 tiny 测试不会自动成为 source-faithful、paper-faithful 或全量运行资格。

## 逐项假设

| ID / 部分 | 已恢复事实 | 暂定实现：不是作者事实 | 所需验证 / 敏感性 | 门槛 |
|---|---|---|---|---|
| A01 · 初始化身份 | 论文原字面为 sup-smi-roberta-large / unsup-smi-roberta-large；官方 SimCSE 候选已锁定，但与作者初始化的精确对应未确认。 | 分别采用锁中两个官方不可变 revision 作为解释性候选，不替换成通用 RoBERTa/MNLI；将作者身份对应始终单独标记为未确认。 | sup/unsup 作为独立行；保存实际权重 SHA、加载键差异和参数差异。候选下载或成功加载本身不证明作者使用它。 | G07, G09 |
| A02 · H 的生成规则 | 论文要求类型约束；RECENT 的纯聚合片段按有序类型对收集关系。精确 CTL H 未确认；现有 primitive 只接受声明 train，尚未绑定物理 split。 | 使用已资格审查的物理训练 split 拟合 observed-pair 支持表，冻结 SHA；NA 无条件加入。此为显式适配，不是已恢复作者 H。 | 改变 dev/test gold 后表及预测输入保持不变；subject/object 交换不隐式同义；不采用 TACRED 常量或 RECENT 二分类/强制正类分支。 | G02, G05, G06 |
| A03 · 未知类型对与缺失 gold | 现有查询无 gold 参数；未知类型对报诊断，known NA-only 合法；gold 覆盖审计与预测路径分离。 | 先做完整覆盖审计；未知 pair 或 excluded gold 均停止该资格检查并报告索引/计数。不给 all-label、NA-only 或 gold 重插的隐式修补。 | heldout 诊断只决定是否具备评估资格，不调 H 或超参、不跳样本；如真实覆盖不完整，该方案不进入完整比较，另立假设后再评审。 | G05, G06, G09 |
| A04 · 实体标记与角色/类型 | 论文指定四个新 role-specific 符号，保留实体名，实体后括号含角色/类型；不是旧 punctuation marker。具体字节、空格和初始化未恢复。 | 暂定 u1/u2 为 subject、u3/u4 为 object 的四个专用注册 token；原 span 边界和角色保留。具体文本模板在 token fixture 前独立冻结。 | 逐字 token-id/offset fixture 覆盖实体顺序互换、Unicode、空格和多词 span；非法/重叠 span 显式诊断而非自动重排。 | G02, G03, G07 |
| A05 · 关系描述与提示拼接 | 附录 46 条描述经显式 alias 覆盖 canonical40；NA 为负叙述。句子与候选描述共同编码；精确拼接边界未恢复。 | 主候选采用 tokenizer 的 sentence-pair 编码：marked sentence 为 A、prompt+description 为 B；描述由已锁 native TeX 提取，不自行改写。 | 对单序列拼接与 pair 边界做有界敏感性比较，先冻结模板与分隔字节；不把 pair 边界称为作者事实；六 extras 不加输出类。 | G02, G03, G08 |
| A06 · 编码与读出 | 每个候选联合编码，取 final-layer first subtoken，使用一套共享 affine scalar head；sup/unsup downstream 同为标签监督。 | 直接读取 last_hidden_state[:,0,:]；不采用 SimCSE pooler、L2 normalization、mean pooling 或缓存句向量后单独点积。 | 固定小张量逐候选联合编码与批处理比较；梯度同时进入联合编码器及共享 head；错误 pooler/独立描述路径须被反例捕获。 | G02, G04, G07 |
| A07 · Eq2 与损失归约 | 单句 allowed candidates+NA 内 softmax/gold NLL；共享 bias 消去；NA-only gold loss/gradient 为零。 | 采用稳定 logsumexp；先算每句 NLL，再按真实句数等权平均。候选数不同不额外加权、不混入其他句的分母。 | neural loss/gradient 对现有 scalar reference；1/2/多候选、极端 logits、NA-only、bias 零梯度及 batch permutation 检验。 | G04 |
| A08 · batch512 的单位 | batch512、LR1e-5、epochs10 被论文明确给出；512 是句数还是 expanded pairs 未恢复。 | 主解释为 effective batch512 句；physical micro-batch 只控制内存，不重定义 Eq2 或句权重。pair512 作为单独解释性敏感性分支。 | 对比两个 batch 单位的 optimizer-step 数、句暴露量、候选对量和指标；若 pair512 模式未定义句组边界，先补规范而非拆断 Eq2。 | G02, G04, G08, G11 |
| A09 · micro-batch 与候选 chunk | 作者 accumulation/chunk 策略未恢复；显存优化不得改变每句归一化候选集合。 | 先实现 FP32、无 AMP 的聚焦基准；候选联合 logits 齐备后算 Eq2。训练期 chunk/重计算优化暂不启用，不能沿用独立 chunk softmax。 | 在 dropout disabled 下检查 loss/gradient/update 等价；启用 dropout 后声明 RNG/统计差异，不伪称逐位等价；尾批按真实句数归约。 | G04, G08, G11 |
| A10 · optimizer 与 scheduler | 仅 LR1e-5 和 10epochs 已恢复；optimizer/betas/eps/decay/warmup/scheduler 未恢复。 | 可测试主候选 AdamW(beta1=.9,beta2=.999,eps=1e-8)，weight_decay=0、constant LR、warmup=0、无额外 clipping。全部为假设，非论文默认。 | 分别对 weight_decay=.01、linear schedule/6% warmup、clip1 做预声明有界敏感性测试；不得以 test 得分选配置。记录实际 update 序列。 | G02, G04, G08 |
| A11 · 长度与溢出 | 候选 tokenizer 上限512；CTL maxlen/截断政策未恢复。tokenizer 512 不是已恢复下游 recipe。 | 聚焦实现以512为明确实验上限，默认检测溢出并报诊断，不静默截断/丢样本；完整数据截断方案尚未采纳。 | 512/513边界；必须同时保留两个实体及完整 prompt/description。若需 entity-preserving crop，记录原位置/窗口，并先冻结与审查，再单独敏感性比较。 | G02, G03, G05, G08 |
| A12 · 新增 token/head 初始化及 dropout | 候选 configs 的 dropout=.1、initializer_range=.02 为候选资产事实，不是 CTL 下游选择；新 token/head 初始化未恢复。 | 主候选保留候选 config dropout；新 embedding/head 用显式 seeded Normal(0,.02)，bias0；仅允许四个新 token 行，不重置已有权重。 | 加载前后已有 embedding 行差异0；固定 seed 重复初始化；初始化与 dropout0/.1 的敏感性结果分开记录。 | G03, G04, G07, G08 |
| A13 · dev 选择与 inference ties | 推理 argmax 已恢复；dev checkpoint policy 和 tie law 未恢复。scalar reference 返回全部并列最大值。 | 主候选使用 dev 非NA micro-F1 选 checkpoint，等分先选较早 update；推理 logits 精确相同时按 canonical ID 选最小 ID，并记录 tie。 | last-epoch vs dev-best 的预声明比较；禁止 test 选 checkpoint/阈值；tie fixture 与 scalar 最大值集合一致。 | G02, G08, G09 |
| A14 · 评测口径 | 论文写 micro-F1，但 null-class 分母及原 scorer 未恢复；论文分数不是项目实测。 | 项目报告非NA micro P/R/F1、all40 accuracy、all40 macro-F1、loss 和每类 TP/FP/FN/support；另列 all-label micro 等于 accuracy 的说明。 | 混淆矩阵已知 fixture；NA-only/错误NA/错正类/零支持定义明确。原论文 scorer 未确认前，不把同名分数视为 exact comparable。 | G09 |
| A15 · seed 与预算 | 论文均值来自 several unspecified seeds，无 seed 列表或 SD。 | 项目候选 seed13/29/53 作为明确项目选择；首个完整 seed 和后续多seed仍交合作者，绝不声称作者 seeds；预算不可默默降低。 | 逐 seed 全指标、mean±sample SD、失败尝试和预算/暴露量；原结果仅用于带口径限制的对照，不承诺复现原值。 | G08, G09, G12 |
| A16 · case study / checkpoint / 导出 | 原论文未恢复运行生命周期；项目要求真实 train/test 数值证据、post-update batch resume 和审计报告。 | 另行实现样本 lineage：raw→marked/prompt→token ids/masks→embedding→layer states→first token→scalar logits→候选概率→pred/gold/loss；train 另含梯度/update 时点。 | 实际数值 tensor 在私有 exchange 可取且 SHA 匹配；摘要/hash/placeholders 不等于张量；恢复绑定 H/data/模板/checkpoint/optimizer/RNG/游标，拒绝漂移。 | G10, G11, G12 |

## 验收矩阵：准备好的检查不是已通过的检查

`bounded_evidence_available` 只复用已完成的 provenance/primitive 证据；
`primitive_available_physical_binding_pending` 表示代码存在但物理数据绑定未完成。
其他项均 `pending`。下面的 acceptance evidence 是**未来需交付的证据**。

| 门槛 / 内容 | 当前状态 | 前置项 | 通过时必须有的实证 | 失败时的处理 |
|---|---|---|---|---|
| G01 · 已恢复证据边界 | bounded_evidence_available | — | 已锁 paper/SimCSE/RECENT 原始响应和 primitive 测试；本矩阵验证这些锁的 SHA，不冒充新增 neural 证据。 | 锁或证据变更时重审受影响条目，保留原记录。 |
| G02 · 解释性机制与假设审查 | pending | G01 | 独立审查报告逐项检查联合编码/H/Eq2/梯度/batch/数据隔离；未恢复选择和敏感性计划明确获批。 | 保留 not_verifiable 状态和历史行；文档存在不等于审查完成。 |
| G03 · 真实 tokenizer/标记/模板 | pending | G01 | 真实 byte-BPE+四注册 token；role/span/offset/prompt边界/512溢出逐字 fixture；最终模板/描述提取代码 SHA。 | 不静默 alias、裁剪或丢样本；输出失败定位。 |
| G04 · tiny neural Eq2 与 update | pending | G01, G03 | 随机 tiny encoder 合成数据 FP32；forward/loss/gradient/一次 update 对独立 reference（初始 atol1e-6/rtol1e-5，调整需证据）；NA-only/bias/批量与分块反例。 | tiny 通过只证明方程翻译，不证明 large checkpoint 或正式训练速度/准确率。 |
| G05 · 物理数据资格 | pending | G01 | qualified task/data manifest、train/dev/test hashes/counts/40-label map、role/type/span校验；全量 overflow/候选覆盖报告。 | 覆盖缺口或资格失败时不进入完整比较；诊断不修 H、不跳样本。 |
| G06 · 训练支持表隔离 | primitive_available_physical_binding_pending | G05 | H 输入绑定物理 train hash；表 hash/counts可复算；变更 heldout gold 不变表/预测输入，unknown/excluded gold计数为0或资格失败。 | 禁止 heldout fitting、gold reinsertion、NA-only未知回退、强制正类及完整 RECENT 脚本复用。 |
| G07 · 实际资产加载与作者身份边界 | pending | G01 | 实际权重本地 SHA/加载差异/旧embedding不变/完整encoder数值证据；另有独立字段证明作者 checkpoint/H 身份，当前仍未确认。 | 可在明确候选身份下做解释性 smoke；仅候选加载通过不升级 exact_author_mapping。 |
| G08 · recipe 与敏感性 | pending | G02, G04, G05, G06, G07 | 冻结完整有效 recipe、batch单位/update数、预声明单因素敏感性/失败记录；只用 train/dev 作选择。 | 未实施敏感性不能用准备好的计划冒充 paper-faithful。 |
| G09 · scorer / 结果证据 | pending | G05, G06, G08 | 项目全指标fixture与真实结果；原paper scorer口径单独确认；合作者完整单seed及多seed不确定性/至少一个reported result对照或明确缺口说明。 | 不将 paper 均值当 single seed，不以相近数值倒推 scorer/H/初始化正确。 |
| G10 · 运行恢复与完整张量 | pending | G03, G04 | train/test完整所选样本链路实际 tensor，私有下载验SHA；未中断与post-update resume等价；上下游 tensor边保持可追踪。 | partial/只有摘要/hash/无下载实值即不满足完整case；汇总/导出不得伪造未采集阶段。 |
| G11 · 资源/CUDA精度 | pending | G04, G07, G10 | 合适内存环境full-weight smoke；CUDA/AMP溢出与optimizer-step/恢复实证；GPU拓扑/physical batch变化不静默改有效batch。 | 当前1GiB限制不变；不降低门槛或操作无关进程；CPU/tiny通过不称CUDA认证。 |
| G12 · 单独正式准入/发布 | pending | G02, G03, G04, G05, G06, G07, G08, G09, G10, G11 | 独立 fidelity report明确paper-faithful是否成立；具备严格资格后单独更新cohort并获发布许可；完整实验仅合作者执行。 | 保持cohort ready5和两CTL行blocked；文档/flags不能自动打开runner或推送main。 |

## 三个不同的许可层级

1. **合成 tiny 实现/测试**：可推进输入 fixture 和方程/梯度验证，不使用真实任务数据、
   large 权重或自动触发 formal runner；其作用仅是验证算法翻译。开发时的随机 tiny
   RoBERTa config 不宣称是 pretrained SimCSE，更不产生 benchmark 得分。
2. **明确候选身份的解释性 smoke**：需实际资产 SHA/加载和资源门槛；所有解释性选择
   明示冻结。作者精确身份尚未确认时，候选仍是候选。该层不纳入严格 baseline 表。
3. **严格比较/正式发布**：需独立 fidelity report、数据/scorer、数值/生命周期和
   敏感性/结果证据，逐项解释残余缺口；再单独更新 cohort/manifest 和发布授权。
   不以“所有 pending 字段填 true”代替证据。完整数据训练及评估由合作者进行。

## 下一步与停止条件

下一步只连接 synthetic tiny 联合编码与 Eq2，验证首 token、共享 scalar head、
按句归一化/归约及一次 FP32 update。先不扩展全量 runner 或启用内存优化。

碰到未知 pair、gold 覆盖缺口、512溢出、未冻结模板或 checkpoint/资源缺口时，
记录具体诊断；不降格替换模型、不用 heldout labels 修 H、不丢样本、不伪造张量。
独立机制审查仍是解释性实现认证前的门槛，本次文档工作没有完成该审查。

本表的 JSON 是专用 design schema，不是 `baseline-manifest.json`、runtime config
或 admission override；现有 runner/selector 不读取它。修改暂定值时必须保留假设
身份并更新受影响敏感性/证据，不能把 proposal 重命名为 recovered fact。
