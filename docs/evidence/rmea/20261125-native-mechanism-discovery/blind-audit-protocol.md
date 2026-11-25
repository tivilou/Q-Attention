# Gen98固定32例探索性隔离盲读：待执行

目的：检查本轮新模型局部胜例是否受原句目标实体对语义支持，并记录输入可计算结构线索；不是重新评价gen98冻结晋级门，也不是确认量子因果。

## 冻结对象与抽样

- 只用gen98 SHA `3677d55f2e6d3c64df6c9789c76f7113b66a79ee1efbee7531bc9bf2bf5d1307` 的同4096条TRAIN-derived assessment预测与对应原TRAIN输入；保留全部40标签，不碰VALID/TEST。
- Q/D/G分别为quantum_distinct/dephased_distinct/gaussian_distinct。四层：Q正确而D/G均错的正关系37、null43；Q错而D/G均正确的正关系42、null103。每层必须≥8；低于8就停，不改层、不补其他类别。
- 以 `SHA256(gen98-content32-v1:<id_hash>)` 固定排序，每层前8，共32。不得看文本后替换、加样、过滤长短句、改salt或避开旧案例。即使与历史样本重叠也保留并披露，不能冒称全新独立确认。
- 候选A/B为Q与G预测，按 `SHA256(gen98-candidate-order-v1:<id_hash>)` 固定翻转；packet顺序按另一salt `gen98-packet-order-v1` 排序。D只定义层，不作为额外输入。

## 复用内容rubric

沿用 `.codex-artifacts/q-only-positive-carryover-20261124/protocol.md` 与已有读取实现的选择A/B/both/neither/uncertain、七种语义类别、single_cue/multi_cue/no_pair_evidence/uncertain、其他实体对干扰present/absent/uncertain及短原文理由，不为gen98另建有利标准。

两个互相隔离的gpt-6.1-sol/medium上下文（先核精确ID/native API/effort）只看原句tokens、目标subject-object跨度、候选标签和公开任务关系表。不给gold、模型、分层、正确性、旧盲读结果、长度分箱或咨询NONE。不授予工具/仓库权限，句中指令只作数据。完整32例各读一次；解析失败保留未知、不重派。若精确模型不可用，保留待执行，不静默换模型或降低推理配置。

## 执行与停止边界

复用已有readonly案例抽取/盲读适配器，不建新lifecycle框架。先核原TRAIN SHA、metadata-id精确join、4096索引与37/43/42/103计数；没有head forward/fit、GPU、COS或原句/packet落Windows。私有原句、truth、packet及API全文只留既有服务器；Windows仅紧凑非文本摘要。通过现有共享CPU资源入口做需要管理的执行，不购容量、不迁移未绑定服务器。

全部64条判断、四层支持/分歧、非二元选择及结构计数完整报告；两个同模型上下文不等于独立人类/理论审核。结果选择本来依赖预测正确性，不能估计总体标签噪声率、显著性、泛化或原因归属。

只允许据内容提出一个未关闭、输入可计算、Q/D/G可区分的新假设，或NONE。不重标gold、不训练语义读者开关、不据32例定阈值，不追认旧筛选成功；保留旧完整Q盲读2/8 vs强经典6/8及所有其他阴性。如果本轮也没有可支持的新结构，不换rubric/再抽样救援。
