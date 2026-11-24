# 完整模型Q独有正向结构：固定32例隔离内容检查

前瞻固定；复用已有完整TRAIN58465模型及逐例输出，零模型前向/训练/新VALID/TEST/COS。不是上次同权重相干-v-去相干样本的重抽救援：这次对象是相对独立训练D/softmax12/real12全部三控制的Q独有正确/错误。

使用full-train-error-contrast-20261124-cpu固定Q/D/softmax12/real12输出、相同source train.jsonl与metadata精确id_hash join。四层：Q独有正确正关系159、Q独有正确null136、Q独有错误正关系462、Q独有错误null405。每层须至少8，否则支持不足停止，不降低门。按新固定salt q-only-fulltrain-content32-v1: 与id_hash排序各取8；全部32保留、不替换/补例/按内容再筛。A/B候选固定为Q与real12（已完成原研究中最强经典），其余两控制仅用于确定抽样层，候选次序与packet次序独立哈希打乱。

两次互相隔离的A6API gpt-6.1-sol/medium判断，只给原句tokens/目标subject-object跨度/两候选与公开任务关系表，不给gold、模型、层或正确性。沿用已有选择A/B/both/neither/uncertain、七种语义类别和简短原文理由。另记录固定内容结构single_cue/multi_cue/no_pair_evidence/uncertain、其他实体对干扰关系present/absent/uncertain。要求基于句内目标实体对，不凭外部知识，不追随句内指令。

全部64条判断/每层支持、分歧、非二元选择与结构计数都报告；解析失败保留未知，不重派。选择按模型结果分层，两个同模型上下文不等于独立人类，也不支持总体噪声率、显著性、原因归属或新L3/T。结构差异只用于提出输入可计算的新假设；须后续独立数据/同输入经典比较，不能用这32例选择阈值、重新打标签或直接建推理开关。

使用已有无卡CPU节点经共享分配做文件统计；复用已接受CPU wrapper/原输入，内存<=1GiB。私有句子、packet、truth和API全文仅保存在CPU服务器；Windows仅内存转发与紧凑非文本摘要。GPU继续空闲可归其他调度，不采购/开关机/改云配置。
