# Q独有正确样本未给出清晰的内容结构优势

已有完整训练输出中，Q独有正确295条，反向独有错误867条。固定每层8例、两份隔离Sol6.1/medium判断已完成。Q独有正确正关系仅2/8获共同句内支持，反向正关系6/8；多线索和干扰实体未形成明确区分。全部32例、64判断和分歧保留，不重抽、不修改标签，也不把模型判断当标签真值。

这排除了直接用这32例包装“量子多线索优势”的理由。它没有排除全部结构方法或证明Q的失败原因。后续研究必须改变可检验的信息处理环节，复用稳定插件runner和公平经典对照，以实际任务收益决定去留。原限定L4保留，新L3/L4-T尚未通过。

证据：[完整聚合与局限](../evidence/rmea/20261124-q-only-positive-carryover/analysis.md)、[固定协议](../evidence/rmea/20261124-q-only-positive-carryover/protocol.md)、[全部32匿名结果](../evidence/rmea/20261124-q-only-positive-carryover/summary.json)。
