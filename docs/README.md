# Q-Attention Documentation

本目录只通过本页导航。当前实验只阅读 `current/`；`archive/` 中的文档用于追溯历史设计，不再作为运行依据。

## 当前文档

| 文档 | 用途 |
| --- | --- |
| [当前方法概览](current/method_overview_zh.md) | Q-VRES 方法、控制组和证据边界 |
| [Q-VRES 正式实验](current/qvres_relation_transfer_full_run_zh.md) | 五 seed、自动多 GPU 和报告提交 |
| [合作者 Git 工作流](current/collaborator_git_workflow_zh.md) | clone、同步 `main`、维护 `1.1`、提交报告 |
| [正式实验协议](current/experiment_protocol_zh.md) | 数据、selector、指标和交接规则 |
| [Q-NESS Toy Gate](current/qness_toy_gate_zh.md) | 历史机制原型测试，不是当前正式入口 |

## 历史文档

- [历史文档索引](archive/README.md)
- `archive/legacy_experiments/`：旧 smoke pipeline、旧重跑指南和早期交接流程。
- `archive/research_notes/`：早期 projector、plugin、Q-NESS 和 Q-RES 研究记录。

## 使用规则

1. 当前实验以 GitHub `main` 和 `current/` 为准。
2. 当前正式入口是 `scripts/run_qvres_relation_transfer_multi_seed.sh`。
3. 正式运行只在 valid 上选择 intervention checkpoint，然后评估独立 test。
4. 不提交 `data/`、`runs/`、checkpoint、predictions、JSONL 或完整日志。
