# Q-Attention Documentation

本目录只通过本页导航。需要运行当前实验时，只阅读 `current/`；`archive/` 中的文档用于追溯历史设计，不再作为运行依据。

## 当前文档

| 文档 | 用途 |
| --- | --- |
| [当前方法概览](current/method_overview_zh.md) | 了解当前 Q-NESS 主线、控制和代码入口 |
| [Q-NESS Toy Gate](current/qness_toy_gate_zh.md) | 运行五 seed 机制原型测试 |
| [合作者 Git 工作流](current/collaborator_git_workflow_zh.md) | clone、同步 `main`、维护 `1.1`、提交报告 |
| [Re-TACRED full 运行指南](current/retacred_dual_qres_full_run_zh.md) | 在完整 train/valid 上运行 seed 13 主实验预跑 |
| [正式实验协议](current/experiment_protocol_zh.md) | 五 seed、消融、指标和 blind test 规则 |

## 历史文档

- [历史文档索引](archive/README.md)
- `archive/legacy_experiments/`：旧 smoke pipeline、旧重跑指南和早期交接流程。
- `archive/research_notes/`：早期 projector、plugin、任务规划和创新构想。

## 使用规则

1. 当前实验以 GitHub `main` 和 `current/` 为准。
2. `run_relation_smoke_pipeline.py` 和 `run_relation_attention_transfer_screen.py` 不是当前 dual Q-RES full 主实验入口。
3. full 主实验只在 train/valid 上选择模型；配置冻结后才允许统一评估 blind test。
4. 不提交 `data/`、`runs/`、checkpoint、predictions、JSONL 或完整日志。
