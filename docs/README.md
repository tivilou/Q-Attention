# Q-Attention Documentation

本目录只通过本页导航。当前实验只阅读 `current/`；`archive/` 中的文档用于追溯历史设计，不再作为运行依据。

## 当前文档

| 文档 | 用途 |
| --- | --- |
| [当前方法概览](current/method_overview_zh.md) | Q-VRES 方法、控制组和证据边界 |
| [Q-TRIAD Re-TACRED 正式单 seed](current/retacred_qtriad_formal_single_seed_zh.md) | 完整数据、seed 13、单 GPU 的 Q-TRIAD 与经典密度控制 |
| [Q-VRES 正式实验](current/qvres_relation_transfer_full_run_zh.md) | 当前 seed 13 validation 诊断与报告提交 |
| [Q-LASS Re-TACRED 正式单 seed](current/retacred_qlass_formal_single_seed_zh.md) | 完整数据、单 seed、单 GPU 的 Q-LASS 与 classical matched control |
| [Q-LASS 冻结验证](current/q_consensus_frozen_multiseed_zh.md) | 项目方执行 seed 7 门禁与五 seed synthetic 验证 |
| [QCCW Stage-0](current/qccw_stage0_zh.md) | connected-correlation successor 的 seed-7 机制门禁；当前 bilinear gate 失败 |
| [QCDD readout 诊断](current/qcdd_readout_diagnostic_zh.md) | coherent-minus-dephased seed-7 门禁；当前 matched control 失败 |
| [合作者 Git 工作流](current/collaborator_git_workflow_zh.md) | clone、同步 `main`、维护 `1.1`、提交报告 |
| [正式实验协议](current/experiment_protocol_zh.md) | 数据、selector、指标和交接规则 |
| [Q-NESS Toy Gate](current/qness_toy_gate_zh.md) | 历史机制原型测试，不是当前正式入口 |

## 历史文档

- [历史文档索引](archive/README.md)
- `archive/legacy_experiments/`：旧 smoke pipeline、旧重跑指南和早期交接流程。
- `archive/research_notes/`：早期 projector、plugin、Q-NESS 和 Q-RES 研究记录。

## 使用规则

1. 当前实验以 GitHub `main` 和 `current/` 为准。
2. Q-VRES 入口是 `scripts/run_qvres_validation_diagnostic.sh`，只读取已有 seed 13 raw run。
3. Q-LASS synthetic 验证由项目方执行：先跑 seed 7 单 GPU 门禁，再跑五 seed。完整真实数据正式实验才交给合作者。
4. QCCW Stage-0 仅跑 seed 7；当前 gate 失败，禁止五 seed、真实数据和硬件实验。
5. QCDD 仅为 seed-7 readout 诊断；当前 matched-control gate 失败，禁止调参、加 seed 和 attention action。
6. 不提交 `data/`、`runs/`、checkpoint、predictions、JSONL 或完整日志。
