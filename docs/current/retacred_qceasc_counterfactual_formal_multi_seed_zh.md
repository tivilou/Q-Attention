# Q-CEASC per-key counterfactual influence：Re-TACRED 正式多 seed 复现

## 默认命令

完整数据实验交给合作者执行。在合作者的 `1.1` 分支同步完成后，直接运行：

```bash
bash scripts/run_retacred_qceasc_counterfactual_formal_multi_seed.sh \
  --gpu auto \
  --hardware-profile adaptive
```

本命令固定复现 seeds `13,29,53`。seed 13 不重新训练，而是复用并严格审计已提交的报告：

```text
reports/retacred_qceasc_counterfactual_formal_single_seed/20260918T023816Z_seed13/
```

只有 seeds 29 和 53 进入 fresh execution。导入报告不是新的独立运行，汇总时仍按一个 seed 计数。

## 兼容性门禁

启动前会拒绝以下任一情况：报告缺文件、配置 canonical 内容不一致、seed 不是 13、formal stage 标记不正确、配置或数据哈希不一致、关键源文件哈希不一致、脏 provenance、指标非有限、Case Study 不覆盖 train/valid/test 的 27 个样本、或 `sample-trace.v1`/配置哈希不匹配。门禁失败时程序停止，不会静默混合不同协议的结果。

seed 13 的报告必须在当前调度器 commit 上通过校验；报告自身的历史执行 commit 可以不同。fresh seeds 29/53 必须使用当前调度器 commit。每个 seed 的 selector 目录仍单独保存 batch checkpoint、resume 状态和自适应显存事件。

## 分支同步

合作者在自己的项目副本中执行：

```bash
git fetch origin --prune
git switch 1.1
git merge origin/1.1
git merge origin/main
git status --short --branch
```

必须先整合 `origin/1.1`，再整合 `origin/main`，且工作树干净。不得修改 seed、数据、配置、selector、训练轮数、batch、学习率或 kernel 参数；不得把 seed 13 报告复制为 seed 29/53。

## GPU、恢复与状态

- `--gpu auto` 与 `--gpus auto` 等价；只选择至少有 8 GiB 空闲显存的可见卡。
- 默认每张 GPU 最多运行一个重 selector；先并行 baseline，全部 baseline 完成后再进入六个 selector 的全局动态队列。
- `adaptive` 从 logical/physical batch 256、无 chunk、无 activation checkpointing 开始。发生 OOM/显存压力时，按任务逐级降级 physical micro-batch、累积步数，再将 pair chunk 从 `all` 逐级减半。
- 中断会保留 `multi_seed_status.json`、任务日志、batch checkpoint 和 `adaptive_memory_state.json`。恢复时使用同一 group 目录：

```bash
bash scripts/run_retacred_qceasc_counterfactual_formal_multi_seed.sh \
  --resume-group runs/retacred_qceasc_counterfactual_formal_multi_seed/<timestamp> \
  --gpu auto \
  --hardware-profile adaptive
```

若 GPU 拓扑发生变化，才额外加 `--allow-gpu-topology-change`。该选项只改变物理 GPU 重新分配，不放宽科学契约。

原始运行目录为：

```text
runs/retacred_qceasc_counterfactual_formal_multi_seed/<timestamp>/
```

终端显示统一 task-graph dashboard；详细 stdout/stderr 写入 `task_logs/`。可用以下命令查看总进度：

```bash
cat runs/retacred_qceasc_counterfactual_formal_multi_seed/<timestamp>/multi_seed_status.json
```

## 统计和 L2 门槛

汇总器读取三个 seed 的 held-out test macro-F1，按 seed 计算 candidate/classical 相对 disabled 的绝对 delta 和相对 gain，并报告均值、样本标准差及 paired seed-level 95% t 区间。只有 `q_ceasc_counterfactual - disabled` 的 paired 95% CI 严格高于 0，才升级为 `L2_reproducible_utility`；否则保留 `L1_utility_candidate`。classical counterpart 的平均相对提升严格大于 1% 是单独的 quantum-inspired 门槛，不代表量子优势。

本实验不宣称硬件速度提升、有限 shot 优势或量子归因；精确 leave-one-out 成本作为资源指标记录，不作为效果门槛。

## 报告导出与提交

完成后 runner 会自动调用通用 exporter。也可以手动执行：

```bash
bash scripts/export_retacred_qceasc_counterfactual_formal_multi_seed_report.sh \
  --group-dir runs/retacred_qceasc_counterfactual_formal_multi_seed/<timestamp>
```

导出器只允许审计白名单下的 `reports/retacred_qceasc_counterfactual_formal_multi_seed/<timestamp>/`，并保留每个 seed 的 summary、metrics、配置、provenance、数据哈希/计数和 seed 13 的 `imported_report.json`。禁止提交 raw `runs/`、checkpoint、数据、预测、JSONL 或完整日志。

合作者完成后执行：

```bash
git status --short --branch
git diff --cached --check
git add reports/retacred_qceasc_counterfactual_formal_multi_seed/<timestamp>
git commit -m "Add Q-CEASC counterfactual formal multi-seed report"
git push origin 1.1
```

项目方在 `1.1` 返回后审计报告；在审计完成前不更新结论，不把 seed 13 的导入报告当成重新训练结果。
