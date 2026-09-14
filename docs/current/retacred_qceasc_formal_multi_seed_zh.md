# Q-CEASC Re-TACRED 正式多 seed 复现

默认命令（完整数据实验交给合作者执行）：

```bash
bash scripts/run_retacred_qceasc_formal_multi_seed.sh \
  --gpus auto \
  --hardware-profile adaptive
```

该命令固定复现 seed `13,29,53`，每张选中的物理 GPU 最多运行一个 seed，采用动态队列。每个 seed 都重新执行 disabled baseline、Q-CEASC 和 classical CEASC；不得把旧的单 seed 结果拼接进本次 L2 汇总，也不得修改数据、模型、训练轮数、batch、学习率、selector 或控制组。

## 执行前同步

合作者在自己的副本执行：

```bash
git fetch origin --prune
git switch 1.1
git merge origin/1.1
git merge origin/main
git status --short --branch
```

必须先合并 `origin/1.1`，再合并 `origin/main`，且工作树干净。实验服务器与项目授权服务器隔离；项目方不访问合作者的 raw `runs/`、checkpoint、日志或 GPU 状态。

## 环境与 GPU

脚本按 `PYTHON_BIN`、`python`、`python3` 顺序寻找解释器。`--gpus auto` 只选择至少有 8 GiB 空闲显存的卡；也可显式写 `--gpus 0,1,2`。启动前应确认 CUDA、数据路径和磁盘空间可用。多 GPU 只并行独立 seed，不代表已经证明速度提升。

## 输出与观察

原始输出写入：

```text
runs/retacred_qceasc_formal_multi_seed/<timestamp>/
  multi_seed_manifest.json
  multi_seed_status.json
  seed_13/
  seed_29/
  seed_53/
```

父进程终端显示统一 dashboard；每个 seed 的完整 stdout/stderr 保存在自己的 `parent-child.log`，训练器还会写 batch checkpoint、heartbeat、metrics、case study 和 `sample_trace.v1`。可用下面命令查看状态：

```bash
cat runs/retacred_qceasc_formal_multi_seed/<timestamp>/multi_seed_status.json
```

收到 `MULTI_SEED_COMPLETE`、`multi_seed_summary.json` 和 `multi_seed_summary.md` 后，脚本才会尝试导出审计报告。

## 暂停、恢复与失败

`Ctrl-C`/终止信号会转发给正在运行的 seed；单 seed 训练器在完成当前 optimizer update 后写 batch checkpoint。若整个父进程在所有 seed 完成前退出，不得手工把目录标成完成，也不得从新鲜 shuffle 的 loader 跳过 batch。使用同一目录恢复：

```bash
bash scripts/run_retacred_qceasc_formal_multi_seed.sh \
  --resume-group runs/retacred_qceasc_formal_multi_seed/<timestamp> \
  --gpus auto \
  --hardware-profile adaptive
```

调度器会跳过已有 `RUN_COMPLETE` 的 seed，只把未完成 seed 交给单 seed runner 的 batch-level `--resume`；已完成 seed 不得复制成另一 seed。

任一 seed 失败时停止派发尚未开始的 seed，并保留 `MULTI_SEED_FAILED`。先检查对应 `seed_<seed>/parent-child.log` 和 run marker，再向项目方返回诊断；不要改 seed、调参或用结果救回失败运行。

## L2 统计口径

汇总脚本读取三个 seed 的 held-out test macro-F1，计算每个 seed 的：

- Q-CEASC − disabled 的绝对 delta；
- classical CEASC − disabled 的绝对 delta；
- 相对于 disabled 的相对 gain；
- 均值、样本标准差和 paired seed-level 95% t 区间。

只有 Q-CEASC 的 paired 95% CI 严格高于 0 才建议升级为 `L2_reproducible_utility`；否则保留 `L1_utility_candidate`。classical 对照单独按项目规则检查平均相对提升是否严格大于 1%，这不等同于量子优势。统计结果不宣称硬件速度提升、有限 shot 优势或 L4 量子归因。

## 审计报告提交

成功完成后脚本自动调用：

```bash
bash scripts/export_retacred_qceasc_formal_multi_seed_report.sh \
  --group-dir runs/retacred_qceasc_formal_multi_seed/<timestamp>
```

导出器再次检查 `1.1` 分支、工作树、祖先关系、全部 seed marker、指标、case study 和 `sample-trace.v1`，只提交 `reports/retacred_qceasc_formal_multi_seed/<timestamp>/` 下的审计白名单。禁止提交 raw runs、数据集、checkpoint、预测和完整日志。若自动 push 失败，只返回终端中的错误，不要重写报告目录。
