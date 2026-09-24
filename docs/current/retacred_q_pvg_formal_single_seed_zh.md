# Q-PVG Re-TACRED 正式单 seed 交接

## 冻结协议

Q-PVG（Phase-Sensitive Quantum Value Gating）在显式 attention context/value hook 上运行。正式入口固定为 seed 13、完整 Re-TACRED 数据和六个 selector：

- `disabled`
- `q_pvg_phase_value`（候选）
- `q_pvg_real_only`（去除虚部通道的消融）
- `q_pvg_classical_complex`（匹配 classical control）
- `q_pvg_random_phase`（结构控制）
- `q_pvg_score_value`（score-value 消融）

正式脚本会先完成 baseline，再把 selector 作为独立任务分配给可用 GPU。每个 selector 独立保存 batch checkpoint、显存自适应状态和 Case Study；`--gpu auto` 只改变调度/显存策略，不改变 seed、数据、预算或 selector 集合。

## Case Study 与报告

每个 selector 固定 train/valid/test 各 3 个样本，并记录 initial、best-valid、final 三个 checkpoint。完整张量留在私有 run 目录，导出报告只包含安全投影、指标、语义 `sample-trace.v1`、数据计数、哈希和 provenance。不得提交 `data/`、`runs/`、checkpoint、模型权重、predictions、JSONL 或完整日志。

## 合作者运行

```bash
bash scripts/run_retacred_q_pvg_formal_single_seed.sh \
  --gpu auto \
  --hardware-profile adaptive
```

如中断，在同一 run 目录上恢复：

```bash
bash scripts/run_retacred_q_pvg_formal_single_seed.sh \
  --gpu auto \
  --hardware-profile adaptive \
  --resume runs/retacred_q_pvg_formal_single_seed/<timestamp>_seed13
```

如果运行在首个 selector 处因旧版梯度检查器报 `missing=scalar_value_logit`
而停止，先同步本次 execution-layer 修复，然后在原 run 目录恢复，保留已完成的
baseline 和 batch checkpoint：

```bash
git fetch origin --prune
git checkout 1.1
git pull --ff-only origin 1.1
git merge origin/main
git push origin 1.1

bash scripts/run_retacred_q_pvg_formal_single_seed.sh \
  --gpu auto \
  --hardware-profile adaptive \
  --resume runs/retacred_q_pvg_formal_single_seed/<timestamp>_seed13 \
  --allow-code-update
```

`--allow-code-update` 只允许已发布的 execution-layer 修复迁移到恢复契约；seed、
数据、selector、batch size、epoch 和科学配置仍必须完全一致。

如果恢复时出现 `ResumeCompatibilityError: resume contract differs`，先运行只读诊断，
不要删除 checkpoint 或新建 run：

```bash
bash scripts/check_retacred_q_pvg_resume.sh \
  --run-dir runs/retacred_q_pvg_formal_single_seed/<timestamp>_seed13 \
  --gpus auto \
  --hardware-profile adaptive
```

诊断会比较原 run 与当前 checkout 的恢复契约，只输出不兼容字段路径和类别，不显示
数据内容或契约字段值。`recommended action` 给出可用的恢复选项；若建议停止，或差异
中包含 `config`、`data`、`materialization`、`selector`、`seed`、`batch_size`、`epochs`
等科学契约字段，应先把诊断输出交回项目方审查，不要改参数绕过检查。

完整数据实验只能由合作者执行。项目方只做代码测试、toy、preflight 和报告审计；本单 seed 审计完成前不得启动 multi-seed，也不得宣称量子优势。

## 交接前检查

```bash
python scripts/check_retacred_q_pvg_formal_single_seed.py --fresh-run
python -m py_compile experiments/run_q_pvg_formal_single_seed.py experiments/run_q_pvg_scheduler_base.py experiments/run_q_pvg_selector_worker.py
bash -n scripts/run_retacred_q_pvg_formal_single_seed.sh
git status --short --branch
git add <intentional-files>
git diff --cached --check
git commit -m "Add Q-PVG formal single-seed handoff"
git push origin main
```

合作者随后将 `main` 合并到 `1.1`，在干净的 `1.1` 工作树中运行脚本；脚本成功生成 `RUN_COMPLETE` 和 summary 后才调用 Q-PVG exporter。导出的报告再通过 `1.1` 返回项目方审计。
