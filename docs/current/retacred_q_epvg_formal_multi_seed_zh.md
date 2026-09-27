# Q-EPVG Re-TACRED 正式多 seed 交接

## 默认运行命令

在干净的 `1.1` 工作树中执行。seed 13 已有审计通过的完整单 seed 报告时，复用它，只新跑
seed 29 和 53：

```bash
bash scripts/run_retacred_q_epvg_formal_multi_seed.sh \
  --gpu auto \
  --hardware-profile adaptive \
  --import-seed13-report reports/retacred_q_epvg_formal_single_seed/<audited_seed13_report_dir>
```

`<audited_seed13_report_dir>` 必须是通过新版 v2/v3 Case Study、`run_summary.data`、
`data_counts.txt`、`data.sha256` 和 `export_manifest.json` 校验的报告目录；旧版 v1 或
缺少身份文件的报告会被拒绝，不能静默复用。

若项目方明确要求三个 seed 都从头执行，可省略 `--import-seed13-report`；seed 集合仍固定为
`13,29,53`。正式完整数据运行只能在合作者服务器执行，不得在本项目服务器启动。

调度器先并行执行 `baseline(seed)`，通过 baseline barrier 后，将
`selector(seed,name)` 放入一个全局动态队列。队列按 seed 轮转，每轮每个 seed 最多进入一个
selector，默认每张物理 GPU 只运行一个重 selector。selector 独立记录 batch checkpoint、
显存档位和 heartbeat；单个 selector OOM 只降低该任务的执行档位，不改变逻辑 batch、seed、
selector、数据或科学门禁。

启动前 runner 会自动执行 `scripts/check_retacred_q_epvg_formal_multi_seed.py --canary`，用
正式 scheduler 生成的同一 argv/cwd 启动 baseline 和 selector 的 `--help` 子进程；该 canary
只验证父子进程路径和退出清理，不读取完整数据，也不开始训练。

## 同步分支

```bash
git fetch origin --prune
git checkout 1.1
git pull --ff-only origin 1.1
git merge origin/main
git status --short --branch
```

必须先整合 `origin/1.1`，再合并 `origin/main`。不要在 `main` 上运行或修改实验。

## 断点续跑

多 seed 组目录由启动时 UTC 时间戳唯一确定。中断后使用同一目录恢复，并在 GPU 拓扑变化时
显式授权：

```bash
bash scripts/run_retacred_q_epvg_formal_multi_seed.sh \
  --gpu auto \
  --hardware-profile adaptive \
  --resume-group runs/retacred_q_epvg_formal_multi_seed/<timestamp>
```

若 seed 13 是通过报告导入的，它会被标记为 immutable audited reuse；seed 29/53 的已完成
selector 会按完整 Case Study 和 checkpoint 证据跳过。恢复契约不允许改变数据、seed、selector、
batch、epoch 或控制组。

## 完整 Case Study 记录

每个非 disabled selector、每个 seed 都必须生成同一组冻结样本：train/valid/test 各 3 条，
并在 `initial_or_pre_training`、`best_valid_or_declared_selection_checkpoint`、`final` 三个
checkpoint replay。每条样本包含原句、tokens、token ids、实体文本与 span、gold/prediction，
以及 data → preprocess → embedding → encoder → training → attention baseline → scoring →
selection → intervention（适用时）→ context → classifier → evaluation → diagnosis 的真实
producer-owned 输入/输出引用。所有决策相关张量都通过 manifest 记录 shape、dtype、轴语义、
摘要、SHA-256、字节数、producer stage 和安全预览；完整 tensor 文件只留在私有 run 中。

缺失阶段、dangling reference、checkpoint/split 覆盖不足、manifest 不一致或私有 tensor
校验失败都会阻止导出，不能标记为 `partial` 后继续发布，也不能从最终 prediction 推断中间状态。

## 导出与回传

训练和 summary 完成后，一键 runner 会自动调用 exporter。exporter 只允许审计白名单：
metrics、Case Study safe projection、sample trace、summary、配置、provenance、数据计数与
哈希；不允许提交 `runs/`、权重、checkpoint、预测、JSONL、tensor 二进制或完整日志。

exporter 使用 staging + 原子发布，并在正式交接前通过一次受控 copy/validation 失败、清理和
同一 immutable run 重试。失败尝试不会重新训练，也不会留下 final/partial 报告目录。

完成后只回传：

```bash
git status --short --branch
git diff --cached --check
git add reports/retacred_q_epvg_formal_multi_seed/<timestamp>
git diff --cached --check
git commit -m "Add Q-EPVG formal multi-seed report"
git push origin 1.1
```

项目方收到 `1.1` 报告后会重新审计每个 seed 的身份、Case Study、selector 对照和 paired 统计，
再决定 L2/quantum-inspired/quantum attribution 的 claim ceiling。不要自行更改 selector 或
在报告未通过审计前发布结论。
