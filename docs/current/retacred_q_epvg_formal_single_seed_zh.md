# Q-EPVG Re-TACRED 正式单 seed 交接

## 默认运行命令

在干净的 `1.1` 工作树中执行：

```bash
bash scripts/run_retacred_q_epvg_formal_single_seed.sh \
  --gpu auto \
  --hardware-profile adaptive
```

脚本固定 seed 13、完整 Re-TACRED 数据和配置中的 28 个 selector。它先训练一次
disabled baseline，再把 27 个 Q-EPVG 变体作为独立 selector 任务动态分配到可见 GPU。
`--gpu auto` 只选择设备和执行内存档位，不改变数据、seed、batch、epoch、selector 或
科学门禁。正式全量实验只能在合作者服务器运行。

## 同步分支

```bash
git fetch origin --prune
git checkout 1.1
git pull --ff-only origin 1.1
git merge origin/main
git status --short --branch
```

不要在 `main` 上直接运行、改配置或改 selector。若 `1.1` 有本地改动，先停止并报告。

## 断点续跑

脚本在 baseline 和每个 selector 中都写入 batch 级 checkpoint、恢复契约和显存档位状态。
中断后在同一目录继续：

```bash
bash scripts/run_retacred_q_epvg_formal_single_seed.sh \
  --gpu auto \
  --hardware-profile adaptive \
  --resume runs/retacred_q_epvg_formal_single_seed/<timestamp>_seed13
```

若从单 GPU 换到多 GPU，必须显式加 `--allow-gpu-topology-change`。只有 execution-layer
修复才允许加 `--allow-code-update`；seed、数据、batch、epoch、selector 和控制组契约
不能改变。出现恢复契约错误时，先运行只读诊断：

```bash
bash scripts/check_retacred_q_epvg_resume.sh \
  --run-dir runs/retacred_q_epvg_formal_single_seed/<timestamp>_seed13 \
  --gpus auto \
  --hardware-profile adaptive
```

不得删除 checkpoint、改参数绕过诊断或新建替代 run。

## 运行产物与 Case Study

每个 selector 固定 train/valid/test 各 3 个样本，并在 initial、best-valid、final 三个
checkpoint replay。Case Study 会生成样本语义字段、token/实体位置、baseline 与 selector
预测、关键 attention/EPVG 张量的安全投影和私有 detached tensor manifest。完整 tensor
文件只留在私有 run 目录，不得提交。

程序只有在 `RUN_COMPLETE`、summary、所有 selector metrics 和 Case Study 均存在时才调用
exporter。exporter 只允许提交 `reports/retacred_q_epvg_formal_single_seed/<timestamp>_seed13/`
下的审计报告子集，并检查 `data.sha256`、`data_counts.txt` 和非空 `run_summary.data`。

## 停止门禁

- `CUDA OOM` 或显存压力：让自适应档位按既定顺序降级，不要手工改 batch 或 selector。
- 数据、配置、seed、selector、训练预算或 test leakage 检查失败：立即停止并回传日志摘要。
- 单 seed 结果未由项目方完成报告审计前，不得开始 multi-seed 或宣称 L1/L2、量子优势。
- 只回传 exporter 生成的 `reports/` 子集；不要提交 `runs/`、`data/`、checkpoint、权重、预测、JSONL 或完整日志。

## 完成后的回传

```bash
git status --short --branch
git diff --cached --check
git add reports/retacred_q_epvg_formal_single_seed/<timestamp>_seed13
git diff --cached --check
git commit -m "Add Q-EPVG formal single-seed report"
git push origin 1.1
```

项目方收到 `1.1` 报告后先审计完整性、数据身份、selector 对照和门禁，再决定是否允许
multi-seed。不要在没有明确授权前自行启动 seed 29/53。
