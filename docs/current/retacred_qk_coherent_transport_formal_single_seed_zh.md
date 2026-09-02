# QK coherent transport Re-TACRED 正式单 seed

本包只授权完整 Re-TACRED、固定 seed 13 的单次正式执行；由合作者在自己的服务器运行。负责人服务器不会启动该完整数据实验。QK 是 pre-softmax 的 query-key 双寄存器 transport score residual；`register_qubits=1` 表示一对两量子比特通道，不代表已验证硬件加速或量子优势。

不要改代码、配置、数据、seed、batch、epoch、selector、门槛或控制组。不要启动多 seed；该决定只能由负责人审计正式报告后作出。

## 1. 同步分支

在项目根目录执行。顺序不可交换：先更新 `1.1`，再合并 `main`。

```bash
git fetch origin --prune
git switch 1.1
git merge origin/1.1
git merge origin/main
git status --short
git merge-base --is-ancestor origin/1.1 HEAD
git merge-base --is-ancestor origin/main HEAD
```

`git status --short` 必须没有输出；最后两条命令必须成功。否则停止，不要运行或导出报告。

## 2. 环境与 GPU 预检

激活已安装 PyTorch/CUDA 的环境；可用 `PYTHON_BIN` 显式指定解释器。脚本优先使用该变量，其次使用 `python`、`python3`。先确认可见 GPU：

```bash
export PYTHON_BIN=python
nvidia-smi
bash scripts/check_retacred_qk_coherent_transport_formal_single_seed.sh --fresh
```

预检为只读，不会创建 `runs/`。`--gpu auto` 只会选择空闲显存不少于 8 GiB 的卡；多个 GPU 时 QK candidate 和 matched classical control 以独立 worker 动态排队，终端显示统一 dashboard。它不会改变科学合同。

## 3. 一条命令运行

如果本机有和该合同完全一致的已完成 baseline，可导入它，避免重复训练 baseline。`OLD_RUN_DIR` 必须是旧的 raw `runs/.../<时间戳>_seed13` 目录，不能是 Git 中的 `reports/` 目录；导入后两个 QK selector 均从 batch 0 开始。

```bash
bash scripts/run_retacred_qk_coherent_transport_formal_single_seed.sh \
  --gpu auto \
  --hardware-profile adaptive \
  --import-baseline-from "OLD_RUN_DIR"
```

若没有可验证的旧 baseline，删除最后一行，让脚本按冻结合同训练 baseline：

```bash
bash scripts/run_retacred_qk_coherent_transport_formal_single_seed.sh \
  --gpu auto \
  --hardware-profile adaptive
```

`adaptive` 从逻辑 batch 256、无 physical micro-batch、all-pairs chunk 起步；CUDA OOM 后从最新 batch checkpoint 原位重试，先逐级将 pair chunk 减半，再引入 micro-batch。它不跳 batch、不换随机顺序、不降低逻辑 batch 或改变优化目标。`RUN_COMPLETE`、`run_summary.json` 和 `run_summary.md` 都存在后，脚本才会自动运行审计导出器。

发生 SIGINT/SIGTERM/SIGHUP 时，脚本会在当前 optimizer update 后安全暂停，写入 batch checkpoint 与 `RUN_PAUSED`，并返回退出码 75；这不是失败。

## 4. 查看或恢复

原始 run 只留在合作者机器，禁止提交。统一 dashboard 已显示阶段、batch、速度、ETA、显存和各 GPU worker；需要只读查看或定位恢复参数时运行：

```bash
bash scripts/check_retacred_qk_coherent_transport_formal_single_seed.sh \
  --run-dir "runs/retacred_qk_coherent_transport_formal_single_seed/<时间戳>_seed13" \
  --gpus auto --device cuda --hardware-profile adaptive
```

正常恢复：

```bash
bash scripts/run_retacred_qk_coherent_transport_formal_single_seed.sh \
  --gpu auto --hardware-profile adaptive \
  --resume "runs/retacred_qk_coherent_transport_formal_single_seed/<时间戳>_seed13"
```

若是明确从单 GPU 改为多 GPU 的 selector 并行恢复，在确认诊断建议后才加 `--allow-gpu-topology-change`。仅发布后的执行层代码更新才可加 `--allow-code-update`；任何科学合同差异都不能靠该参数绕过。

## 5. 导出并提交允许的报告

正常完成时运行器已自动导出。若自动导出因 `git` 状态停止，在 raw run 完成且工作树干净后手工执行：

```bash
bash scripts/export_retacred_qk_coherent_transport_formal_single_seed_report.sh \
  --run-dir "runs/retacred_qk_coherent_transport_formal_single_seed/<时间戳>_seed13"
```

默认报告目录是 `reports/retacred_qk_coherent_transport_formal_single_seed/<同一时间戳>_seed13/`。导出器会拒绝非 `1.1`、脏工作树、未包含 `origin/1.1` 与 `origin/main`、不完整 marker/metrics 或私有产物。它仅允许报告摘要、配置、metrics、GPU 分配、数据计数/hash 与 provenance。

自动导出会提交并 push；若使用 `--no-commit` 或需人工提交，确认只暂存报告目录：

```bash
git status --short
git diff --cached --check
git diff --cached --name-only
git commit -m "Add QK coherent-transport Re-TACRED formal single-seed report"
git push origin 1.1
```

绝不提交 `data/`、`runs/`、checkpoint、权重、预测、JSONL 或完整日志。报告提交后停止并通知负责人审计；即使 single-seed 指标看起来为正，也不能自行运行多 seed 或声称任务增益、量子特异优势、有限-shot 稳健性或硬件速度优势。
