# Q-CFRA Re-TACRED 正式单 seed 交付

本任务由合作者执行完整数据 seed13。我们只做实现、审查、小规模 canary 和结果审计；全量尚未运行。五 seed toy 已通过预定晋级门槛，平均 paired Macro-F1 增量 +0.005149，4/5 为正，不能替代正式结果。

## 同步分支和环境

在干净工作区执行，先整合远端 `1.1`，再合入发布的 `main`；冲突、脏文件或预检失败时停止并返回诊断，不改实验配方。

```bash
git fetch origin --prune
git switch 1.1
git merge origin/1.1
git merge origin/main
git status --short
git merge-base --is-ancestor origin/1.1 HEAD
git merge-base --is-ancestor origin/main HEAD
python --version
python -c 'import torch,numpy; print(torch.__version__,numpy.__version__,torch.cuda.is_available())'
nvidia-smi
```

需要 Linux、Python 3.10+、PyTorch 2.7.1 和 NumPy 1.26.4，以及兼容 CUDA。保留原 Git 身份，不加 AI 作者或协作者信息。脚本按 `PYTHON_BIN`、`python`、`python3` 顺序选择解释器；可先 `export PYTHON_BIN=/your/env/bin/python`。私有证据使用项目既有 HTTPS 交换服务；在执行环境安全设置 `PROJECT_EXCHANGE_URL` 和 `PROJECT_EXCHANGE_TOKEN`，不得写入 Git、聊天或报告。

## 预检和一键执行

数据默认位于 `data/relation/retacred/{train,valid,test}.jsonl`。预检逐行检查全部数据、实体范围和 SHA256，数量必须为 58465/19584/13418。全部记录参与对应阶段，不裁剪为 toy，不截短句子。最长96 tokens；训练40类，测试39类有支持，仍按固定40类计算 Macro-F1。词表、标签映射和中性实体表示只从训练集构造。

```bash
bash scripts/check_retacred_q_cfra_formal_single_seed.sh --gpu auto
bash scripts/run_retacred_q_cfra_formal_single_seed.sh --gpu auto
```

`--gpu auto` 只用可见且空闲显存至少8GiB的物理 GPU；也可指定 `--gpu 0,2`。先训练一次共享 warm-start，再由先空闲的 GPU 接下一个条件，不采用 DDP。最低空闲显存决定物理 micro-batch：8–12GiB用1，12–24GiB用2，24GiB起用4；逻辑 batch 始终16。梯度累积按样本数和有效 attention query 数分别加权，不改变科学预算。没有符合条件的 GPU 则停止。

正式启动前，程序自动在执行者机器完成合成12/6/6样本的真实父子进程 canary：探测全部选中 GPU 的复数反传，在第一块 GPU 验证 fresh-process 恢复和导出失败后重试。canary 失败就不建正式运行目录，不进入全量训练。本项目开发端的 CPU canary 不等于 GPU 验证。

固定配方见 `configs/retacred_q_cfra_formal_single_seed.json`：seed13；dim32、2层、4 heads、FF64、dropout0；warm-start8 epochs，所有7条件各6 epochs，Adam lr0.001、逻辑 batch16。条件为 Disabled、Quantum、classical_cosine、classical_rbf、diagonal、shuffled、entity_permuted。共享 warm-start 的验证集最优权重初始化各条件，冻结 encoder、训练 classifier；除 Disabled 外增加150个名义参数。第二层 attention 替换和三视图 CE/预测JS/attentionJS 使用已审计 toy 的同一实现，JS系数均0.05。经典对照仅名义参数预算匹配，存在冗余仿射方向，不能据此声称严格量子归因。

## 暂停、恢复和故障

运行保存在 `runs/retacred_q_cfra_formal_single_seed/<启动时间>_seed13/`，每30秒打印调度摘要，逐作业保存 heartbeat 和日志。每50个完整优化步、120秒、epoch边界和安全暂停保存模型、Adam、RNG、数据排列及游标；保留 latest 和上一代，best 单独嵌入提交的状态。验证只选 best，测试不参与训练或选择。Ctrl-C/SIGTERM 请求当前完整优化步后暂停，退出码85表示暂停。

```bash
bash scripts/run_retacred_q_cfra_formal_single_seed.sh --gpu auto --resume runs/retacred_q_cfra_formal_single_seed/原启动时间_seed13
```

恢复必须使用同一源码、配置和数据，沿用原 micro-batch；已完成作业保持原完成标记。latest 损坏默认停止。只有确认损坏且接受回滚到上一代提交时，才显式加 `--recover-previous quantum`（或对应作业名）；回滚有审计记录，未提交工作可能重算。OOM 停止并保留最近提交，不自动减逻辑 batch、减长度、改 epoch 或换条件；返回 OOM 记录和日志中的非敏感诊断，由我们处理。

已完成作业的 latest 损坏也会被检测。显式修复要求已有指标、案例和 best 身份完整，且上一代已经完成全部优化步和验证选择；仅重放收尾，不执行新的优化步。程序保存旧完成标记和恢复依据，要求指标、案例及最终 summary 完全不变，其他作业完成标记不变；拒绝覆盖不同身份的公开报告。上一代尚未完成预算时停止交由我们审计。回滚已生效后再次重试使用普通 `--resume`，不要重复请求回滚。

## 完整案例和报告

固定各 split 官方记录0，在 initial/best/final 做重放，并记录第一个真实训练更新；每个作业10条完整数值 trace。原始文本、预测、参数、梯度和全部中间 NPY 私有保存，任意坐标可通过 `q_attention.analysis.ecpa_trace.read_values` 读取，内容、维度、计数、完整性和 SHA256 均受审计。截图或缩略值不能替代完整私有案例。

成功后自动重新审计所有8作业、上传完整私有 trace，默认导出到 `reports/retacred_q_cfra_formal_single_seed/<同一启动时间>_seed13/` 并提交、推送至 `1.1`。公开文件仅为 `run_summary.{json,data,md}`、`run_config.json`、`data.sha256`、`data_counts.txt`、去路径的 `run_manifest.json`、私有交换收据 `exchange_manifest.json`、8作业案例索引 `case_study.json`、`REPORT_COMPLETE.json`。不上传原始 runs、数据、checkpoint、预测或日志到 Git。

导出或推送失败后只重试原运行，不重训：

```bash
bash scripts/run_retacred_q_cfra_formal_single_seed.sh --export-only --resume runs/retacred_q_cfra_formal_single_seed/原启动时间_seed13
```

网络上传最多3次重试，已验证分片可复用；失败保留私有状态，清除临时公开目录，拒绝覆盖不同身份的已有报告。若希望人工检查后推送，在正常运行或导出命令加 `--no-push`，随后只提交对应报告目录：

```bash
git status --short
git add -- reports/retacred_q_cfra_formal_single_seed/原启动时间_seed13
git diff --cached --check
git diff --cached --name-only
git diff --cached --stat
git commit -m "reports: Q-CFRA formal seed13"
git push origin HEAD:refs/heads/1.1
```

存在任何无关改动或未发布的无关提交时，停止报告发布，不强推、不清理他人文件。报告返回后由我们审计；执行成功不等于科学门槛通过。Quantum 相对 Disabled 的正式主指标正增量且 intervention/leakage 验证通过，最多成为单 seed L1 候选；经典 QI 候选要求相对增益严格大于1%。是否追加多 seed 必须等待我们的审计决定，不自动启动。新颖性、严格量子归因和硬件优势仍未确立。
