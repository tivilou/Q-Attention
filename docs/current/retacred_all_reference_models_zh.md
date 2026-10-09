# Re-TACRED 全套 reference：合作者一键运行

## 默认指令（发布到 main 后执行）

```bash
bash scripts/run_retacred_all_references.sh --gpu auto --models all --seeds 13
```

这次执行十五个配置：MetaEntail、Sequence Routing H3、CTL-DRP supervised / unsupervised、2025 typed-marker 重建、syntax curriculum、四种 Improved marker、正常 PTR、LUKE CoRE / CFIE / resample / focal。它们不是十五个独立 SOTA 架构，也不是已经复现出论文分数的全球 Top15。原历史候选中的 PTR Reversed 改变任务方向，已排除；同时保留普通 entity-marker 锚点。历史 strict 审计记录不改写，新入口按锁定的实际实现和解释性配方运行。

## 1. 同步代码：main → 合作者 1.1

在自己项目根目录中（不依赖开发服务器的绝对路径）：

```bash
git fetch origin --prune
git switch 1.1
git merge --no-edit origin/1.1
git merge --no-edit origin/main
git status --short --branch
git merge-base --is-ancestor origin/1.1 HEAD
git merge-base --is-ancestor origin/main HEAD
```

先整合返回结果分支再整合 main。两条 ancestor 命令退出码均应为 0。先保存自己的修改，不混入本次报告提交。

## 2. 环境、GPU、文件交换

使用已有 CUDA PyTorch 环境；不替换 Torch/CUDA。允许用 `PYTHON_BIN` 指定自己的 Python。安装其余锁定依赖：

```bash
python -m pip install -r baselines/retacred/requirements-reference-complete.txt
nvidia-smi
bash scripts/check_retacred_all_references.sh --gpu auto --models all --seeds 13
```

需要完整 canonical 数据 `data/relation/retacred/{train,valid,test}.jsonl`，记录数为 58465 / 19584 / 13418。需至少 12 GiB 进程可用 RAM。每张 GPU 默认仅一个重进程，建议至少 12 GiB 显存，物理 micro-batch 默认 1；逻辑 batch 和优化器不因 GPU 数量变化。FP32 和激活 checkpointing 降低显存峰值，不承诺所有 GPU 都能承受任意最长输入。

默认十五配置单 seed 的保守空闲磁盘预检为 **352 GiB**：包含权重下载及分片、完整 Adam batch checkpoint 的临时写入/保留、best 和完整数值 trace，curriculum 还保留三个 reviewer。实际占用依运行而定。多 seed 按配置线性增加，避免运行很久后才磁盘写满。

配置项目提供的 `PROJECT_EXCHANGE_URL`（18084 的 HTTPS 地址）和 `PROJECT_EXCHANGE_TOKEN`。证书由代码带入，禁止关闭 TLS 校验。缺失的哈希锁定源码、RoBERTa / MNLI / LUKE / SimCSE 权重、依存模型会自动从服务下载，逐分片及拼装后校验大小/SHA256；无需上传 Git。已有缓存先校验再复用。

## 3. 实际训练、记录和默认发布

默认上述一条 runner 完成预检 → 数据转换/资格校验 → 资产获取 → 独立 `(model, seed)` GPU 队列 → 完整 dev/test 评价 → 原始张量上传 18084 → 白名单报告导出 → commit → push `1.1`。不另敲导出命令。`--prepare-only` 只准备，`--dry-run` 只列配方，`--no-push` 仍提交报告但不推送。

保存 micro precision / recall / F1、accuracy、macro-F1、逐类 P/R/F1/support、confusion matrix、loss 和 loss_scope。CoRE 的 loss 是 factual 分支 CE，不是修正分数的 NLL；MetaEntail 的 loss 是预测后统计的 NLI pair CE，不是关系 NLL；弃答保留在全样本分母和独立预测列中。不把缺值伪造为 0，也不把不同 loss 当成同一口径排名。

所有解释性补齐见每个报告中的 recipe/adaptations：CFIE 是明确的 LUKE 移植重建；curriculum 是 train-only own-subset cross-review + 真实 dependency GAT + bilinear；H3 保留源三段路由但使用 canonical40 和共同编码器；marker 统一 FP32。不存在对未知作者参数或论文分数的保证。

所选 train/test 样本记录原句、tokens、两个实体及位置/类型、预测与预测后的 gold 对照、真实 embedding / attention / routing / dependency / logits，训练阶段还记录反向梯度与分类头更新前后值。所有实际观测数值保存在私有 `.pt`，不是 shape/hash 的占位。NLI 会记录所选样本全部可行候选和组级最终关系/弃答。服务接收数值和原始 trace；Git 只接收紧凑哈希、完整指标、data.sha256、data_counts.txt、run_summary.data 和白名单报告。尚未把新结果自动导入门户的动作算作本次 runner 工作。

## 4. 中止、恢复、只补导出

原始路径为 `runs/retacred_reference_complete/<启动时间戳>/`，每个配置/seed 有独立 checkpoint。Ctrl-C 在当前逻辑 batch 保存后停下；强杀会回到最近已提交 batch。已完成任务校验后跳过；未完成恢复模型、优化器、调度器、RNG 和 batch 游标。

```bash
bash scripts/run_retacred_all_references.sh --gpu auto --models all --seeds 13 \
  --resume runs/retacred_reference_complete/<原启动时间戳>
```

单 GPU ↔ 多 GPU 只改变任务分配；固定物理 micro、源码、资产、数据和配方保持不变。不把此恢复机制误认为同一重模型的 DDP 切分。出现 OOM 保留 checkpoint 并停止，禁止静默改变科学配方。新实现的长输入超过512时明确停止而非丢弃样本；marker/PTR 保留源代码的512上限和截断规则，但缺实体标记/非法prompt位置会停止。不同 overflow 口径在 recipe 中明确披露；每个 worker 先编码全 split 再训练。

训练完成后若上传/推送失败，只执行：

```bash
bash scripts/run_retacred_all_references.sh --models all --seeds 13 \
  --resume runs/retacred_reference_complete/<原启动时间戳> --export-only
```

此路径不加载模型、不训练、不重评价。复用匹配的上传回执/已有报告/已有提交，然后补推送；不覆盖不同身份的报告。

## 5. 完成检查与后续决定

终端应显示 `exported, committed and pushed`。报告位于 `reports/retacred_reference_suite/<同一启动时间戳>/`。检查：

```bash
git status --short --branch
git diff --cached --name-only
git log --oneline origin/1.1..HEAD
git show --stat --oneline HEAD
git fetch origin 1.1
git merge-base --is-ancestor HEAD origin/1.1
```

最后退出码 0 才确认远端包含 HEAD。训练完成不等于上传或 push 已成功。仅 seed13 完整结果审核后再决定追加 29/53。不得私改数据、label 方向、配方、筛选测试集或从非 baseline 同口径结果宣称量子优势。
