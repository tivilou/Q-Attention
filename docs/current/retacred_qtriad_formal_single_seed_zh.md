# Q-TRIAD Re-TACRED 正式单 seed 实验

这是完整 Re-TACRED 的 seed 13 正式实验。Q-TRIAD candidate 使用 label-free 的 query、key 和 subject/object relation anchor；`classical_density_tensor` 是参数和输入匹配的经典密度控制，`quantum_product` 是无三方纠缠的量子 product control，`disabled` 是基线。完整数据正式运行必须由合作者在 `1.1` 分支执行。

## 1. 同步并执行

在项目根目录执行。下面的默认命令会检查分支、数据、环境和测试，运行 baseline 及三个 selector，写入摘要，并在完成标记生成后自动导出报告、提交并推送到 `origin/1.1`。不要在标准命令中加入 `--report-dir`。

```bash
git fetch origin --prune
git switch 1.1
git merge origin/1.1
git merge origin/main
git status --short
bash scripts/run_retacred_qtriad_formal_single_seed.sh --gpu 0
```

也可以让脚本自动发现可用 GPU，并按启动时的总显存和空闲显存选择执行 profile：

```bash
bash scripts/run_retacred_qtriad_formal_single_seed.sh --gpu auto
```

`auto` 会选择空闲显存至少 8 GiB 的可见物理 GPU；单 GPU 环境也适用。自动 profile 只调整显存执行策略，不改变 seed、数据、epoch、batch size、selector 或控制组：低显存（总显存小于 16 GiB，或空闲显存小于 12 GiB）使用 `pair_chunk_size=64` 并启用 activation checkpointing；中等和高显存使用 `pair_chunk_size=256` 并启用 checkpointing。Q-TRIAD 的训练反向会对 pair chunk 逐块重算并累积梯度，不能通过关闭 checkpointing 换取速度；这样可避免把所有 query-key chunk 的计算图长期保留在显存中。实际 GPU、显存、profile 和生效参数会写入 `run_summary.json` 与 `gpu_assignments.json`，供审计复核。

脚本在创建 raw run 之前、以及 baseline 完成准备 selector 之前各检查一次显存。显式 `--gpu 0` 也必须至少有 8 GiB 空闲；若某个 GPU 被其他 PID 占用，脚本会列出 `nvidia-smi` 的进程并停止，不会把这次失败写成实验结果。不要强制杀掉不属于本实验的进程；确认占用进程可以停止后再重跑同一命令。

脚本在创建 raw run 之前、以及 baseline 完成准备 selector 之前各检查一次显存。显式 `--gpu 0` 也必须至少有 8 GiB 空闲；若某个 GPU 被其他 PID 占用，脚本会列出 `nvidia-smi` 的进程并停止，不会把这次失败写成实验结果。不要强制杀掉不属于本实验的进程；确认占用进程可以停止后再重跑同一命令。

默认单 GPU 路径会使用同一套调度器串行运行三个 selector。若有多张已验证的 GPU，可以在同一 seed 内并行运行相互独立的 selector worker，例如：

```bash
bash scripts/run_retacred_qtriad_formal_single_seed.sh --gpu 0,1
```

多 GPU 模式先训练一次共享 baseline，再按空闲 GPU 动态分配 `q_triad`、`classical_density_tensor` 和 `quantum_product`。每个 worker 独立加载同一 baseline checkpoint；这不是 DDP，也不会把两张卡的显存合并成一张卡。任一 worker 失败时，调度器会终止其余 worker、写入 `RUN_FAILED`，且不会写 `RUN_COMPLETE`。GPU ID 必须唯一且在 `nvidia-smi` 中可用。

`git status --short` 在启动前必须没有输出。脚本在启动时记录一个 UTC 时间戳 `YYYYMMDDTHHMMSSZ`，raw run 默认写入：

```text
runs/retacred_qtriad_formal_single_seed/<timestamp>_seed13/
```

脚本从当前环境解析 Python：优先使用 `PYTHON_BIN`，否则依次查找 `python`、`python3`；不得写入或假定负责人服务器的解释器路径。默认报告目录由同一个启动时间戳确定：

```text
reports/retacred_qtriad_formal_single_seed/<timestamp>_seed13/
```

运行期间不要改动 seed、数据、selector、epoch、batch size、控制组或代码，也不要并行启动第二个完整 run。

### 可选的模型级并行

当单个模型阶段需要跨卡放置时，可以显式启用按完整 encoder layer 切分的模型级并行：

```bash
bash scripts/run_retacred_qtriad_formal_single_seed.sh --model-parallel-gpus 0,1
```

该选项要求至少两张可见 GPU。embedding 和前部 layer 放在首卡，后部 layer 与 classifier 放在末卡，中间 hidden state 跨卡传输；Q-TRIAD 每层子核跟随对应 attention layer。它与独立 selector-parallel 不同，启用后 selector 按序运行，不会为每个 selector 复制整套模型。摘要会记录物理 GPU ID、进程内 CUDA 映射和模块 device map。

模型级并行首先用于验证显存可行性；本模型层间依赖和跨卡通信可能抵消并行收益，不能预先宣称提速。合作者必须先做双 GPU canary，记录单卡/双卡吞吐、峰值显存、数值等价、checkpoint 恢复和退出后的残留进程，再决定是否用于完整 formal run。双 GPU canary 也不能替代 seed-13 正式实验的单 seed 门禁。

## 2. 显存与并行说明

Q-TRIAD 的 attention hook 会把 query-key 对分块计算，避免一次性物化 `batch x query_tokens x key_tokens` 的 statevector 输入；训练前向只保留最终 score 张量，反向传播时逐块重算 statevector 并累积梯度，不会为每个 chunk 保留完整 autograd 图。`kernel.pair_chunk_size` 是已发布配置的一部分，不能在正式运行中临时修改。多 GPU 只降低每张卡承载的 selector 数量，不改变单个 selector 的峰值显存；因此每张卡仍必须满足单个 selector 的显存要求。

运行摘要会记录 `gpu_assignments.json`、请求和解析到的 GPU ID、每个 worker 的 PID/状态/耗时、`pair_chunk_size` 以及 CUDA 设备信息。先用小规模 canary 验证 GPU 拓扑和显存，再运行完整正式实验；不得用降低 batch 或改变 selector 的临时命令冒充正式结果。

运行期间主终端会每 30 秒输出一份可读的 selector 面板，按 GPU 显示当前 selector、phase、epoch/batch、速率和 ETA；完成、排队和失败的 selector 也会单独列出。worker 的原始 JSON 进度仍保存在各自的 `selectors/<selector>/worker.log`，调度器事件另存为 raw run 根目录的 `scheduler_events.jsonl`，便于审计。终端被重定向或通过 `tee` 保存时仍使用普通追加文本，不依赖 ANSI 光标控制。

## 3. 完成检查

运行成功后 raw run 必须包含：

```text
RUN_COMPLETE
run_summary.json
run_summary.md
baseline/metrics.json
selectors/q_triad/metrics.json
selectors/classical_density_tensor/metrics.json
selectors/quantum_product/metrics.json
```

摘要必须声明 seed 13、完整数据计数 `58465/19584/13418`，且 test 未用于训练或 valid checkpoint 选择。报告 exporter 还会检查 branch ancestry、clean tree、三份数据行数、完整 selector 指标、provenance 和私有文件禁带。

## 4. 报告提交

一键脚本已经调用 exporter。exporter 只复制 `reports/retacred_qtriad_formal_single_seed/<timestamp>_seed13/` 下的审计子集，并自动执行 `git add`、`git diff --cached --check`、commit 和 `git push origin 1.1`。不得提交 `runs/`、`data/`、checkpoint、预测、JSONL 或完整日志。

完成后检查提交结果：

```bash
git status --short
git diff --cached --name-only
git log -1 --oneline
git ls-remote --heads origin 1.1
```

正常情况下前两条没有输出，最后两条应显示本次报告提交和更新后的 `origin/1.1`。如果自动 push 失败，保留 raw run 和已生成的报告目录，先记录错误；不要重跑、改写报告、强推或切换到 `main`。修复网络或分支同步后，只能在 clean `1.1` 上重新执行 exporter。

只有在需要审计而不提交时，才使用 exporter 的 `--no-commit`；显式 `--report-dir` 仅用于经负责人确认的例外目录，不是标准路径。

## 5. 停止门禁

负责人审计报告中的 valid/test 指标、控制组、数据 hash、provenance、test isolation 和 staged 文件后，才形成项目结论。单 seed 结果未通过预声明的 candidate-vs-disabled 实用增益门禁或 matched classical comparator 门禁时，立即停止；即使中间日志看起来有提升，也不得启动 multi-seed。只有负责人完成审计并明确授权后，才会发布后续 multi-seed handoff。
