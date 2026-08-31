# Q-RPEC Re-TACRED 正式单 seed 实验

这是完整 Re-TACRED 的 seed 13 正式实验。Q-RPEC candidate 使用 label-free 的 query、key 和 subject/object relation anchor 做对称三点关系曲率回声；`classical_local_echo` 使用相同局部编码与参数但移除 relation-key 受控相位，`disabled` 是基线。完整数据正式运行必须由合作者在 `1.1` 分支执行。

## 1. 同步并执行

在项目根目录执行。下面的默认命令会检查分支、数据、环境和测试，运行 baseline 及三个 selector，写入摘要，并在完成标记生成后自动导出报告、提交并推送到 `origin/1.1`。不要在标准命令中加入 `--report-dir`。

```bash
git fetch origin --prune
git switch 1.1
git merge origin/1.1
git merge origin/main
git status --short
bash scripts/run_retacred_qrpec_formal_single_seed.sh --gpu 0
```

也可以让脚本自动发现可用 GPU，并按启动时的总显存和空闲显存选择执行 profile：

```bash
bash scripts/run_retacred_qrpec_formal_single_seed.sh --gpu auto
```

`auto` 会选择空闲显存至少 8 GiB 的可见物理 GPU；单 GPU 环境也适用。默认 profile 是 `adaptive`：每个 selector 先试探 10 倍于原最高档的 `adaptive_max`（`pair_chunk_size=163840`），这是吞吐优先的激进试探档；只有该 selector 确认 CUDA OOM 或显存压力仍未解除，才独立回退到 `adaptive_fast=16384 -> 4096 -> 1024 -> 256 -> 64`，并在需要时启用 checkpointing。OOM worker 会退出，父调度器释放 GPU 后用该 selector 的最近 batch checkpoint 重启；回退过程写入 `adaptive_memory_state.json` 与 `scheduler_events.jsonl`。一个 selector 降档不会降低其他 selector 的初始档位，因此不会无故牺牲它们的速度。没有 checkpoint 或同一 selector 的所有档位都失败时才写 `RUN_FAILED`。这不是把两张 GPU 的显存合并。该机制只调整执行策略，不改变 seed、数据、batch size、epoch、selector、控制组或优化契约；不恢复旧的无限 autograd graph retaining 实现。固定 `low_memory`/`balanced`/`high_memory` profile 不会自动回退。当前 activation checkpointing 字段保留在执行契约中，Q-RPEC 的核心显存保护来自 streamed backward。实际 GPU、显存、profile 和生效参数会写入 `run_summary.json` 与 `gpu_assignments.json`，供审计复核。

每个 CUDA selector worker 还会在完成 optimizer 更新后的安全边界按 20 个 batch 轮询显存压力。压力触发时只执行本进程的 `gc.collect()` 和 `torch.cuda.empty_cache()`：前者回收不可达 Python 对象，后者只归还本 worker PyTorch allocator 的闲置缓存；不会删除仍在计算图中的活跃 Tensor，也不会触碰其他 CUDA 进程。每次回收会输出 `memory_pressure_reclaim` 事件及回收前后 `free/allocated/reserved` 诊断。如果清理后仍低于保留阈值，自适应 worker 会先保存当前 batch checkpoint，再以退出码 `87` 退出，由父调度器仅降低该 selector 的下一档并从 checkpoint 继续；固定 profile 则记录事件后继续运行。自适应重试写入 `memory_pressure_event.json`、`adaptive_memory_state.json` 和 `scheduler_events.jsonl`；固定 profile 只输出回收事件，不会擅自改变执行档位。因此该机制不能释放活跃模型状态导致的真实峰值显存，仍需依靠 streamed backward、pair chunking 和必要的 checkpointing 降低峰值。

脚本在创建 raw run 之前、以及 baseline 完成准备 selector 之前各检查一次显存。显式 `--gpu 0` 也必须至少有 8 GiB 空闲；若某个 GPU 被其他 PID 占用，脚本会列出 `nvidia-smi` 的进程并停止，不会把这次失败写成实验结果。不要强制杀掉不属于本实验的进程；确认占用进程可以停止后再重跑同一命令。

默认单 GPU 路径会使用同一套调度器串行运行三个 selector。若有多张已验证的 GPU，可以在同一 seed 内并行运行相互独立的 selector worker，例如：

```bash
bash scripts/run_retacred_qrpec_formal_single_seed.sh --gpu 0,1
```

多 GPU 模式先训练一次共享 baseline，再按空闲 GPU 动态分配 `q_rpec` 与 `classical_local_echo`。每个 worker 独立加载同一 baseline checkpoint；这不是 DDP，也不会把两张卡的显存合并成一张卡。adaptive 模式只重试确认 OOM 且有 batch checkpoint 的 worker，其他 selector 可以继续运行；普通非零退出仍立即失败。GPU ID 必须唯一且在 `nvidia-smi` 中可用。

`git status --short` 在启动前必须没有输出。脚本在启动时记录一个 UTC 时间戳 `YYYYMMDDTHHMMSSZ`，raw run 默认写入：

```text
runs/retacred_qrpec_formal_single_seed/<timestamp>_seed13/
```

脚本从当前环境解析 Python：优先使用 `PYTHON_BIN`，否则依次查找 `python`、`python3`；不得写入或假定负责人服务器的解释器路径。默认报告目录由同一个启动时间戳确定：

```text
reports/retacred_qrpec_formal_single_seed/<timestamp>_seed13/
```

运行期间不要改动 seed、数据、selector、epoch、batch size、控制组或代码，也不要并行启动第二个完整 run。

### 安全暂停与 batch 级恢复

训练默认每隔 `--checkpoint-every-batches` 个 batch，在完整 `optimizer.step()` 后原子写入 checkpoint，并保存 optimizer、RNG、epoch permutation 和 next-batch cursor。按 `Ctrl+C` 或向 runner 发送 `SIGTERM`/`SIGHUP` 会请求安全暂停；当前更新完成后写入 `RUN_PAUSED` 并返回退出码 `75`。不要使用 `kill -9`，否则无法保证最后一个更新有 checkpoint。

暂停后继续同一 run：

```bash
bash scripts/run_retacred_qrpec_formal_single_seed.sh --gpu 0 --resume runs/retacred_qrpec_formal_single_seed/<timestamp>_seed13
```

恢复会复用原 run 的 `data/*.jsonl` 和 `data/data_manifest.json`，跳过已有有效指标的 baseline/selector，并仅从兼容的 batch checkpoint 继续。恢复命令必须使用原 run 相同的物理 GPU 列表、并行模式和显存 profile；adaptive run 还会读取 `adaptive_memory_state.json`，从上次已验证的 tier 继续，不会重新回到最快档。若最初使用 `--gpu auto`，请从 `run_summary.json` 取出已解析的 GPU 与 profile，并显式传入，例如 `--gpu 0,1 --hardware-profile adaptive`。`--resume` 不能与 `--output-dir` 同时使用；恢复期间不能修改算法参数、数据、代码、seed、batch size、epoch、学习率、selector、显存 profile 或 checkpoint 契约。原始 `RUN_PAUSED` 保留完整 worker 状态，每次恢复会额外写入 `resume_state.json`；wrapper 的恢复日志写入新的 `logs/run.resume-<timestamp>.log`，不会覆盖首次日志。旧目录若没有 `run_manifest.json`、`data_manifest.json` 或 batch checkpoint，不能宣称 batch 级恢复，只能新建目录从 baseline 级重新开始。

如果旧 run 是单 GPU selector-parallel，暂停后希望改用多张 GPU 并行剩余 selector，必须显式授权执行层拓扑迁移，并保持其它训练合同不变：

```bash
bash scripts/run_retacred_qrpec_formal_single_seed.sh \
  --gpu 0,1 \
  --hardware-profile adaptive \
  --resume runs/retacred_qrpec_formal_single_seed/<timestamp>_seed13 \
  --allow-gpu-topology-change
```

该迁移只支持“原来恰好 1 张 GPU、现在至少 2 张 GPU”的 selector-parallel 恢复；baseline 和每个未完成 selector 仍从各自最近的 batch checkpoint 恢复，已完成 selector 按指标文件跳过。它不合并显存、不改变模型并行布局，也不允许 `--model-parallel-gpus`、新建 run 或改变 seed、数据、batch size、epoch、学习率、selector、控制组和科学训练参数。脚本会在 `resume_state.json` 和 `scheduler_events.jsonl` 记录 `elastic_gpu_topology_change`，普通不兼容变更仍会拒绝恢复。

### 旧版已完成 baseline 的迁移

如果师弟运行的是没有新 manifest/checkpoint 机制的旧版脚本，且旧 run 的 baseline 已完成、当前停在 `q_triad`，不要把旧目录作为 `--resume` 目录。`--resume` 会严格校验新代码指纹，拒绝不兼容的旧 run；请新建一个 run，只显式导入旧 baseline：

```bash
bash scripts/run_retacred_qrpec_formal_single_seed.sh \
  --gpu 0 \
  --import-baseline-from runs/retacred_qrpec_formal_single_seed/<旧时间戳>_seed13
```

该命令仍会按当前 frozen contract 重新 materialize 数据，并逐项核对旧 run 的 `baseline/model.pt`、`metrics.json`、`vocab.json`、`labels.json`、三份 `data/*.jsonl` 的完整性、SHA-256、seed 13、baseline/model 超参和模型结构。校验失败会停止，不会把旧结果当成可恢复状态。通过后只复制四个已完成 baseline 文件到新 run；旧 `selectors/`、旧 selector 中间 checkpoint 和日志一律不复制。新 run 会写 `baseline_import.json`，并让 `disabled`、`q_rpec`、`classical_local_echo` 按新代码从 batch 0 重新开始，因此不能继承旧版 selector 已经跑过的部分，但可以跳过 baseline 训练。

导入后的新 run 是独立 run，后续暂停请使用新目录的 `--resume <新目录>`；不要再次对同一目录使用 `--import-baseline-from`，也不要同时指定 `--resume` 和 `--import-baseline-from`。

### GPU 使用边界

Q-RPEC 正式交接只支持串行或 selector-parallel。`--model-parallel-gpus` 明确拒绝，因为当前 Q-RPEC kernel 没有模型级跨卡布局实现；多卡请使用 `--gpu 0,1` 的 selector-parallel 模式。selector-parallel 只并行独立 selector，不合并显存，也不改变单个 selector 的峰值显存需求。

## 2. 显存与并行说明

Q-RPEC 的 attention hook 会把 query-key 对分块计算，避免一次性物化 `batch x query_tokens x key_tokens` 的三寄存器输入；训练时可按执行 profile 使用 activation checkpointing，避免为每个 chunk 保留完整 autograd 图。`kernel.pair_chunk_size` 是已发布配置的一部分，adaptive 模式只在确认 OOM/显存压力后降低它。多 GPU 只降低每张卡承载的 selector 数量，不改变单个 selector 的峰值显存；因此每张卡仍必须满足单个 selector 的显存要求。

运行摘要会记录 `gpu_assignments.json`、请求和解析到的 GPU ID、每个 worker 的 PID/状态/耗时、`pair_chunk_size` 以及 CUDA 设备信息。先用小规模 canary 验证 GPU 拓扑和显存，再运行完整正式实验；不得用降低 batch 或改变 selector 的临时命令冒充正式结果。

运行期间主终端会输出可读的 baseline 进度和每 30 秒一份 selector 面板。baseline 显示 phase、epoch、batch、速率、ETA、当前/峰值进程显存和每轮 valid 指标；selector 面板按物理 GPU 显示当前 selector、phase、epoch/batch、速率、ETA、整卡已用/空闲显存以及该 worker 的 allocated/reserved/peak 显存，完成、排队和失败的 selector 也会单独列出。显存采样由 `nvidia-smi` 和 PyTorch allocator 提供，batch 进度最多每 5 秒查询一次，采样失败不会阻断训练。worker 的原始 JSON 进度仍保存在各自的 `selectors/<selector>/worker.log`，baseline 原始输出保存在 `baseline_train.log`，调度器事件另存为 raw run 根目录的 `scheduler_events.jsonl`，便于审计。终端被重定向或通过 `tee` 保存时仍使用普通追加文本，不依赖 ANSI 光标控制。

## 3. 完成检查

运行成功后 raw run 必须包含：

```text
RUN_COMPLETE
run_summary.json
run_summary.md
baseline/metrics.json
selectors/q_rpec/metrics.json
selectors/classical_local_echo/metrics.json
```

摘要必须声明 seed 13、完整数据计数 `58465/19584/13418`，且 test 未用于训练或 valid checkpoint 选择。报告 exporter 还会检查 branch ancestry、clean tree、三份数据行数、完整 selector 指标、provenance 和私有文件禁带。

## 4. 报告提交

一键脚本已经调用 exporter。exporter 只复制 `reports/retacred_qrpec_formal_single_seed/<timestamp>_seed13/` 下的审计子集，并自动执行 `git add`、`git diff --cached --check`、commit 和 `git push origin 1.1`。不得提交 `runs/`、`data/`、checkpoint、预测、JSONL 或完整日志。

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
