# Re-TACRED LUKE / MetaEntail 一键运行预检

## 一键入口

在项目根目录、已激活实验 Python 环境后运行：

```bash
bash scripts/check_retacred_reference_runtime.sh --upload
```

此入口只做小规模运行诊断，不读完整数据集、不启动全量实验、不导出正式指标，也不执行 Git commit/push。当前是开发分支准备内容；同步到 `main` 并完成正式交接后，合作者再从更新后的代码执行。

已有的 `PROJECT_EXCHANGE_URL` 和 `PROJECT_EXCHANGE_TOKEN` 用于缺失权重的下载及证据上传，复用本项目 18084 服务。凭据只由环境变量读取，不写入命令、Git、结果摘要或进度记录。代码不自动安装依赖、不修改容器限制、不结束其他进程。

## 自动执行的步骤

1. 读取两类模型的锁定信息，按 LUKE tiny、LUKE full、MetaEntail tiny、MetaEntail full 顺序串行处理。
2. `--device auto` 在未被 `CUDA_VISIBLE_DEVICES` 禁用时，用 `nvidia-smi` 探测显卡；子进程仍会核实 Torch 的 CUDA 可用性。多卡环境也只同时运行一个诊断进程。
3. 每项任务在下载源码、读取权重或导入神经网络后端前先检查可用主存。tiny 要求至少 768 MiB，full 要求至少 24 GiB；full 权重构造器还有自己的检查。GPU 显存不替代这些主存检查。
4. LUKE 自动准备官方仓库固定 commit 的 15 个声明源码文件，按长度和 SHA-256 校验；已存在且损坏的缓存保留并报错，不覆盖。源码来源不是合作者机器上的某个固定绝对目录。
5. full 模式复用现有 LUKE/MNLI 下载器，从私有交换服务获取锁定分片并恢复权重。已有缓存也要校验；不切换其他预训练模型。tiny 使用小型随机架构，不代表预训练权重/BPE 已通过验证。
6. 调用已有诊断：检查前向、loss、梯度及记录透明性，做两步诊断更新，并比较断点恢复后的第二步；记录选定训练和测试样本的实际张量。MetaEntail 每个所选样本保留全部三个候选关系句对。
7. 核验子进程退出码和证据状态，以及两种阶段的 `.pt` 文件、JSON、checkpoint 的完整性。缺文件、shape-only、错误哈希或未完成的检查都不算通过。
8. 生成私有结果摘要；使用 `--upload` 时，把声明的日志、JSON、张量和诊断 checkpoint 上传到本项目命名空间。大于 512 MiB 的文件分片，并验证分片拼接后与原文件的哈希相同。

只有 tiny 被选中且通过，才会继续同一模型的 full 阶段；另一类模型的检查不受它的失败影响。显式只选 `--mode full` 可单独执行 full。每项诊断默认最多 1800 秒，超时只终止该任务自己的子进程，保留输出。

## 结果与退出码

默认结果目录：

```text
runs/retacred_reference_runtime/<UTC时间戳>_<随机短ID>/
```

主要文件：

- `summary.md`：每项模型、模式、设备和状态的可读表格。
- `result.json`：命令、退出码、源码哈希、资源或失败原因、仍缺少的认证项。
- `artifact_manifest.json`：只声明本次预检产物的长度和 SHA-256。
- `luke/{tiny,full}/`、`metaentail/{tiny,full}/`：实际调用后产生的私有诊断证据；主存检查未通过时，仅记录父入口的阻塞原因。
- `logs/`：被调用任务的 stdout/stderr；主存检查挡住而未调用子任务时不生成这些日志。
- `exchange_pending.json` / `exchange_receipt.json`：上传状态和经核验的回执。

终端逐项打印进度和状态，最后打印 `summary.md` 的路径。

| 退出码 | 含义 |
| --- | --- |
| 0 | 所选诊断全部通过；若要求上传，则上传也完成。只认证本次诊断范围。 |
| 1 | 检查失败、证据缺失/损坏、超时或上传失败。已有结果保留。 |
| 2 | 资源不足、离线缓存缺失或 tiny 前置任务未通过，所选诊断尚未全部完成。 |

`pending_memory_gate` 表示内存检查在模型执行前挡住了该任务，不是成功训练，也不是一次模型 OOM。不要为了获得退出码 0 调低检查阈值。上传成功不改变诊断状态：资源仍不足时依然返回 2。

远端证据目录：

```text
q-attention/retacred-reference-runtime/<本次目录名>/
```

这些产物包含私有样本、张量和 checkpoint，不直接加入 GitHub 的 `reports/`。全量实验报告的发布流程保持独立。

## 可选命令

仅查看计划，不下载、不导入模型、不写文件：

```bash
bash scripts/check_retacred_reference_runtime.sh --describe
```

只检查 MetaEntail 的 tiny 架构：

```bash
bash scripts/check_retacred_reference_runtime.sh --models metaentail --mode tiny
```

严格使用已有源码和权重，禁止下载；不上传：

```bash
bash scripts/check_retacred_reference_runtime.sh --offline --device cpu
```

上传失败后，不重新执行模型，只重试同一结果目录的上传：

```bash
bash scripts/check_retacred_reference_runtime.sh \
  --upload-only runs/retacred_reference_runtime/<本次目录名>
```

通过 `--output-dir` 指定的目录必须不存在，避免覆盖旧证据。再次执行诊断会创建新目录，源码和权重缓存仍可复用。`PYTHON_BIN` 可指定当前环境的 Python；脚本不内置我们服务器的解释器路径。

## 证据边界

诊断更新使用明确记录的检查配置，不是恢复出来的作者训练配方。即使 tiny/full 全部通过，也不自动认证论文分数、原始优化器等价性、AllenNLP 包装等价性、CoRE/CFIE 等机制，或 MetaEntail 的评估分母。对应源码/论文契约中的未解决项继续保留。

本入口仅覆盖 LUKE 和 MetaEntail 基础层；其通过不代表 15 项参考配置全部就绪，也不改变正式套件默认的五项配置。后续完成模型机制、训练与评估契约后，才接入总的一键全量实验脚本。
