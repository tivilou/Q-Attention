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

需要完整 canonical 数据 `data/relation/retacred/{train,valid,test}.jsonl`，记录数为 58465 / 19584 / 13418。需至少 12 GiB 进程可用 RAM。每张 GPU 默认仅一个重进程，建议至少 12 GiB 显存。物理 micro-batch 默认 `auto`：先尝试该卡实际分配到的全部逻辑样本，不预先开启激活重计算；只有实际 CUDA OOM 才逐级减半。H3 的上游 routing 未屏蔽 padding，必须保持单句分区，不能靠拼句改变模型语义。逻辑 batch 和优化器不因 GPU 数量变化。

### CTL 多卡与普通任务混合调度

`--gpu auto` 下，默认每个 CTL supervised / unsupervised 任务占用两张 GPU，其他模型各用一张。六卡可同时运行 CTL双卡 + CTL双卡 + PTR单卡 + H3单卡；三卡运行一个CTL双卡和一个单卡任务，另一个CTL等待；只检测到一张卡时退回原单卡训练。队列先启动预计耗时较长的任务，释放整组GPU后再补下一个任务。

CTL 的全局逻辑 batch 仍是512句，一句的全部候选关系放在同一张卡，按全局句数归一化损失后求和梯度，各卡各执行一次相同AdamW更新。默认双卡的第一档实际micro是256句/卡，然后实际OOM才降到128、64、…、1，再开启激活重计算；没有重复的512/256空降档。完整评估、样本记录、保存和上传由rank0负责，其余卡保留训练状态。

`--ctl-gpus-per-task 1` 可显式选择全部任务单卡；`--ctl-gpus-per-task 3` 可选择每个CTL三卡。双卡和三卡已纳入Titan X的有界训练验证；4–6卡接受显式配置，但未做该规模硬件验证。每张卡只属于一个任务组，固定GPU行会显示 `DDP1/2` 等rank标记，Pending/Completed按任务计数而非rank计数。

验证证据见 [Titan X DDP检查记录](../validation/reference_ctl_ddp_20261011.json)：完整24层预训练unsupervised双卡、supervised三卡各完成2次更新，均通过所选训练253项/测试249项真实数值表示校验；同时验证尾批、梯度归一化、全rank随机状态恢复、实际OOM降一档重试及47项回归。收尾恢复修复后另通过40项相关回归及真实三卡完成后重试，已提交样本文件哈希不变且不重复训练/评估。两组测试有重叠，不相加为87项独立测试。它是有界正确性检查，不是全量分数或DDP加速倍率测量，不能直接把早先RTX3060单卡时长估算当作新版多卡实测。

[独立审查记录](../validation/reference_ctl_ddp_review_20261011.json) 的最终结论为PASS：Opus5.5与Fable5最终调用均超时，按约定回退独立subagent审查，发现并复核修复了原生完成标记之后重试误隔离测试trace的问题。没有声称这两次外部调用审查了最终版本。小模型OOM canary验证物理micro逐级降档，但因小模型已开启重计算，未单独认证从无重计算切换到重计算的硬件行为。

### 动态磁盘管理（不再要求固定 352 GiB）

磁盘不是显存。预检按缺失的共享权重、单个文件拼装周转、实际 GPU 并行数和任务峰值估算；种子/配置数量不再线性乘成固定空闲门槛。调度器只启动当前磁盘能容纳的任务，空间偏紧时自动减少同时运行的任务数。完整最长输入的 case 张量大小并非事先已知，估计值不是最低容量保证；每次实际大文件写入还会检查实测张量存储量与其他写入者的预留。

- 预训练权重共用一个哈希锁定缓存。拼装后核对最终文件及分片 SHA256，再删除该文件的已验证下载分片；不动服务端原件和其他缓存。
- 活跃普通任务只保留完整 `latest` 和当前权重版 `best`；默认每120秒在完整 optimizer update 边界原子保存，采样、epoch结束、正常暂停和最终完成强制保存。不保存全部 batch 历史；旧 latest 留到新文件完整落盘替换成功。`--checkpoint-seconds 0` 可恢复每步保存，磁盘开销会增加。
- CTL 保留两代滚动 checkpoint；采样事件关联的前后 checkpoint 经上传及完整回读 SHA 校验后可移到18084，保留紧凑出处回执。活跃 latest 不删除，下一次保存仍需要临时周转空间。
- curriculum 每个 reviewer 完成后，把全部 heldout votes、难度所需信息和所选样本实际 trace 写成独立完成账本，再归档 reviewer checkpoint。恢复时核对账本后跳过该 reviewer，不因优化器已移出本地而重新训练。
- 已提交训练 batch 的所选 case 张量及每个完成任务的 latest/best 都滚动归档。训练期间使用有界后台单写入队列，积压达到上限时施加背压；上传失败继续作为真实错误报告，不丢弃证据。上传采用一次最多512MiB分片周转，服务端回执和逐字节 GET 回读分别校验分片与完整原文件，持久化目录账本后才清理本地数值副本。原句/生产者 JSON 与指标留在本地；数值不删减、不截断、不降精度。
- 完成任务即时处理，不等十五配置全跑完才释放空间。上传失败保留尚未归档的本地文件与恢复账本；不会把失败当成功。未提交的 trace 尝试单独隔离并私有归档，不冒充正式 case。
- 空间仍不足时退出码75，并在 suite 下写 `STORAGE_PAUSED.json`。释放空间后用原目录 `--resume`；已完成任务跳过，未完成任务从最近落盘 batch 恢复。正在写入的临时文件、唯一活跃 latest、其他进程/其他 run 的内容均不清理。

私有数值读取器可按账本下载并校验完整原 `.pt`，展示坐标切片；完成任务的优化器等原始恢复文件也保留在服务上。若需要把某个文件恢复到原位置（通常默认续跑只会跳过完成任务，无需手工操作）：

```bash
python -m baselines.retacred.reference_archive --restore \
  runs/retacred_reference_complete/<时间戳>/tasks/<model>_seed13/checkpoints/latest.pt
```

这会恢复原始文件字节，不重训练或重建张量。服务可达且仍保有对应归档是读取已移出本地数值的前提。

配置项目提供的 `PROJECT_EXCHANGE_URL`（18084 的 HTTPS 地址）和 `PROJECT_EXCHANGE_TOKEN`。证书由代码带入，禁止关闭 TLS 校验。缺失的哈希锁定源码、RoBERTa / MNLI / LUKE / SimCSE 权重、依存模型会自动从服务下载，逐分片及拼装后校验大小/SHA256；无需上传 Git。已有缓存先校验再复用。

## 3. 实际训练、记录和默认发布

交互终端固定显示一行总进度和每张 GPU 一行状态，原位刷新任务、阶段、batch、速率、ETA 和显存，不再不断追加重复快照。GPU 行下方固定显示 Pending 和 Completed 两块队列区域，分别按等待顺序和最新完成优先展示任务名、序号和总数；每块保留三行任务位置，空队列也不收起。任务过多时显示 `+N more`，完整名单保存在当前 run 的 `queue_status.json`。重定向到日志或非交互终端时自动改用普通文本快照。MetaEntail 的速率与显存来自实际采样；未提供的数据显示 `--`，不会伪装成零。任务失败时主终端显示失败任务、GPU、退出码、日志位置及末尾异常，再退出。

2026-10-11 的速度更新保持 FP32、全部候选关系和原逻辑 batch。执行档位是 `该卡最大句数 → /2 → … → 1 → 1+非重入激活checkpointing`；显式 `--micro-batch N` 将N作为单卡物理上限。只有实际 `torch.cuda.OutOfMemoryError` 推进一档，子进程退出76后调度器在同一GPU或同一DDP GPU组自动重建进程、读取该任务的档位账本和最后完整恢复状态，再执行未落盘批次。非OOM错误不降档，档位耗尽会明确失败。分类损失、候选分母、优化器和总步数不改；physical partition变化会改变dropout随机数消费和浮点归约，不能声称换档前后训练结果逐位一致。

CTL 不再返回训练未使用的全层集合，H3 跳过未用于其损失的MLM词表预测头；完整样本的实际forward钩子与反向梯度仍保留。CTL全数据编码预检按数据、分词器、描述和源码身份缓存，OOM重启不重复预检；每epoch洗牌顺序缓存，编码LRU限制在256MiB。普通worker最终只做全量dev/test评价，不重复全量train推理；训练loss和所选训练样本的张量记录仍保留。此次旧运行 `20261011T014753Z` 按项目方决定不导入；更新后运行默认命令生成新时间戳，不使用旧目录 `--resume`。哈希校验通过的共享资产缓存可以复用。

开发实测：RTX3060、完整预训练24层SimCSE RoBERTa-large CTL、FP32、固定3句/8候选对、1步预热+3步计时。旧micro1+激活重计算+每步保存，平均计算1.06秒/步、含保存8.41秒/步；新micro3+无重计算+3步末保存一次，分别0.28和2.56秒/步，计算约3.85倍、含保存约3.29倍。输入是有明确标记的合成小样本，此结果不代表58465句全量实验、最长输入或整套15模型的加速保证；没有测量网络上传时间。

默认上述一条 runner 完成预检 → 数据转换/资格校验 → 资产获取 → 独立 `(model, seed)` GPU 队列 → 完整 dev/test 评价及逐任务滚动归档 → 汇总18084完整值回执 → 白名单报告导出 → commit → push `1.1`。不另敲导出命令。`--prepare-only` 只准备，`--dry-run` 只列配方，`--no-push` 仍提交报告但不推送。

保存 micro precision / recall / F1、accuracy、macro-F1、逐类 P/R/F1/support、confusion matrix、loss 和 loss_scope。CoRE 的 loss 是 factual 分支 CE，不是修正分数的 NLL；MetaEntail 的 loss 是预测后统计的 NLI pair CE，不是关系 NLL；弃答保留在全样本分母和独立预测列中。不把缺值伪造为 0，也不把不同 loss 当成同一口径排名。

所有解释性补齐见每个报告中的 recipe/adaptations：CFIE 是明确的 LUKE 移植重建；curriculum 是 train-only own-subset cross-review + 真实 dependency GAT + bilinear；H3 保留源三段路由但使用 canonical40 和共同编码器；marker 统一 FP32。不存在对未知作者参数或论文分数的保证。

所选 train/test 样本记录原句、tokens、两个实体及位置/类型、预测与预测后的 gold 对照、真实 embedding / attention / routing / dependency / logits，训练阶段还记录反向梯度与分类头更新前后值。所有实际观测数值保存在私有 `.pt`，不是 shape/hash 的占位。NLI 会记录所选样本全部可行候选和组级最终关系/弃答。服务接收数值和原始 trace；Git 只接收紧凑哈希、完整指标、data.sha256、data_counts.txt、run_summary.data 和白名单报告。尚未把新结果自动导入门户的动作算作本次 runner 工作。

## 4. 中止、恢复、只补导出

原始路径为 `runs/retacred_reference_complete/<启动时间戳>/`，每个配置/seed 有独立checkpoint及持久化执行档位。Ctrl-C在当前逻辑batch保存后停下；OOM或强杀回到最近已落盘完整update，最近保存之后的批次可能重做，不跳过数据。120秒是保存检查间隔，不是最长重做时间：单个慢batch、归档背压、写入本身可能延长间隔。已完成任务校验后跳过；未完成恢复模型、优化器、调度器、RNG和batch游标。

```bash
bash scripts/run_retacred_all_references.sh --gpu auto --models all --seeds 13 \
  --resume runs/retacred_reference_complete/<原启动时间戳>
```

执行策略、源码、资产、数据和科学配方在同一run恢复时必须保持不变。OOM自动重试固定在原GPU或GPU组；CTL恢复核对world size、各卡物理身份和执行环境，并保存每个rank的Python/NumPy/CPU/CUDA随机状态。DDP恢复目前要求原卡组、原卡数和原rank顺序；更换卡数、自动改用另一组卡或旧单卡checkpoint导入均未认证。重启suite后应显式保持原GPU列表和任务分组；不匹配时拒绝恢复，避免静默改变训练。旧实验继续按项目方要求使用新时间戳重新开始。长输入超过512时明确停止而非丢弃样本；marker/PTR保留源代码的512上限和截断规则，但缺实体标记/非法prompt位置会停止。不同overflow口径在recipe中披露；各worker训练前核查完整split。

CTL已写入原生完成标记、但尚未生成通用RUN_COMPLETE时，同版本重试先校验完整原生结果和train/test张量记录，再补结果格式转换；不会重新训练、评估或执行未完成checkpoint的trace隔离清理。损坏的已提交样本记录会明确失败，不通过重训练掩盖。

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
