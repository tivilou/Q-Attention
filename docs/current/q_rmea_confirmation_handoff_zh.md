# Q-RMEA 五个新种子的独立确认

本轮只执行已通过 seed 13 审计的方法的独立确认。种子固定为 **29、53、71、97、113**；旧 seed 13 不计入本轮统计。完整实验由合作者在 `1.1` 执行。负责人只做实现、有界测试和结果审计。

冻结的机器可读规则见 [确认契约](../contracts/q_rmea_confirmation_contract.json)，已完成的技术验收见 [验收记录](../validation/q_rmea_confirmation_acceptance.json)。

## 同步与一键运行

以下命令均从 Q-Attention 项目根目录运行。使用上一轮已经成功的环境与数据路径。

```bash
git fetch origin --prune
git switch 1.1
git merge origin/1.1
git merge origin/main
bash scripts/run_q_rmea_confirmation_handoff.sh \
  --data-dir data/relation/retacred \
  --assets .external/assets/722cf37b1afa9454edce342e7895e588b6ff1d59/model \
  --gpu auto
```

运行前保持工作区干净。Python 入口依次使用 `PYTHON_BIN`、`python`、`python3`；需要时通过 `PYTHON_BIN` 指向上一轮的解释器。环境固定 Python 3.10、PyTorch 2.5.1 或 2.7.1、NumPy 1.26.4、Transformers 4.44.2。预检同时校验精确源码审查记录、两个远程分支的祖先关系、现有私有 HTTPS 交换授权、数据与模型资产。不得将令牌发到聊天或提交 Git。

GPU 自动选择一张至少有 8 GiB 可用显存的卡，串行运行，记录实际卡与环境。无卡满足要求时停止等待资源，不自动修改批大小或训练预算。负责人使用的 P4 技术小测例外仅对开发 canary 生效，不能用于完整训练。

## 固定实验内容

先运行种子 29 的 16/8/8 样本、1 epoch 技术 canary；其追踪通过后，依次运行五个新种子的完整数据：train 58465、valid 19584、test 13418。每个种子均包含 uniform、relation_softmax、real_overlap、quantum、dephased 五种条件，每种 20 epochs、batch 128、AdamW lr 0.001、weight_decay 0.01、clip 1。每个种子内部共用编码缓存和 value/decoder 初始参数；不同种子分别沿用原来的确定顺序编码流程，避免引入新的缓存共享规则。

模型、数据、分词、最大长度、解码器、标签集合和优化预算均沿用旧实验。只有种子改变，以及增加环境记录和样本追踪。每种条件按 valid 的最大 full40 Macro-F1 选择检查点，相同分数取最早 epoch；选定之后才计算 test。历史 15 个参考配置已完成，本轮不重跑。三个对照是本轮方法所需的同接口比较条件。

## 预先固定的判断规则

每个新种子的主要效用差值为 quantum 减 uniform 的 test full40 Macro-F1。五个差值的均值必须大于零，至少 4/5 为正，且配对种子 t 区间的双侧 95% 下界大于零。固定 df=4、临界值 2.7764451051977987。

主要机制诊断在同一个 valid-selected 检查点，对完整 test 的非实体上下文权重进行固定置换，计算平均 `CE(shuffled)-CE(actual)`。仍保留没有足够上下文的样本。五个种子的该差值使用相同均值、至少 4/5 为正、t 区间下界规则；每个种子的 uniform CE 差值绝对值必须不超过 2e-5，各条件的恒等式残差不超过 2e-5。

统计单位是五个新种子，t 区间依赖独立、近似正态的种子差值假设，小样本限制必须保留。脚本同时报告精确双侧符号翻转 p 值；五对数据能达到的最小 p 为 0.0625，不能据此声称 p<0.05。旧 seed 13 不进入确认统计，不挑选成功种子，不更改门槛，也不根据 test 改训练参数。

quantum 与 relation_softmax、real_overlap、dephased 的差值单独报告，用于讨论机制归因。参数量并不相等，不声称容量匹配、量子独有优势、SOTA 或文献新颖性。自动生成的 gate 字段只是待审计结果，不自动将证据提升到 L2/L3。

## 私有证据和自动交付

启动时打印 `PRIVATE_RUN=...`，目录默认在仓库外，名称为一次启动时间戳加 `_fresh5`。这是原单种子 `_seed13` 命名约定的独立确认扩展。每个阶段单独封存完整清单与 `RUN_COMPLETE`。每个完整种子的缓存约 6.9 GB，五种子约 35 GB，另需检查点、追踪和安全余量；原模型资产不重复上传。

新实验使用 `sample-trace.v2`：每个 split 固定两个样本，五条件、initial/valid-selected 检查点均捕获完整关键数值，显式连接分词、编码、词级输入、value、权重、预测和诊断。完整张量、原文、逐样本预测和检查点仅保存在私有证据中。其范围是冻结编码器最后输出与输出头推理回放；不声称保存每层编码器状态或完整优化轨迹。首个训练批次的参数、梯度和更新另存。

完整阶段全部成功后，脚本先将每个封存阶段上传现有私有 HTTPS 交换服务并核对分块回执；这不是 COS 独立恢复认证。随后只导出以下七个文件到 `reports/q_rmea_confirmation/<启动时间戳>_fresh5/`：summary.json、manifest.json、trace_validation.json、receipt.json、run_summary.json、data.sha256、data_counts.txt。

脚本在临时目录完成复制与校验，再原子提交到报告目录；检查 Git 状态、精确暂存差异、提交内容与运行版本关系，创建一个只增加这七个文件的提交，普通推送到 `1.1`。**正常运行无需手动提交或推送。** 不上传原始数据、缓存、追踪 NPZ、检查点、日志或凭据到 Git。

## 失败与重试

任何阶段失败都停止，不自动重训、重选种子或改配置。保留打印出的私有目录和失败信息，交负责人判断；部分种子完成不能当作完整确认结果，也不能直接重新运行整套脚本覆盖它。当前没有训练断点恢复或跨服务器恢复保证。

若五个完整种子已经封存，仅上传、报告导出或 Git 推送失败，可使用配套 `export_q_rmea_confirmation_handoff.sh --run-dir` 指向同一个实际私有目录重试。此入口只校验、上传及导出，不启动训练；已验证的私有分块按相同内容检查后复用；若报告提交已经存在，普通推送重试复用该提交，不生成第二个提交。发生分支冲突或运行中出现新的主分支代码时停止报告，不擅自合并或 force push。将错误交负责人处理。

开发测试会注入一次导出失败，验证没有残留最终/临时报告，再从同一封存源重试；源码与证据见 [确认验收记录](../validation/q_rmea_confirmation_acceptance.json)。

负责人可在取得私有证据后，运行 `scripts/view_rmea_confirmation_trace.py --run-dir` 指向某个阶段目录，在本机 127.0.0.1:4191 阅读固定样本及任意坐标真实数值。该页面只监听 loopback，不对外暴露原文和张量，不通过公共 Git 分发数值。
