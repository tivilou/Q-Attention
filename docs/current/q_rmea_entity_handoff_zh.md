# Q-RMEA 有序实体条件测量：正式单种子交付

本轮研究“同一句话的主客体内容与顺序，是否应改变关系的词级测量”。冻结 RoBERTa-large 输出和旧解码器，新增主语、宾语的共享投影和有序变换；不是重新跑历史十五组 benchmark。`static` 是当前新实验中关闭实体条件分支的同接口比较条件，其实现与旧 quantum 头已验证等价。

## 同步与运行

在 Q-Attention 根目录，使用上一轮成功的环境、数据和模型资产：

```bash
git fetch origin --prune
git switch 1.1
git merge origin/1.1
git merge origin/main
bash scripts/run_q_rmea_entity_handoff.sh
```

数据默认 `data/relation/retacred`，模型资产默认 `.external/assets/722cf37b1afa9454edce342e7895e588b6ff1d59/model`，GPU 默认 `auto`。这些是上一轮已使用的真实相对路径；必要时可用原样命名的 `--data-dir`、`--assets`、`--gpu` 参数指定实际路径，不能换数据或权重。单独预检用 `bash scripts/check_q_rmea_entity_handoff.sh`。

解释器依次取 `PYTHON_BIN`、当前环境的 `python`、`python3`。环境固定 Python3.10、Torch2.5.1 或2.7.1、NumPy1.26.4、Transformers4.44.2；不自动安装或更改环境。预检验证精确源码审查、`origin/1.1` 和 `origin/main` 的祖先关系及干净工作区。`auto` 选择至少8GiB空闲显存的一张可见卡，串行运行；资源不足就停止，不改 batch 或科学预算。

## 冻结实验

先跑16/8/8样本、1 epoch的技术 canary，完整追踪验收通过后，才跑完整 train58465 / valid19584 / test13418。只运行 **seed13**，五个条件各20epochs、batch128、AdamW lr0.001、weight_decay0.01、clip1。五路共享同一特征缓存与全部初始可训练参数，均为81028个活跃参数。历史 seed13 证据不是本轮独立重复。

条件为 `static`、`ordered`、`no_content`、`symmetric`、`classical_pair`。有序概率为 `|phi† U_object U_subject psi|²`，等价地把两个逆变换施于关系态，避免两侧同乘同一酉变换而抵消。`no_content` 使用全 train 的等样本实体均值，冻结为缓冲区；`symmetric` 平均两个顺序的原始概率，再做原词级归一化；经典对照使用同维度与同参数量的实体条件查询，但不声称表达能力或优化难度完全匹配。

每一路仅按 valid full40 Macro-F1 选择检查点，相同取最早 epoch。选定后才计算完整 test 指标；initial/selected 的固定两条 test 案例只作预先确定的推理追踪，不参与选择。`ordered` 在选定检查点另做分支干预：train均值、主客体互换、整个同一 split 按样本ID固定错配；保留全部样本。只替换条件分支，原词特征、角色特征、解码器和任务标签不变；不是把任务中的主客体交换后要求标签不变。

主要效用是 `ordered-static` 的 test full40 Macro-F1；大于零仅为待负责人审计的单种子候选门槛。均值、交换、错配各自的 `CE(intervention)-CE(actual)`、重新训练的消融差值和经典对照单独报告。单种子不自动达到L3，不证明新颖性、量子独有优势或统计显著性。负责人收到报告后才决定继续、修改或发多种子合同；脚本绝不自动开多种子。沿用官方 split 的重叠与历史 test 暴露限制。

## 私有证据与自动报告

启动打印 `PRIVATE_RUN=...`，默认目录是仓库外 `../q-rmea-entity-private/<一次启动时间戳>_seed13`。完整阶段只建立一份约6.9GB的编码缓存供五路复用，另留检查点、固定样本追踪及安全余量。**本轮不自动上传大文件，不需要交换服务授权。** 私有证据保留在执行服务器；不要删除，迁移或释放服务器前按负责人指定的选取与COS归档流程处理。

`sample-trace.v2` 保存各 split 固定两个样本、五路 initial/valid-selected 的完整关键数值，包括实际分词ID与mask、已缓存的编码器末层输出、原词均值、实体条件、values、权重、logits及有序分支干预。每个值连接到私有文件哈希与任意坐标读取接口；验证实际 encoder-output→word 聚合、checkpoint 重放、同输入对照与全部数值坐标。范围不包含重跑编码器、每个内部层或完整训练轨迹；首批参数/梯度/更新另存。私有读取命令是 `python -B scripts/view_rmea_entity_trace.py --run-dir <打印目录>/discovery`，仅监听127.0.0.1:4192；原文和张量不进公共Git。

每阶段通过完整清单封存和 `RUN_COMPLETE` 后，自动原子导出七文件到 `reports/q_rmea_entity/<同一时间戳>_seed13/`：summary.json、manifest.json、trace_validation.json、receipt.json、run_summary.json、data.sha256、data_counts.txt。包含完整 source SHA、训练配置、原始数量/哈希和实际环境；不包含原文、逐样本预测、tensor、checkpoint或日志。脚本检查 status、精确暂存和提交差异，只生成一个七文件提交，普通推送到1.1；正常结束无需手动 commit/push。

## 失败处理

训练或canary失败就停下，保留私有目录与错误，交负责人判断；不改seed/数据/预算，不自动重训或续训。只有全部阶段已经封存、仅导出或推送失败时，使用 `bash scripts/export_q_rmea_entity_handoff.sh --run-dir` 加**本次实际打印目录**重试。该入口不导入训练模块、不调用训练，核对同一闭合清单；报告已提交时复用原提交，不增加第二个提交。不可拿历史fresh5目录替代本轮目录，也不 force push 或自行解决运行后的代码漂移。

技术验收见 [交付验收](../validation/q_rmea_entity_handoff_acceptance.json)，冻结配置见 [实验契约](../contracts/q_rmea_entity_contract.json)，精确源码审查见 [审查记录](../validation/q_rmea_entity_implementation_review.json)。
