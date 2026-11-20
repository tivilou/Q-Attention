# Q-RMEA 三阶相干词图输运：完整数据单 seed 验证

状态：实现准备中，尚未通过精确版本审查与 2080 小规模验收，尚未发布正式运行许可。本页命令在发布验收完成前仅供审阅。

这次检验三阶词图输运的弱正向初筛能否在完整 Re-TACRED 上成立。已有初筛的正类 Macro 增益只有 0.000551；固定模型换回 native 或 dephased 后 Macro 还更高。因此本实验不预设相干性必要，也不继承历史 Q-RMEA 的限定 L4，更不代表达到 L5。

## 同步与运行

默认已在 Q-Attention 项目根目录，使用自己已有的 Python 环境。先执行：

```bash
git fetch origin --prune
git switch 1.1
git merge --no-edit origin/1.1
git merge --no-edit origin/main
bash scripts/run_q_rmea_token_transport_handoff.sh --gpu auto
```

默认数据目录是 `data/relation/retacred`；固定 RoBERTa-large 目录是 `.external/assets/722cf37b1afa9454edce342e7895e588b6ff1d59/model`，均相对项目根目录。模型目录仅允许资产锁中的五个文件，避免额外 safetensors 或分词配置被自动选中。脚本不自动下载数据或安装环境。若环境中的默认 `python` 不正确，用 `PYTHON_BIN` 指向已有解释器；脚本依次选择 `PYTHON_BIN`、`python`、`python3`。

只检查环境、权限与 GPU 时可执行：

```bash
bash scripts/check_q_rmea_token_transport_handoff.sh --gpu auto
```

## 固定协议

- 官方 train/valid/test 为 58465/19584/13418，检查既有固定数据哈希和 train-derived40标签，`no_relation` 是索引0。历史测试集已暴露，单 seed 不构成新总体独立确认。
- seed13；四路分别为 native、cubic coherent、cubic initial-dephased、linear coherent，各81028参数，共同初始化、样本顺序和优化器。只改路由。
- 每路4轮、batch32、AdamW，学习率0.001、weight decay0.01、betas0.9/0.999、eps1e-8、clip1。每路7312步，共29248步；float64/complex128；tau0.25，Born eps1e-6。
- 统一使用第4轮最终模型。valid 仅描述，不挑选 epoch；完成训练后才评估 test。不能换成旧 Q-RMEA 的20轮/batch128或按 valid 选模型。
- 主指标为排除 `no_relation`、固定分母39的 Macro F1。cubic 必须严格高于另外三路，且 CE 不高于 native。完整40类 Macro、准确率和非空 Micro 为辅助指标。
- 在最终 cubic 模型上，对完整 test 执行四路冻结状态交换；对 hash 顺序前32条执行坐标上界、读出上界和行范数结构诊断。诊断不进入任务通过门槛，也不直接证明机制带来收益。

## 环境与资源

Python3.10；已明确支持的 Torch/NumPy 组合为2.4.1/1.24.3、2.5.1/1.26.4、2.7.1/1.26.4；真实特征抽取需要 Transformers4.44.2。不同组合的训练轨迹不保证逐字节一致，实际环境记录入报告，并先运行同一环境的小规模四路 canary。版本不符时停止，报告问题。

仅使用一张至少8GiB空闲显存的可见 GPU，四路串行；固定 batch 不变。主进程的预检在独立子进程完成，释放 Torch/显存后进入实验。容器内存至少4GiB，私有输出磁盘可用空间至少12GiB。OOM、非有限梯度、数据或来源不符均停止，不悄悄缩小数据或改科学参数。

## 保存与返回

启动时输出唯一 `PRIVATE_RUN=...`，默认 `../q-rmea-token-transport-private/<UTC时间>_seed13`，私有目录位于 Git checkout 外。保留一份特征缓存、四个小型最终 checkpoint、固定各 split 前2条的最终状态完整 v2 回放及私有几何诊断。不保留初始/逐轮模型或训练全轨迹；不自动上传完整缓存，也不把原始样本、逐样本预测、权重或张量提交 GitHub。本脚本没有自动 COS 备份或中断续训能力；实例释放前应另走既有最小恢复对象归档流程。

训练、回放和私有清单验收成功后，脚本自动导出 `reports/q_rmea_token_transport/<UTC时间>_seed13/` 的七文件报告，验证工作区状态及暂存差异，只提交这些新增报告，自动普通推送至 `1.1`。不要手动提交原始运行目录，不必另执行 Git 提交命令。

若仅报告发布失败，保存启动时打印的实际私有目录，在同一代码版本上执行导出入口的 `--run-dir` 参数即可重试。它核验原 `RUN_COMPLETE`，重用同一不可变运行，不重新训练，不生成重复提交，不强制推送。不要为了发布失败再运行训练入口。

## 停止与后续

任一路缺失、执行/回放/导出校验失败，或主门槛未过，都先返回现有报告和错误，不改 seed、tau、预算、控制组或指标。单 seed 通过也只进入我们审阅；师弟不自动开多种子。历史 baseline 不重跑。新鲜种子确认及更强归因将由独立协议决定。
