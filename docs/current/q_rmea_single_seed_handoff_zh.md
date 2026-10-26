# Q-RMEA：Re-TACRED 单 seed 实验

本轮研究冻结 RoBERTa-large 输出上的关系专属证据池。已有十五组 reference baseline 已完成，不重跑；下面五个同接口条件属于本轮新 idea 的必要对照。历史微调模型分数只作背景。

执行前须确认本版本的实现审查及有界验证记录已经发布并通过。我们负责代码、有限测试和结果审计；完整实验由合作者执行。本轮只跑 seed13，不自行扩展多 seed。

## 同步与运行

在已有仓库中同步，先合并合作者远端，再合并发布分支：

```bash
git fetch origin --prune
git switch 1.1
git merge origin/1.1
git merge origin/main
bash scripts/run_q_rmea_handoff.sh --data-dir /你的路径/retacred-jsonl --assets /你的路径/roberta-large
```

数据为既有项目规范化的 `train.jsonl`、`valid.jsonl`、`test.jsonl`，源文件 SHA 必须匹配。完整样本数为58465/19584/13418。权重和四个分词配置须匹配 `baselines/retacred/assets.lock.json`；脚本使用本机已持有资产，不自动改模型或联网下载。若路径不同，只修改上述两个路径。

默认使用当前 `python`，找不到则用 `python3`；可设置 `PYTHON_BIN` 指定已有环境。要求 Python3.10、Torch2.5.1或2.7.1、NumPy1.26.4、Transformers4.44.2。运行前可使用相同参数执行 `bash scripts/check_q_rmea_handoff.sh ...`。环境不符或 Git 冲突时停止并反馈，不自行改版本、预算或协议。

自动选择当前可见且空闲显存至少8GiB的一张 GPU，使用串行低内存配置。也可用 `--gpu 可见编号`。它不缩减数据、batch、epochs、条件或 seed。原 GPU 不空闲时等待，不将正式实验降为 CPU。

## 固定科学协议

同一冻结特征、相同 value/decoder 初始化、数据顺序和训练预算，运行：uniform、relation_softmax、real_overlap、quantum、dephased。后三者的参数量会如实记录，不宣称容量完全相等。训练均为20epochs、batch128、AdamW学习率0.001、weight_decay0.01、梯度裁剪1，seed13。仅 valid 的全40类 Macro-F1 选择检查点，平分取最早；test 只在选择后评估。训练前执行16/8/8样本、1epoch的技术 canary。

每个词制备三量子比特态，候选关系各有独立二元测量效应；测量概率归一化成词权重，直接形成该关系的分类分数。dephased 仅去掉词态在计算基中的相干项。它们是40个二元测量，不是一个40输出 POVM。没有经典残差旁路，也不向权重计算传入真实关系标签。

固定检查点上只置换有效非实体上下文词的权重，保留值向量、decoder、实体/填充权重和归一化质量。同一句的所有关系行使用同一、与标签和结果无关的固定置换。上下文不足两个词时保留原样并计入全体分母。主机制指标为全 test 的平均 `CE(置换)-CE(原始)`；uniform 必须不变。关系间平均权重仅作次要描述，不能挽救主诊断失败。

## 私有输出、自动回传和故障

启动时打印 `PRIVATE_RUN=.../<timestamp>_seed13`。特征、原文案例、逐样本预测、检查点、梯度和完整 `sample-trace.v1` 都留在仓库外。预计完整 FP16 词缓存约6.33GiB，另需资产、临时文件和案例空间。不要删除这个目录或把它 `git add`。

每个阶段关闭后写入完整 SHA/大小清单和 `RUN_COMPLETE`。正式阶段完成后，一键脚本自动检查样本数、源代码/数据/协议身份、案例结构及数值回放记录，导出 `reports/q_rmea/<timestamp>_seed13/`。只暂存并提交七个白名单文件，再普通 push 到 `1.1`；不需要手动提交。

导出或 push 失败时保留原运行，不重新训练；从同一代码/环境执行：

```bash
bash scripts/export_q_rmea_handoff.sh --run-dir /完整路径/PRIVATE_RUN
```

脚本会清理未完成的临时报告；已存在的最终报告必须逐字节相同。已生成的单个报告提交会复用，不重复提交。不自动合并、不强推。若运行期间远端分支变化导致祖先检查失败，请反馈，由我们决定同步方式。

本版本不支持中途训练断点恢复；未关闭的失败阶段应保留并反馈，不使用 `--resume`、不自行补跑或换预算。以上仅导出恢复已通过专门的失败注入测试，不等于服务器被释放后的训练恢复或 COS 备份验证。

## 验收与停止条件

首先审计本轮 seed13：主效用为 quantum 相对 uniform 的全40类 test Macro-F1 差值。任何正差值只可能登记 L1 候选，不能单 seed 宣称 L2/L3。无效、负值或来源/泄漏/回放失败即停止自动推进。

通过后才会另发五个新 seed（29/53/71/97/113）的确认任务，不把探索 seed13混入确认均值。拟采用平均效用差值大于0、至少4/5为正、seed层配对 t 区间下界大于0；L3 还需预声明机制诊断得到同方向的重复证据。具体确认协议在新交付前冻结；五对精确符号翻转双侧检验最小 p=0.0625，不声称其 p<0.05。

该研究暂不建立新颖性、量子独占性、硬件加速、Transformer内部注意力改进、实体/文档独立或跨数据集泛化结论。官方 split 的既有重叠保留并披露。同一 benchmark 上的可重复效用与机制实验可以支持普通方法的 L3，但数据集名气本身不能代替这些证据。
