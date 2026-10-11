# Re-TACRED Improved Baseline 外部 reference

## 默认执行命令

合作者在自己的 `1.1` 工作区同步 `origin/main` 后，直接运行：

```bash
git fetch origin --prune
git merge origin/1.1
git merge origin/main
bash scripts/run_retacred_reference_improved_baseline.sh \
  --seed 13 \
  --gpu 0
```

runner 会自动：

1. 在 `.external/RE_improved_baseline` 准备并锁定官方源码 commit `fe00781116ab4c34cbcd9a2ae5b02513afae87ee`；
2. 检查 Re-TACRED 三个 split 的记录数、标签映射、实体类型和 CUDA/依赖环境；
3. 将项目 canonical JSONL 转为上游 TACRED JSON 格式，记录源数据与转换后文件的 hash；
4. 按上游 `RoBERTa-large + typed_entity_marker_punct` 路径训练和评估；
5. 输出完整 train/valid/test 指标、checkpoint provenance 和 train/test case study；
6. 导出不包含 checkpoint、tensor `.pt`、数据集或预测文件的安全报告，并提交、推送到 `origin/1.1`。

正式实验必须由合作者 GPU 执行。项目方服务器只做代码审查、source gate、转换器测试和 preflight，不启动全量训练。

## 证据与适配边界

- 论文：[An Improved Baseline for Sentence-level Relation Extraction](https://arxiv.org/abs/2102.01373)
- 官方仓库：[RE_improved_baseline](https://github.com/wzhouad/RE_improved_baseline)
- 机制契约：[`docs/reference_baselines/improved_entity_marker_roberta_large_mechanism_contract.md`](../reference_baselines/improved_entity_marker_roberta_large_mechanism_contract.md)
- 适配边界：[`docs/reference_baselines/improved_entity_marker_roberta_large_adaptation_diff.md`](../reference_baselines/improved_entity_marker_roberta_large_adaptation_diff.md)
- manifest：[`configs/reference_baselines/improved_entity_marker_roberta_large.manifest.json`](../../configs/reference_baselines/improved_entity_marker_roberta_large.manifest.json)

上游模型的 marker 构造、RoBERTa encoder、实体位置 readout、分类器和训练语义保持不变。项目新增的只是数据格式适配、完整指标、provenance、checkpoint 和 case-study 记录。任何上游机制改动都必须使 manifest 保持 `planned`，不能继续标记为 source-faithful。

## 输出目录

原始私有 run：

```text
runs/retacred_reference_improved_baseline/<timestamp>_seed13/
```

包含 `case_study_tensors.pt` 和 `checkpoint/model.pt`，不得提交 Git。报告目录：

```text
reports/retacred_reference_improved_baseline/<timestamp>_seed13/
```

包含完整标量和逐类指标、混淆矩阵、数据 hash、上游 source commit、case-study JSON/张量 manifest 与日志 tail，但不包含私有权重、数据、预测或 full tensor artifact。

## 停止门禁

遇到以下任一情况必须停止，不要把结果标成严格 reference：

- 上游 commit、origin 或 source 文件 hash 不匹配；
- `transformers`、`ujson`、`tqdm`、CUDA 环境不满足要求；
- 三个 split 记录数或 40 类 label map 不匹配；
- entity marker/tokenizer/entity-position readout 的固定输入 gate 失败；
- test 被用于训练、调参或 checkpoint 选择；
- case study 缺少 train/test 样本或完整 tensor manifest；
- report exporter 发现 checkpoint、`.pt`、数据集或预测文件进入公开报告。

只有 source/paper fidelity gate、共同协议运行和报告审计全部通过后，才可将 manifest 的 `strict_comparison_eligible` 从 `false` 改为 `true`。论文报告的 91.1% 不是本项目实测分数。

## 失败重试

训练完成但导出或推送失败时，使用原始 run 目录重试，不重复训练：

```bash
bash scripts/export_retacred_reference_improved_baseline_report.sh \
  runs/retacred_reference_improved_baseline/<timestamp>_seed13
bash scripts/publish_retacred_reference_improved_baseline_report.sh \
  --run-dir runs/retacred_reference_improved_baseline/<timestamp>_seed13
```
