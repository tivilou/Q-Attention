# Improved Entity Marker 适配边界

## 允许的适配

- 将项目 canonical JSONL 字段映射到上游 TACRED 风格的 `token`、`subj_start`、`subj_end`、`obj_start`、`obj_end`、`subj_type`、`obj_type`、`relation`。
- 将项目的 `valid` 文件映射为上游的 `dev` 文件名；内容和顺序保持不变。
- 将上游输出转换为项目统一的 train/valid/test 指标、逐类结果、混淆矩阵和 provenance 报告。
- 使用环境变量或命令行显式指定 Hugging Face cache、GPU 和 upstream checkout 路径。

## 禁止的改变

- 不能把 typed punctuation marker 改成项目自定义 marker。
- 不能按项目的字母序 label map 替换上游 Re-TACRED label IDs；必须保留上游映射并在报告中记录。
- 不能把 entity representation 改成平均池化、CLS 或项目当前小型 Transformer 的 span pooling。
- 不能改变上游 loss、optimizer、scheduler、AMP 或 checkpoint-selection 逻辑后仍称为 source-faithful。
- 不能用 test split 调参、选择 checkpoint 或 rejection-sampling 修正预测。

## 未决项

- 上游 checkpoint 是否可取得、hash 是否稳定。
- 上游依赖版本与当前协作者环境是否兼容。
- 上游的评估脚本只输出 F1 时，如何在不改变预测路径的前提下补齐项目要求的 precision/recall/accuracy/loss。

这些未决项完成前，manifest 的 `strict_comparison_eligible` 必须保持 `false`。
