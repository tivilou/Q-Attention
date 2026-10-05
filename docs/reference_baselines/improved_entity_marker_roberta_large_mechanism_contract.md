# Improved Entity Marker / RoBERTa-large 机制契约

## 证据边界

这是第一优先级的外部 reference baseline。本文档只冻结上游代码和论文中可以核对的事实；在 fidelity gates 通过前，不把论文中的 91.1% 当作本项目结果。

- 论文：[An Improved Baseline for Sentence-level Relation Extraction](https://arxiv.org/abs/2102.01373)
- 官方仓库：[RE_improved_baseline](https://github.com/wzhouad/RE_improved_baseline)
- 已锁定 source commit：`fe00781116ab4c34cbcd9a2ae5b02513afae87ee`
- 项目 manifest：`configs/reference_baselines/improved_entity_marker_roberta_large.manifest.json`

## 上游前向路径

1. 输入为 TACRED/Re-TACRED 的 token 序列、subject/object 的 token span、实体类型和关系标签。
2. `typed_entity_marker_punct` 在两个实体边界加入上游定义的标记和实体类型 token。
3. RoBERTa-large 编码带标记的序列。
4. 读取 subject/object 标记位置的 hidden state，拼接成实体对表示。
5. 通过两层 MLP 输出关系 logits，并使用交叉熵训练。

## 训练事实

- 上游入口：`train_retacred.py`。
- 默认 backbone：`roberta-large`。
- 默认输入格式：`typed_entity_marker_punct`。
- 上游使用 AdamW、线性 warmup/decay、AMP 和 gradient checkpointing。
- test split 只能用于最终评估，不能用于训练或 checkpoint 选择。

## 必须完成的门禁

- 用固定输入核对 tokenizer、marker 插入位置、entity-position readout 和 logits。
- 核对 checkpoint 缺失/多余键、权重 hash 和加载后的输出。
- 用项目固定的 train/valid/test、seed、指标和报告契约跑一次共同协议。
- exporter 只发布审计后的 `reports/`，不发布 checkpoint、预测或完整日志。

任何门禁失败时，manifest 保持 `planned`/`not_verifiable`，不得把上游论文分数写入项目结果页。
