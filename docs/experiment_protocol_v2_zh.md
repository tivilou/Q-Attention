# Re-TACRED 实验协议 v2

本协议用于后续正式实验。核心要求是：验证集只用于模型选择和超参数选择，测试集只用于最终报告，禁止在测试集上扫描 projector、filter 或 steering gain。

## 单次运行

配置中的三个数据路径含义如下：

```text
train_path -> 训练 baseline、收集 projector key
valid_path -> 选择 checkpoint 和 spectral filter
test_path  -> classical/quantum/routing 最终评价，以及所选 spectral filter 的一次性评价
```

运行命令保持不变：

```bash
python experiments/run_relation_smoke_pipeline.py \
  --config configs/retacred_low_resource_gpu.json \
  --device cuda
```

pipeline 会自动生成：

```text
pipeline_summary.json   命令、耗时、Git commit、环境、seed 和数据 SHA-256
run_summary.json        最终 split 上的结构化指标
run_summary.md          最终 split 上的可读表格
```

当配置存在 `test_path` 时，`run_summary.*` 中的 baseline 和所有 variant 都来自 test。spectral filter 的 `best_by_macro_f1` 是 validation 选择结果，`best_on_test` 才是最终测试结果。

## 五随机种子运行

正式结果使用固定 seed 集：

```bash
python experiments/run_relation_seed_matrix.py \
  --config configs/retacred_low_resource_gpu.json \
  --seeds 13,17,23,29,31 \
  --output_root runs/retacred_low_resource_multiseed \
  --device cuda
```

Full 设置只需替换配置和输出目录：

```bash
python experiments/run_relation_seed_matrix.py \
  --config configs/retacred_full_gpu.json \
  --seeds 13,17,23,29,31 \
  --output_root runs/retacred_full_multiseed \
  --device cuda
```

`--seed` 会同时覆盖模型训练、projector 抽样、quantum feature map、spectral sweep 和 routing 的随机种子。中断后可添加 `--skip_existing`，已存在 `run_summary.json` 的 seed 不会重跑。

## 输出目录

```text
runs/retacred_*_multiseed/
  seed_13/
  seed_17/
  seed_23/
  seed_29/
  seed_31/
  seed_matrix_manifest.json
  seed_summary.json
  seed_summary.md
```

`seed_summary.json` 保存均值、样本标准差、近似 95% 置信区间，以及每个方法获得正向 macro-F1 增益的 seed 数量。

## 提交结果

不提交数据集、模型权重或完整 `runs/`。提交以下文件即可：

```text
seed_matrix_manifest.json
seed_summary.json
seed_summary.md
每个 seed 的 pipeline_summary.json
每个 seed 的 run_summary.json
每个 seed 的 run_summary.md
各模块 metrics.json
spectral_filter_sweep/summary.json
spectral_filter_sweep/test_predictions.jsonl
```

在 multi-seed 汇总完成前，不将单个 seed 的微小提升作为论文结论。
