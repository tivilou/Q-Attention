# CTL-DRP 独立 reference 接口（待准入）

## 当前边界

此页是待审查接口说明，不是已经交给合作者的全量运行授权。默认可执行
cohort 仍是五个配置。CTL 两个配置仍保持原有阻塞状态；实现入口、通过
小规模 fixture 检查，不代表完成真实数据复现或正式发布。

CTL 采用独立联合句子/关系描述编码器与句子内候选分母，不复用 marker/PTR
trainer 或它们的 batch64 配方。登记 ID 精确为 `ctl_drp_supervised` 和
`ctl_drp_unsupervised`，不是 `ctl_drp_sup/unsup`。

## 无副作用查看

在自己的项目根目录运行：

```bash
bash scripts/run_retacred_reference_suite.sh ctl-run --describe --variant sup
```

会列出登记状态、审查要求、锁定文件哈希以及固定 recipe，不加载 Torch、
GPU 或权重，不训练、不联网、不写报告。真实运行前需要逐项通过：

1. 对应 cohort 配置进入可执行状态；本改动不改变该状态。
2. 独立机制与发布审查给出 `ctl-drp-independent-review.v1` 外部记录：
   `status=accepted`、`reviewer_role=independent`、非空 reviewer、models 包含
   精确 ID、`fidelity=paper_interpretive`、`source_sha256` 与 describe 给出的
   全部代码/锁定文件哈希完全一致。实现 agent 不自行签发此记录。
3. 有资格完整数据及逐行 H/编码审计。句长溢出、未知训练类型对、哈希变化、
   重复/无效样本 ID 都报错；不丢样本、不截断、不使用 heldout gold 修补 H。
4. 指定的官方 checkpoint 通过锁定哈希、大小及加载一致性检查。
5. 合作者使用现有 `1.1` 分支发布策略，保留正常 Git 身份，无无关 dirty/
   staged 内容，且配置 HTTPS `PROJECT_EXCHANGE_URL` 和
   `PROJECT_EXCHANGE_TOKEN`。token 不进入命令记录或报告。

## 数据资格接口

推荐直接传 `ctl-drp-physical-splits.v1` 外部资格记录：`status=qualified`、
`task=retacred_sentence_relation_40`、`synthetic=false`、canonical_labels 为
锁定 ID 顺序，train/dev/test 各包含记录目录内的相对 file、显式 format、
sha256 和 count。真实数量必须是 58465/19584/13418。

已有统一 suite 的 `dataset_qualification.json` 也可使用，但必须另外指定
`--prepared-data` 指向同一次转换的目录。目录有 `data_manifest.json` 和
train.json/dev.json/test.json；manifest 中 valid 条目对应已重命名的 dev.json。
桥接核对外部 qualified receipt 与 manifest 全部 split 条目相同，读取当前
prepared-data 下的物理文件，而不是他人机器上的绝对 output_path。缺失 ID
才生成稳定 split/序号/内容哈希 ID；已有错误 ID 不自动修复。资格不是由
文件夹名字推断出来的。原始声明哈希、转换后实测哈希与桥接哈希分别保存。

## 准入后的命令模板

以下变量由合作者指向自己机器的文件；没有写死开发服务器路径。

```bash
bash scripts/run_retacred_reference_suite.sh ctl-run \
  --variant sup --seed 13 --micro-batch 4 --gpu auto \
  --qualification "$QUALIFICATION" \
  --review-receipt "$INDEPENDENT_REVIEW" \
  --cache "$SIMCSE_CACHE" --paper-source "$PAPER_NATIVE_TEX" \
  --weight "$PINNED_WEIGHT" --run-dir "$PRIVATE_RUN"
```

若资格记录来自旧 suite，追加 `--prepared-data "$PREPARED_DATA"`。固定逻辑
batch 是 512 个句子，十个 epoch，即 1150 次完整逻辑更新；LR 1e-5、FP32、
无 warmup/clipping，最终 epoch 评估，不自动搜索最佳 checkpoint。物理
micro-batch 按句子数划分、加权累积；恢复时不可改变，因为 dropout 分区
会影响轨迹。`--gpu auto` 选择一张可见 GPU；当前不提供 CTL 拓扑迁移、
AMP 或 OOM 自动降档。

每次逻辑更新原子保存 checkpoint；保留两个滚动 generation、所选训练
事件前后 generation 和一个写入暂存空间。正式入口保守检查至少
`18×checkpoint权重文件字节数+2 GiB` 的额外可用磁盘，不在磁盘不足时开跑。
完整 `.pt` 张量留在私有 run/交换系统，Git 只收紧凑白名单。

## 恢复和发布

- 正常暂停或训练异常：原命令追加 `--resume`，去掉 `--pause-after`。
  若初始 worker 在创建 checkpoint 前失败，复用父级配置，重新执行初始化，
  不伪称已恢复数值状态。配方/数据/权重/GPU/runtime/source 改变都拒绝续跑。
- 全部更新完成后才做 dev/test，所选 train/test 样本的完整关键张量必须实际
  记录、可读取且逐值验证。训练只写 COMPUTE_COMPLETE；不是 RUN_COMPLETE。
- 必需上传按精确文件名、哈希、大小、数量、namespace 校验，大文件验证有序
  shard 重组哈希。上传成功且原子报告导出后才写 RUN_COMPLETE。
- 已算完但上传/导出/commit/push 失败，执行下列命令，不再次训练或测试：

```bash
bash scripts/run_retacred_reference_suite.sh ctl-run --export-only \
  --variant sup --seed 13 --run-dir "$PRIVATE_RUN" \
  --review-receipt "$INDEPENDENT_REVIEW"
```

`--export-only` 不需要 GPU、cache、weight 或资格文件，只验证冻结记录及
实际完成证据。已有报告但完成标记缺失也先核验交付，再修复标记。已完成
上传回执可复用；不生成虚假成功回执。`--no-push` 只禁 Git push，仍上传、
导出并提交准确的报告白名单。

公开目录是仓库根目录下 `reports/retacred_ctl_drp_reference/<run目录名>/`：
metrics.json、run_summary.json、data_identity.json、data.sha256、
data_counts.txt、run_summary.data、export_manifest.json。原始句子、数值张量、
checkpoint、predictions、token 和完整 worker 日志不进入 Git。微平均
precision/recall/F1、accuracy、macro-F1、逐类统计以及真实 loss 状态来自
全部 heldout 数据；gold 不在 H 时 NLL 显式为 infinite/null，不伪造有限 loss。
父级命令/状态保存在私有 `<run目录名>.ctl-formal-control/`。

这仍是 `paper_interpretive` reference，不是 strict/source-faithful baseline。
全量真实数据、unsupervised 权重运行、实际交付与门户接入、独立审查均需要
单独证据；此接口的 fixture 成功不替代这些步骤。
