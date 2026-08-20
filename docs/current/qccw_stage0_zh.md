# QCCW Stage-0 机制门禁

QCCW（Quantum Connected-Correlation Consensus Witness）在现有 synthetic dynamic-address task 上枚举全部 15 个无序 key pair，读取两组 connected `XX`。推理不接收 label 或 evidence pair；固定 witness 和 bounded zero-sum action 不变。

## 运行

```bash
python -m pytest -q tests/test_q_connected_consensus_witness.py
python experiments/run_q_connected_consensus_witness_stage0.py --device cuda
```

配置固定为 `configs/q_connected_consensus_witness_stage0.json`，只跑 seed 7。结果写入 `runs/q_connected_consensus_witness_stage0/`，不提交 runs。

## 预声明门禁

- product connected null `<=1e-6`；pair AUC `>=0.75`；valid/test gain `>=0.05`。
- QCCW 超等参数 bilinear `>=0.02`，超 entangler-cut 和 key-shuffle `>=0.03`。
- harm `<=0.02`，connected score/gradient/residual 均有限。
- 任一核心条件失败即停止，不调参、不换 seed、不进入五 seed 或真实数据。

## 当前结果

seed 7 CPU smoke：QCCW valid/test accuracy delta `+0.1836/+0.1895`，pair AUC `0.9977/0.9975`，product null 最大 `3.9e-7`，harm `0`。entangler-cut 和 key-shuffle 均产生预期下降。

但 bilinear valid/test delta 为 `+0.1836/+0.1875`，未达到量子超 classical margin，Stage-0 gate 为 `fail`。这说明当前 pair witness 有任务效用，但其信号可由等参数 classical bilinear 表达；不得据此宣称量子优势或继续多 seed。
