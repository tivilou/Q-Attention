# QCDD seed-7 readout 诊断

QCDD（Coherence-Destruction Differential）复用 QCCW 的四量子比特状态制备，读取 coherent-minus-dephased connected `YYYY`。本轮只诊断 readout，不施加 attention action；量子模型与 sine/cosine control 均为 4 个可训练参数。

## 运行

```bash
python -m pytest -q \
  tests/test_q_coherence_destruction_differential.py \
  tests/test_run_q_coherence_destruction_differential.py \
  tests/test_q_connected_consensus_witness.py

python experiments/run_q_coherence_destruction_differential.py --device cpu
```

配置固定为 `configs/q_coherence_destruction_differential.json`，只跑 seed 7。结果写入 `runs/q_coherence_destruction_differential/`，不提交 runs。

## 当前结果

| 指标 | valid | test |
| --- | ---: | ---: |
| QCDD pair AUC | 0.998930 | 0.998645 |
| sine/cosine control pair AUC | 0.999069 | 0.998754 |
| QCDD - control | -0.000139 | -0.000109 |
| key-shuffle pair AUC | 0.520844 | 0.518831 |

product null 最大值为 `4.47e-8`，dephased null 为 `0`；梯度、四参数预算和 deterministic replay 均通过。shot p95 为 `246`，低于上限 `4096`。

唯一失败的机制条件是 `quantum_control_margin`：QCDD 没有超过等参数 sine/cosine control。因此 gate 为 `fail`。不得调参、增加 seed、施加 attention action，或进入真实数据、硬件和完整实验。下一步重新筛选机制，不继续修饰 QCDD。
