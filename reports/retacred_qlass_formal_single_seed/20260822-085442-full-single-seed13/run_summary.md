# Q-LASS Re-TACRED 单 seed 汇总

- seed: `13`
- selection: `macro_f1_then_loss`（仅 valid）
- test 未用于训练、checkpoint 选择或调参。

| split | Q-LASS macro-F1 | classical macro-F1 | Q-LASS - classical |
| --- | ---: | ---: | ---: |
| valid | 0.294543 | 0.294721 | -0.000178 |
| test | 0.251820 | 0.252165 | -0.000345 |

test Q-LASS - classical： accuracy -0.000224，loss -0.000679，correct-label margin +0.000368。
