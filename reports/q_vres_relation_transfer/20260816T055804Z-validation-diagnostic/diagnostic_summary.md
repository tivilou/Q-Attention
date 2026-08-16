# Q-VRES Validation Diagnostic

Split: `valid`
Records: `19584`
Hypothesis assessment: `mixed_correlational_support`

| selector | macro recall | delta recall | macro F1 | delta F1 | correct->wrong | wrong->correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| disabled | 0.214727 | 0.000000 | 0.217253 | 0.000000 | - | - |
| q_causal_transport | 0.207161 | -0.007566 | 0.215264 | -0.001989 | 96 | 210 |
| classical_causal_transport | 0.209019 | -0.005708 | 0.217354 | 0.000102 | - | - |
| q_causal_key_only | 0.213761 | -0.000966 | 0.218519 | 0.001266 | 58 | 81 |

## Evidence And Intervention

| selector | layer | evidence vs abs grad | evidence vs positive grad | attention delta vs positive grad | attention TV |
| --- | ---: | ---: | ---: | ---: | ---: |
| q_causal_transport | 0 | 0.607720 | 0.425764 | 0.212330 | 0.021593 |
| q_causal_transport | 1 | 0.644825 | 0.475918 | 0.051842 | 0.016564 |
| q_causal_key_only | 0 | 0.053060 | 0.047278 | -0.154208 | 0.013350 |
| q_causal_key_only | 1 | -0.129952 | -0.067191 | -0.364771 | 0.031613 |

## Largest Q-Causal Recall Drops

| relation | support | delta recall | delta F1 |
| --- | ---: | ---: | ---: |
| per:other_family | 34 | -0.058824 | -0.071429 |
| per:siblings | 33 | -0.030303 | 0.019744 |
| per:date_of_death | 234 | -0.029915 | -0.036331 |
| per:parents | 69 | -0.028986 | -0.009279 |
| org:founded | 36 | -0.027778 | -0.033333 |
| per:stateorprovinces_of_residence | 37 | -0.027027 | -0.007682 |
| org:website | 94 | -0.021277 | -0.022399 |
| org:country_of_branch | 338 | -0.020710 | -0.014641 |
| org:stateorprovince_of_branch | 98 | -0.020408 | -0.019740 |
| org:top_members/employees | 462 | -0.019481 | -0.023584 |
| per:children | 114 | -0.017544 | -0.016395 |
| per:title | 998 | -0.017034 | -0.005586 |
| per:city_of_death | 148 | -0.013514 | -0.018344 |
| org:founded_by | 76 | -0.013158 | -0.021256 |
| per:cities_of_residence | 98 | -0.010204 | -0.003986 |
| per:countries_of_residence | 192 | -0.005208 | 0.000981 |
| per:cause_of_death | 193 | -0.005181 | 0.003851 |
| per:age | 256 | -0.003906 | 0.011248 |
| per:employee_of | 576 | -0.001736 | 0.000339 |
| org:dissolved | 7 | 0.000000 | 0.000000 |

## Interpretation

- Suggested next step: `inspect_relation_rows_before_selecting_q_causal_or_key_only`
- Minority-recall subhypothesis: `correlational_support`
- All method-selection evidence in this report comes from validation unless the report is explicitly marked test-only.
- Evidence correlations and first-order margin deltas are attribution proxies, not causal proof.
- The report contains no raw text, per-example predictions, token sequences, checkpoints, or gradient tensors.
