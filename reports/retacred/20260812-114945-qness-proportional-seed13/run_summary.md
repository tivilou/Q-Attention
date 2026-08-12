# Re-TACRED Q-NESS Proportional Gate

- revision: `2ef4fc7e775496f87de1d9e8f48b15d19ccf598d`
- seed: `13`
- overall gate: `true`
- evaluation: validation only; blind test was not read.

| Stage | Loss | Macro-F1 | Correct-label margin | Selectivity |
| --- | ---: | ---: | ---: | :---: |
| baseline | 2.395113 | 0.187350 | -1.024377 | false |
| core_quantum | 2.389279 | 0.187095 | -1.017602 | false |
| selector_qness | 2.328071 | 0.189050 | -0.936665 | true |
| selector_qness_classical | 2.355604 | 0.194521 | -0.983383 | true |
| selector_qness_commuting | 2.322715 | 0.176527 | -0.944025 | true |
| selector_qness_separable | 2.409092 | 0.182848 | -1.068788 | false |
| selector_qness_phase_scrambled | 2.325297 | 0.183036 | -0.930553 | true |
| selector_qness_dephased | 2.360708 | 0.190905 | -0.978293 | false |

## Decision

- task checks: `{"baseline_usable": true, "qness_f1_guardrail": true, "qness_has_task_checkpoint": true, "qness_improves_over_classical_control": true, "qness_improves_over_quantum_core": true}`
- resource checks: `{"commuting_commutator_removed": true, "dephased_off_diagonal_removed": true, "qness_commutator_nonzero": true, "qness_mutual_information_nonzero": true, "qness_off_diagonal_nonzero": true, "separable_mutual_information_removed": true}`
- gains: `{"qness_over_classical_control_loss": 0.027533266693353653, "qness_over_classical_control_macro_f1": -0.005471004265232421, "qness_over_quantum_core_loss": 0.0612080879509449, "qness_over_quantum_core_macro_f1": 0.0019554165324002504}`

This is a screening gate, not evidence of statistical significance or quantum advantage.
