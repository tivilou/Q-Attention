# Q-VRES Formal Single-Seed Selector-Parallel Run

Seed: `13`
Pilot validation gate: `fail`

| selector | valid macro-F1 | test macro-F1 | delta test macro-F1 | parameters |
| --- | ---: | ---: | ---: | ---: |
| disabled | 0.217253 | 0.182597 | 0.000000 | 0 |
| q_causal_transport | 0.215264 | 0.181179 | -0.001418 | 72 |
| classical_causal_transport | 0.217354 | 0.183178 | 0.000581 | 72 |
| q_causal_key_only | 0.218519 | 0.184319 | 0.001722 | 72 |

The pilot gate uses validation macro-F1 only:
- Q-VRES minus baseline: `-0.001989`
- Q-VRES minus classical transport: `-0.002090`

Test metrics are reported but must not be used to tune the method or decide hyperparameters.
