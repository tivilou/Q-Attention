# Re-TACRED reference suite

Only requested configurations completed; not a full top-15 reproduction.

| model | seed | benchmark micro-F1 | accuracy | macro-F1 | loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| metaentail | 13 | 0.909123 | 0.922716 | 0.797941 | 0.416426 |
| sequence_routing_h3 | 13 | 0.893541 | 0.920256 | 0.737188 | 0.335827 |
| ctl_drp_supervised | 13 | 0.908273 | 0.927933 | 0.807943 | N/A (see loss_scope) |
| typed_marker_comparison_2025 | 13 | 0.908834 | 0.930168 | 0.796330 | 0.283146 |
| curriculum | 13 | 0.892121 | 0.918468 | 0.754959 | 0.310449 |
| ctl_drp_unsupervised | 13 | 0.897586 | 0.917946 | 0.795069 | N/A (see loss_scope) |
| typed_marker_punct | 13 | 0.907819 | 0.929721 | 0.790997 | 0.276014 |
| ptr | 13 | 0.906707 | 0.927262 | 0.813732 | 1.514148 |
| luke_core | 13 | 0.912948 | 0.932106 | 0.805022 | 0.343401 |
| luke_cfie | 13 | 0.909316 | 0.930243 | 0.803213 | 0.620038 |
| luke_resample | 13 | 0.848532 | 0.866075 | 0.737440 | 0.834352 |
| typed_marker | 13 | 0.901726 | 0.924504 | 0.795559 | 0.302878 |
| entity_marker_punct | 13 | 0.906024 | 0.928454 | 0.797835 | 0.292482 |
| luke_focal | 13 | 0.906297 | 0.928231 | 0.800339 | 0.137623 |
| entity_marker | 13 | 0.904800 | 0.927560 | 0.792033 | 0.299680 |
