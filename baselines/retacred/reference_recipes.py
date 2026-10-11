"""Explicit reconstructions, not the papers' unrecovered author configurations."""
from __future__ import annotations
import copy
from .common import cohort

BASE = dict(epochs=5, logical_batch=32, learning_rate=1e-5, weight_decay=.01,
            betas=[.9,.999], eps=1e-8, warmup_ratio=.06, max_grad_norm=1.,
            optimizer='torch_AdamW', scheduler='linear_warmup_decay', precision='fp32',
            selection='dev_all_example_non_null_micro_f1', overflow='reject', max_length=512,
            fidelity='paper_interpretive', exact_author_recipe=False)

def recipe(model):
    ids={r['id'] for r in cohort()['models'] if r.get('run_eligible',True)}
    if model not in ids: raise ValueError('unknown or excluded reference')
    r=copy.deepcopy(BASE);r['model']=model
    if model.startswith('ctl_drp_'):
        from .ctl_drp_task import RECIPE
        r.update(epochs=10,logical_batch=512,learning_rate=1e-5,weight_decay=0.,
                 warmup_ratio=0.,scheduler='constant',selection='final_epoch',native_recipe=RECIPE)
    elif model=='metaentail':
        r.update(epochs=3,logical_batch=32,learning_rate=2e-5,warmup_ratio=.1,
                 training_unit='expanded_NLI_pair',inference='group_argmax_entail_then_confidence',
                 abstention='explicit_separate_prediction_column_no_null_fallback')
    elif model=='sequence_routing_h3':
        r.update(epochs=5,logical_batch=24,learning_rate=1e-5,weight_decay=0.,
                 optimizer='torch_Adam',scheduler='source_LinearLR_default_start1over3_epoch_step',
                 warmup_ratio=0.,max_grad_norm=.5,selection='dev_all_example_non_null_micro_f1',
                 heads=['head_3'],routing_hidden=256,routing_output=512,
                 adaptations=['canonical40_not_source41','locked_common_roberta_checkpoint',
                              'canonical_train_only_relation_token_order','source24x5_not_paper64x6'])
    elif model=='curriculum':
        r.update(epochs=6,logical_batch=32,learning_rate=5e-4,layer_decay=.7,
                 reviewer_count=3,reviewer_epochs=2,reviewer_training='own_subset_only',
                 curriculum_pacing='cumulative_easiest_third_then_two_thirds_then_all',
                 graph_heads=4,graph_head_dim=128,graph_layers=2,
                 bilinear='concat(marker_hidden,GAT_entity_mean)_subject_vs_object',
                 parser='spacy_en_core_web_sm_3.7.1_pretokenized')
    elif model.startswith('luke_'):
        r['adaptations']=['canonical40','modern_torch_AdamW','weighted_micro_tail',
                         'common_dev_selection_not_unknown_CoRE_author_port']
        if model=='luke_focal': r.update(loss='focal',focal_gamma=2.,focal_alpha=1.)
        elif model=='luke_resample':r.update(sampling='inverse_class_frequency_with_replacement_N_draws')
        elif model=='luke_core':r.update(inference='factual_probability_minus_entity_only_minus_blank',
            core_grid=[round(-2+i*.1,1) for i in range(41)],
            restriction='train_observed_ordered_entity_type_pairs_only; unseen_pair_unrestricted')
        elif model=='luke_cfie':r.update(inference='Y_minus_alpha_counterfactual_Y',
            alpha_grid=[round(i*.2,1) for i in range(11)],beta=0.,
            auxiliary_loss='main_CE+.9_sentence_CE+.9_tag_CE+entity_CE',
            counterfactual='mask_entities_and_dependency_one_hop_neighbors',
            causal_branches='sentence_CLS,role_entity_hidden,entity_type_and_POS',
            parser='spacy_en_core_web_sm_3.7.1_pretokenized')
    else:
        r.update(logical_batch=64,learning_rate=3e-5,eps=1e-6,weight_decay=0.,warmup_ratio=.1,overflow='source_truncate512_reject_missing_entity_marker')
        if model=='ptr':
            r.update(overflow='native_PTR_source_limit512_reject_invalid_prompt_masks',precision='native_PTR_fp32',optimizer='native_PTR_dual_AdamW',
                     selection='final_epoch',fidelity='source_faithful_adapted')
        elif model!='typed_marker_comparison_2025':
            r['fidelity']='paper_interpretive'
            r['adaptations']=['pinned_author_marker_mechanism','common_FP32_no_upstream_AMP','common_dev_checkpoint_selection']
        else:r['adaptations']=['typed_punctuation_common_large_marker_pooling_choice',
                              'paper_grid_unavailable_no_91_53_replay_claim']
    return r

def inventory():
    return [dict(id=r['id'],recipe=recipe(r['id'])) for r in cohort()['models'] if r.get('run_eligible',True)]
