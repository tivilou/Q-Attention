"""Actual tiny Torch/Roberta mechanism tests, not reported-score reproduction."""
import copy
import os
from pathlib import Path
import pytest

from baselines.retacred import ctl_drp_neural as n
from baselines.retacred.ctl_drp_smoke import examples, synthetic_table
from baselines.retacred.ctl_drp_paper import candidate_reference
from baselines.retacred.common import cohort, select_models

CACHE=Path(os.environ["CTL_DRP_BPE_CACHE"])
PAPER=Path(os.environ["CTL_DRP_NATIVE"])

@pytest.fixture(scope="module")
def encoded():
    tok=n.tokenizer_from_cache(CACHE)
    descriptions=n.descriptions_from_native(PAPER)
    groups=n.encode_groups(examples(),synthetic_table(),tok,descriptions,split="train")
    return tok,descriptions,groups

def test_marker_role_order_and_gold_isolation():
    original=examples()[0]
    modified=copy.deepcopy(original);modified["relation"]="no_relation"
    assert n.marked_sentence(original)==n.marked_sentence(modified)
    assert n.marked_sentence(original)["text"]=="[u1] Alice (subject person) [u2] joined [u3] Acme (object organization) [u4] ."
    modified["subj_start"],modified["obj_start"]=modified["obj_start"],modified["subj_start"]
    modified["subj_end"],modified["obj_end"]=modified["obj_end"],modified["subj_end"]
    assert n.marked_sentence(modified)["text"].startswith("[u3] Alice")
    modified["subj_start"]=modified["obj_start"]
    modified["subj_end"]=modified["obj_end"]
    with pytest.raises(ValueError,match="overlapping"):n.marked_sentence(modified)

def test_actual_bpe_prompt_gold_isolation_overflow(encoded):
    tok,descriptions,groups=encoded
    assert len(descriptions)==40 and tok.get_vocab_size()==50269
    changed=examples();changed[0]["relation"]="per:spouse"
    one=n.encode_groups(examples(),synthetic_table(),tok,descriptions,split="test")
    two=n.encode_groups(changed,synthetic_table(),tok,descriptions,split="test")
    assert one==two and all("gold" not in g for g in one)
    assert groups[0]["pairs"][0]["input_ids"][0]==0
    assert groups[0]["pairs"][0]["prompt"]=="There is no relation between subject and object"
    assert all(len(g["pairs"])==len(g["candidates"]) for g in groups)
    with pytest.raises(ValueError,match="overflow"):
        n.encode_groups(examples(),synthetic_table(),tok,descriptions,split="test",max_length=8)
    with pytest.raises(ValueError,match="gold excluded"):
        n.encode_groups(changed,synthetic_table(),tok,descriptions,split="train")

def test_actual_joint_encoder_loss_and_head_gradient_vs_independent_scalar(encoded):
    import torch
    torch.set_num_threads(1)
    model=n.tiny_model();batch=n.collate_groups(encoded[2])
    gold=[g["gold"] for g in encoded[2]]
    result=model(**batch,gold=gold)
    refs=[];cursor=0
    for names,label in zip(batch["candidate_groups"],gold):
        h={name:result["first_token"][cursor+i].detach().tolist() for i,name in enumerate(names)}
        reference=candidate_reference(h,model.head.weight.detach()[0].tolist(),float(model.head.bias.detach()[0]),names,label)
        refs.append(reference)
        actual=result["probabilities"][len(refs)-1].detach().tolist()
        assert actual==pytest.approx(list(reference["probabilities"].values()),abs=1e-6)
        cursor+=len(names)
    assert float(result["loss"].detach())==pytest.approx(sum(r["loss"] for r in refs)/len(refs),abs=1e-6)
    result["loss"].backward()
    expected=[sum(r["weight_gradient"][j] for r in refs)/len(refs) for j in range(32)]
    assert model.head.weight.grad[0].tolist()==pytest.approx(expected,abs=1e-6)
    assert float(model.head.bias.grad)==pytest.approx(0.,abs=1e-6)
    assert model.encoder.pooler is None
    params=dict(model.named_parameters())
    for name in ("encoder.embeddings.word_embeddings.weight",
                 "encoder.encoder.layer.0.attention.self.query.weight",
                 "encoder.encoder.layer.0.attention.self.key.weight",
                 "encoder.encoder.layer.0.attention.self.value.weight"):
        assert params[name].grad is not None and bool(torch.count_nonzero(params[name].grad))

def test_sentence_grouping_not_global_softmax_and_prediction_gold_independence(encoded):
    import torch
    model=n.tiny_model();model.eval();groups=encoded[2]
    batch=n.collate_groups(groups);gold=[g["gold"] for g in groups]
    one=model(**batch,gold=gold)
    two=model(**batch,gold=["no_relation"]*len(groups))
    assert torch.equal(one["scores"],two["scores"]) and one["predictions"]==two["predictions"]
    for i,group in enumerate(groups):
        alone=model(**n.collate_groups([group]),gold=[group["gold"]])
        assert torch.allclose(one["probabilities"][i],alone["probabilities"][0],atol=1e-6,rtol=1e-5)
    with pytest.raises(ValueError,match="grouping"):
        model(**{**batch,"candidate_groups":batch["candidate_groups"][:1]})

def test_na_only_has_probability_one_zero_loss_and_zero_grad():
    import torch
    model=n.tiny_model()
    result=model(torch.tensor([[0,10,2]]),torch.ones((1,3),dtype=torch.long),[["no_relation"]],gold=["no_relation"])
    assert result["probabilities"][0].tolist()==[1.] and float(result["loss"].detach())==0.
    result["loss"].backward()
    assert all(p.grad is None or not bool(torch.count_nonzero(p.grad)) for p in model.parameters())

def test_ragged_loss_is_mean_over_sentences_not_candidate_rows(encoded):
    import torch
    groups=copy.deepcopy(encoded[2]);groups[1]["candidates"]=["no_relation"]
    groups[1]["pairs"]=groups[1]["pairs"][:1];groups[1]["gold"]="no_relation"
    model=n.tiny_model();batch=n.collate_groups(groups)
    result=model(**batch,gold=[g["gold"] for g in groups])
    assert torch.equal(result["loss"],result["per_sentence_loss"].mean())
    assert result["per_sentence_loss"][1]==0
    assert float(result["loss"].detach())==pytest.approx(float(result["per_sentence_loss"][0].detach())/2)

def test_drift_rejected_before_any_state_mutation(tmp_path,encoded):
    import torch
    model=n.tiny_model();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5)
    batch=n.collate_groups(encoded[2]);gold=[g["gold"] for g in encoded[2]]
    model(**batch,gold=gold)["loss"].backward();optimizer.step()
    p=tmp_path/"checkpoint.pt";contract={"data":"synthetic", "batch_unit":"sentences"}
    n.save_checkpoint(p,model,optimizer,contract=contract,cursor=1)
    before={k:v.clone() for k,v in model.state_dict().items()};rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError,match="before mutation"):
        n.load_checkpoint(p,model,optimizer,contract={"data":"other"})
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    assert torch.equal(rng,torch.get_rng_state())

def test_excluded_ptr_is_preserved_but_not_an_all_execution_candidate():
    registry=cohort();row=next(r for r in registry["models"] if r["id"]=="ptr_reversed")
    assert row["run_eligible"] is False and row["status"]=="excluded_noncomparable"
    assert row["comparison_role"]=="changed_task_reference_only"
    assert len([r for r in registry["models"] if r.get("run_eligible",True)])==15
    with pytest.raises(ValueError,match="not runnable") as error:select_models("all")
    assert "ptr_reversed:" not in str(error.value)
    with pytest.raises(ValueError,match="no jobs were launched"):select_models("ptr_reversed")
