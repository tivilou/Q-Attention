"""Bounded physical-file fixtures and real tiny neural task checks, not formal data."""
import copy
import json
import os
from pathlib import Path
import random

import pytest

from baselines.retacred import ctl_drp_task as task
from baselines.retacred import ctl_drp_neural as neural
from baselines.retacred.common import cohort, digest, metrics, select_models
from baselines.retacred.ctl_drp_smoke import examples


def fixture_receipt(directory, *, jsonl=False):
    train = examples()
    extra = copy.deepcopy(train[0]); extra.update(id="synthetic-c", relation="per:schools_attended")
    train.append(extra)
    rows = {"train": train, "dev": examples(), "test": examples()}
    spec = {"schema": "ctl-drp-physical-splits.v1", "status": "qualified",
            "task": "retacred_sentence_relation_40", "canonical_labels": task.LABELS,
            "synthetic": True, "splits": {}}
    for split, records in rows.items():
        if split != "train":
            for i, row in enumerate(records): row["id"] = f"synthetic-{split}-{i}"
        path = directory/(split+(".jsonl" if jsonl else ".json"))
        path.write_text(("\n".join(json.dumps(r) for r in records) if jsonl else json.dumps(records))+"\n")
        spec["splits"][split] = {"file": path.name, "format": "jsonl" if jsonl else "json_array",
                                "sha256": digest(path), "count": len(records)}
    receipt = directory/"qualification.json"
    receipt.write_text(json.dumps(spec))
    return receipt


@pytest.fixture
def physical(tmp_path):
    return task.PhysicalSplits(fixture_receipt(tmp_path))


@pytest.fixture(scope="module")
def assets():
    import torch
    torch.set_num_threads(1)
    return (neural.tokenizer_from_cache(Path(os.environ["CTL_DRP_BPE_CACHE"])),
            neural.descriptions_from_native(Path(os.environ["CTL_DRP_NATIVE"])))


def make(data, assets, *, micro=1, dropout=0., device="cpu", seed=13):
    return task.CTLDRPTask(data, *assets, neural.tiny_model(seed=seed, dropout=dropout),
                           micro_batch=micro, device=device, seed=seed)


def test_physical_split_binding_train_only_and_jsonl(tmp_path, assets):
    data = task.PhysicalSplits(fixture_receipt(tmp_path, jsonl=True))
    assert data.binding["physical_split_identity_verified"] is True
    assert data.table["physical_split_identity_verified"] is False  # historical primitive never relabeled
    assert data.table["training_rows"] == 3
    assert task.candidate_labels(data.table, "PERSON", "PERSON") == ["no_relation", "per:spouse"]
    assert data.coverage["test"]["samples"] == 2
    a = task.audit_encodings(data, *assets)
    assert a["train"]["sentences"] == 3 and a["train"]["candidate_pairs"] == 8
    data.records["test"][0]["relation"] = "per:parents"
    assert task.audit_encodings(data, *assets) == a


@pytest.mark.parametrize("kind", ["hash", "count", "task", "labels", "reuse", "duplicate_id", "unseen_pair", "real_subset", "unqualified"])
def test_bad_physical_receipt_rejected_before_model_load(tmp_path, kind):
    receipt = fixture_receipt(tmp_path)
    spec = json.loads(receipt.read_text())
    if kind == "hash": spec["splits"]["train"]["sha256"] = "0"*64
    elif kind == "count": spec["splits"]["dev"]["count"] = 3
    elif kind == "task": spec["task"] = "TACRED"
    elif kind == "labels": spec["canonical_labels"] = spec["canonical_labels"][::-1]
    elif kind == "reuse": spec["splits"]["dev"] = spec["splits"]["test"]
    elif kind == "real_subset": spec["synthetic"] = False
    elif kind == "unqualified": spec["status"] = "pending"
    else:
        path = tmp_path/"test.json"; rows = json.loads(path.read_text())
        if kind == "duplicate_id": rows[0]["id"] = "synthetic-a"
        else: rows[0]["obj_type"] = "CITY"
        path.write_text(json.dumps(rows)); spec["splits"]["test"]["sha256"] = digest(path)
    receipt.write_text(json.dumps(spec))
    with pytest.raises(ValueError): task.PhysicalSplits(receipt)


def test_path_and_format_escape_rejected(tmp_path):
    receipt = fixture_receipt(tmp_path); spec = json.loads(receipt.read_text())
    spec["splits"]["train"]["file"] = "../train.json"; receipt.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="inside receipt"): task.PhysicalSplits(receipt)
    receipt = fixture_receipt(tmp_path); spec = json.loads(receipt.read_text())
    spec["splits"]["train"]["format"] = "auto"; receipt.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="explicit json"): task.PhysicalSplits(receipt)


def test_train_support_drift_rejected(physical, assets):
    physical.records["train"][0]["relation"] = "per:parents"
    with pytest.raises(ValueError, match="frozen H drift"): make(physical, assets)


def test_cursor_every_epoch_and_tail_no_global_rng_consumption():
    before = random.getstate(); plan = task.UpdatePlan(5, 13, batch_size=2, epochs=3)
    assert plan.total == 9 and plan.at(9) is None
    for epoch in range(3):
        batches = [plan.at(epoch*3+i)["indices"] for i in range(3)]
        assert list(map(len, batches)) == [2, 2, 1]
        assert sorted(sum(batches, [])) == list(range(5))
        assert task.UpdatePlan(5, 13, batch_size=2, epochs=3).at(epoch*3+1) == plan.at(epoch*3+1)
    assert before == random.getstate()
    with pytest.raises(ValueError): plan.at(-1)
    with pytest.raises(ValueError): plan.at(10)
    with pytest.raises(ValueError): plan.at(True)


def test_weighted_micro_and_short_tail_matches_full_sentence_objective(physical, assets):
    import torch
    # Dropout off to test the algebra; no claim of dropout partition invariance.
    one = make(physical, assets, micro=3); two = make(physical, assets, micro=2)
    groups = one.encode("train", [0, 1, 2]); one.model.train()
    one.make_optimizer().zero_grad(set_to_none=True)
    full = one.model(**neural.collate_groups(groups), gold=[g["gold"] for g in groups])
    full["loss"].backward()
    target = {k: p.grad.clone() for k, p in one.model.named_parameters() if p.grad is not None}
    actual = two.train_update([0, 1, 2], two.make_optimizer())
    assert actual["sentences"] == 3 and actual["pairs"] == 8 and actual["micro_batches"] == 2
    assert actual["loss"] == pytest.approx(float(full["loss"].detach()), abs=1e-6)
    for name, p in two.model.named_parameters():
        if name in target: assert torch.allclose(p.grad, target[name], atol=2e-6, rtol=1e-4), name
    assert one.contract_sha256 != two.contract_sha256  # physical partition frozen in resume identity


def test_eval_all_40_metrics_and_no_gold_input(physical, assets):
    adapter = make(physical, assets, micro=2)
    before = adapter.encode("test", [0, 1]); report, predictions = adapter.evaluate("test")
    physical.records["test"][0]["relation"] = "per:parents"
    after = adapter.encode("test", [0, 1]); changed, repeated = adapter.evaluate("test")
    assert before == after and predictions == repeated
    assert changed["items"] == 2 and changed["samples_dropped"] == 0 and changed["loss"] is None
    assert changed["loss_status"] == "infinite_zero_gold_support" and changed["excluded_gold_indices"] == [0]
    assert len(changed["per_class"]) == 40 and len(changed["confusion_matrix"]) == 40
    for key in ("accuracy", "micro_precision", "micro_recall", "micro_f1", "macro_f1"): assert key in changed
    assert report["loss_status"] == "finite_mean_all_sentences" and report["loss"] >= 0
    for item in predictions: assert len(item["probabilities_canonical40"]) == 40 and sum(item["probabilities_canonical40"]) == pytest.approx(1., abs=1e-6)
    assert adapter.model.training  # eval restores preceding mode


def test_common_scorer_null_loss_is_not_zero_imputed():
    null = metrics([0, 1, 1], [0, 1, 0], task.LABELS, None)
    finite = metrics([0, 1, 1], [0, 1, 0], task.LABELS, .25)
    assert null["loss"] is None and finite["loss"] == .25
    assert {k:v for k,v in null.items() if k != "loss"} == {k:v for k,v in finite.items() if k != "loss"}
    assert null["accuracy"] == pytest.approx(2/3) and null["micro_f1"] == pytest.approx(2/3)
    for bad in (float("inf"), float("nan"), -.1):
        with pytest.raises(ValueError): metrics([0], [0], task.LABELS, bad)


def test_optimizer_drift_and_synthetic_pretrained_factory_rejected(physical, assets, tmp_path):
    adapter = make(physical, assets); optimizer = adapter.make_optimizer()
    optimizer.param_groups[0]["lr"] = .001
    with pytest.raises(ValueError, match="frozen AdamW"): adapter.train_update([0], optimizer)
    with pytest.raises(ValueError, match="real qualified"):
        task.CTLDRPTask.from_assets(tmp_path/"qualification.json", "absent", "absent", "absent")


def test_post_update_plan_resume_has_identical_next_batch_loss_and_state(physical, assets, tmp_path):
    import torch
    adapter = make(physical, assets, dropout=.1); optimizer = adapter.make_optimizer()
    first = adapter.plan.at(0); adapter.train_update(first["indices"], optimizer)
    checkpoint = tmp_path/"checkpoint.pt"
    neural.save_checkpoint(checkpoint, adapter.model, optimizer, contract=adapter.contract, cursor=1)
    second = adapter.plan.at(1); expected = adapter.train_update(second["indices"], optimizer)
    state = {k:v.clone() for k,v in adapter.model.state_dict().items()}; expected_rng = torch.get_rng_state().clone()
    replay = make(physical, assets, dropout=.1); other = replay.make_optimizer()
    cursor = neural.load_checkpoint(checkpoint, replay.model, other, contract=replay.contract)
    assert replay.plan.at(cursor) == second
    actual = replay.train_update(replay.plan.at(cursor)["indices"], other)
    assert actual == expected and torch.equal(expected_rng, torch.get_rng_state())
    assert all(torch.equal(v, replay.model.state_dict()[k]) for k,v in state.items())
    for i, value in optimizer.state_dict()["state"].items():
        assert all(torch.equal(t, other.state_dict()["state"][i][k]) for k,t in value.items())


def test_dropout_is_part_of_resume_identity(physical, assets):
    one = make(physical, assets, dropout=0.)
    two = make(physical, assets, dropout=.1)
    assert one.contract_sha256 != two.contract_sha256
    assert one.contract["model_configuration"]["hidden_dropout_prob"] == 0.
    assert two.contract["model_configuration"]["hidden_dropout_prob"] == .1


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_actual_task_train_test_complete_numeric_fixture(physical, assets, tmp_path, device):
    import torch
    from baselines.retacred.improved import SampleCapture
    if device.startswith("cuda"): assert torch.cuda.is_available(), "CUDA diagnostic expected on selected server"
    adapter = make(physical, assets, micro=3, device=device); optimizer = adapter.make_optimizer()
    directory = Path(os.environ.get("CTL_DRP_TASK_EVIDENCE", str(tmp_path)))/device.replace(":", "-")
    directory.mkdir(parents=True, exist_ok=False)
    traces = []
    for phase in ("train", "test"):
        with SampleCapture(adapter.model, row=0, batch_size=-1) as capture:
            result = adapter.train_update([0, 1, 2], optimizer) if phase == "train" else adapter.evaluate("test")[0]
            values = {k:v.contiguous() for k,v in capture.values.items()}
        assert any("word_embeddings" in k for k in values) and any("query" in k for k in values)
        assert any("attention_scores_to_probabilities" in k for k in values)
        if phase == "train": assert any("gradient" in k for k in values)
        path = directory/(phase+".pt"); torch.save(values, path)
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        assert set(loaded) == set(values) and all(torch.equal(v, loaded[k]) for k,v in values.items())
        traces.append({"phase":phase, "numeric_artifact":path.name, "sha256":digest(path),
                       "tensor_count":len(values), "actual_result":result})
    (directory/"task-evidence.json").write_text(json.dumps({"scope":"synthetic_tiny_task_not_formal_case",
        "device":device, "contract_sha256":adapter.contract_sha256, "traces":traces,
        "full_numeric_roundtrip_equal":True, "formal_training":False}, indent=2)+"\n")


def test_formal_registry_and_pending_stage_unchanged():
    assert len(cohort()["default_models"]) == 5
    for variant in ("ctl_drp_sup", "ctl_drp_unsup"):
        with pytest.raises(ValueError): select_models(variant)
    assert task.describe()["formal_comparison_admitted"] is False and task.describe()["model_jobs"] == 0
