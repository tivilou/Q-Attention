"""Normal PTR source mechanism, independent objective, traces and dual recovery."""
from __future__ import annotations

import argparse
import copy
import json
import signal
import subprocess
import sys

import pytest

from baselines.retacred import common, fidelity, orchestrate, train
from baselines.retacred.ptr import PTR
from baselines.retacred.ptr_fidelity import compare_ptr
from test_retacred_reference_suite import RECORD, tiny_checkpoint


def adapter(checkpoint, *, checkpointing=False):
    import torch
    torch.set_num_threads(2)
    return PTR(common.pin_source("ptr", fetch=False), checkpoint, device="cpu",
               checkpointing=checkpointing, max_seq_length=64)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_source_objective_and_actual_projection(tiny_checkpoint, batch_size):
    value = adapter(tiny_checkpoint)
    records = [copy.deepcopy(RECORD)]
    if batch_size == 2:
        records.append({**copy.deepcopy(RECORD), "relation": "no_relation"})
    result = compare_ptr(value, value.batch(value.encode(records)))
    assert result["relation_ce"] == result["two_optimizers"] == "passed"
    assert result["training_dropout_trace_transparency"] == "passed"


def test_no_target_input_leak_and_source_numeric_label_order(tiny_checkpoint):
    import torch
    value = adapter(tiny_checkpoint)
    features = value.encode([{**copy.deepcopy(RECORD), "relation": label} for label in value.labels])
    assert sorted(item["labels"] for item in features) == list(range(40))
    for name in features[0]:
        if name != "labels":
            for item in features[1:]:
                torch.testing.assert_close(item[name], features[0][name], rtol=0, atol=0)
    assert features[0]["input_flags"].count_nonzero() == 0
    assert features[0]["mlm_labels"].ge(0).sum() == 5
    assert value.labels[value.processor.rel2id[RECORD["relation"]]] == RECORD["relation"]


def test_dual_optimizer_warmup_and_missing_state_rejection(tiny_checkpoint):
    import torch
    value = adapter(tiny_checkpoint)
    first, scheduler = value.make_updates(600, 1)
    assert len(first.members) == len(scheduler.members) == 2
    assert first.members["encoder"].param_groups[0]["weight_decay"] == .01
    assert first.members["encoder"].param_groups[1]["weight_decay"] == 0
    assert first.members["extra"].param_groups[0]["weight_decay"] == 0
    batch = value.batch(value.encode([copy.deepcopy(RECORD)]))
    for _ in range(3):
        first.zero_grad(set_to_none=True)
        loss, _ = value.forward(batch); loss.backward()
        first.step(); scheduler.step()
    assert first.members["encoder"].param_groups[0]["lr"] == pytest.approx(3e-5*3/500)
    assert first.members["extra"].param_groups[0]["lr"] == pytest.approx(1e-5*3/500)
    assert all(item["last_epoch"] == 3 for item in scheduler.state_dict().values())
    assert all(item["state"] for item in first.state_dict().values())
    # The released zero-flag branch remains dormant; don't activate it to make
    # the extra optimizer's parameter changes look more interesting.
    assert all(parameter.grad is not None and not parameter.grad.count_nonzero()
               for parameter in value.model.mlp.parameters())
    with pytest.raises(ValueError, match="both encoder and extra"):
        first.load_state_dict({"encoder": first.state_dict()["encoder"]})
    with pytest.raises(ValueError, match="both encoder and extra"):
        scheduler.load_state_dict({"extra": scheduler.state_dict()["extra"]})


def test_fidelity_fresh_source_rejects_adapter_mutation(tiny_checkpoint):
    value = adapter(tiny_checkpoint)
    handle = value.model.register_forward_hook(lambda module, inputs, output: [item + .5 for item in output])
    try:
        with pytest.raises(AssertionError):
            compare_ptr(value, value.batch(value.encode([copy.deepcopy(RECORD)])))
    finally:
        handle.remove()


def test_arguments_import_isolation(tiny_checkpoint):
    sentinel = object()
    previous = sys.modules.get("arguments")
    sys.modules["arguments"] = sentinel
    try:
        adapter(tiny_checkpoint)
        assert sys.modules["arguments"] is sentinel
    finally:
        if previous is None:
            sys.modules.pop("arguments", None)
        else:
            sys.modules["arguments"] = previous


def test_public_fidelity_module_entry(tiny_checkpoint, tmp_path):
    output = tmp_path/"fidelity.json"
    command = [sys.executable, "-m", "baselines.retacred.fidelity", "--model", "ptr",
               "--checkpoint", str(tiny_checkpoint), "--output", str(output), "--device", "cpu"]
    result = subprocess.run(command, cwd=common.ROOT, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(output.read_text())
    assert payload["two_optimizers"] == payload["relation_ce"] == "passed"
    assert payload["formal_certificate_status"] == "not_verifiable"
    assert len(payload["source_files"]) == 7


def test_full_weight_memory_gate_stops_before_model_allocation(tmp_path, monkeypatch):
    weight = tmp_path/"pytorch_model.bin"
    with weight.open("wb") as handle:
        handle.truncate(2*1024**3)  # sparse identity fixture, never loaded
    monkeypatch.setattr(sys, "argv", ["fidelity", "--model", "ptr", "--checkpoint", str(tmp_path),
                                      "--output", str(tmp_path/"result.json")])
    import os
    monkeypatch.setattr(os, "sysconf", lambda name: 4096 if name == "SC_PAGE_SIZE" else 1024)
    with pytest.raises(RuntimeError, match="at least 24 GiB"):
        fidelity.main()
    assert not (tmp_path/"result.json").exists()


def assert_nested_equal(first, second):
    import torch
    if torch.is_tensor(first):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_nested_equal(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for left, right in zip(first, second):
            assert_nested_equal(left, right)
    else:
        assert first == second


@pytest.mark.parametrize("checkpointing", [False, True])
def test_actual_train_test_complete_tensors_and_mid_batch_resume(tmp_path, tiny_checkpoint, monkeypatch, checkpointing):
    import torch
    torch.set_num_threads(2)
    data = tmp_path/"data"; data.mkdir()
    records = [{**copy.deepcopy(RECORD), "token": ["Alice", "works", "for", "Acme", str(i)]} for i in range(5)]
    manifest = {"splits": {}}
    for split, name in (("train", "train"), ("valid", "dev"), ("test", "test")):
        path = data/(name+".json"); path.write_text(json.dumps(records))
        manifest["splits"][split] = {"records":5,"source_sha256":common.digest(path),"output_sha256":common.digest(path)}
    (data/"data_manifest.json").write_text(json.dumps(manifest))
    args = argparse.Namespace(model="ptr", seed=13, data_dir=data, checkpoint=tiny_checkpoint,
        run_dir=tmp_path/"whole", resume=False, device="cpu", epochs=1, logical_batch=2,
        micro_batch=1, checkpointing=checkpointing, checkpoint_every=1, canary=False)
    monkeypatch.setattr(train, "parse_args", lambda: args)
    saved_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP)}
    try:
        train.main()
        whole = torch.load(args.run_dir/"checkpoints/latest.pt", map_location="cpu", weights_only=False)
        assert whole["contract"]["precision"] == "source_fp32_no_autocast"
        assert whole["scaler"] == {}
        args.run_dir = tmp_path/"paused"
        real_atomic = train.atomic_json
        def pause_after_first_update(path, value):
            real_atomic(path, value)
            if path.name == "heartbeat.json" and value.get("update") == 1:
                signal.raise_signal(signal.SIGTERM)
        monkeypatch.setattr(train, "atomic_json", pause_after_first_update)
        with pytest.raises(SystemExit) as paused:
            train.main()
        assert paused.value.code == train.PAUSED
        paused_state = torch.load(args.run_dir/"checkpoints/latest.pt", map_location="cpu", weights_only=False)
        assert paused_state["cursor"]["cursor"] == 1
        assert set(paused_state["optimizer"]) == set(paused_state["scheduler"]) == {"encoder", "extra"}
        monkeypatch.setattr(train, "atomic_json", real_atomic)
        args.resume = True
        train.main()
        final = torch.load(args.run_dir/"checkpoints/latest.pt", map_location="cpu", weights_only=False)
        for key in ("model", "optimizer", "scheduler", "scaler", "cursor"):
            assert_nested_equal(whole[key], final[key])
        orchestrate.validate_job(args.run_dir)
        summary = json.loads((args.run_dir/"run_summary.json").read_text())
        assert summary["updates"] == 3
        for item in summary["case_study"]["traces"]:
            trace = json.loads((args.run_dir/"case_study"/item["path"]).read_text())
            values = torch.load(args.run_dir/"case_study"/trace["tensor_artifact"], weights_only=True)
            assert len(trace["prompt_mask_positions"]) == 5
            assert values["task_output.logits"].shape == (1,40)
            assert values["functional.ptr_embedding_selection.selected"].shape == (1,512,16)
            torch.testing.assert_close(values["functional.ptr_embedding_selection.selected"],
                                       values["functional.ptr_embedding_selection.raw"], rtol=0, atol=0)
            assert any("attention_scores_to_probabilities" in key for key in values)
            for position in range(5):
                prefix = f"functional.ptr_verbalizer_projection.position{position}"
                assert values[prefix+".hidden"].shape == (1,16)
                torch.testing.assert_close(values[prefix+".hidden"] @ values[prefix+".decoder"],
                                           values[prefix+".logits"], rtol=1e-5, atol=1e-6)
                if trace["phase"] == "train":
                    assert "source_model.call0.output."+str(position)+".gradient" in values
            if trace["phase"] == "train":
                assert "verbalizer.parameter_gradient" in values
                assert trace["activation_gradient_scale"] == 1
        for split in summary["splits"].values():
            assert split["items"] == 5 and len(split["per_class"]) == 40
    finally:
        for sig, handler in saved_handlers.items():
            signal.signal(sig, handler)
