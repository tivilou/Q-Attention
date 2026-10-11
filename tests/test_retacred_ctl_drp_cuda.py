"""Focused diagnostic tests; a tiny CUDA test is not pretrained certification."""
from pathlib import Path
import random

import pytest
import torch

from baselines.retacred.ctl_drp_pretrained import load_encoder
from baselines.retacred.ctl_drp_cuda_smoke import describe, state_hash
from baselines.retacred.ctl_drp_cuda_checkpoint import decode_numpy_rng, numpy_rng


def test_pretrained_rejects_bad_asset_before_load(tmp_path):
    p = tmp_path / "bad.bin"; p.write_bytes(b"not a model")
    with pytest.raises(ValueError, match="hash/size"):
        load_encoder(p, tmp_path)
    with pytest.raises(ValueError, match="locked variant"):
        load_encoder(p, tmp_path, variant="unlocked")
    link = tmp_path / "link.bin"; link.symlink_to(p)
    with pytest.raises(ValueError, match="regular asset"):
        load_encoder(link, tmp_path)


def test_identity_and_streamed_hash():
    assert describe()["formal_training"] is False
    assert describe()["formal_comparison_admitted"] is False
    x = torch.tensor(3.)  # scalar optimizer step also hashes correctly
    assert state_hash({"x": x}) == state_hash({"x": x.clone()})
    assert state_hash(x) != state_hash(torch.tensor([3.]))
    assert state_hash(x) != state_hash(torch.tensor(4.))


def test_numpy_rng_schema_is_safe():
    import numpy as np
    before = numpy_rng()
    decode_numpy_rng(before)
    assert before == numpy_rng()
    broken = before.copy(); broken[1] = [-1] * 624
    with pytest.raises(ValueError, match="schema"):
        decode_numpy_rng(broken)
    assert before == numpy_rng()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="actual CUDA required")
def test_actual_cuda_checkpoint_replay_and_rejection(tmp_path, monkeypatch):
    from baselines.retacred.ctl_drp_cuda_checkpoint import load, save
    from baselines.retacred.ctl_drp_neural import tiny_model
    # CPU torch import above did not create a CUDA context before workspace config.
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = tiny_model(seed=13, dropout=.1).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0., foreach=False, fused=False)
    batch = {"input_ids": torch.tensor([[0,31414,2],[0,232,2]], device="cuda"),
             "attention_mask": torch.ones((2,3), dtype=torch.long, device="cuda"),
             "candidate_groups": [["no_relation", "per:spouse"]]}
    def update():
        opt.zero_grad(set_to_none=True)
        result = model(**batch, gold=["per:spouse"])
        result["loss"].backward(); opt.step()
        return result["loss"].item()
    update()
    p = tmp_path / "ckpt.pt"; contract = {"tiny_test": True}
    save(p, model, opt, contract=contract, cursor=1)
    expected_loss = update()
    expected = state_hash([model.state_dict(), opt.state_dict(), torch.get_rng_state(),
                           torch.cuda.get_rng_state(), random.getstate(), numpy_rng()])
    # Wrong contract and malformed state reject without mutating live model/RNG.
    with pytest.raises(ValueError, match="identity/runtime"):
        load(p, model, opt, contract={"wrong": True})
    assert expected == state_hash([model.state_dict(), opt.state_dict(), torch.get_rng_state(),
                                  torch.cuda.get_rng_state(), random.getstate(), numpy_rng()])
    value = torch.load(p, weights_only=True)
    key = next(iter(value["optimizer"]["state"]))
    value["optimizer"]["state"][key]["exp_avg"] = torch.zeros(1)
    bad = tmp_path / "bad-state.pt"; torch.save(value, bad)
    with pytest.raises(ValueError, match="shape/dtype"):
        load(bad, model, opt, contract=contract)
    assert expected == state_hash([model.state_dict(), opt.state_dict(), torch.get_rng_state(),
                                  torch.cuda.get_rng_state(), random.getstate(), numpy_rng()])
    opt.zero_grad(set_to_none=True)
    assert load(p, model, opt, contract=contract) == 1
    assert expected_loss == update()
    assert expected == state_hash([model.state_dict(), opt.state_dict(), torch.get_rng_state(),
                                  torch.cuda.get_rng_state(), random.getstate(), numpy_rng()])
    with pytest.raises(ValueError, match="fresh regular"):
        save(p, model, opt, contract=contract, cursor=2)
