import copy
import importlib.util
import json
from pathlib import Path
import torch
import pytest

from q_attention.adapters.q_ecpa import ECPAToyModel
from q_attention.analysis.ecpa_trace import read_values, validate_trace
from q_attention.models.relation_transformer import RelationExtractionModel, RelationTransformerConfig
from q_attention.plugins.q_ecpa import EntityCoherentPath, ECPAConfig, MODES, propagate
from q_attention.tasks.ecpa_toy_data import datasets, collate, WORDS


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(2)
    torch.manual_seed(13)


def example():
    q = torch.randn(2,2,6,8,dtype=torch.double)
    mask = torch.tensor([[1,1,1,1,1,0],[1,1,1,1,0,0]],dtype=torch.bool)
    s,o = torch.zeros_like(mask),torch.zeros_like(mask)
    s[:,0],o[:,3] = True,True
    scores = q @ q.transpose(-1,-2)
    return q,scores,mask,s,o


def test_hermiticity_unitarity_probabilities_and_coherence():
    q,scores,mask,s,o = example()
    model = EntityCoherentPath(ECPAConfig(),"quantum").double()
    adjusted,state = model(q,scores,mask,s,o,capture=True)
    torch.testing.assert_close(state["hamiltonian"],state["hamiltonian"].mH)
    eye = torch.eye(6,dtype=torch.cdouble).expand(2,2,6,6)
    torch.testing.assert_close(state["unitary"].mH@state["unitary"],eye)
    for name in ("pq","pd"):
        torch.testing.assert_close(state[name].sum(-1),torch.ones(2,2,6,dtype=torch.double))
        assert (state[name] >= 0).all()
    assert (state["pq"]-state["pd"]).abs().max() > 1e-3
    flux = state["phase"][...,0,1]+state["phase"][...,1,2]+state["phase"][...,2,0]
    assert flux.abs().max() > 1e-3
    assert (adjusted-scores).abs().max() <= 0.4+1e-12
    invalid = ~(mask[:,None,:,None]&mask[:,None,None,:]).expand_as(scores)
    assert torch.equal(adjusted[invalid],scores[invalid])


def test_gauge_invariance_and_single_step_null():
    q,scores,mask,s,o = example()
    model = EntityCoherentPath(ECPAConfig(steps=1),"quantum").double()
    _,state = model(q,scores,mask,s,o,capture=True)
    torch.testing.assert_close(state["pq"],state["pd"])
    h = state["hamiltonian"]
    diagonal = torch.diag_embed(torch.exp(1j*torch.randn(2,2,6,dtype=torch.double)))
    _,a,_,d = propagate(h,3,0.7)
    _,ga,_,gd = propagate(diagonal@h@diagonal.mH,3,0.7)
    torch.testing.assert_close(a.abs().square(),ga.abs().square())
    torch.testing.assert_close(d,gd)


def test_controls_identical_h_parameter_budget_and_entity_diagnostic():
    args = example()
    reference = EntityCoherentPath(ECPAConfig(),"quantum").double()
    records = {}
    for mode in ("quantum","dephased","classical"):
        model = EntityCoherentPath(ECPAConfig(),mode).double()
        model.load_state_dict(reference.state_dict())
        records[mode] = model(*args,capture=True)[1]
        assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in reference.parameters())
    torch.testing.assert_close(records["quantum"]["hamiltonian"],records["dephased"]["hamiltonian"])
    torch.testing.assert_close(records["quantum"]["single"],records["dephased"]["single"])
    torch.testing.assert_close(records["dephased"]["probability"],records["dephased"]["pd"])
    torch.testing.assert_close(records["classical"]["probability"],records["classical"]["classical_steps"][...,-1,:,:])
    q,scores,mask,s,o = args
    _,swapped = reference(q,scores,mask,o,s,capture=True)
    assert not torch.allclose(swapped["hamiltonian"],records["quantum"]["hamiltonian"])


@pytest.mark.parametrize("mode",["quantum","dephased","classical"])
def test_nonzero_gradient_and_finite_difference(mode):
    q,scores,mask,s,o = example()
    q.requires_grad_()
    model = EntityCoherentPath(ECPAConfig(),mode).double()
    weight = torch.randn_like(scores)
    def loss():
        return (model(q,scores,mask,s,o)[0]*weight).sum()
    loss().backward()
    assert torch.isfinite(q.grad).all() and q.grad.norm() > 0
    gradient = model.condition.bias.grad.clone()
    assert torch.isfinite(gradient).all() and gradient.norm() > 1e-6
    index = int(gradient.abs().argmax())
    original = float(model.condition.bias[index].detach())
    estimates = []
    for delta in (1e-5,-1e-5):
        with torch.no_grad():
            model.condition.bias[index] = original+delta
        estimates.append(float(loss().detach()))
    with torch.no_grad():
        model.condition.bias[index] = original
    assert (estimates[0]-estimates[1])/(2e-5) == pytest.approx(float(gradient[index]),rel=1e-4,abs=1e-6)


def test_disabled_exact_baseline_and_scope_cleanup():
    base = RelationExtractionModel(RelationTransformerConfig(vocab_size=len(WORDS),num_labels=3,
        dim=16,num_heads=2,num_layers=1,ff_dim=32,dropout=0,max_length=8))
    control = copy.deepcopy(base)
    model = ECPAToyModel(base,EntityCoherentPath(ECPAConfig(),"disabled"))
    batch = collate(datasets()["test"][:2])
    torch.testing.assert_close(model(**batch),control(**batch),rtol=0,atol=0)
    torch.testing.assert_close(model(**batch,capture=True),control(**batch),rtol=0,atol=0)
    bad = dict(batch, subject_mask=torch.zeros_like(batch["subject_mask"]))
    with pytest.raises(ValueError):
        model(**bad)
    assert model.intervention.inputs is None
    torch.testing.assert_close(model(**batch),control(**batch),rtol=0,atol=0)


def test_dataset_no_cross_split_duplicate_and_target_boundary():
    data = datasets()
    sets = [set((tuple(r["tokens"]),r["subject"],r["object"]) for r in rows) for rows in data.values()]
    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    assert "gold" not in collate(data["train"][:2])


def runner():
    path = Path(__file__).resolve().parents[1]/"experiments/run_ecpa_toy.py"
    spec = importlib.util.spec_from_file_location("ecpa_runner",path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_interrupted_update_replay_trace_values_and_tamper(tmp_path):
    run = runner()
    data = datasets(13)
    config = dict(seed=13,updates=10,batch_size=4,lr=0.003,source_sha256="test-source",
                  dataset_sha256="same-fixture",primary="macro_f1")
    uninterrupted = run.run_one(tmp_path/"a","quantum",config,data)
    interrupted = run.run_one(tmp_path/"b","quantum",config,data,stop_after=4)
    assert interrupted["status"] == "interrupted"
    resumed = run.run_one(tmp_path/"b","quantum",config,data,resume=True)
    assert resumed["test"] == uninterrupted["test"]
    left = torch.load(tmp_path/"a/quantum/latest.pt",weights_only=False)
    right = torch.load(tmp_path/"b/quantum/latest.pt",weights_only=False)
    for key in left["model"]:
        torch.testing.assert_close(left["model"][key],right["model"][key],atol=0,rtol=0)
    path = tmp_path/"b/quantum/sample_trace.json"
    report = validate_trace(path)
    assert report["samples"] == 10 and report["full_value_artifacts"] > 400
    trace = json.loads(path.read_text())
    selected = trace["samples"][-1]
    assert selected["roles"]["checkpoint_identity"]["optimizer_updates_completed"] == 10
    # Bind final capture to the exact latest model, beyond a phase label.
    witness = run.make_model("quantum",13)
    witness.load_state_dict(right["model"])
    assert selected["roles"]["checkpoint_identity"]["model_state_sha256"] == run.model_identity(witness)
    values = read_values(path,selected["sample_id"],"amplitudes",(1,2),component="imag")
    assert values.shape == (8,8)
    artifact = next(r for r in selected["representations"] if r["id"] == "amplitudes")["artifact_refs"][0]
    source = path.parent/artifact["path"]
    original = source.read_bytes()
    source.write_bytes(original+b"tamper")
    with pytest.raises(ValueError,match="hash"):
        validate_trace(path)
    source.write_bytes(original)
    validate_trace(path)
    # Export/readiness can be retried without retraining completed updates.
    assert run.run_one(tmp_path/"b","quantum",config,data,resume=True)["test"] == resumed["test"]
    with pytest.raises(ValueError,match="identity mismatch"):
        run.run_one(tmp_path/"b","quantum",dict(config,lr=0.01),data,resume=True)


def test_padding_invariance_and_original_qk_distinct():
    q,scores,mask,s,o = example()
    model = EntityCoherentPath(ECPAConfig(),"quantum").double()
    original = model(q,scores,mask,s,o)[0]
    q2 = q.clone()
    q2[~mask[:,None,:,None].expand_as(q)] = 1e4
    torch.testing.assert_close(original,model(q2,scores,mask,s,o)[0])
    # Existing pairwise QK control is executable, not renamed as ECPA.
    from q_attention.plugins.q_query_key_coherent_transport import QueryKeyCoherentTransportConfig, QueryKeyCoherentTransportKernel
    kernel = QueryKeyCoherentTransportKernel(QueryKeyCoherentTransportConfig(
        num_layers=1,num_heads=2,head_dim=8,pair_chunk_size=None)).double()
    residual = kernel(q,q,scores=scores,attention_mask=mask,subject_mask=s,object_mask=o,layer_index=0)
    assert residual.shape == scores.shape
    assert not torch.allclose(original-scores,residual)
