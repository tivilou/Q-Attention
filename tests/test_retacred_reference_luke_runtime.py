"""Lightweight gate contracts; real Torch tiny checks run in one separate child."""
import json
import ast
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from baselines.retacred import luke_runtime_fidelity as gate

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("mode,minimum", [("tiny", 768*1024**2), ("full", 24*1024**3)])
def test_read_only_plan_is_honest_and_keeps_admission_closed(mode, minimum):
    result = gate.description(mode, "cpu")
    assert result["minimum_host_headroom_bytes"] == minimum
    assert not result["formal_training"] and not result["formal_comparison_admitted"]
    assert result["optimizer_updates"] == 2 and result["replayed_updates"] == 1
    assert result["diagnostic_optimizer"]["warmup_steps"] == 0
    assert "not_source_recipe" in result["optimizer_claim"]
    assert "upstream_optimizer_scheduler_equivalence" in result["pending"]


def test_describe_no_backend_weights_network_or_writes(tmp_path):
    code = '''
import runpy,sys
def audit(event,args):
    if event=='import' and args[0].split('.')[0] in ('torch','transformers','numpy'):
        raise AssertionError('backend import in read-only plan')
    if event.startswith('socket.'):
        raise AssertionError('network in read-only plan')
    if event=='open' and (str(args[0]).endswith(('.bin','.safetensors')) or isinstance(args[1],str) and any(x in args[1] for x in 'wax+')):
        raise AssertionError('write/weight read in read-only plan')
sys.addaudithook(audit)
runpy.run_module('baselines.retacred.luke_runtime_fidelity',run_name='__main__')
'''
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run([sys.executable, "-c", code, "--describe"], cwd=ROOT,
                               env=env, text=True, capture_output=True, timeout=30)
    assert completed.returncode == 0, completed.stdout+completed.stderr
    assert json.loads(completed.stdout)["weight_scope"] == "hash_locked_luke_large_pretrained"


def test_memory_gate_happens_before_backend_and_writes_pending(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "verify_source", lambda _: {str(i): "fixture" for i in range(15)})
    def blocked(_):
        assert "torch" not in sys.modules
        raise RuntimeError("fixture memory blocked")
    monkeypatch.setattr(gate, "require_runtime_memory", blocked)
    output = tmp_path/"memory"
    assert gate.main(["--output-dir", str(output)]) == 2
    result = json.loads((output/"runtime-evidence.json").read_text())
    assert result["status"] == "pending_memory_gate"
    assert result["backend_imported"] is False and result["weights_read"] is False


def test_existing_artifacts_never_overwritten(tmp_path):
    existing = tmp_path/"owned.json"
    existing.write_text("preserved")
    with pytest.raises(SystemExit) as error:
        gate.main(["--output-dir", str(tmp_path)])
    assert error.value.code == 2 and existing.read_text() == "preserved"


def test_symlinked_output_never_used(tmp_path):
    target = tmp_path/"target"
    target.mkdir()
    alias = tmp_path/"alias"
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(SystemExit):
        gate.main(["--output-dir", str(alias)])
    assert not list(target.iterdir())


def test_capture_rejects_missing_intermediates():
    class Empty:
        values = {}
    with pytest.raises(AssertionError, match="required numeric"):
        gate.validate_capture(Empty(), phase="train")


def test_trace_observer_supports_both_head_interfaces_without_fallback():
    content = (ROOT/"baselines/retacred/improved.py").read_text()
    assert 'getattr(classifier, "in_features", None)' in content
    assert "classifier[0].in_features" not in content
    assert '"luke.entity_embeddings"' in content


def probe_observer(content, sequential=False):
    """Execute the actual observer method with tensor-shape interface carriers."""
    method = next(node for node in ast.walk(ast.parse(content))
                  if isinstance(node, ast.FunctionDef) and node.name == "__torch_dispatch__")
    class Linear:
        in_features = 8
        def children(self):
            return iter(())
    class Sequential:
        def children(self):
            return iter((Linear(),))
        def __getitem__(self, _):
            return Linear()
    capture = SimpleNamespace(model=SimpleNamespace(classifier=Sequential() if sequential else Linear()),
                              counts={}, nodes=[], batch_size=1, save=lambda name, _: name)
    namespace = {"capture": capture}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "actual_observer", "exec"), namespace)
    result = SimpleNamespace(ndim=2, shape=(1,8))
    class Concatenate:
        def __str__(self):
            return "aten.cat.default"
        def __call__(self, *_args, **_kwargs):
            return result
    operands = [SimpleNamespace(ndim=2, shape=(1,4)) for _ in range(2)]
    actual = namespace["__torch_dispatch__"](None, Concatenate(), (), (operands,1))
    assert actual is result
    assert capture.counts["entity_readout"] == 1
    assert capture.nodes[0]["inputs"] == ["functional.entity_readout.subject", "functional.entity_readout.object"]
    assert capture.nodes[0]["outputs"] == ["functional.entity_readout.concatenated"]
    return "entity_readout_observed"


@pytest.mark.parametrize("sequential", [False, True])
def test_actual_observer_dispatch_handles_linear_and_sequential(sequential):
    assert probe_observer((ROOT/"baselines/retacred/improved.py").read_text(), sequential) == "entity_readout_observed"


def test_portable_shell_entry():
    completed = subprocess.run(["bash", "-n", str(ROOT/"scripts/check_retacred_luke_runtime.sh")], capture_output=True)
    assert completed.returncode == 0
    content = (ROOT/"scripts/check_retacred_luke_runtime.sh").read_text()
    assert "BASH_SOURCE" in content and "PYTHON_BIN" in content
    assert "/home/" not in content and "/usr/local/" not in content
