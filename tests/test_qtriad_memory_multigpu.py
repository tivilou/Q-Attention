from __future__ import annotations

import json
import torch
import pytest
import sys
import time
from types import SimpleNamespace

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import run_qtriad_relation_transfer as formal_runner

from q_attention.plugins.q_triad import QTriadAttentionScoreKernel


def _write_selector_metrics(output_dir, selector: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps({"selector": selector, "valid": {}, "test": {}}),
        encoding="utf-8",
    )


def _inputs() -> dict[str, torch.Tensor]:
    torch.manual_seed(41)
    batch, heads, tokens, dim = 2, 2, 5, 4
    attention_mask = torch.ones(batch, tokens, dtype=torch.bool)
    attention_mask[1, -1] = False
    subject_mask = torch.zeros_like(attention_mask)
    object_mask = torch.zeros_like(attention_mask)
    subject_mask[:, 1] = True
    object_mask[:, 3] = True
    return {
        "query": torch.randn(batch, heads, tokens, dim),
        "key": torch.randn(batch, heads, tokens, dim),
        "scores": torch.randn(batch, heads, tokens, tokens),
        "attention_mask": attention_mask,
        "subject_mask": subject_mask,
        "object_mask": object_mask,
        "layer_index": torch.tensor(0),
    }


def _kernel(pair_chunk_size: int) -> QTriadAttentionScoreKernel:
    return QTriadAttentionScoreKernel(
        num_layers=1,
        num_heads=2,
        head_dim=4,
        num_qubits=2,
        circuit_depth=1,
        seed=29,
        pair_chunk_size=pair_chunk_size,
    )


def _call(kernel: QTriadAttentionScoreKernel, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return kernel(
        inputs["query"],
        inputs["key"],
        scores=inputs["scores"],
        layer_index=int(inputs["layer_index"].item()),
        attention_mask=inputs["attention_mask"],
        subject_mask=inputs["subject_mask"],
        object_mask=inputs["object_mask"],
    )


def test_chunked_score_matches_single_chunk_and_gradients() -> None:
    inputs = _inputs()
    reference = _kernel(10_000)
    chunked = _kernel(3)
    chunked.load_state_dict(reference.state_dict())
    reference_residual = _call(reference, inputs)
    chunked_residual = _call(chunked, inputs)
    assert torch.allclose(chunked_residual, reference_residual, atol=2e-6, rtol=2e-6)

    reference.zero_grad(set_to_none=True)
    chunked.zero_grad(set_to_none=True)
    _call(reference, inputs).square().mean().backward()
    _call(chunked, inputs).square().mean().backward()
    for reference_parameter, chunked_parameter in zip(reference.parameters(), chunked.parameters()):
        assert reference_parameter.grad is not None
        assert chunked_parameter.grad is not None
        assert torch.allclose(chunked_parameter.grad, reference_parameter.grad, atol=2e-5, rtol=2e-5)


def test_streamed_backward_matches_direct_pair_loop() -> None:
    torch.manual_seed(43)
    query = torch.randn(2, 4, 4)
    relation = torch.randn(2, 4)
    key = torch.randn(2, 4, 4)
    kernel = _kernel(3)
    streamed = kernel._score_pairs(kernel._kernel(0, 0), query, relation, key)
    (streamed.square().sum()).backward()
    streamed_grads = [parameter.grad.detach().clone() for parameter in kernel._kernel(0, 0).parameters()]

    kernel.zero_grad(set_to_none=True)
    direct_chunks = []
    total_pairs = query.shape[0] * query.shape[1] * key.shape[1]
    for start in range(0, total_pairs, 3):
        indices = torch.arange(start, min(start + 3, total_pairs))
        batch_index = indices // (query.shape[1] * key.shape[1])
        remainder = indices % (query.shape[1] * key.shape[1])
        query_index = remainder // key.shape[1]
        key_index = remainder % key.shape[1]
        direct_chunks.append(
            kernel._kernel(0, 0)(
                query[batch_index, query_index],
                relation[batch_index],
                key[batch_index, key_index],
            ).score
        )
    direct = torch.cat(direct_chunks).reshape_as(streamed)
    direct.square().sum().backward()
    assert torch.allclose(streamed.detach(), direct.detach(), atol=2e-6, rtol=2e-6)
    for streamed_grad, parameter in zip(streamed_grads, kernel._kernel(0, 0).parameters()):
        assert torch.allclose(streamed_grad, parameter.grad, atol=2e-5, rtol=2e-5)


def test_pair_chunk_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="pair_chunk_size"):
        _kernel(0)


def test_selector_dashboard_renders_multi_gpu_heartbeat(tmp_path) -> None:
    heartbeat = tmp_path / "q_triad.heartbeat"
    heartbeat.write_text(
        '{"event":"batch_progress","phase":"train","epoch":3,"epochs":12,'
        '"batch":147,"batches":229,"percent":64.19,"eta_seconds":4200,'
        '"batches_per_second":0.42,"gpu_memory":[{"physical_index":0,'
        '"nvidia_used_mib":4096,"nvidia_free_mib":4096,"nvidia_total_mib":8192,'
        '"allocated_mib":1024,"reserved_mib":2048,"peak_reserved_mib":3072}]}\n',
        encoding="utf-8",
    )
    statuses = {
        "q_triad": {
            "status": "running",
            "gpu": 0,
            "heartbeat_file": str(heartbeat),
        },
        "classical_density_tensor": {"status": "pending", "gpu": None},
        "quantum_product": {"status": "complete", "gpu": 1},
    }
    active = {"q_triad": {"started_monotonic": time.monotonic() - 65}}

    rendered = formal_runner._render_selector_dashboard(statuses, active)

    assert "Q-TRIAD selectors: 1/3 complete" in rendered
    assert "GPU 0 | q_triad" in rendered
    assert "epoch 3/12" in rendered
    assert "batch 147/229 64.2%" in rendered
    assert "ETA 01:10:00" in rendered
    assert "GPU 0 VRAM 4.0/8.0 GiB used, free 4.0 GiB" in rendered
    assert "Queued: classical_density_tensor" in rendered
    assert "Completed: quantum_product" in rendered


def test_baseline_console_renderer_formats_progress_and_epoch() -> None:
    progress = formal_runner._render_baseline_line(
        '{"event":"batch_progress","phase":"train","epoch":2,"batch":8,'
        '"batches":16,"percent":50,"elapsed_seconds":12,"eta_seconds":12,'
        '"batches_per_second":0.67,"gpu_memory":[{"physical_index":0,'
        '"nvidia_used_mib":4096,"nvidia_free_mib":4096,"nvidia_total_mib":8192,'
        '"allocated_mib":1024,"reserved_mib":2048,"peak_reserved_mib":3072}]}',
        epochs=8,
    )
    epoch = formal_runner._render_baseline_line(
        '{"epoch":2,"train_loss":0.321,"valid":{"loss":0.456,"macro_f1":0.789}}',
        epochs=8,
    )

    assert progress is not None
    assert "[baseline][train] epoch 2/8" in progress
    assert "batch 8/16" in progress and "ETA 00:12" in progress
    assert "proc 1.0/2.0 GiB alloc/reserved, peak 3.0 GiB" in progress
    assert epoch == (
        "[baseline] epoch 2/8 complete | train_loss=0.3210 | "
        "valid_loss=0.4560 | valid_macro_f1=0.7890"
    )


def test_selector_scheduler_reuses_first_free_gpu(tmp_path, monkeypatch) -> None:
    seen_environments = []

    class FakeProcess:
        _next_pid = 4100

        def __init__(self, command, **kwargs):
            self.pid = FakeProcess._next_pid
            FakeProcess._next_pid += 1
            self.return_code = 0
            seen_environments.append(kwargs["env"])
            selector = command[command.index("--selector") + 1]
            output_dir = __import__("pathlib").Path(
                command[command.index("--output-dir") + 1]
            )
            _write_selector_metrics(output_dir, selector)

        def poll(self):
            return self.return_code

        def wait(self, **_kwargs):
            return self.return_code

    monkeypatch.setattr(formal_runner.subprocess, "Popen", FakeProcess)
    args = SimpleNamespace(python_bin=sys.executable, log_every_batches=1)
    statuses = formal_runner.run_selector_workers(
        selectors=["q_triad", "classical_density_tensor", "quantum_product"],
        gpu_ids=[2, 5],
        args=args,
        config_path=tmp_path / "config.json",
        baseline_dir=tmp_path / "baseline",
        data_dir=tmp_path / "data",
        run_dir=tmp_path / "run",
        seed=13,
    )
    assert [statuses[name]["status"] for name in statuses] == ["complete"] * 3
    assert statuses["q_triad"]["gpu"] == 2
    assert statuses["classical_density_tensor"]["gpu"] == 5
    assert statuses["quantum_product"]["gpu"] == 2
    assert all(environment["Q_ATTENTION_HEARTBEAT_FILE"] for environment in seen_environments)
    assert all(environment["PYTHONUNBUFFERED"] == "1" for environment in seen_environments)
    assert (tmp_path / "run" / "scheduler_events.jsonl").is_file()


def test_selector_resume_skips_valid_completed_metrics(tmp_path, monkeypatch) -> None:
    selector_dir = tmp_path / "run" / "selectors" / "q_triad"
    _write_selector_metrics(selector_dir, "q_triad")

    def unexpected_popen(*_args, **_kwargs):
        raise AssertionError("completed selector must not be relaunched")

    monkeypatch.setattr(formal_runner.subprocess, "Popen", unexpected_popen)
    statuses = formal_runner.run_selector_workers(
        selectors=["q_triad"],
        gpu_ids=[0],
        args=SimpleNamespace(python_bin=sys.executable, log_every_batches=1),
        config_path=tmp_path / "config.json",
        baseline_dir=tmp_path / "baseline",
        data_dir=tmp_path / "data",
        run_dir=tmp_path / "run",
        seed=13,
        resume=True,
    )

    assert statuses["q_triad"]["status"] == "complete"
    assert statuses["q_triad"]["resumed_skip"] is True


def test_selector_resume_rejects_partial_directory_without_checkpoint(
    tmp_path, monkeypatch
) -> None:
    selector_dir = tmp_path / "run" / "selectors" / "q_triad"
    selector_dir.mkdir(parents=True)
    (selector_dir / "worker.log").write_text("partial", encoding="utf-8")

    def unexpected_popen(*_args, **_kwargs):
        raise AssertionError("unsafe partial selector must not be relaunched")

    monkeypatch.setattr(formal_runner.subprocess, "Popen", unexpected_popen)
    with pytest.raises(formal_runner.ResumeCompatibilityError, match="no batch checkpoint"):
        formal_runner.run_selector_workers(
            selectors=["q_triad"],
            gpu_ids=[0],
            args=SimpleNamespace(python_bin=sys.executable, log_every_batches=1),
            config_path=tmp_path / "config.json",
            baseline_dir=tmp_path / "baseline",
            data_dir=tmp_path / "data",
            run_dir=tmp_path / "run",
            seed=13,
            resume=True,
        )

    assert (tmp_path / "run" / "RUN_FAILED").is_file()
    assert not (tmp_path / "run" / "RUN_PAUSED").exists()


def test_selector_resume_passes_resume_only_with_checkpoint(tmp_path, monkeypatch) -> None:
    commands = []
    checkpoint = tmp_path / "run" / "selectors" / "q_triad" / "checkpoints" / "latest.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")

    class FakeProcess:
        def __init__(self, command, **_kwargs):
            self.pid = 4200
            self.return_code = 0
            commands.append(command)
            selector = command[command.index("--selector") + 1]
            output_dir = __import__("pathlib").Path(
                command[command.index("--output-dir") + 1]
            )
            _write_selector_metrics(output_dir, selector)

        def poll(self):
            return self.return_code

        def wait(self, **_kwargs):
            return self.return_code

    monkeypatch.setattr(formal_runner.subprocess, "Popen", FakeProcess)
    formal_runner.run_selector_workers(
        selectors=["q_triad"],
        gpu_ids=[0],
        args=SimpleNamespace(python_bin=sys.executable, log_every_batches=1),
        config_path=tmp_path / "config.json",
        baseline_dir=tmp_path / "baseline",
        data_dir=tmp_path / "data",
        run_dir=tmp_path / "run",
        seed=13,
        resume=True,
    )

    assert len(commands) == 1
    assert "--resume" in commands[0]


def test_paused_selector_writes_paused_marker_not_failed(tmp_path, monkeypatch) -> None:
    class FakeProcess:
        def __init__(self, _command, **_kwargs):
            self.pid = 4300

        def poll(self):
            return formal_runner.PAUSED_EXIT_CODE

        def wait(self, **_kwargs):
            return formal_runner.PAUSED_EXIT_CODE

    monkeypatch.setattr(formal_runner.subprocess, "Popen", FakeProcess)
    with pytest.raises(formal_runner.RunPaused, match="paused"):
        formal_runner.run_selector_workers(
            selectors=["q_triad"],
            gpu_ids=[0],
            args=SimpleNamespace(python_bin=sys.executable, log_every_batches=1),
            config_path=tmp_path / "config.json",
            baseline_dir=tmp_path / "baseline",
            data_dir=tmp_path / "data",
            run_dir=tmp_path / "run",
            seed=13,
        )

    marker = json.loads((tmp_path / "run" / "RUN_PAUSED").read_text(encoding="utf-8"))
    assert marker["workers"]["q_triad"]["status"] == "paused"
    assert not (tmp_path / "run" / "RUN_FAILED").exists()


def test_parent_pause_waits_for_all_active_selector_workers(tmp_path, monkeypatch) -> None:
    processes = {}
    signaled = []

    class FakeProcess:
        _next_pid = 4400

        def __init__(self, _command, **_kwargs):
            self.pid = FakeProcess._next_pid
            FakeProcess._next_pid += 1
            self.return_code = None
            self.polls_after_signal = 0
            processes[self.pid] = self

        def poll(self):
            if self.pid in signaled:
                self.polls_after_signal += 1
                if self.polls_after_signal >= self.pid - 4399:
                    self.return_code = formal_runner.PAUSED_EXIT_CODE
            return self.return_code

        def wait(self, **_kwargs):
            return self.return_code

    pause = SimpleNamespace(requested=False, reason=None)
    sleep_calls = 0

    def fake_sleep(_seconds):
        nonlocal sleep_calls
        sleep_calls += 1
        if len(processes) == 2 and not signaled:
            pause.requested = True
            pause.reason = "SIGTERM"

    def fake_killpg(pid, signum):
        assert signum == formal_runner.signal.SIGTERM
        signaled.append(pid)

    monkeypatch.setattr(formal_runner.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(formal_runner.time, "sleep", fake_sleep)
    monkeypatch.setattr(formal_runner.os, "killpg", fake_killpg, raising=False)

    with pytest.raises(formal_runner.RunPaused, match="scheduler pause requested"):
        formal_runner.run_selector_workers(
            selectors=["q_triad", "classical_density_tensor"],
            gpu_ids=[0, 1],
            args=SimpleNamespace(python_bin=sys.executable, log_every_batches=1),
            config_path=tmp_path / "config.json",
            baseline_dir=tmp_path / "baseline",
            data_dir=tmp_path / "data",
            run_dir=tmp_path / "run",
            seed=13,
            pause=pause,
        )

    assert sorted(signaled) == [4400, 4401]
    assert processes[4400].return_code == formal_runner.PAUSED_EXIT_CODE
    assert processes[4401].return_code == formal_runner.PAUSED_EXIT_CODE
    assert sleep_calls >= 2
    marker = json.loads((tmp_path / "run" / "RUN_PAUSED").read_text(encoding="utf-8"))
    assert {item["status"] for item in marker["workers"].values()} == {"paused"}
    assert not (tmp_path / "run" / "RUN_FAILED").exists()


def test_baseline_safe_pause_timeout_terminates_child(tmp_path, monkeypatch) -> None:
    terminated = []

    class FakeProcess:
        def __init__(self, *_args, **_kwargs):
            self.pid = 4500
            self.stdout = []
            self.return_code = None

        def poll(self):
            return self.return_code

        def wait(self, **_kwargs):
            return self.return_code

    def fake_terminate(process):
        terminated.append(process.pid)
        process.return_code = -9

    monkeypatch.setattr(formal_runner.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(formal_runner, "SAFE_PAUSE_TIMEOUT_SECONDS", 0)
    monkeypatch.setattr(formal_runner, "_terminate_worker", fake_terminate)
    monkeypatch.setattr(formal_runner.os, "killpg", lambda *_args: None, raising=False)

    with pytest.raises(RuntimeError, match="safe-pause timeout"):
        formal_runner._run_baseline_logged_command(
            [sys.executable, "train.py"],
            tmp_path / "baseline.log",
            tmp_path / "heartbeat.json",
            epochs=2,
            pause=SimpleNamespace(requested=True),
        )

    assert terminated == [4500]


def test_gpu_ids_follow_physical_ids_exposed_by_cuda_visible_devices(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,5")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert formal_runner.resolve_gpu_ids("2,5", "cuda") == [2, 5]


def test_auto_gpu_selects_all_sufficient_physical_devices(monkeypatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    inventory = [
        {"index": 0, "name": "A100", "memory_total_mib": 40 * 1024, "memory_free_mib": 39 * 1024, "memory_used_mib": 1024},
        {"index": 1, "name": "A100", "memory_total_mib": 40 * 1024, "memory_free_mib": 38 * 1024, "memory_used_mib": 2 * 1024},
    ]
    assert formal_runner.resolve_gpu_ids("auto", "cuda", inventory) == [0, 1]


def test_auto_gpu_resolution_does_not_initialize_torch_cuda(monkeypatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: (_ for _ in ()).throw(AssertionError("CUDA initialized")))
    inventory = [
        {"index": 0, "name": "A100", "memory_total_mib": 40 * 1024, "memory_free_mib": 39 * 1024, "memory_used_mib": 1024},
    ]
    assert formal_runner.resolve_gpu_ids("auto", "cuda", inventory) == [0]


def test_auto_profile_is_conservative_on_small_or_busy_gpu() -> None:
    config = {"kernel": {"pair_chunk_size": 256}}
    low_inventory = [
        {"index": 0, "name": "3080 Ti", "memory_total_mib": 12 * 1024, "memory_free_mib": 11 * 1024, "memory_used_mib": 1024},
    ]
    low = formal_runner.choose_hardware_profile("auto", config, [0], low_inventory)
    assert low["name"] == "low_memory"
    assert low["pair_chunk_size"] == 64
    assert low["activation_checkpointing"] is True

    high_inventory = [
        {"index": 0, "name": "A100", "memory_total_mib": 80 * 1024, "memory_free_mib": 78 * 1024, "memory_used_mib": 2 * 1024},
    ]
    high = formal_runner.choose_hardware_profile("auto", config, [0], high_inventory)
    assert high["name"] == "high_memory"
    assert high["pair_chunk_size"] == 256
    assert high["activation_checkpointing"] is True


def test_auto_profile_uses_the_weakest_selected_gpu() -> None:
    config = {"kernel": {"pair_chunk_size": 256}}
    inventory = [
        {"index": 0, "name": "A100", "memory_total_mib": 80 * 1024, "memory_free_mib": 70 * 1024, "memory_used_mib": 10 * 1024},
        {"index": 1, "name": "A100", "memory_total_mib": 40 * 1024, "memory_free_mib": 20 * 1024, "memory_used_mib": 20 * 1024},
    ]
    profile = formal_runner.choose_hardware_profile("auto", config, [0, 1], inventory)
    assert profile["name"] == "balanced"
    assert profile["minimum_memory_total_mib"] == 40 * 1024
    assert profile["minimum_memory_free_mib"] == 20 * 1024


def test_gpu_capacity_guard_accepts_sufficient_memory() -> None:
    formal_runner.validate_gpu_capacity(
        [0],
        [{"index": 0, "memory_total_mib": 12 * 1024, "memory_free_mib": 8 * 1024}],
        phase="test",
    )


def test_gpu_capacity_guard_reports_competing_process(monkeypatch) -> None:
    monkeypatch.setattr(
        formal_runner,
        "query_compute_apps",
        lambda: [{"pid": 29066, "process_name": "python", "used_memory_mib": 47 * 1024}],
    )
    with pytest.raises(RuntimeError, match=r"before workers.*pid=29066"):
        formal_runner.validate_gpu_capacity(
            [0],
            [{"index": 0, "memory_total_mib": 48 * 1024, "memory_free_mib": 300}],
            phase="before workers",
        )
