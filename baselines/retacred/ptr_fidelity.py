"""Fresh upstream PTR forward/objective/dual-update and tracing comparisons."""
from __future__ import annotations

from .improved import SampleCapture


def compare_ptr(adapter, batch):
    import torch
    reference, runtime = adapter.fresh_source_model()
    reference.load_state_dict(adapter.model.state_dict(), strict=True)
    reference.eval(); adapter.model.eval()
    reference.zero_grad(); adapter.model.zero_grad()
    direct_loss, direct_logits, prompt_logits = runtime.objective(reference, adapter.processor, batch, adapter.criterion)
    # An independent formula guard catches omission of relation CE even if both
    # calls use the same AST extraction helper.
    targets = adapter.processor.prompt_id_2_label[batch["labels"]]
    independent_scores = torch.stack([sum(prompt_logits[j][:, indices[j]] for j in range(5))
                                      for indices in adapter.processor.prompt_id_2_label], dim=1)
    independent_loss = sum(adapter.criterion(z, targets[:, j]) for j, z in enumerate(prompt_logits))/5
    independent_loss = independent_loss + adapter.criterion(independent_scores, batch["labels"])
    torch.testing.assert_close(direct_logits, independent_scores, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(direct_loss, independent_loss, rtol=1e-5, atol=1e-6)
    with SampleCapture(adapter.model, 0, len(batch["labels"])) as capture:
        loss, logits = adapter.forward(batch)
        loss.backward()
    direct_loss.backward()
    def check_gradients():
        count = 0
        for (name, parameter), (other_name, other) in zip(adapter.model.named_parameters(), reference.named_parameters()):
            assert name == other_name
            assert (parameter.grad is None) == (other.grad is None), name
            if parameter.grad is not None:
                torch.testing.assert_close(parameter.grad, other.grad, rtol=1e-5, atol=1e-6)
                count += 1
        return count
    torch.testing.assert_close(logits, direct_logits, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(loss, direct_loss, rtol=1e-5, atol=1e-6)
    gradients = check_gradients()
    torch.nn.utils.clip_grad_norm_(adapter.model.parameters(), 1.)
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
    # Explicit nonzero-LR toy update: original warmup500 would make the very
    # first update LR0, which is a vacuous parameter-update equivalence test.
    first, first_schedule = adapter.make_updates(2, 1, warmup=0)
    second, second_schedule = adapter.make_updates(2, 1, model=reference, runtime=runtime, warmup=0)
    before = adapter.model.model.embeddings.word_embeddings.weight.detach().clone()
    first.step(); first_schedule.step(); second.step(); second_schedule.step()
    for name, value in adapter.model.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name], rtol=1e-5, atol=1e-6)
    if torch.equal(before, adapter.model.model.embeddings.word_embeddings.weight):
        raise ValueError("PTR one-update comparison was vacuous")
    reference.train(); adapter.model.train()
    reference.zero_grad(); adapter.model.zero_grad()
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    direct_train_loss, direct_train_logits, _ = runtime.objective(reference, adapter.processor, batch, adapter.criterion)
    direct_train_loss.backward()
    torch.set_rng_state(cpu_rng)
    if cuda_rng:
        torch.cuda.set_rng_state_all(cuda_rng)
    with SampleCapture(adapter.model, 0, len(batch["labels"])) as training_capture:
        train_loss, train_logits = adapter.forward(batch)
        train_loss.backward()
    torch.testing.assert_close(train_logits, direct_train_logits, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(train_loss, direct_train_loss, rtol=1e-5, atol=1e-6)
    check_gradients()
    for position in range(5):
        if f"functional.ptr_verbalizer_projection.position{position}.hidden" not in capture.values:
            raise ValueError("actual masked-state projection missing")
    if (not any("embeddings" in key and ".output." in key for key in capture.values)
            or not any(key.endswith(".gradient") for key in training_capture.values)):
        raise ValueError("numeric embeddings or actual training gradients absent")
    return {"forward":"passed", "loss":"passed", "relation_ce":"passed", "gradients":"passed",
            "optimizer_update":"passed", "two_optimizers":"passed", "trace_transparency":"passed",
            "training_dropout_trace_transparency":"passed", "gradient_parameters":gradients,
            "captured_tensors":len(capture.values), "rtol":1e-5, "atol":1e-6,
            "gate_override":{"warmup_steps":0,"scope":"nonzero_learning_rate_one_update_fixture_only"},
            "formal_warmup_steps":500, "mixed_precision":"not_applicable_source_FP32",
            "limitations":["fixed input and matched dropout RNG", "not a full-data published-score reproduction"]}
