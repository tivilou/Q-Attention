"""Runtime mechanism gate: direct upstream vs adapter and trace transparency."""
from __future__ import annotations

import argparse
from pathlib import Path

from .common import atomic_json, digest, pin_source, select_models
from .improved import Improved, SampleCapture, load_module


def compare(adapter, batch) -> dict:
    import torch
    from transformers import AutoConfig
    # Construct a fresh upstream instance, not a deepcopy of the adapter:
    # copying would also copy an erroneous adapter mutation or extra hook.
    config = AutoConfig.from_pretrained(adapter.args.model_name_or_path, local_files_only=True,num_labels=40)
    config._attn_implementation = "eager"
    reference = load_module(adapter.source,"model").REModel(adapter.args,config)
    vocabulary_size = adapter.model.encoder.get_input_embeddings().num_embeddings
    if reference.encoder.get_input_embeddings().num_embeddings != vocabulary_size:
        reference.encoder.resize_token_embeddings(vocabulary_size)
    reference.to(adapter.device)
    reference.load_state_dict(adapter.model.state_dict(),strict=True)
    # Disable dropout in both copies for a fixed-input numerical mechanism gate.
    # The actual training protocol retains upstream dropout.
    reference.eval(); adapter.model.eval()
    reference.zero_grad(); adapter.model.zero_grad()
    direct_loss, direct_logits = reference(**batch)
    with SampleCapture(adapter.model, 0, len(batch["labels"])) as capture:
        loss, logits = adapter.forward(batch)
        loss.backward()
    direct_loss.backward()
    torch.testing.assert_close(logits, direct_logits, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(loss, direct_loss, rtol=1e-5, atol=1e-6)
    gradient_count = 0
    for (name, parameter), (_, other) in zip(adapter.model.named_parameters(), reference.named_parameters()):
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad, rtol=1e-5, atol=1e-6)
            gradient_count += 1
    from transformers.optimization import AdamW
    first = AdamW(adapter.model.parameters(), lr=3e-5, eps=1e-6, weight_decay=0)
    second = AdamW(reference.parameters(), lr=3e-5, eps=1e-6, weight_decay=0)
    first.step(); second.step()
    for name, parameter in adapter.model.state_dict().items():
        torch.testing.assert_close(parameter, reference.state_dict()[name], rtol=1e-5, atol=1e-6)
    # Repeat the transparency comparison in actual train mode with identical
    # dropout RNG, not only the easy eval-mode fixture.
    reference.train(); adapter.model.train()
    reference.zero_grad(); adapter.model.zero_grad()
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    direct_train_loss, direct_train_logits = reference(**batch)
    direct_train_loss.backward()
    torch.set_rng_state(cpu_rng)
    if cuda_rng:
        torch.cuda.set_rng_state_all(cuda_rng)
    with SampleCapture(adapter.model,0,len(batch["labels"])) as training_capture:
        observed_train_loss, observed_train_logits = adapter.forward(batch)
        observed_train_loss.backward()
    torch.testing.assert_close(observed_train_logits,direct_train_logits,rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(observed_train_loss,direct_train_loss,rtol=1e-5,atol=1e-6)
    for (_, parameter), (_, other) in zip(adapter.model.named_parameters(),reference.named_parameters()):
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad,other.grad,rtol=1e-5,atol=1e-6)
    keys = list(capture.values)
    if not any("embeddings" in key and ".output." in key for key in keys) or not any(key.endswith(".gradient") for key in keys):
        raise ValueError("trace does not contain numeric embeddings and backward gradients")
    return {"forward": "passed", "loss": "passed", "gradients": "passed", "optimizer_update": "passed",
            "trace_transparency": "passed", "training_dropout_trace_transparency":"passed", "gradient_parameters":gradient_count,
            "captured_tensors":len(keys), "rtol":1e-5, "atol":1e-6,
            "limitations":["matched RNG required for train-mode numerical gate", "not a reproduction of the reported full-data score"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    row = select_models(args.model)[0]
    import os
    available = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    # A full RoBERTa-large double-model/gradient/update comparison has a much
    # higher RAM footprint than a tiny architecture fixture. Fail before any
    # model allocation, not by killing a low-memory exchange host.
    if (args.checkpoint/"pytorch_model.bin").exists() and (args.checkpoint/"pytorch_model.bin").stat().st_size > 1024**3 and available < 12*1024**3:
        raise RuntimeError("full-weight fidelity gate requires at least 12 GiB available RAM; no formal training started")
    source = pin_source("improved", fetch=False)
    adapter = Improved(source, args.checkpoint, row["input_format"], device=args.device)
    record = {"token":["Alice", "works", "for", "Acme", "."], "subj_start":0, "subj_end":0,
              "obj_start":3, "obj_end":3, "subj_type":"PERSON", "obj_type":"ORGANIZATION", "relation":"per:employee_of"}
    result = compare(adapter, adapter.batch(adapter.encode([record])))
    result.update({"schema":"retacred-source-fidelity.v1", "status":"source_faithful_adapted",
                   "scope":"fixed_input_source_mechanism_only_not_full_formal_certification",
                   "device":args.device, "formal_certificate_status":"not_verifiable",
                   "model":args.model, "source_files":adapter.source_manifest(),
                   "checkpoint_files":{path.name:digest(path) for path in args.checkpoint.iterdir() if path.is_file()}})
    atomic_json(args.output, result)
    print(f"[fidelity] {args.model}: forward/loss/gradient/update and trace gates passed", flush=True)


if __name__ == "__main__":
    main()
