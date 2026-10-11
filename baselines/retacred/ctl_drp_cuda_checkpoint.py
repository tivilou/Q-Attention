"""Narrow diagnostic resume: one visible CUDA GPU, FP32 AdamW, post-update."""
from __future__ import annotations
import os
from pathlib import Path
import random


def runtime():
    import torch
    if torch.cuda.device_count() != 1:
        raise ValueError("this diagnostic certificate requires exactly one visible CUDA GPU")
    p = torch.cuda.get_device_properties(0)
    return {"torch": str(torch.__version__), "cuda": torch.version.cuda,
            "device": p.name, "device_uuid": str(getattr(p, "uuid", "not_exposed")),
            "capability": [p.major, p.minor], "total_memory": p.total_memory,
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "precision": "FP32_no_autocast"}


def check_scope(model, optimizer):
    import torch
    if (type(optimizer) is not torch.optim.AdamW or any(p.device != torch.device("cuda:0")
            or p.dtype != torch.float32 for p in model.parameters())):
        raise ValueError("single CUDA FP32 AdamW diagnostic scope required")
    if any(g.get("foreach") is not False or g.get("fused") is not False
           or g.get("capturable") or g.get("differentiable") for g in optimizer.param_groups):
        raise ValueError("explicit unfused non-capturable AdamW required")


def numpy_rng():
    import numpy as np
    name, keys, pos, has_gauss, cached = np.random.get_state()
    return [name, keys.tolist(), int(pos), int(has_gauss), float(cached)]


def decode_numpy_rng(value):
    import numpy as np
    if (not isinstance(value, list) or len(value) != 5 or value[0] != "MT19937"
            or not isinstance(value[1], list) or len(value[1]) != 624
            or any(type(x) is not int or not 0 <= x < 2**32 for x in value[1])
            or type(value[2]) is not int or not 0 <= value[2] <= 624
            or type(value[3]) is not int or value[3] not in (0, 1)
            or type(value[4]) is not float):
        raise ValueError("NumPy RNG schema drift")
    state = (value[0], np.asarray(value[1], dtype=np.uint32), *value[2:])
    np.random.RandomState().set_state(state)
    return state


def cpu(value):
    import torch
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu(v) for v in value)
    return value


def save(path, model, optimizer, *, contract, cursor):
    import torch
    check_scope(model, optimizer)
    path = Path(path).absolute()
    if path.exists() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("fresh regular diagnostic checkpoint path required")
    if type(cursor) is not int or cursor < 1:
        raise ValueError("post-update positive cursor required")
    value = {"schema": "ctl-drp-single-cuda-fp32.v1", "phase": "post_optimizer_update",
             "runtime": runtime(), "contract": contract, "cursor": cursor, "training": model.training,
             "model": cpu(model.state_dict()), "optimizer": cpu(optimizer.state_dict()),
             "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(0),
             "python_rng": random.getstate(), "numpy_rng": numpy_rng()}
    temporary = path.with_name(path.name + ".part")
    # Exclusive temporary + hard link prevents overwriting an existing checkpoint.
    with temporary.open("xb") as f:
        torch.save(value, f)
        f.flush(); os.fsync(f.fileno())
    os.link(temporary, path)
    temporary.unlink()


def load(path, model, optimizer, *, contract):
    import torch
    import numpy as np
    check_scope(model, optimizer)
    path = Path(path).absolute()
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("regular diagnostic checkpoint required")
    value = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if (value.get("schema") != "ctl-drp-single-cuda-fp32.v1"
            or value.get("phase") != "post_optimizer_update" or value.get("runtime") != runtime()
            or value.get("contract") != contract or type(value.get("cursor")) is not int
            or value["cursor"] < 1 or type(value.get("training")) is not bool):
        raise ValueError("resume identity/runtime/phase rejected before mutation")
    current = model.state_dict()
    if set(value["model"]) != set(current) or any(not torch.is_tensor(t)
            or t.shape != current[k].shape or t.dtype != current[k].dtype
            or not torch.isfinite(t).all() for k, t in value["model"].items()):
        raise ValueError("checkpoint model architecture/numeric drift")
    old = optimizer.state_dict()
    new = value["optimizer"]
    if new["param_groups"] != old["param_groups"]:
        raise ValueError("optimizer settings or parameter order drift")
    params = [p for g in optimizer.param_groups for p in g["params"]]
    indices = [i for g in new["param_groups"] for i in g["params"]]
    if len(indices) != len(set(indices)) or set(new["state"]) != set(indices):
        raise ValueError("complete post-update AdamW state required")
    for index, param in zip(indices, params):
        state = new["state"][index]
        if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("unexpected AdamW state fields")
        for name, tensor in state.items():
            if (not torch.is_tensor(tensor) or tensor.dtype != torch.float32
                    or not torch.isfinite(tensor).all() or name != "step" and tensor.shape != param.shape
                    or name == "step" and (tensor.numel() != 1 or tensor.item() != value["cursor"])):
                raise ValueError("AdamW state shape/dtype/cursor drift")
    for name, expected in (("torch_rng", torch.get_rng_state()), ("cuda_rng", torch.cuda.get_rng_state(0))):
        t = value[name]
        if not torch.is_tensor(t) or t.dtype != torch.uint8 or t.shape != expected.shape:
            raise ValueError("checkpoint RNG drift")
    torch.Generator().set_state(value["torch_rng"])
    random.Random().setstate(value["python_rng"])
    np_state = decode_numpy_rng(value["numpy_rng"])
    # Identity, numerics, topology and RNG validated before any live mutation.
    model.load_state_dict(value["model"], strict=True)
    optimizer.load_state_dict(new)
    model.train(value["training"])
    torch.set_rng_state(value["torch_rng"])
    torch.cuda.set_rng_state(value["cuda_rng"], 0)
    random.setstate(value["python_rng"])
    np.random.set_state(np_state)
    return value["cursor"]
