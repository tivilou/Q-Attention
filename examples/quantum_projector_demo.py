from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q_attention import QuantumFeatureMapConfig, SpectralProjectorConfig, apply_key_steering, build_quantum_projector  # noqa: E402


def main() -> None:
    torch.manual_seed(23)
    keys = torch.randn(2, 6, 8)
    anchor_keys = keys.reshape(-1, keys.shape[-1])
    result = build_quantum_projector(
        anchor_keys,
        quantum_config=QuantumFeatureMapConfig(num_qubits=3, angle_scale=1.25, seed=5),
        projector_config=SpectralProjectorConfig(rank=3),
    )
    mask = torch.tensor([[True, False, True, False, False, False], [False, True, False, False, True, False]])
    steered = apply_key_steering(keys, result.projector, mask=mask, gain=0.2)

    print("projector_shape", tuple(result.projector.shape))
    print("state_dim", result.metadata["state_dim"])
    print("kernel_mean", round(result.metadata["kernel_mean"], 6))
    print("masked_changed", not torch.allclose(steered[mask], keys[mask]))
    print("unmasked_same", torch.allclose(steered[~mask], keys[~mask]))


if __name__ == "__main__":
    main()