from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q_attention import RouterConfig, apply_key_steering, route_projectors, stack_projector_bank  # noqa: E402


def main() -> None:
    projectors = [torch.eye(3), 2.0 * torch.eye(3), 0.5 * torch.eye(3)]
    prototypes = [
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([0.0, 1.0, 0.0]),
        torch.tensor([0.0, 0.0, 1.0]),
    ]
    bank = stack_projector_bank(["identity", "double", "half"], projectors, prototypes)
    anchors = torch.tensor([[1.0, 0.1, 0.0], [0.0, 1.0, 0.2]])
    routed = route_projectors(anchors, bank, RouterConfig(temperature=0.5))

    keys = torch.ones(2, 4, 3)
    mask = torch.tensor([[True, False, True, False], [False, True, True, False]])
    steered = apply_key_steering(keys, routed.projectors, mask=mask, gain=0.25)

    print("weights", [[round(float(value), 4) for value in row] for row in routed.weights])
    print("dynamic_projector_shape", tuple(routed.projectors.shape))
    print("masked_changed", not torch.allclose(steered[mask], keys[mask]))
    print("unmasked_same", torch.allclose(steered[~mask], keys[~mask]))


if __name__ == "__main__":
    main()