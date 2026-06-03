from __future__ import annotations

import torch
import torch.nn as nn

from q_attention.adapters import EncoderKeySteeringAdapter, KeySteeringHookConfig
from q_attention.spans import batched_span_mask


class ToyEncoderLayer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.key_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.last_keys: torch.Tensor | None = None

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        keys = self.key_proj(hidden)
        self.last_keys = keys.detach().clone()
        return self.out_proj(keys)


class ToyEncoder(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([ToyEncoderLayer(dim)])

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.layers[0](hidden)


def main() -> None:
    torch.manual_seed(17)
    batch, tokens, dim = 2, 5, 6
    model = ToyEncoder(dim)
    hidden = torch.randn(batch, tokens, dim)
    projector = torch.eye(dim)
    mask = batched_span_mask(batch, tokens, [[(1, 3)], [(3, 5)]])

    baseline = model(hidden)
    baseline_keys = model.layers[0].last_keys

    adapter = EncoderKeySteeringAdapter(model, ["layers.0.key_proj"])
    config = KeySteeringHookConfig(projector=projector, mask=mask, gain=0.25)
    with adapter.steering(config):
        steered = model(hidden)
        steered_keys = model.layers[0].last_keys

    restored = model(hidden)

    print("baseline_shape", tuple(baseline.shape))
    print("steered_shape", tuple(steered.shape))
    print("masked_changed", bool(not torch.allclose(steered_keys[mask], baseline_keys[mask])))
    print("unmasked_same", bool(torch.allclose(steered_keys[~mask], baseline_keys[~mask])))
    print("restored", bool(torch.allclose(restored, baseline)))


if __name__ == "__main__":
    main()
