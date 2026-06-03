from __future__ import annotations

import torch

from q_attention import SpectralProjectorConfig, apply_key_steering, batched_span_mask, build_projector


def main() -> None:
    torch.manual_seed(13)
    batch, tokens, dim = 2, 6, 8

    neutral_keys = torch.randn(32, dim)
    positive_keys = neutral_keys + 0.25 * torch.randn(32, dim)
    projector = build_projector(
        neutral_keys,
        positive_keys,
        config=SpectralProjectorConfig(mode="hard_topk", rank=3),
    )

    keys = torch.randn(batch, tokens, dim)
    mask = batched_span_mask(batch, tokens, [[(1, 3)], [(4, 6)]])
    result = apply_key_steering(keys, projector, mask=mask, gain=0.5, return_delta=True)

    print("projector_shape", tuple(projector.shape))
    print("changed_count", result.changed_count)
    print("delta_norm", round(float(result.delta.norm()), 6))
    print("unchanged_ok", bool(torch.allclose(result.keys[~mask], keys[~mask])))


if __name__ == "__main__":
    main()
