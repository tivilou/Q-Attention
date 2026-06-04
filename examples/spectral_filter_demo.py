from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q_attention import SpectralProjectorConfig, build_projector, cross_covariance, spectral_filter_diagnostics  # noqa: E402


def main() -> None:
    torch.manual_seed(29)
    keys = torch.randn(16, 8)
    omega = cross_covariance(keys, keys)
    basis, singular_values, _ = torch.linalg.svd(omega, full_matrices=False)
    configs = [
        SpectralProjectorConfig(mode="hard_topk", rank=2),
        SpectralProjectorConfig(mode="high_pass", threshold=0.5, sharpness=8.0),
        SpectralProjectorConfig(mode="band_pass", threshold=0.5, sharpness=8.0),
        SpectralProjectorConfig(mode="soft_energy"),
    ]

    for config in configs:
        projector = build_projector(basis=basis, singular_values=singular_values, config=config)
        diagnostics = spectral_filter_diagnostics(singular_values, config)
        print(
            config.mode,
            "active",
            diagnostics["active_directions"],
            "weight_sum",
            round(diagnostics["weight_sum"], 6),
            "projector_norm",
            round(float(torch.linalg.norm(projector).item()), 6),
        )


if __name__ == "__main__":
    main()