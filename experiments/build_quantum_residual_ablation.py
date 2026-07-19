from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q_attention.experiments import load_projector  # noqa: E402
from q_attention.quantum import build_quantum_residual_projector  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the optional classical-plus-quantum residual projector ablation.")
    parser.add_argument("--classical_projector_path", required=True)
    parser.add_argument("--quantum_projector_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")
    classical, classical_metadata = load_projector(args.classical_projector_path, device)
    quantum, quantum_metadata = load_projector(args.quantum_projector_path, device)
    projector = build_quantum_residual_projector(classical, quantum, alpha=args.alpha)
    metadata = {
        "projector_family": "classical_quantum_residual_ablation",
        "main_method": False,
        "alpha": args.alpha,
        "classical_projector_path": args.classical_projector_path,
        "quantum_projector_path": args.quantum_projector_path,
        "classical_metadata": dict(classical_metadata),
        "quantum_metadata": dict(quantum_metadata),
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"projector": projector, "metadata": metadata}, output_path)
    metadata_path = output_path.with_name(f"{output_path.stem}_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_path": str(output_path), "metadata_path": str(metadata_path), "shape": list(projector.shape)}, sort_keys=True))


if __name__ == "__main__":
    main()
