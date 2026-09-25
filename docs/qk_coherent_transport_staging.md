# QK Coherent Transport Staging

This directory contains the isolated `qk-coherent-transport-v1` implementation
for mechanism and toy validation. It is not the authorized server checkout.

## Checks

Run the focused regression suite:

```powershell
conda run -n hf python -m pytest .codex-qk-transport-staging/tests/test_qk_coherent_transport.py -q
```

Run the three-seed CPU toy screen:

```powershell
conda run -n hf python .codex-qk-transport-staging/experiments/run_qk_coherent_transport_toy.py `
  --config .codex-qk-transport-staging/configs/qk_coherent_transport_toy.json `
  --output .codex-qk-transport-staging/runs/qk_coherent_transport_toy
```

The toy manifest records mechanism diagnostics only. A passing manifest does
not authorize a complete-data run or establish task utility, novelty, or a
quantum resource advantage.

The plugin implements one independent two-qubit transport circuit per channel
and depth. `register_qubits > 1` therefore means multiple independent channel
pairs, not a fully entangled multi-qubit register.
