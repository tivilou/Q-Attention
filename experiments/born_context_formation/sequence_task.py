"""Original frozen-word task adapter for the recurrent role probe.

Reuse accepted microbatch loss/evaluation and optimizer semantics; no lifecycle.
"""
from pathlib import Path
import json
import numpy as np
import torch
from baselines.retacred.rmea_task_runtime import backward, score_batch


class FrozenWords:
    def __init__(self, cache, cohort_file):
        self.cache = Path(cache)
        self.cohort = json.loads(Path(cohort_file).read_bytes())
        self.data = {k: np.load(self.cache/f'train-{k}.npy', mmap_mode='r', allow_pickle=False)
                     for k in ('h', 'roles', 'offsets', 'targets')}
        if self.data['h'].dtype != np.float16 or self.data['roles'].shape[-1] != 4:
            raise ValueError('Original frozen cache required')
        if len(self.data['offsets']) != len(self.data['targets'])+1:
            raise ValueError('Word row boundaries differ')

    def batch(self, indices, device):
        # Same original FP16 h -> FP32 promotion and word/role collation as gen91.
        lengths = [int(self.data['offsets'][i+1]-self.data['offsets'][i]) for i in indices]
        if not lengths or min(lengths) <= 0:
            raise ValueError('Nonempty original sentences required')
        h = np.zeros((len(indices), max(lengths), 1024), np.float32)
        roles = np.zeros(h.shape[:2]+(4,), np.float32)
        mask = np.zeros(h.shape[:2], bool)
        for j, i in enumerate(indices):
            a, b = map(int, self.data['offsets'][i:i+2])
            h[j, :b-a] = self.data['h'][a:b]
            roles[j, :b-a] = self.data['roles'][a:b]
            mask[j, :b-a] = True
        inputs = tuple(torch.from_numpy(x).to(device) for x in (h, roles, mask))
        labels = torch.tensor(np.array(self.data['targets'][indices]), dtype=torch.long, device=device)
        return inputs, labels


def optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=.01)


def check_gradients(model):
    for name, p in model.named_parameters():
        if p.grad is None or not torch.isfinite(p.grad).all():
            raise ValueError('Missing/nonfinite gradient: '+name)
