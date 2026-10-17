"""Public synthetic relation task; targets are derived outside the model.

Disjoint ordered triples per split, randomized entity names, variable padding,
forward/reversed entity roles, and a distractor relation phrase. All conditions
receive the same sentence, ordered spans and minibatches. This is NOT Re-TACRED.
"""
from __future__ import annotations
import random
import torch

LABELS = ("employee_of", "resident_of", "founder_of")
WORDS = ("<pad>", "<unk>", "joined", "lives_in", "founded", "while", "and") + tuple(
    f"entity_{i}" for i in range(16))
VOCAB = {word: i for i, word in enumerate(WORDS)}


def datasets(seed=13):
    rng = random.Random(seed)
    triples = [(a,b,c) for a in range(16) for b in range(16) for c in range(16)
               if len({a,b,c}) == 3]
    rng.shuffle(triples)
    output = {}
    cursor = 0
    for split, count in (("train",192), ("valid",48), ("test",48)):
        rows = []
        for i, (a,b,c) in enumerate(triples[cursor:cursor+count]):
            label, noise = i % 3, (i+1) % 3
            verbs = ("joined", "lives_in", "founded")
            # Model must select the relation belonging to the supplied pair.
            tokens = [f"entity_{a}", verbs[label], f"entity_{b}", "while",
                      f"entity_{b}", verbs[noise], f"entity_{c}"]
            if rng.random() < 0.5:
                tokens = tokens[4:] + ["while"] + tokens[:3]
                positions = (4,6)
            else:
                positions = (0,2)
            if rng.random() < 0.5:
                # A second query on the same observed sentence is possible,
                # without any gold-dependent feature construction.
                positions = (0,2) if positions == (4,6) else (4,6)
                label = noise
            rows.append(dict(id=f"{split}-{i:04d}", tokens=tokens,
                             subject=positions[0], object=positions[1], gold=label))
        cursor += count
        output[split] = rows
    return output


def collate(rows, device="cpu"):
    # Eight positions include a padded position to exercise mask handling.
    ids = torch.zeros((len(rows),8), dtype=torch.long, device=device)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    subject, obj = torch.zeros_like(mask), torch.zeros_like(mask)
    for i, row in enumerate(rows):
        ids[i,:len(row["tokens"])] = torch.tensor([VOCAB[x] for x in row["tokens"]], device=device)
        mask[i,:len(row["tokens"])] = True
        subject[i,row["subject"]], obj[i,row["object"]] = True, True
    return dict(input_ids=ids, attention_mask=mask, subject_mask=subject, object_mask=obj)
