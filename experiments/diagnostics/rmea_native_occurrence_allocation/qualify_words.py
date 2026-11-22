"""Identity/coverage qualification of the frozen occurrence design; no model calls."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import sys
import time
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    data = (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()
    if path.exists():
        assert path.read_bytes() == data
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def two_split_metadata(path):
    """Decode only the TRAIN/VALID arrays, stopping before decoding TEST metadata."""
    decoder = json.JSONDecoder(); buffer = ''; result = {}
    with path.open(encoding='utf8') as f:
        def consume_space():
            nonlocal buffer
            while not buffer.strip():
                chunk = f.read(65536)
                assert chunk, 'Unexpected metadata EOF'
                buffer += chunk
            buffer = buffer.lstrip()

        def char(expected):
            nonlocal buffer
            consume_space(); assert buffer[0] == expected
            buffer = buffer[1:]

        def value():
            nonlocal buffer
            consume_space()
            while True:
                try:
                    v, n = decoder.raw_decode(buffer); buffer = buffer[n:]; return v
                except json.JSONDecodeError:
                    chunk = f.read(65536); assert chunk, 'Incomplete metadata value'
                    buffer += chunk

        char('{')
        for split in ('train', 'valid'):
            assert value() == split, 'Pinned producer writes TRAIN then VALID then TEST'
            char(':'); result[split] = value()
            if split == 'train':
                char(',')
    return result


def main():
    started = time.time(); payload = json.load(sys.stdin)
    root = Path('/hy-tmp/Q-Attention/inputs/native-word-metadata-20261123')
    cache = Path('/hy-tmp/Q-Attention/inputs/native-baseline/discovery/cache')
    pins = dict(train='f1051d574a44df3ecbd036c1de4fbbe0a372c768f6ec990330e2cea4a657269f',
                valid='9d6965f984d7485e2e81d380a0b0de2922d3da1e2685393ea3e46869805c2b55')
    counts = dict(train=8192, valid=19584)
    assert payload['source_hashes'] == pins and set(payload['splits']) == set(counts)
    metadata = two_split_metadata(cache / 'metadata.json')
    grouped = {}; integrity = {}
    for split, count in counts.items():
        rows = payload['splits'][split]; assert len(rows) == count
        roles = np.load(cache / (split+'-roles.npy'), mmap_mode='r', allow_pickle=False)
        offsets = np.load(cache / (split+'-offsets.npy'), mmap_mode='r', allow_pickle=False)
        pairs = []; actual_count = 0
        for i, row in enumerate(rows):
            a, b = map(int, offsets[i:i+2]); words = row['tokens']; n = len(words)
            assert row['id_hash'] == metadata[split][i]['id_hash']
            assert n == b-a == metadata[split][i]['words']
            positions = np.arange(n, dtype=np.float32); s, o = row['subject'], row['object']
            expected = np.stack([np.clip((positions-(s[0]+s[1]-1)/2)/32,-1,1),
                np.clip((positions-(o[0]+o[1]-1)/2)/32,-1,1),
                ((positions>=s[0])&(positions<s[1])).astype(np.float32),
                ((positions>=o[0])&(positions<o[1])).astype(np.float32)], -1)
            assert np.array_equal(expected, roles[a:b]), 'Original word/role/order mismatch'
            groups = defaultdict(list)
            for j, word in enumerate(words):
                assert isinstance(word, str)
                groups[(word, *map(float, roles[a+j]))].append(j)
            selected = [(key, tuple(idx)) for key,idx in groups.items() if len(idx)==2]
            pairs.append(selected); actual_count += len(selected)
        grouped[split] = pairs
        write(root/(split+'.private.json'), rows)
        integrity[split] = dict(items=count, all_ids_matched=True, all_wordcounts_matched=True,
            all_roles_exact=True, qualifying_pairs_before_train_support=actual_count,
            metadata_sha256=digest(root/(split+'.private.json')))
    support = Counter(key for row in grouped['train'] for key, _ in row)
    supported = {key for key,n in support.items() if n>=32}
    eligible = [i for i,row in enumerate(grouped['valid']) if any(key in supported for key,_ in row)]
    valid_targets = np.load(cache/'valid-targets.npy', mmap_mode='r', allow_pickle=False)
    positive = int(np.count_nonzero(valid_targets[eligible] != 0))
    result = dict(status='qualified' if len(eligible)>=1000 and positive>=250 else 'coverage_failed',
        supported_train_keys=len(supported), eligible_valid=len(eligible), eligible_positive_valid=positive,
        required_valid=1000, required_positive_valid=250, coverage_gate=len(eligible)>=1000 and positive>=250,
        source_hashes=pins, integrity=integrity, qualification_seconds=time.time()-started,
        model_forward_calls=0, optimizer_updates=0, test_targets_or_metadata_decoded=False, cos_writes=0,
        word_key='Exact original word string plus all four stored role coordinates; exactlytwo occurrences',
        grouping_selection='Allpairs in first8192TRAIN/all19584VALID; TRAINsupport>=32; no label/outcome selection',
        stop='Coverage failure closes frozen test; no key, multiplicity, support, estimator, cohort or threshold rescue',
        scientific_outcome='Not an effect estimate; coverage qualification only', L5='NOT_ACCEPTED')
    write(root/'qualification.json',result)
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
