"""Bounded CPU technical runner. Never a full-data or collaborator formal runner."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def finalize(output):
    """Idempotently finalize the same closed fixture; never construct/train a head."""
    output = output.resolve()
    closed = json.loads((output/'CLOSED.json').read_text())
    for name, expected in closed['files'].items():
        p = output/name
        if p.is_symlink() or not p.resolve().is_relative_to(output):
            raise ValueError('Unsafe closed artifact')
        if hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            raise ValueError('Closed artifact changed: '+name)
    result = json.loads((output/'closed-summary.json').read_text())
    final = output/'summary.json'
    if final.exists():
        if json.loads(final.read_text()) != result:
            raise ValueError('Final summary conflicts with closed source')
    else:
        temporary = output/'.summary.pending'
        try:
            temporary.write_text(json.dumps(result, indent=2)+'\n')
            temporary.replace(final)
        finally:
            temporary.unlink(missing_ok=True)
    print(json.dumps(result))


def load_cache_archive(archive_path, expected_sha256):
    """Read only the old owner's tiny train/valid cache, never test members."""
    import io
    import tarfile
    import numpy as np
    import torch
    path = archive_path.resolve()
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    digest = h.hexdigest()
    if digest != expected_sha256:
        raise ValueError('Pinned owner cache archive hash differs')
    batches = {}
    identities = {}
    with tarfile.open(path, 'r:gz') as archive:
        manifest_member = archive.getmember('canary/manifest.json')
        if not manifest_member.isfile():
            raise ValueError('Owner canary manifest must be a regular file')
        manifest = json.load(archive.extractfile(manifest_member))
        if manifest['config']['profile'] != 'canary' or manifest['config']['counts'] != dict(train=16, valid=8, test=8):
            raise ValueError('Only the previously closed tiny owner canary is allowed')
        for split, count in (('train', 16), ('valid', 8)):
            arrays = {}
            for kind in ('h', 'roles', 'offsets', 'targets'):
                name = f'canary/cache/{split}-{kind}.npy'
                member = archive.getmember(name)
                if not member.isfile() or member.size > 16*1024*1024:
                    raise ValueError('Unexpected tiny cache member')
                raw = archive.extractfile(member).read()
                identities[name] = hashlib.sha256(raw).hexdigest()
                arrays[kind] = np.load(io.BytesIO(raw), allow_pickle=False)
            offsets = arrays['offsets']
            labels = arrays['targets']
            if (offsets.shape != (count+1,) or labels.shape != (count,) or offsets.dtype.kind not in 'iu'
                    or labels.dtype.kind not in 'iu' or offsets[0] != 0 or (np.diff(offsets) <= 0).any()
                    or labels.min() < 0 or labels.max() >= 40):
                raise ValueError('Bad tiny cache offsets/targets')
            words = int(offsets[-1])
            if arrays['h'].shape != (words, 1024) or arrays['roles'].shape != (words, 4):
                raise ValueError('Tiny cache shapes differ')
            lengths = np.diff(offsets)
            n = int(lengths.max())
            if n > 512:
                raise ValueError('Unexpected word population')
            h = np.zeros((count, n, 1024), dtype=np.float32)
            roles = np.zeros((count, n, 4), dtype=np.float32)
            mask = np.zeros((count, n), dtype=bool)
            for i in range(count):
                lo, hi = map(int, offsets[i:i+2])
                h[i, :hi-lo] = arrays['h'][lo:hi]
                roles[i, :hi-lo] = arrays['roles'][lo:hi]
                mask[i, :hi-lo] = True
            batches[split] = (tuple(torch.from_numpy(a) for a in (h, roles, mask)), torch.from_numpy(labels).long())
    return batches, dict(archive_sha256=digest, selected_member_sha256=identities,
        original_source_sha256=manifest['source_sha256'], member_scope='train/valid cache and manifest only; no test member loaded')


def run(output, inject_summary_failure=False, cache_archive=None, archive_sha256=None):
    import torch
    from baselines.retacred.rmea_entity import VARIANTS, make_head
    from baselines.retacred.rmea_entity_trace import capture, close
    from tests.test_rmea_entity import fixture
    if output.exists() or output.resolve().is_relative_to(ROOT):
        raise ValueError('Fresh private output outside checkout required; no training resume')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output.mkdir(parents=True)
    config = dict(seed=307, variants=list(VARIANTS), epochs=2, batch_size=8,
                  train_count=16, valid_count=8, data_scope='deterministic synthetic features; no benchmark/test access')
    revision = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    if cache_archive is None:
        train = fixture(16, dtype=torch.float32)
        valid = fixture(8, dtype=torch.float32)
        # Separate synthetic feature population; never claim independent real data.
        valid = (valid[0]+.03, valid[1], valid[2])
        labels = torch.arange(16) % 4
        valid_labels = torch.arange(8) % 4
    else:
        batches, cache_identity = load_cache_archive(cache_archive, archive_sha256)
        train, labels = batches['train']
        valid, valid_labels = batches['valid']
        config.update(data_scope='Owner historical canary frozen features; train16/valid8 only, technical check',
                      cache_identity=cache_identity)
    records = []
    results = {}
    common = []
    for variant in VARIANTS:
        torch.manual_seed(config['seed'])
        head = make_head(variant)
        if variant == 'no_content':
            head.fit_role_means([train], split='train')
        common.append(hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in
            (head.value.weight, head.value.bias, head.decoder, head.bias))).hexdigest())
        for split, data in (('train', train), ('valid', valid)):
            records += capture(output, head, data, split, 'initial')
        optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.01)
        first = None
        for epoch in range(2):
            head.train()
            for start in range(0, len(labels), 8):
                optimizer.zero_grad(set_to_none=True)
                data = tuple(t[start:start+8] for t in train)
                loss = torch.nn.functional.cross_entropy(head(*data)[0], labels[start:start+8])
                loss.backward()
                if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in head.parameters()):
                    raise ValueError('Nonfinite or disconnected training path')
                if first is None:
                    first = float(loss.detach())
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
        for split, data in (('train', train), ('valid', valid)):
            records += capture(output, head, data, split, 'trained')
        with torch.no_grad():
            final = float(torch.nn.functional.cross_entropy(head(*train)[0], labels))
            valid_ce = float(torch.nn.functional.cross_entropy(head(*valid)[0], valid_labels))
        results[variant] = dict(parameters=sum(p.numel() for p in head.parameters()),
            first_batch_ce=first, final_train_ce=final, valid_ce=valid_ce, updates=4, finite=True)
        if not all(__import__('math').isfinite(v) for v in (first, final, valid_ce)):
            raise ValueError('Nonfinite bounded technical evaluation')
    if len(set(common)) != 1:
        raise ValueError('Common initialization differs')
    trace = close(output, records, config, revision)
    result = dict(scope='head technical-only; no utility tier or handoff authorization',
                  revision=revision, config=config, results=results, trace=trace,
                  common_initialization_equal=True, formal_runs=0, test_access=False)
    (output/'closed-summary.json').write_text(json.dumps(result, indent=2)+'\n')
    files = {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(output.rglob('*')) if p.is_file()}
    (output/'CLOSED.json').write_text(json.dumps(dict(files=files), indent=2)+'\n')
    if inject_summary_failure:
        raise RuntimeError('Injected summary failure after immutable technical fixture closed')
    finalize(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    r = sub.add_parser('run')
    r.add_argument('--output', type=Path, required=True)
    r.add_argument('--inject-summary-failure', action='store_true')
    r.add_argument('--cache-archive', type=Path)
    r.add_argument('--archive-sha256')
    f = sub.add_parser('finalize')
    f.add_argument('--run-dir', type=Path, required=True)
    v = sub.add_parser('value')
    v.add_argument('--run-dir', type=Path, required=True)
    v.add_argument('--sample', required=True)
    v.add_argument('--representation', required=True)
    v.add_argument('--coordinates', required=True, help='JSON list of coordinate lists, max64')
    a = p.parse_args()
    if a.action == 'run':
        if bool(a.cache_archive) != bool(a.archive_sha256):
            p.error('--cache-archive and --archive-sha256 must be supplied together')
        run(a.output, a.inject_summary_failure, a.cache_archive, a.archive_sha256)
    elif a.action == 'finalize':
        finalize(a.run_dir)
    else:
        from baselines.retacred.rmea_entity_trace import read_value
        trace = json.loads((a.run_dir/'sample-trace.json').read_text())
        sample = next(s for s in trace['samples'] if s['sample_id'] == a.sample)
        print(json.dumps(read_value(a.run_dir, sample, a.representation, json.loads(a.coordinates))))


if __name__ == '__main__':
    main()
