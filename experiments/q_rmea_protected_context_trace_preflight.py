"""Real bounded final-head trace fixture, using existing pinned canary inputs."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from baselines.retacred.common import atomic_json, digest
from baselines.retacred.rmea_protocol import git, identity
from baselines.retacred.rmea_protected_context import KINDS, ProtectedContextHead
from baselines.retacred.rmea_protected_context_trace import ROOT, READER, capture, close, require, validate_trace

CHECKPOINT = '2da52b1ce826046a6f443269f47f49e4e9fa63e10f22f9ebdcde6f890e12ce51'
CANARY = 'd1e17dfd0f8f75d2d90ab1b9c5365b7a752b7e44a70c950c3c2b94ad28d207ea'


def run(args):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source = args.baseline_run.parent/'canary'
    checkpoint = args.baseline_run/'checkpoints/native-final.pt'
    require(digest(checkpoint) == CHECKPOINT and digest(source/'manifest.json') == CANARY,
            'Immutable preflight model/cache manifest mismatch')
    original = json.loads((source/'manifest.json').read_bytes())
    counts = dict(train=16, valid=8, test=8)
    require(original['config']['counts'] == counts, 'Tiny input population mismatch')
    names = ['metadata.json']+[split+'-'+field+'.npy' for split in counts for field in ('h', 'roles', 'offsets', 'targets')]
    cache = {name: original['cache'][name] for name in names}
    for name, item in cache.items():
        path = source/'cache'/name
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == item['bytes']
                and digest(path) == item['sha256'], 'Pinned tiny cache mismatch: '+name)
    existing = {p.name for p in args.output.iterdir()} if args.output.exists() else set()
    require(existing <= {'progress.json'}, 'Preserve existing scientific output')
    args.output.mkdir(exist_ok=True)
    for folder in ('cache', 'checkpoints'):
        (args.output/folder).mkdir()
    for name in names:
        os.link(source/'cache'/name, args.output/'cache'/name)
    for kind in KINDS:
        os.link(checkpoint, args.output/'checkpoints'/(kind+'-final.pt'))
    metadata = json.loads((args.output/'cache/metadata.json').read_bytes())
    require(all(len(metadata[s]) == counts[s] for s in counts), 'Metadata count mismatch')
    data = {s: {field: np.load(args.output/'cache'/f'{s}-{field}.npy', mmap_mode='r', allow_pickle=False)
                for field in ('h', 'roles', 'offsets', 'targets')} for s in counts}
    def batch(split, indices):
        require(len(indices) == 1 and indices[0] in (0, 1), 'Only fixed trace indices admitted')
        index = indices[0]
        lo, hi = map(int, data[split]['offsets'][index:index+2])
        h = torch.from_numpy(np.array(data[split]['h'][lo:hi], dtype=np.float64))[None]
        roles = torch.from_numpy(np.array(data[split]['roles'][lo:hi], dtype=np.float64))[None]
        mask = torch.ones(1, hi-lo, dtype=torch.bool)
        return (h, roles, mask), torch.tensor([int(data[split]['targets'][index])])
    cfg = dict(stage='source_trace_preflight_only', conditions=list(KINDS), counts=counts,
               selection='Existing immutable native checkpoint; no training or future-model selection',
               sample_rule='First2 cache units per split, fixed before outputs', seed=13)
    atomic_json(args.output/'manifest.json', dict(schema='q-rmea.protected-context.trace-preflight-manifest.v1',
        git_revision=git('rev-parse', 'HEAD'), config=cfg, config_sha256=identity(cfg), cache=cache,
        source_input_manifest=CANARY, checkpoint_sha256=CHECKPOINT))
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    samples = []
    for kind in KINDS:
        head = ProtectedContextHead(kind).double().eval()
        head.load_state_dict(state, strict=True)
        require(sum(p.numel() for p in head.parameters()) == 81028, 'Parameter count mismatch')
        for split in counts:
            samples += capture(args.output, head, kind, split, batch, metadata)
        print(json.dumps(dict(stage='trace_capture', condition=kind, samples=6)), flush=True)
    hashes = {kind: CHECKPOINT for kind in KINDS}
    report = close(args.output, samples, cfg, metadata, cache_manifest=cache, checkpoint_hashes=hashes)
    payload = json.loads((args.output/'sample-trace.json').read_bytes())
    negative = {}
    for name in ('incomplete_values', 'source_index', 'label_in_routing', 'legacy_native_lineage'):
        broken = copy.deepcopy(payload)
        if name == 'incomplete_values':
            broken['samples'][0]['representations'][0]['value_capture']['stored_element_count'] -= 1
        elif name == 'source_index':
            broken['samples'][0]['roles']['provenance']['cache_index'] = 1
        elif name == 'label_in_routing':
            prediction = next(s for s in broken['samples'][0]['stages'] if s['stage_id'] == 'prediction')
            prediction['input_refs'].append('source:pinned_split_target')
        else:
            paired = next(s for s in broken['samples'][0]['stages'] if s['stage_id'] == 'paired_native')
            paired['input_refs'] = ['psi', 'phi', 'mask', 'utilities', 'bias']
        try:
            validate_trace(args.output, broken, cfg=cfg, metadata=metadata,
                           cache_manifest=cache, checkpoint_hashes=hashes)
        except ValueError:
            negative[name] = 'rejected'
        else:
            raise ValueError('Invalid trace admitted: '+name)
    # Isolated negative fixtures: interruption cannot mutate validated evidence.
    negative_interrupts = {}
    for name in ('producer_dtype', 'extra_target_element'):
        broken = copy.deepcopy(payload)
        sample = next(s for s in broken['samples'] if s['sample_id'] == 'protected_context:final:train:0')
        fixture = args.output/'negative-fixtures'/name
        fixture.mkdir(parents=True)
        relative = Path('trace/protected_context-final-train-0.npz')
        published = args.output/relative
        published_hash = digest(published)
        for folder in ('cache', 'checkpoints', 'trace'):
            (fixture/folder).mkdir()
            for source_path in (args.output/folder).iterdir():
                destination = fixture/folder/source_path.name
                if source_path == published:
                    destination.write_bytes(source_path.read_bytes())
                else:
                    os.link(source_path, destination)
        os.link(args.output/'manifest.json', fixture/'manifest.json')
        path = fixture/relative
        require(path.stat().st_ino != published.stat().st_ino, 'Negative fixture shares validated inode')
        key = 'mention' if name == 'producer_dtype' else 'target'
        # Abrupt child exit after mutation deliberately bypasses Python cleanup.
        child_code = ('import numpy as np,os,sys;from pathlib import Path;'
            'p=Path(sys.argv[1]);a=np.load(p,allow_pickle=False);'
            'v={k:a[k].copy() for k in a.files};a.close();k=sys.argv[2];'
            'v[k]=v[k].astype(np.int64) if k=="mention" else np.repeat(v[k],2);'
            'np.savez_compressed(p,**v);os._exit(73)')
        child = subprocess.run([sys.executable, '-B', '-c', child_code, str(path), key],
                               capture_output=True, timeout=30)
        require(child.returncode == 73 and not child.stdout and not child.stderr,
                'Negative fixture interruption failed')
        require(digest(published) == published_hash == sample['representations'][0]['artifact_refs'][0]['sha256'],
                'Interrupted negative check changed validated archive')
        negative_interrupts[name] = dict(returncode=73, validated_archive_unchanged=True, separate_inode=True)
        with np.load(path, allow_pickle=False) as archive:
            array = archive[key].copy()
        ref = dict(path=relative.as_posix(), sha256=digest(path), bytes=path.stat().st_size, visibility='private')
        for stage in sample['stages']:
            stage['artifact_refs'] = [ref]
        for rep in sample['representations']:
            rep['artifact_refs'] = [ref]
            rep['value_capture']['artifact_refs'] = [ref]
            rep['value_capture']['reader_projection']['source_sha256'] = [ref['sha256']]
            if rep['id'] == key:
                rep.update(dtype=str(array.dtype), shape=list(array.shape))
                rep['value_capture'].update(shape=list(array.shape),
                    logical_element_count=int(array.size), stored_element_count=int(array.size))
        try:
            validate_trace(fixture, broken, cfg=cfg, metadata=metadata,
                           cache_manifest=cache, checkpoint_hashes=hashes)
        except ValueError:
            negative[name] = 'rejected'
        else:
            raise ValueError('Invalid complete values admitted: '+name)
    first = samples[0]
    length = len(metadata['train'][0]['tokens'])
    coordinates = [[0, 0, 0], [0, 39, length-1]]
    command = [sys.executable, '-B', str(ROOT/READER), '--run-dir', str(args.output),
               '--sample', first['sample_id'], '--representation', 'alpha',
               '--coordinates', json.dumps(coordinates)]
    reader = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
    require(reader.returncode == 0 and not reader.stderr, 'Real reader CLI failed')
    from baselines.retacred.rmea_protected_context_trace import read_value
    require(json.loads(reader.stdout) == read_value(args.output, first, 'alpha', coordinates), 'Reader CLI projection mismatch')
    private = {p.relative_to(args.output).as_posix(): dict(bytes=p.stat().st_size, sha256=digest(p))
               for p in sorted((args.output/'trace').iterdir())}
    summary = dict(schema='q-rmea.protected-context.trace-preflight.v1', status='complete',
        scope='Existing frozen model on tiny source cache, no optimizer', checks=report,
        negative_checks=negative, abrupt_negative_checks=negative_interrupts,
        actual_reader_cli=dict(command=command, returncode=reader.returncode,
            stdout_sha256=hashlib.sha256(reader.stdout.encode()).hexdigest(), source_sha256=digest(ROOT/READER)),
        input_manifest_sha256=CANARY, checkpoint_sha256=CHECKPOINT,
        sample_trace_sha256=digest(args.output/'sample-trace.json'),
        private_arrays=private, private_array_bytes=sum(v['bytes'] for v in private.values()),
        same_sample_condition_inputs='verified', paired_native='same checkpoint, unchanged native routing',
        optimizer_steps=0, COS_written_bytes=0, source_trace_readiness=True,
        formal_training_admission=False, L5='NOT_ACCEPTED')
    atomic_json(args.output/'trace-preflight.json', summary)
    atomic_json(args.output/'RUN_COMPLETE', dict(artifacts={name: digest(args.output/name) for name in
        ('manifest.json', 'sample-trace.json', 'trace-validation.json', 'trace-preflight.json')}))
    print(json.dumps(dict(status='complete', checks=report, negative_checks=negative,
                          private_array_bytes=summary['private_array_bytes'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
