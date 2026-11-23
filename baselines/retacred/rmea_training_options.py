"""Explicit training options; absent options retain the accepted legacy behavior."""


def batch_size(cfg):
    n = cfg.get('batch_size', 32)
    if type(n) is not int or n < 1:
        raise ValueError('Positive integer logical batch size required')
    return n


def per_epoch(cfg):
    n = cfg['counts']['train']
    if type(n) is not int or n < 1:
        raise ValueError('Positive integer training count required')
    b = batch_size(cfg)
    return (n + b - 1) // b


def dtype_name(cfg):
    n = cfg.get('training_dtype', 'float64')
    if n not in ('float32', 'float64'):
        raise ValueError('Declared float32 or float64 training required')
    return n


def selection_metric(cfg):
    n = cfg.get('validation_metric', 'positive39_macro_f1')
    if n not in ('macro_f1', 'positive39_macro_f1'):
        raise ValueError('Unknown validation selection metric')
    return n


def selection_description(cfg):
    name = 'all40 Macro' if selection_metric(cfg) == 'macro_f1' else 'positive39 Macro'
    return 'validation ' + name + '; strict improvement; first tie'


def reported_checkpoint(cfg):
    n = cfg.get('reported_checkpoint', 'final')
    if n not in ('final', 'validation_best'):
        raise ValueError('Unknown reported checkpoint policy')
    return n


def exact_tensors(actual, expected):
    import torch
    return (torch.is_tensor(actual) and torch.is_tensor(expected) and
            actual.dtype==expected.dtype and actual.shape==expected.shape and
            actual.layout==expected.layout==torch.strided and
            torch.equal(actual.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                        expected.detach().cpu().contiguous().reshape(-1).view(torch.uint8)))


def exact_tree(actual,expected):
    import torch
    if torch.is_tensor(actual):return exact_tensors(actual,expected)
    if isinstance(actual,dict):return isinstance(expected,dict) and set(actual)==set(expected) and all(exact_tree(actual[k],expected[k]) for k in actual)
    if isinstance(actual,(list,tuple)):return type(actual)==type(expected) and len(actual)==len(expected) and all(exact_tree(a,b) for a,b in zip(actual,expected))
    return type(actual)==type(expected) and actual==expected


def validate_retained_checkpoints(output, state, modes, cfg):
    """Validate retained evidence against a proposed commit before any repair."""
    from pathlib import Path
    import torch
    output=Path(output)
    for kind in modes:
        final=output/'checkpoints'/(kind+'-final.pt')
        selected=output/'checkpoints'/(kind+'-selected.pt')
        for path in (final,selected):
            if path.is_symlink() or (path.exists() and not path.is_file()):
                raise ValueError('Nonregular retained checkpoint entry')
        if selected.exists() and reported_checkpoint(cfg)!='validation_best':
            raise ValueError('Selected evidence contradicts final-reporting policy')
        if not final.exists() and not selected.exists():continue
        if state is None:raise ValueError('Retained checkpoint without proposed worker commit')
        if final.exists():
            if kind in state['completed']:expected=state['completed'][kind]['state']
            else:
                cursor=state['cursor'];index=cursor['mode']
                if (index>=len(modes) or modes[index]!=kind or cursor['epoch']!=cfg['epochs'] or cursor['next_start']!=0):
                    raise ValueError('Retained final precedes available post-validation commit')
                expected=state['active']
            _retained_state(final,expected)
        if selected.exists():
            if kind not in state['completed'] or kind not in state['best']:
                raise ValueError('Retained selected precedes available completed-condition commit')
            _retained_state(selected,state['best'][kind]['state'])


def _retained_state(path, expected):
    import torch
    if path.is_symlink() or not path.is_file():raise ValueError('Nonregular retained checkpoint')
    actual=torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(actual,dict) or not isinstance(expected,dict) or set(actual)!=set(expected) or any(not exact_tensors(actual[k],v) for k,v in expected.items()):
        raise ValueError('Retained checkpoint tensor bytes differ from proposed commit')


def expected_training_identities(cfg):
    """Reconstruct only the common initial head and seeded epoch permutations on CPU."""
    import hashlib
    import numpy as np
    import torch
    from .rmea import RMEAHead
    from .rmea_token_transport_study import state_identity
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(cfg['seed'])
        initial=RMEAHead('quantum').to(dtype=getattr(torch,dtype_name(cfg))).state_dict()
    generator=cfg.get('order_generator','numpy_default_rng')
    if generator=='torch_epoch_seed':
        orders=[torch.randperm(cfg['counts']['train'],generator=torch.Generator().manual_seed(cfg['seed']+e)).numpy() for e in range(cfg['epochs'])]
    elif generator=='numpy_default_rng':
        rng=np.random.default_rng(cfg['seed']);orders=[rng.permutation(cfg['counts']['train']) for _ in range(cfg['epochs'])]
    else:raise ValueError('Unknown epoch permutation generator')
    order=hashlib.sha256(b''.join(o.astype('<i8').tobytes() for o in orders)).hexdigest()
    return state_identity(initial),order


def validate_training_envelope(state,cfg):
    initial,order=expected_training_identities(cfg)
    if state.get('schema')!='q-rmea.resume.v1' or state.get('initial_sha')!=initial or state.get('order_sha')!=order:
        raise ValueError('Resume schema/initial/order identity differs')
