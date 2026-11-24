"""Focused offline checks of the changed control admission boundaries only."""
from pathlib import Path, PurePosixPath
import argparse
import ast
import copy
import json

r = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('--baseline', action='store_true')
args = ap.parse_args()
folder = r / ('before' if args.baseline else 'modified')


def definitions(name, wanted):
    tree = ast.parse((folder / name).read_text('utf8'))
    nodes = [n for n in tree.body if
             isinstance(n, ast.FunctionDef) and n.name in wanted or
             isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id.startswith('PARENT_') for t in n.targets)]
    ns = {'re': __import__('re'), 'Path': PurePosixPath}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), name, 'exec'), ns)
    return ns


old = json.loads((r.parent / 'prepared.json').read_bytes())['launch']
root = '/hy-tmp/Q-Attention/control/number-coherent-optical-full-20261125'
binding = definitions('allocation_publisher.py', {'require', 'optical_binding', 'validate_attempt_paths'})
if args.baseline:
    binding['optical_binding']('number-coherent-optical', old['allocation']['run_id'], root)
    assert '--resume' not in old['argv'] and '--restore-pin' not in old['argv']
    print(json.dumps(dict(baseline_fresh_contract_preserved=True, baseline_continuation_supported=False)))
else:
    ns = definitions('full_host_run.py', {'require', 'optical_binding', 'validate_number_optical_continuation'})
    v = copy.deepcopy(old)
    v['allocation']['runner_attempt_id'] = 'a' * 32
    v['allocation']['generation'] = 84
    v['output'] = '/hy-tmp/Q-Attention/owner/attempts/' + 'a' * 32 + '/output'
    v['run_root'] = v['output'] + '/' + v['allocation']['run_id']
    v['argv'][v['argv'].index('--output') + 1] = v['output']
    v['argv'].append('--resume')
    v.update(resume=True, parent_attempt_id=ns['PARENT_ATTEMPT'], parent_launch_sha256=ns['PARENT_LAUNCH_SHA'])
    ns['validate_number_optical_continuation'](v)
    bad = []
    for field, value in [('resume', False), ('experiment', 'relative-return'),
                         ('parent_attempt_id', 'b' * 32), ('parent_launch_sha256', '0' * 64),
                         ('revision', '0' * 40)]:
        x = copy.deepcopy(v)
        x[field] = value
        bad.append(x)
    for value in (83, 82):
        x = copy.deepcopy(v)
        x['allocation']['generation'] = value
        bad.append(x)
    x = copy.deepcopy(v)
    x['allocation']['runner_attempt_id'] = ns['PARENT_ATTEMPT']
    bad.append(x)
    x = copy.deepcopy(v)
    x['output'] = '/hy-tmp/Q-Attention/owner/attempts/' + 'a' * 32 + '/scratch'
    x['run_root'] = x['output'] + '/' + x['allocation']['run_id']
    x['argv'][x['argv'].index('--output') + 1] = x['output']
    bad.append(x)
    for flags in ([], ['--resume', '--restore-pin'], ['--resume', '--resume'], ['--resume', '--condition', 'number_mixed02']):
        x = copy.deepcopy(v)
        x['argv'] = v['argv'][:-1] + flags
        bad.append(x)
    rejected = 0
    for x in bad:
        try:
            ns['validate_number_optical_continuation'](x)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError('Incorrect continuation accepted')
    cfg = dict(experiment='number-coherent-optical', run_id=v['allocation']['run_id'],
               remote_run=v['output'] + '/' + v['allocation']['run_id'],
               control_root=root + '/continuation-20261125')
    assert len(binding['validate_attempt_paths'](cfg, v['allocation'])) == 4
    for key, value in [('control_root', root), ('remote_run', old['run_root']), ('publication_lineage', 'continuation-' + 'a' * 32)]:
        x = dict(cfg)
        x[key] = value
        try:
            binding['validate_attempt_paths'](x, v['allocation'])
        except ValueError:
            rejected += 1
        else:
            raise AssertionError('Incorrect publication binding accepted')
    ns['optical_binding']('number-coherent-optical', cfg['run_id'], cfg['control_root'])
    for old_id, generation in [(ns['PARENT_ATTEMPT'], 84), ('a' * 32, 83)]:
        a = dict(v['allocation'], runner_attempt_id=old_id, generation=generation)
        x = dict(cfg, remote_run='/hy-tmp/Q-Attention/owner/attempts/' + old_id + '/output/' + cfg['run_id'])
        try:
            binding['validate_attempt_paths'](x, a)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError('Stale publisher allocation accepted')
    print(json.dumps(dict(explicit_local_resume_accepted=True, frozen_parent_source_and_argv=True,
                          wrong_contracts_rejected=rejected, actual_gpu_or_cos_execution=False)))
