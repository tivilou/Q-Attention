"""Assess candidate timing sensitivity and check the installed release contract without scheduling."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


ARM_INDEX = 'phase_presence_q'
HOST_RUNNER = '238f7e68ea94250ecd786e64c269c8006c7225e40e638879515e545bf7795662'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_bytes())


def assess(terminal_path, registry_budget_path, native_terminal_path,
           cost_path, allocation_source, wrapper_path):
    assert sha(wrapper_path) == HOST_RUNNER
    terminal = load(terminal_path)
    b = load(registry_budget_path)
    cost = load(cost_path)
    native = load(native_terminal_path)
    assert b['terminal_sha256'] == sha(terminal_path)
    assert b['gpu_seconds_requested'] == 0 and b['scientific_admission'] is False
    assert terminal['generation'] == native['generation'] == 130
    assert terminal['allocation'] == native['state'] == 'released'
    assert terminal['cumulative_reserved_gpu_seconds'] == 551.457
    assert terminal['remaining_candidate_gpu_seconds'] == b['remaining_gpu_seconds']
    assert terminal['remaining_candidate_wall_seconds'] == b['remaining_wall_seconds']
    assert cost['budget']['gpu_s'] == b['remaining_gpu_seconds']
    assert terminal['updates_confirmed'] == 4 and terminal['science_updates'] == 0
    charge = terminal['charge_gpu_seconds']
    wall_history = terminal['request_to_release_seconds']
    wrapper_history = terminal['terminal_wrapper_seconds']
    assert 0 < wrapper_history < wall_history <= charge < b['remaining_gpu_seconds']
    request, reserve = {}, {}
    tree = ast.parse(allocation_source.read_text(encoding='utf-8'))
    methods = {m.name:m for cls in tree.body if isinstance(cls,ast.ClassDef) and cls.name=='Allocations'
               for m in cls.body if isinstance(m,ast.FunctionDef)}
    assert {'request','reserve','release','ack'} <= methods.keys()
    for dest, key in ((request,'request'),(reserve,'reserve')):
        dest['time_calls'] = [x.attr for x in ast.walk(methods[key])
                              if isinstance(x,ast.Attribute) and isinstance(x.value,ast.Name) and x.value.id=='time']
        dest['timestamp_checks'] = ast.unparse(methods[key]).count('deadline')+ast.unparse(methods[key]).count('expire')
    assert request['time_calls'] == ['time'] and reserve['time_calls'] == ['time']
    assert request['timestamp_checks'] == reserve['timestamp_checks'] == 0
    host_script = ast.parse(wrapper_path.read_text(encoding='utf-8'))
    child = [n for n in ast.walk(host_script) if isinstance(n, ast.Call) and isinstance(n.func,ast.Attribute)
             and isinstance(n.func.value,ast.Name) and n.func.value.id=='subprocess' and n.func.attr=='run']
    assert len(child) == 1 and 'timeout' not in {x.arg for x in child[0].keywords}
    # Reuse fixed two-stage estimates, but swap the measurement charge
    # placeholder for the preceding actual request-to-release charge.
    script=registry_budget_path.with_name('barrier_budget.py')
    spec=importlib.util.spec_from_file_location('barrier_budget',script)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    terminal_raw_path=terminal_path.parent/'presence-fit128-native-canary-20261128-results/output/summary.json'
    raw=load(terminal_raw_path)
    assert terminal['pinned_sha256']['summary']==sha(terminal_raw_path)
    score=raw['eval_seconds_fit_batch_only']
    scores=[1.1*(16*score[arm]+(16*score['block_D'] if arm==ARM_INDEX else 0)) for arm in mod.ARMS]
    base=[1.1*terminal['projection']['per_arm_base_seconds'][arm]-scores[i] for i,arm in enumerate(mod.ARMS)]
    thresholds={}
    for devices in (1,2):
        headroom=min((b['remaining_gpu_seconds']-charge)/devices,b['remaining_wall_seconds']-wall_history)-10.0
        cap=mod.limit(base,scores,devices,headroom)
        thresholds[str(devices)]=dict(headroom=headroom,min_hypothetical_equal_qd_full_fit_selection_reduction_pct=100*(1-cap))
    return dict(status='DEFER_NO_MEASURED_OR_ENFORCEABLE_COST_ADMISSION',measurement_charged_new=False,
                installed_allocation_source_sha256=sha(allocation_source),installed_host_wrapper_sha256=sha(wrapper_path),
                observed_previous_wrapper_s=wrapper_history,observed_previous_request_to_release_s=wall_history,
                difference_previous_request_to_release_minus_wrapper_s=wall_history-wrapper_history,
                old_charge_gpu_s=charge,cumulative_consumed_gpu_s=terminal['cumulative_reserved_gpu_seconds'],
                remaining_gpu_s=b['remaining_gpu_seconds'],remaining_wall_s=b['remaining_wall_seconds'],
                hypothetical_thresholds_if_next_short_probe_costs_same_as_previous=thresholds,
                runtime_findings=dict(installed_request_reserve_deadline=False,host_child_subprocess_timeout=False,
                                      trusted_matching_terminal_release_required=True),
                scientific_admission=False,new_gpu_allocations=0,optimizer_updates=0,
                scope='Source inspection of current bound adapter and prior terminal arithmetic. No claim that all GPU measurements are impossible; old charge is neither a new spend nor a guaranteed upper bound.')


if __name__ == '__main__':
    print(json.dumps(assess(*(Path(x) for x in sys.argv[1:])),indent=2,allow_nan=False))