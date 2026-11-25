"""Source-transfer boundary checks; no dataset, trained head or model modification."""
from pathlib import Path
import argparse, ast, hashlib, json, math
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument('--native-source', required=True)
p.add_argument('--out', required=True)
a = p.parse_args()
source = Path(a.native_source).read_bytes()
source_sha = hashlib.sha256(source).hexdigest()
assert source_sha == '9f4bfc57a1a966c3383c1a2e82fe406ac3d115b653b062290fcf42d3c4683ee0'
node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'circuit')
ns = {'torch': torch, 'DEPTH': 2, 'QUBITS': 3}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(a.native_source), 'exec'), ns)
circuit = ns['circuit']
eps = 1e-6
rng = np.random.default_rng(261034682)
groups = ([0, 1], [2, 3, 4], [5])
max_density_error = max_cov_error = max_d_error = max_norm_error = 0.
for trial in range(8):
    keys = circuit(torch.tensor(rng.uniform(-2.4, 2.4, (6, 2, 3, 2)), dtype=torch.float64)).numpy()
    queries = circuit(torch.tensor(rng.uniform(-2.4, 2.4, (40, 2, 3, 2)), dtype=torch.float64)).numpy()
    max_norm_error = max(max_norm_error, float(np.max(abs(np.sum(abs(keys)**2, -1)-1))))
    direct = abs(queries.conj() @ keys.T)**2
    utilities = rng.normal(size=(40, 6))
    cov_sum = np.zeros(40)
    denom = np.zeros(40)
    corrected_numerator = np.zeros(40)
    collapsed_numerator = np.zeros(40)
    for g in groups:
        k = keys[g]
        rho = np.einsum('ti,tj->ij', k, k.conj()) / len(g)
        mixed = np.einsum('ri,ij,rj->r', queries.conj(), rho, queries).real
        mean_probability = direct[:, g].mean(-1)
        max_density_error = max(max_density_error, float(np.max(abs(mixed-mean_probability))))
        dephased = np.sum(abs(queries)**2 * np.diag(rho).real, -1)
        direct_d = (abs(queries)**2 @ abs(k).T**2).mean(-1)
        max_d_error = max(max_d_error, float(np.max(abs(dephased-direct_d))))
        u = utilities[:, g]
        corrected_numerator += ((direct[:, g]+eps)*u).mean(-1)
        collapsed_numerator += (mixed+eps)*u.mean(-1)
        cov_sum += (direct[:, g]*u).mean(-1)-mean_probability*u.mean(-1)
        denom += mixed+eps
    max_cov_error = max(max_cov_error, float(np.max(abs(
        (corrected_numerator-collapsed_numerator)/denom-cov_sum/denom))))
assert max(max_density_error, max_cov_error, max_d_error, max_norm_error) < 1e-12

def state(y):
    angles = torch.zeros((2, 3, 2), dtype=torch.float64)
    angles[1, 0, 0] = y
    return circuit(angles).numpy()

k = np.stack([state(math.pi/2), state(-math.pi/2)])
coherent = k.sum(0); coherent /= np.linalg.norm(coherent)
coherent_cases = []
for y in [0., 2*math.pi/3]:
    phi = state(y)
    mean_p = float(np.mean(abs(k @ phi.conj())**2))
    coherent_p = float(abs(np.vdot(phi, coherent))**2)
    target_p = float(abs(np.vdot(phi, phi))**2)
    count_mass = (target_p+eps)/(target_p+eps+mean_p+eps)
    coherent_mass = (target_p+eps)/(target_p+eps+coherent_p+eps)
    coherent_cases.append(dict(effect_last_wire0_ry=y, mean_probability=mean_p,
        coherent_probability=coherent_p, count_corrected_target_mass=count_mass,
        coherent_target_mass=coherent_mass, delta=coherent_mass-count_mass))
assert coherent_cases[0]['delta'] < -.1 and coherent_cases[1]['delta'] > .1

# Same Born scores and denominator, opposite utility pairing: no signed-output ordering.
p_pair = np.array([.9, .1]); target_raw = .5+eps
utility_cases = []
for sign in [1, -1]:
    u_pair = sign*np.array([1., -1.])
    z_corrected = float(np.mean((p_pair+eps)*u_pair)/(target_raw+p_pair.mean()+eps))
    z_mean_value = float((p_pair.mean()+eps)*u_pair.mean()/(target_raw+p_pair.mean()+eps))
    utility_cases.append(dict(sign=sign, count_corrected_logit=z_corrected,
        mean_value_logit=z_mean_value, delta=z_corrected-z_mean_value))
assert utility_cases[0]['delta'] > .39 and utility_cases[1]['delta'] < -.39

best_case_mass_bounds = {str(n):(1+eps)/(1+eps+n*eps) for n in [1, 10, 10**6, 10**9]}
out = dict(status='PASS_SOURCE_TRANSFER_BOUNDARIES_ONLY', native_source_sha256=source_sha,
    circuit_port_checks=8*40, random_trials=8, variable_group_sizes=[2,3,1], eps=eps,
    max_density_linearity_error=max_density_error, max_dephased_linearity_error=max_d_error,
    max_covariance_identity_error=max_cov_error, max_native_state_norm_error=max_norm_error,
    coherent_counterexamples=coherent_cases, signed_utility_counterexamples=utility_cases,
    fixed_eps_single_target_upper_bounds=best_case_mass_bounds,
    source_claim_refuted=False, new_theory_claim=False, quantum_advantage_claim=False,
    task_rows_read=0, trained_head_forwards=0, training_updates=0, gpu_allocations=0,
    limitations=['Synthetic circuit ports, not empirical encoder reachability or utility feasibility.',
        'No rejection of arbitrary clustering, all Born models, or EPS schedules.',
        'Known linearity/covariance/bounded-weight algebra; no new T or accuracy theorem.'])
Path(a.out).write_text(json.dumps(out, indent=2)+'\n', encoding='utf8')
print(json.dumps(out))
