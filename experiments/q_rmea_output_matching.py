"""Fixed-readout K1/K8 oracle matching; no fitting or target-label access."""
import argparse
from fractions import Fraction as F
import hashlib
import math
import os
from pathlib import Path
import subprocess
import sys
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baselines.retacred import rmea_token_transport_protocol as p
from experiments import q_rmea_query_alignment as qa

EVIDENCE = Path("docs/evidence/rmea/20261121-task-factorization")
CHECKPOINT_SHA = "2da52b1ce826046a6f443269f47f49e4e9fa63e10f22f9ebdcde6f890e12ce51"
ETA = F(1, 10**7)
STATUSES = ("approximate_match", "separated_above_tolerance", "unresolved")
LP_OPTIONS = dict(time_limit=3., primal_feasibility_tolerance=1e-9,
                  dual_feasibility_tolerance=1e-9)


def design():
    proposal = p.read(ROOT/EVIDENCE/"proposed-design.json")
    return dict(schema="q-rmea.output-matching.design.v1", proposal=proposal,
        proposal_sha256=p.digest(ROOT/EVIDENCE/"proposed-design.json"),
        tolerance="1/10000000 absolute pre-bias logit infinity error",
        arithmetic="Exact Fraction.from_float inputs and witnesses; outward float bounds for reporting",
        LP=dict(method="highs", options=LP_OPTIONS, row_scaling="max(abs(M_row),1e-12)",
            primal="one zero-objective simplex feasibility solve; clip negative candidate entries to0 then exactly verify",
            dual="if no verified approximate primal: one max-min separator with scaled y in[-1,1]"),
        scope="Per-passage oracle with one fixed common A, not learned/generalizing or all-frame exclusion",
        progress="96 samples, three disjoint32-case workers, no mixed units",
        trace="Private complete LP inputs/witnesses plus first2 per-shard native arrays; partial, not sample-trace.v2",
        recovery="Cheap replay from original protected immutable inputs; no checkpoint/COS writes")


def fraction(x):
    value = float(x)
    if not math.isfinite(value):
        raise ValueError("Nonfinite certificate coefficient")
    return F.from_float(value)


def outward(value, upper):
    if not value:
        return 0.
    return max(0., math.nextafter(float(value), math.inf if upper else -math.inf))


def ratio_record(value, upper):
    return dict(numerator=str(value.numerator), denominator=str(value.denominator),
                bound=outward(value, upper))


def record_fraction(row):
    return F(int(row["numerator"]), int(row["denominator"]))


def exact_inputs(a, u, z):
    return ([[fraction(v) for v in row] for row in a],
            [[fraction(v) for v in row] for row in u],
            [fraction(v) for v in z])


def primal(a, u, z, x):
    """Verify nonnegative nonzero B directly, independently of LP status."""
    rows, k, t = len(a), len(a[0]), len(u[0])
    if len(x) != k*t:
        raise ValueError("Primal shape differs")
    x = [fraction(v) for v in x]
    if min(x) < 0 or sum(x) <= 0:
        raise ValueError("Primal must be nonnegative and nonzero")
    aa, uu, zz = exact_inputs(a, u, z)
    errors = []
    for r in range(rows):
        weights = [sum(aa[r][j]*x[j*t+i] for j in range(k)) for i in range(t)]
        denominator = sum(weights)
        if denominator <= 0:
            raise ValueError("Nonpositive row denominator")
        residual = sum(weights[i]*(uu[r][i]-zz[r]) for i in range(t))
        errors.append(abs(residual)/denominator)
    return ratio_record(max(errors), True)


def dual(a, u, z, y):
    """Exact strict separation against real values of stored binary64 inputs."""
    if len(y) != len(a):
        raise ValueError("Dual shape differs")
    aa, uu, zz = exact_inputs(a, u, z)
    yy = [fraction(v) for v in y]
    delta = min(sum(yy[r]*aa[r][j]*(uu[r][i]-zz[r]) for r in range(len(a)))
                for j in range(len(a[0])) for i in range(len(u[0])))
    denominator = sum(abs(yy[r])*max(aa[r]) for r in range(len(a)))
    if delta <= 0 or denominator <= 0:
        return None
    return ratio_record(delta/denominator, False)


def solve(a, u, z):
    import numpy as np
    from scipy.optimize import linprog
    a, u, z = (np.asarray(v, np.float64) for v in (a, u, z))
    if (a.ndim != 2 or u.ndim != 2 or z.shape != (len(a),)
            or len(a) != len(u) or min(a.shape+u.shape) < 1
            or not all(np.isfinite(v).all() for v in (a, u, z)) or np.any(a <= 0)):
        raise ValueError("Invalid fixed-frame inputs")
    m = (a[:, :, None]*(u[:, None, :]-z[:, None, None])).reshape(len(a), -1)
    if not np.isfinite(m).all():
        raise ValueError("Nonfinite search matrix")
    scale = np.maximum(np.abs(m).max(1), 1e-12)
    scaled = m/scale[:, None]
    n = m.shape[1]
    first = linprog(np.zeros(n), A_eq=np.vstack((scaled, np.ones(n))),
        b_eq=np.r_[np.zeros(len(a)), 1.], bounds=(0., None),
        method="highs", options=LP_OPTIONS)
    row = dict(status="unresolved", primal_solver_status=int(first.status),
               dual_solver_status=None, upper=None, lower=None)
    private = {}
    if first.x is not None and np.isfinite(first.x).all():
        x = np.maximum(first.x, 0.)
        private["primal"] = x
        if x.sum() > 0:
            row["upper"] = primal(a, u, z, x)
            if record_fraction(row["upper"]) <= ETA:
                row["status"] = "approximate_match"
                return row, private
    # The solver proposes y only; certificate arithmetic uses unscaled A,U,z.
    second = linprog(np.r_[np.zeros(len(a)), -1.],
        A_ub=np.c_[-scaled.T, np.ones(n)], b_ub=np.zeros(n),
        bounds=[(-1., 1.)]*len(a)+[(None, None)], method="highs", options=LP_OPTIONS)
    row["dual_solver_status"] = int(second.status)
    if second.x is not None and np.isfinite(second.x).all():
        y = second.x[:-1]/scale
        private["dual"] = y
        if np.isfinite(y).all():
            row["lower"] = dual(a, u, z, y)
            if row["lower"] and record_fraction(row["lower"]) > ETA:
                row["status"] = "separated_above_tolerance"
    return row, private


def event(out, phase, current, total=96, **extra):
    p.atomic_json(out/"progress.json", dict(schema="research.progress.v1",
        attempt=Path(os.environ["RESEARCH_OUTPUT_DIR"]).parent.name,
        phase=phase, stage=phase, unit="sample", current=current, completed=current,
        total=total, observed_at=time.time(), batch_id="rmea-output-matching-20261121", **extra))


def report(out, phase, states):
    event(out, phase, sum(x["current"] for x in states), jobs=states)


def validate_review():
    row = p.read(ROOT/EVIDENCE/"implementation-review.json")
    names = ["experiments/q_rmea_output_matching.py", "experiments/q_rmea_query_alignment.py"]
    if (row.get("verdict") != "PASS" or row.get("review_scope") != "both" or row.get("must_fix")
            or row.get("hashes") != {n:p.digest(ROOT/n) for n in names}
            or row.get("design_sha256") != p.identity(design())
            or row.get("core_sha256") != p.identity(p.sources())):
        raise ValueError("Exact mechanism/lifecycle review required")


def validate_inputs(run):
    manifest, summary, seal = (p.read(run/n) for n in ("manifest.json", "summary.json", "RUN_COMPLETE"))
    if manifest["source_sha256"] != qa.ORIGINAL or manifest["config"] != p.config("discovery"):
        raise ValueError("Original source/config differs")
    for name in ("manifest.json", "summary.json"):
        if p.digest(run/name) != seal["artifacts"][name]["sha256"]:
            raise ValueError("Original seal differs")
    proposal = design()["proposal"]
    checkpoint = run/"checkpoints/native-final.pt"
    if (p.digest(checkpoint) != CHECKPOINT_SHA or proposal["checkpoint_sha256"] != CHECKPOINT_SHA
            or summary["results"]["native"]["checkpoint_sha256"] != CHECKPOINT_SHA):
        raise ValueError("Native checkpoint differs")
    for name, value in proposal["cache"].items():
        f = run/"cache"/name
        if (f.is_symlink() or f.stat().st_size != value["bytes"]
                or value != manifest["cache"][name] or p.digest(f) != value["sha256"]):
            raise ValueError("Validation input differs: "+name)
    return dict(checkpoint_sha256=CHECKPOINT_SHA, cache=proposal["cache"], original_source=qa.ORIGINAL)


def command(args, shard):
    return [sys.executable, "-B", str(Path(__file__).resolve()), "worker",
        "--run", str(args.run), "--output", str(args.output),
        "--parent-pid", str(os.getpid()), "--shard", str(shard)]


def worker(args):
    import ctypes
    import signal
    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Parent-death fencing failed")
    if os.getppid() != args.parent_pid:
        raise RuntimeError("Parent identity changed")
    import numpy as np
    import torch
    from baselines.retacred.rmea import EPS
    from baselines.retacred.rmea_token_transport import TrainableTokenTransportHead, route
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.device_count() != 1 or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("Exactly one reserved GPU per child")
    head = TrainableTokenTransportHead("native").double()
    head.load_state_dict(torch.load(args.run/"checkpoints/native-final.pt", map_location="cpu", weights_only=True))
    head = head.to("cuda").eval()
    out = args.output/("shard"+str(args.shard))
    out.mkdir()
    values = out/"values"
    values.mkdir()
    data = qa.dataset(args.run, "valid")
    ids = design()["proposal"]["selection"]["shards"][args.shard]
    rows = []
    with torch.no_grad():
        for position, index in enumerate(ids):
            start, stop = map(int, data["offsets"][index:index+2])
            h = torch.from_numpy(np.asarray(data["h"][start:stop], np.float64).copy())[None].to("cuda")
            roles = torch.from_numpy(np.asarray(data["roles"][start:stop], np.float64).copy())[None].to("cuda")
            mask = torch.ones((1,stop-start), dtype=torch.bool, device="cuda")
            d = head(h, roles, mask, details=True)
            expected = route(d["psi"],d["phi"],mask,kind="native")["alpha"]
            if not torch.allclose(expected, d["alpha"], atol=2e-12, rtol=2e-12):
                raise ValueError("Original native route parity differs")
            u = d["utilities"][0].cpu().numpy()
            # Match exactly the native emitted pre-bias logits, not gold labels.
            z = (d["logits"][0]-head.bias).cpu().numpy()
            a8 = (d["phi"].abs().square()+EPS).cpu().numpy()
            arrays = dict(utilities=u, target_logits=z, frame=a8)
            row = dict(index=index, tokens=stop-start, controls={})
            for name, a in (("K1", np.ones((40,1))), ("K8", a8)):
                result, witness = solve(a, u, z)
                row["controls"][name] = result
                arrays.update({name+"_"+k:v for k,v in witness.items()})
            if position < 2:
                arrays.update(psi=d["psi"][0].cpu().numpy(), phi=d["phi"].cpu().numpy(),
                    alpha=d["alpha"][0].cpu().numpy(), bias=head.bias.cpu().numpy(),
                    logits=d["logits"][0].cpu().numpy())
            f = values/(str(index)+".npz")
            np.savez_compressed(f, **arrays)
            row["private_values_sha256"] = p.digest(f)
            rows.append(row)
            event(out, "output_matching", len(rows), len(ids), shard=args.shard)
    p.atomic_json(out/"cases.json", rows)
    p.atomic_json(out/"WORKER_COMPLETE", dict(cases_sha256=p.digest(out/"cases.json"),
        count=len(rows), physical_gpu=os.environ["CUDA_VISIBLE_DEVICES"]))


def run(args):
    from baselines.retacred.orchestrate import gpu_ids
    validate_review()
    devices = gpu_ids("auto")
    if len(devices) != 3 or p.git("status", "--porcelain"):
        raise ValueError("Clean exact3GPU allocation required")
    args.run, args.output = args.run.resolve(), args.output.resolve()
    if any(args.output.is_relative_to(d) or d.is_relative_to(args.output) for d in (ROOT,args.run.parent)):
        raise ValueError("Independent new attempt output required")
    qa.fresh_output(args.output)
    started = time.time()
    ids = design()["proposal"]["selection"]["indices"]
    expected = sorted(range(qa.N),key=lambda i:hashlib.sha256(("qattn-task-output-v1:"+str(i)).encode()).digest())[:96]
    shards = design()["proposal"]["selection"]["shards"]
    if ids != expected or shards != [ids[i::3] for i in range(3)]:
        raise ValueError("Frozen96case selection differs")
    p.atomic_json(args.output/"design.json",design())
    p.atomic_json(args.output/"inputs.json",validate_inputs(args.run))
    qa.execute([("shard"+str(i),command(args,i),gpu) for i,gpu in enumerate(devices)],
        args.output,"matching",report=report)
    cases = []
    for i in range(3):
        out = args.output/("shard"+str(i))
        seal = p.read(out/"WORKER_COMPLETE")
        rows = p.read(out/"cases.json")
        if seal["cases_sha256"] != p.digest(out/"cases.json") or [r["index"] for r in rows] != shards[i]:
            raise ValueError("Worker coverage/seal differs")
        for row in rows:
            if p.digest(out/"values"/(str(row["index"])+".npz")) != row["private_values_sha256"]:
                raise ValueError("Private witness changed")
        cases.extend(rows)
    if len(cases) != 96 or len({r["index"] for r in cases}) != 96:
        raise ValueError("Aggregate96case coverage differs")
    counts = {name:{s:sum(r["controls"][name]["status"]==s for r in cases) for s in STATUSES}
        for name in ("K1","K8")}
    distinguishing = sum(r["controls"]["K1"]["status"]=="separated_above_tolerance"
        and r["controls"]["K8"]["status"]=="approximate_match" for r in cases)
    summary = dict(schema="q-rmea.output-matching.result.v1",source=p.git("rev-parse","HEAD"),
        count=96,counts=counts,distinguishing_cases=distinguishing,gate=distinguishing>0,
        seconds=time.time()-started,checkpoint_sha256=CHECKPOINT_SHA,
        scope="Fixed-native-frame oracle matching on96 development passages; no utility/phase necessity/all-frame or L5 claim")
    p.atomic_json(args.output/"case-status.json",cases)
    p.atomic_json(args.output/"summary.json",summary)
    p.atomic_json(args.output/"RUN_COMPLETE",dict(artifacts={n:p.digest(args.output/n)
        for n in ("design.json","inputs.json","case-status.json","summary.json")}))
    event(args.output,"complete",96)
    print(__import__("json").dumps(summary),flush=True)


def unit(args):
    import numpy as np
    validate_review()
    qa.fresh_output(args.output)
    a = np.ones((2,1));u = np.array([[0.,1.],[1.,0.]]);z = np.array([.9,.9])
    certificate = dual(a,u,z,[-1.,-1.])
    assert abs(float(record_fraction(certificate))-.4)<1e-15
    assert dual(a,u,z,[1.,1.]) is None
    row, _ = solve(a,u,z)
    assert row["status"]=="separated_above_tolerance"
    feasible, _ = solve(a,u,np.array([.25,.75]))
    assert feasible["status"]=="approximate_match"
    assert primal(a,u,[.25,.75],[3.,1.])["bound"]==0
    # Independent constructed B verifies the K-major flattening and denominators.
    aa=np.array([[.2,.8],[.7,.3]])
    uu=np.array([[1.,-2.,3.],[4.,1.,-1.]])
    xx=np.array([[1.,2.,3.],[3.,2.,1.]])
    ss=aa@xx
    zz=(ss*uu).sum(1)/ss.sum(1)
    assert primal(aa,uu,zz,xx.ravel())["bound"]<1e-14
    assert solve(aa,uu,zz)[0]["status"]=="approximate_match"
    try:
        primal(a,u,z,[0.,0.])
    except ValueError:
        pass
    else:
        raise AssertionError("Zero primal accepted")
    dummy=argparse.Namespace(run=Path("/unused/run"),output=Path("/unused/output"))
    for i in range(3):
        parsed=parser().parse_args(command(dummy,i)[3:])
        assert parsed.action=="worker" and parsed.shard==i and parsed.parent_pid==os.getpid()
    check=subprocess.run([sys.executable,"-B",str(Path(__file__).resolve()),"--help"],
        capture_output=True,timeout=20)
    assert check.returncode==0
    report(args.output,"matching",[dict(current=1),dict(current=2),dict(current=3)])
    e=p.read(args.output/"progress.json")
    assert e["current"]==6 and e["total"]==96 and e["unit"]=="sample"
    p.atomic_json(args.output/"unit-validation.json",dict(status="PASS",exact_dual_fixture=certificate,
        positive_and_negative_certificates=True,feasible_K1_and_K8=True,actual_CLI_and_callback=True,
        scope="Focused arithmetic and launch-contract checks, not GPU/data acceptance"))
    p.atomic_json(args.output/"RUN_COMPLETE",dict(sha256=p.digest(args.output/"unit-validation.json")))


def parser():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action",choices=("unit","worker","run"))
    ap.add_argument("--run",type=Path)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--parent-pid",type=int)
    ap.add_argument("--shard",type=int,choices=range(3))
    return ap


def main():
    ap=parser();args=ap.parse_args()
    if args.action!="unit" and args.run is None:
        ap.error("--run required")
    if args.action=="worker" and (args.parent_pid is None or args.shard is None):
        ap.error("Exact parent and shard required")
    dict(run=run,worker=worker,unit=unit)[args.action](args)


if __name__=="__main__":
    main()
