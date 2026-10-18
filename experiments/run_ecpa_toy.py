"""Bounded CPU-only Q-ECPA screen. No formal-data runner is implied."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import torch
from q_attention.adapters.q_ecpa import ECPAToyModel
from q_attention.analysis.ecpa_trace import TraceWriter, atomic_json, validate_trace
from q_attention.models.relation_transformer import RelationExtractionModel, RelationTransformerConfig
from q_attention.plugins.q_ecpa import ECPAConfig, EntityCoherentPath, MODES
from q_attention.tasks.ecpa_toy_data import datasets, collate, LABELS, WORDS


def source_identity():
    files = [Path(__file__), ROOT/"scripts/run_ecpa_toy.sh"]
    files += [ROOT/"src/q_attention"/path for path in (
        "plugins/q_ecpa.py","adapters/q_ecpa.py","analysis/ecpa_trace.py",
        "tasks/ecpa_toy_data.py","models/relation_transformer.py")]
    h = hashlib.sha256()
    for path in sorted(files):
        h.update(str(path.relative_to(ROOT)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def make_model(mode, seed):
    torch.manual_seed(seed)
    base = RelationExtractionModel(RelationTransformerConfig(
        vocab_size=len(WORDS),num_labels=len(LABELS),dim=16,num_heads=2,num_layers=1,
        ff_dim=32,dropout=0,max_length=8))
    # A separate seed makes backbone initialization independent of plugin size.
    torch.manual_seed(seed+1000)
    return ECPAToyModel(base,EntityCoherentPath(ECPAConfig(),mode))


def metrics(logits, gold):
    predictions = logits.argmax(-1)
    confusion = torch.zeros((3,3),dtype=torch.long)
    for target,pred in zip(gold,predictions):
        confusion[target,pred] += 1
    tp = confusion.diag().double()
    precision = tp/confusion.sum(0).clamp_min(1)
    recall = tp/confusion.sum(1).clamp_min(1)
    f1 = 2*precision*recall/(precision+recall).clamp_min(1e-15)
    accuracy = float((predictions==gold).double().mean())
    return dict(accuracy=accuracy,micro_precision=accuracy,micro_recall=accuracy,
                micro_f1=accuracy,macro_f1=float(f1.mean()),
                loss=float(torch.nn.functional.cross_entropy(logits,gold)),
                per_class=[dict(label=label,precision=float(precision[i]),recall=float(recall[i]),
                                f1=float(f1[i]),support=int(confusion[i].sum())) for i,label in enumerate(LABELS)],
                confusion=confusion.tolist())


def evaluate(model, rows):
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(**collate(rows[start:start+16])) for start in range(0,len(rows),16)])
    return metrics(logits,torch.tensor([row["gold"] for row in rows]))


def gradient_values(model):
    parameters = [p for p in model.intervention.kernel.parameters() if p.requires_grad]
    gradient = torch.cat([p.grad.detach().flatten().cpu() if p.grad is not None else
                          torch.zeros(p.numel()) for p in parameters]) if parameters else torch.empty(0)
    capture = model.last_capture
    difference = float((capture["pq"]-capture["pd"]).abs().max()) if "pq" in capture else 0.0
    phase = capture.get("phase")
    flux = float((phase[...,0,1]+phase[...,1,2]+phase[...,2,0]).abs().max()) if phase is not None else 0.0
    return dict(classifier_gradient=model.base.classifier[-1].weight.grad.detach().cpu().clone(),
                plugin_gradient=gradient,diagnostics=torch.tensor([difference,flux,float(gradient.norm())]))


def model_identity(model):
    h = hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(str((tuple(tensor.shape),tensor.dtype)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def capture_case(model, writer, row, split, checkpoint, update, lane="replay"):
    model.eval()
    model.zero_grad(set_to_none=True)
    logits = model(**collate([row]),capture=True)
    loss = torch.nn.functional.cross_entropy(logits,torch.tensor([row["gold"]]))
    loss.backward()
    tensors = {name:value[0] for name,value in model.last_capture.items()}
    tensors.update(loss=loss.detach(),**gradient_values(model))
    writer.add(row,split,checkpoint,lane,tensors,update=update,model_sha256=model_identity(model))
    model.zero_grad(set_to_none=True)


def save_checkpoint(path, value):
    temporary = path.with_suffix(".pt.tmp")
    torch.save(value,temporary)
    temporary.replace(path)


def run_one(root,mode,config,data,stop_after=None,resume=False):
    directory = root/mode
    directory.mkdir(parents=True,exist_ok=True)
    latest,best_path = directory/"latest.pt",directory/"best.pt"
    config_hash = hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    experiment = dict(run_id=root.name,dataset="ecpa-public-synthetic-v1",code_revision=config["source_sha256"],
                      config_sha256=config_hash,model_identity=mode,seed=config["seed"])
    if (directory/"RUN_COMPLETE.json").exists():
        summary = json.loads((directory/"summary.json").read_text())
        if summary["config_sha256"] != config_hash:
            raise ValueError("completed run identity mismatch")
        validate_trace(directory/"sample_trace.json")
        return summary
    model = make_model(mode,config["seed"])
    optimizer = torch.optim.Adam(model.parameters(),lr=config["lr"])
    writer = TraceWriter(directory,mode,experiment,resume=resume)
    step,best_metric = 0,float("-inf")
    if latest.exists():
        if not resume:
            raise ValueError("existing run requires --resume")
        state = torch.load(latest,map_location="cpu",weights_only=False)
        if state["config_sha256"] != config_hash or state["dataset_sha256"] != config["dataset_sha256"]:
            raise ValueError("checkpoint source/config/dataset mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        step,best_metric = state["next_update"],state["best_metric"]
    elif resume and writer.trace["samples"]:
        # A crash before first durable update can replay the initial capture.
        writer = TraceWriter(directory,mode,experiment)
    if step == 0:
        for split in data:
            capture_case(model,writer,data[split][0],split,"initial",0)
    # Minibatch identity is a pure function of (seed, next_update), not the
    # number of workers or how many prior processes were restarted.
    started = time.monotonic()
    while step < config["updates"]:
        if stop_after is not None and step >= stop_after:
            return dict(status="interrupted",next_update=step)
        generator = torch.Generator().manual_seed(config["seed"]+step*37)
        indices = torch.randperm(len(data["train"]),generator=generator)[:config["batch_size"]].tolist()
        if step == 0:
            indices = [0]  # real training trace has an unambiguous per-sample gradient
        rows = [data["train"][i] for i in indices]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(**collate(rows),capture=step==0)
        loss = torch.nn.functional.cross_entropy(logits,torch.tensor([row["gold"] for row in rows]))
        loss.backward()
        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError("nonfinite loss/gradient")
        if step == 0:
            tensors = {name:value[0] for name,value in model.last_capture.items()}
            tensors.update(loss=loss.detach(),**gradient_values(model))
            writer.add(rows[0],"train","update-0","training",tensors,update=0,model_sha256=model_identity(model))
        optimizer.step()
        step += 1
        if step % 10 == 0 or step == config["updates"]:
            score = evaluate(model,data["valid"])["macro_f1"]
            if score > best_metric:
                best_metric = score
                save_checkpoint(best_path,dict(model=model.state_dict(),update=step,metric=score))
        # Latest only, atomic at each optimizer boundary. The tiny toy model
        # makes this cheap; formal cadence must be measured independently.
        save_checkpoint(latest,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                        torch_rng=torch.get_rng_state(),next_update=step,best_metric=best_metric,
                        config_sha256=config_hash,dataset_sha256=config["dataset_sha256"]))
        if step % 20 == 0 or step == config["updates"]:
            print(f"[{mode:12s}] {step:3d}/{config['updates']} | {step/(time.monotonic()-started):.1f} updates/s | loss {float(loss.detach()):.4f}",flush=True)
    final = copy.deepcopy(model.state_dict())
    best = torch.load(best_path,map_location="cpu",weights_only=False)
    model.load_state_dict(best["model"])
    validation,test = evaluate(model,data["valid"]),evaluate(model,data["test"])
    for split in data:
        capture_case(model,writer,data[split][0],split,"best",best["update"])
    model.load_state_dict(final)
    for split in data:
        capture_case(model,writer,data[split][0],split,"final",step)
    writer.complete()
    summary = dict(status="complete",mode=mode,config_sha256=config_hash,updates=step,
                   best_update=best["update"],valid=validation,test=test,
                   trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                   trace=validate_trace(directory/"sample_trace.json"),
                   scope="Synthetic toy only; no formal Re-TACRED utility or quantum advantage claim.")
    atomic_json(directory/"summary.json",summary)
    atomic_json(directory/"RUN_COMPLETE.json",dict(config_sha256=config_hash,trace_status="passed"))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--updates",type=int,default=100)
    parser.add_argument("--seed",type=int,default=13)
    parser.add_argument("--modes",nargs="+",choices=MODES,default=list(MODES))
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--stop-after",type=int)
    args = parser.parse_args(argv)
    if not 10 <= args.updates <= 500 or args.stop_after is not None and not 1 <= args.stop_after <= args.updates:
        parser.error("bounded toy requires 10..500 updates and a positive stop boundary")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    data = datasets(args.seed)
    config = dict(seed=args.seed,updates=args.updates,batch_size=16,lr=0.003,
                  dataset_sha256=hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest(),
                  source_sha256=source_identity(),primary="best-validation macro_f1; report held-out test macro_f1")
    args.output.mkdir(parents=True,exist_ok=True)
    config_path = args.output/"config.json"
    if config_path.exists():
        if not args.resume or json.loads(config_path.read_text()) != config:
            raise ValueError("existing output requires --resume with identical source/config")
    atomic_json(config_path,config)
    results = {}
    for mode in args.modes:
        print(f"ECPA TOY | seed {args.seed} | mode {mode} | CPU | {args.updates} updates",flush=True)
        results[mode] = run_one(args.output,mode,config,data,args.stop_after,args.resume)
        if results[mode]["status"] == "interrupted":
            print(f"Stopped safely at optimizer boundary {results[mode]['next_update']}",flush=True)
            return 0
    atomic_json(args.output/"summary.json",results)
    print("Complete: selected samples contain actual full numeric states at initial/best/final.",flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
