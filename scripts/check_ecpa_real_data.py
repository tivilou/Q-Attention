#!/usr/bin/env python3
"""Bounded real-data forward/backward qualification; never a full experiment."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import torch
from q_attention.adapters.ecpa_formal import ECPAFormalModel
from q_attention.experiments import ecpa_protocol as p
from q_attention.models.relation_transformer import RelationExtractionModel,RelationTransformerConfig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path,default=p.ROOT/"data/relation/retacred")
    parser.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    config=p.load_config()
    rows,sets,vocab,labels,maximum,manifest=p.load_data(args.data_dir,config)
    report=dict(schema_version="ecpa-real-data-probe.v1",full_experiment=False,data=manifest,
        max_tokens=maximum,device=args.device,torch=torch.__version__,source_sha256=p.identity(p.source_contract()),modes={})
    for mode in p.MODES:
        p.seed_all(config["seed"])
        base=RelationExtractionModel(RelationTransformerConfig(vocab_size=len(vocab),num_labels=len(labels),max_length=maximum,**config["model"]))
        # The initial base is intentionally untrained: this is data/gradient
        # compatibility evidence, not a measurement of a trained intervention.
        model=ECPAFormalModel(base,mode,config).to(args.device)
        parameters=[x for x in model.parameters() if x.requires_grad]
        optimizer=torch.optim.Adam(parameters,lr=.001) if parameters else None
        started=time.monotonic();steps=0
        for split in ("train","valid","test"):
            value={k:v.to(args.device) for k,v in p.batch(sets[split],[0]).items()}
            model.zero_grad(set_to_none=True)
            logits=model(**p.inputs(value),capture=True)
            loss=torch.nn.functional.cross_entropy(logits,value["labels"])
            if not torch.isfinite(loss):raise ValueError("nonfinite probe output")
            if split=="train" and optimizer:
                loss.backward()
                if not all(torch.isfinite(x.grad).all() for x in parameters if x.grad is not None):
                    raise ValueError("nonfinite probe gradient")
                optimizer.step();steps+=1
            if args.device=="cuda":torch.cuda.synchronize()
        report["modes"][mode]=dict(status="passed",optimizer_steps=steps,seconds=time.monotonic()-started,
            layers_observed=config["model"]["num_layers"],representations=len(model.observed),baseline="untrained diagnostic only")
        del model,base,parameters,optimizer,value,logits,loss
    p.atomic_json(args.output,report)
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
