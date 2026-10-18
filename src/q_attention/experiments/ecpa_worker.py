"""One resumable baseline/selector process for the Q-ECPA parent scheduler."""
from __future__ import annotations

import copy
import math
import signal
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from q_attention.adapters.ecpa_formal import ECPAFormalModel
from q_attention.analysis.ecpa_formal_trace import FormalTrace, gradients, state_identity
from q_attention.models.relation_transformer import RelationExtractionModel, RelationTransformerConfig
from q_attention.experiments import ecpa_protocol as p


class Worker:
    def __init__(self, run_dir, mode, *, device="cuda", inject_oom=False, stop_after=None):
        if mode not in ("baseline",*p.MODES):
            raise ValueError("unknown job")
        self.root,self.mode = Path(run_dir),mode
        self.config,self.contract = p.read(self.root/"run_config.json"),p.read(self.root/"run_manifest.json")
        self.directory = self.root/"jobs"/mode
        self.directory.mkdir(parents=True,exist_ok=True)
        self.paused,self.started,self.update = False,time.monotonic(),0
        self.device = torch.device(device)
        self.inject_oom,self.stop_after = inject_oom,stop_after
        if (inject_oom or stop_after is not None) and self.config["profile"] != "synthetic_canary":
            raise ValueError("fault injection is canary-only")
        self.rows,self.datasets,self.vocab,self.labels,maximum,manifest = p.load_data(self.contract["data_dir"],self.config)
        if manifest != self.contract["data"] or p.identity(p.source_contract()) != self.contract["source_sha256"]:
            raise ValueError("worker source/data differs from immutable run contract")
        p.seed_all(self.config["seed"])
        base = RelationExtractionModel(RelationTransformerConfig(vocab_size=len(self.vocab),num_labels=len(self.labels),max_length=maximum,**self.config["model"]))
        if mode != "baseline":
            selected = torch.load(self.root/"jobs/baseline/best.pt",map_location="cpu",weights_only=False)
            if selected["fingerprint"] != self.contract["fingerprint"]:
                raise ValueError("upstream baseline checkpoint identity mismatch")
            base.load_state_dict({name[5:]:value for name,value in selected["model"].items() if name.startswith("base.")})
            p.seed_all(self.config["seed"]+3000)
        self.model = ECPAFormalModel(base,mode,self.config,train_base=mode=="baseline").to(self.device)
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        self.budget = self.config["baseline" if mode=="baseline" else "kernel"]
        self.optimizer = torch.optim.Adam(parameters,lr=self.budget["lr"]) if parameters else None
        self.total = math.ceil(len(self.rows["train"])/self.budget["batch_size"])*self.budget["epochs"] if parameters else 0
        tiers = p.memory_tiers(self.budget["batch_size"])
        self.tier_index = p.read(self.directory/"memory_state.json")["tier"]
        self.tier = tiers[self.tier_index]
        self.model.recompute = self.tier["recompute"]
        self.best,self.best_metric,self.best_update,self.selection_update = None,-1.0,0,0
        self.latest = self.directory/"latest.pt"
        self.writer = FormalTrace(self.directory,mode,self.config,self.contract,self.labels)
        self.writer.trace["parameter_layout"] = [dict(name=name,shape=list(v.shape),elements=v.numel())
            for name,v in self.model.named_parameters() if v.requires_grad]
        if self.latest.exists():
            state = torch.load(self.latest,map_location="cpu",weights_only=False)
            if state["fingerprint"] != self.contract["fingerprint"] or state["mode"] != mode:
                raise ValueError("checkpoint scientific identity mismatch")
            self.model.load_state_dict(state["model"])
            if self.optimizer:
                self.optimizer.load_state_dict(state["optimizer"])
            self.update,self.best,self.best_metric,self.best_update = state["next_update"],state["best_model"],state["best_metric"],state["best_update"]
            self.selection_update = state["selection_update"]
            p.restore_rng(state["rng"])
            self.saved_update,self.saved_at = self.update,time.monotonic()
        else:
            self.save()

    def save(self):
        self.model.zero_grad(set_to_none=True)
        p.save_state(self.latest,dict(schema_version="ecpa-checkpoint.v1",fingerprint=self.contract["fingerprint"],mode=self.mode,
            model=self.model.state_dict(),optimizer=self.optimizer.state_dict() if self.optimizer else None,
            next_update=self.update,best_model=self.best,best_metric=self.best_metric,best_update=self.best_update,selection_update=self.selection_update,rng=p.rng_state(),
            logical_batch=self.budget["batch_size"],tier_at_save=self.tier_index))
        self.saved_update,self.saved_at = self.update,time.monotonic()

    def heartbeat(self, phase, done=None, total=None):
        done,total = (self.update,self.total) if done is None else (done,total)
        now=time.monotonic()
        if phase==getattr(self,"last_heartbeat_phase",None) and now-getattr(self,"last_heartbeat_at",0)<1:
            return
        rate_key="optimizer" if phase in ("accumulate","training") else phase
        if rate_key!=getattr(self,"rate_key",None):
            self.rate_key,self.rate_start,self.rate_done=rate_key,now,done
        rate=(done-self.rate_done)/max(now-self.rate_start,1e-9)
        elapsed = now-self.started
        memory = {}
        if self.device.type == "cuda":
            memory = dict(allocated_gib=torch.cuda.memory_allocated()/2**30,reserved_gib=torch.cuda.memory_reserved()/2**30,
                peak_gib=torch.cuda.max_memory_allocated()/2**30,total_gib=torch.cuda.get_device_properties(0).total_memory/2**30)
        p.atomic_json(self.directory/"heartbeat.json",dict(phase=phase,done=done,total=total,rate=rate,
            elapsed=elapsed,tier=self.tier_index,micro=self.tier["micro"],checkpoint_update=self.saved_update,
            checkpoint_age=time.monotonic()-self.saved_at,**memory))
        self.last_heartbeat_phase,self.last_heartbeat_at=phase,now

    def evaluate(self, split):
        self.model.eval()
        predictions,gold,loss_sum = [],[],0.0
        count = len(self.datasets[split])
        micro = self.tier["micro"]
        with torch.no_grad():
            for start in range(0,count,micro):
                value = {k:v.to(self.device) for k,v in p.batch(self.datasets[split],range(start,min(count,start+micro))).items()}
                logits = self.model(**p.inputs(value))
                loss_sum += float(F.cross_entropy(logits,value["labels"],reduction="sum"))
                predictions.extend(logits.argmax(-1).cpu().tolist())
                gold.extend(value["labels"].cpu().tolist())
                self.heartbeat(f"evaluate_{split}",len(gold),count)
                if self.paused:
                    raise InterruptedError("safe pause during evaluation")
        return p.metrics(predictions,gold,self.labels,loss_sum)

    def snapshot(self, checkpoint, update):
        self.model.eval()
        model_sha = state_identity(self.model)
        for split in self.config["case_study"]["splits"]:
            for index in self.config["case_study"]["indices"]:
                self.heartbeat(f"trace_{checkpoint}_{split}_{index}")
                value = {k:v.to(self.device) for k,v in p.batch(self.datasets[split],[index]).items()}
                self.model.zero_grad(set_to_none=True)
                logits = self.model(**p.inputs(value),capture=True)
                captured = dict(self.model.observed)
                loss = F.cross_entropy(logits,value["labels"])
                if loss.requires_grad:
                    loss.backward()
                captured.update(loss=loss.detach().cpu(),parameter_gradient=gradients(self.model))
                self.writer.add(self.rows[split][index],split,index,checkpoint,"replay",captured,update,model_sha)
                self.model.zero_grad(set_to_none=True)
                if self.paused:
                    raise InterruptedError("safe pause during snapshot")

    def run(self):
        if self.update == 0:
            self.snapshot("initial",0)
        else:
            needed = {f"{split}-{i:04d}:initial:replay" for split in self.config["case_study"]["splits"] for i in self.config["case_study"]["indices"]}
            if not needed.issubset({s["sample_id"] for s in self.writer.trace["samples"]}):
                raise ValueError("committed training cursor without complete initial cases")
        per_epoch = math.ceil(len(self.rows["train"])/self.budget["batch_size"])
        def select_boundary():
            # Persist the optimizer boundary BEFORE interruptible validation.
            # A resumed boundary repeats selection if it was not committed.
            self.save()
            valid = self.evaluate("valid")["metrics"][self.config["primary"]]
            if valid > self.best_metric:
                self.best = {k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()}
                self.best_metric,self.best_update = valid,self.update
            self.selection_update = self.update
            self.save()
        if self.update and self.update%per_epoch==0 and self.selection_update<self.update:
            select_boundary()
        while self.update < self.total:
            self.model.train()
            indexes = p.update_indexes(len(self.rows["train"]),self.budget["batch_size"],self.update,self.config["seed"]+4000)
            logical = p.batch(self.datasets["train"],indexes)
            self.optimizer.zero_grad(set_to_none=True)
            first_capture,first_loss,first_identity = None,None,None
            cursor = 0
            while cursor < len(indexes):
                # Actual selected-sample observation is bounded to one example;
                # all remaining microbatches follow the active execution tier.
                observing = self.update==0 and cursor==0
                size = 1 if observing else self.tier["micro"]
                part = {k:v[cursor:cursor+size].to(self.device) for k,v in logical.items()}
                if self.inject_oom and self.mode=="quantum" and self.update==1 and not (self.directory/"injected_once.json").exists():
                    p.atomic_json(self.directory/"injected_once.json",dict(update=self.update))
                    raise torch.cuda.OutOfMemoryError("injected canary OOM")
                if observing:
                    first_identity = state_identity(self.model)
                logits = self.model(**p.inputs(part),capture=observing)
                loss = F.cross_entropy(logits,part["labels"],reduction="sum")/len(indexes)
                if observing:
                    first_capture,first_loss = dict(self.model.observed),F.cross_entropy(logits,part["labels"]).detach().cpu()
                loss.backward()
                cursor += len(part["labels"])
                self.heartbeat("accumulate")
            if first_capture is not None:
                first_capture.update(loss=first_loss,parameter_gradient=gradients(self.model))
                self.writer.add(self.rows["train"][0],"train",0,"update-0","training",first_capture,0,first_identity)
            if not all(torch.isfinite(parameter.grad).all() for parameter in self.model.parameters() if parameter.grad is not None):
                raise ValueError("nonfinite training gradient")
            self.optimizer.step()
            self.update += 1
            if self.update % per_epoch == 0:
                select_boundary()
            cadence = self.config["checkpoint"]
            if (self.paused or self.update==self.total or self.update%per_epoch==0
                    or self.update-self.saved_update>=cadence["every_updates"]
                    or time.monotonic()-self.saved_at>=cadence["every_seconds"]):
                self.save()
            self.heartbeat("training")
            print(f"[{self.mode}] {self.update}/{self.total} micro={self.tier['micro']}",flush=True)
            if self.paused or self.stop_after is not None and self.update>=self.stop_after:
                self.save()
                return p.PAUSED_EXIT
        if self.best is None:  # fixed controls have no optimizer
            self.best = {k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()}
            self.best_metric,self.best_update = self.evaluate("valid")["metrics"][self.config["primary"]],self.update
        self.save()
        p.atomic_json(self.directory/"TRAIN_COMPLETE.json",dict(update=self.update,fingerprint=self.contract["fingerprint"]))
        p.save_state(self.directory/"best.pt",dict(model=self.best,update=self.best_update,metric=self.best_metric,fingerprint=self.contract["fingerprint"]))
        final = {k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()}
        self.snapshot("final",self.update)
        self.model.load_state_dict(self.best)
        self.snapshot("best",self.best_update)
        reports = {split:self.evaluate(split) for split in ("valid","test")}
        trace = self.writer.complete()
        self.model.load_state_dict(final)
        p.atomic_json(self.directory/"metrics.json",dict(mode=self.mode,fingerprint=self.contract["fingerprint"],reports=reports,updates=self.update,
            best_update=self.best_update,trainable_parameters=sum(v.numel() for v in self.model.parameters() if v.requires_grad),
            effective_tier=self.tier_index,micro=self.tier["micro"],trace=trace,test_used_for_training_or_selection=False))
        p.atomic_json(self.directory/"RUN_COMPLETE.json",dict(fingerprint=self.contract["fingerprint"],metrics_sha256=p.sha(self.directory/"metrics.json"),trace_sha256=p.sha(self.directory/"sample_trace.json")))
        self.heartbeat("complete",self.total,self.total)
        return 0


def main(run, mode, **kwargs):
    worker = None
    try:
        torch.set_num_threads(2)
        worker = Worker(run,mode,**kwargs)
        def pause(signum, frame):
            worker.paused = True
        for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):
            signal.signal(sig,pause)
        return worker.run()
    except InterruptedError:
        # Snapshot/evaluation may have temporarily loaded BEST. Only explicit
        # optimizer boundaries can write LATEST; never save arbitrary state here.
        return p.PAUSED_EXIT
    except torch.cuda.OutOfMemoryError as exc:
        directory = Path(run)/"jobs"/mode
        p.atomic_json(directory/"OOM.json",dict(error=str(exc),trigger="actual_cuda_oom" if not kwargs.get("inject_oom") else "canary_injection",
            allocated_bytes=torch.cuda.memory_allocated() if torch.cuda.is_available() else None,
            reserved_bytes=torch.cuda.memory_reserved() if torch.cuda.is_available() else None,
            peak_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None))
        return p.OOM_EXIT
