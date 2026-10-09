"""Synthetic tiny CTL-DRP neural/trace/resume canary; never full-data training."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path

from .common import atomic_json, digest, fingerprint
from .ctl_drp_candidates import fit_train_pairs
from .ctl_drp_neural import (RENDERING, collate_groups, descriptions_from_native, encode_groups,
                            load_checkpoint, save_checkpoint, tiny_model, tokenizer_from_cache)


def examples():
    return [dict(id="synthetic-a", token=["Alice","joined","Acme","."],
                 subj_start=0,subj_end=0,obj_start=2,obj_end=2,
                 subj_type="PERSON",obj_type="ORGANIZATION",relation="per:employee_of"),
            dict(id="synthetic-b",token=["Bob","married","Eve","."],
                 subj_start=0,subj_end=0,obj_start=2,obj_end=2,
                 subj_type="PERSON",obj_type="PERSON",relation="per:spouse")]


def synthetic_table():
    support=examples()
    extra=copy.deepcopy(support[0]);extra["relation"]="per:schools_attended"
    return fit_train_pairs(support+[extra],source_split="train")


def describe():
    return {"schema":"ctl-drp-tiny-neural-smoke.v1", "formal_training":False,
            "formal_comparison_admitted":False, "device":"cpu", "precision":"FP32",
            "initialization":"random_tiny_Roberta_not_pretrained_SimCSE",
            "synthetic_sentences":2, "encoder_layers":2, "hidden_size":32,
            "optimizer_updates":2, "replayed_updates":1,
            "rendering":RENDERING,
            "pending":["large_checkpoint_load", "complete_task_adapter", "CUDA",
                       "qualified_physical_train_H", "collaborator_formal_results"]}


def run(output, cache, paper_source):
    import torch
    from .improved import SampleCapture
    torch.set_num_threads(1)
    output=Path(output)
    if output.exists():
        raise ValueError("fresh private output directory required")
    tokenizer=tokenizer_from_cache(cache)
    descriptions=descriptions_from_native(paper_source)
    table=synthetic_table(); records=examples()
    groups=encode_groups(records,table,tokenizer,descriptions,split="train")
    batch=collate_groups(groups); gold=[group["gold"] for group in groups]
    model=tiny_model(seed=13,dropout=.1)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=0.,eps=1e-8)
    contract={"rendering":RENDERING,"table_hash":fingerprint(table),"synthetic_scope":True,
              "architecture":model.encoder.config.to_dict(),
              "optimizer":{"name":"AdamW","lr":1e-5,"weight_decay":0.,"eps":1e-8}}
    output.mkdir(parents=True)
    traces=[]

    def trace(phase, data, input_batch, labels):
        model.train(phase=="train")
        optimizer.zero_grad(set_to_none=True)
        # batch_size=-1 deliberately keeps ALL candidate rows for these two
        # selected synthetic samples. Nothing is recomputed to fabricate a trace.
        with SampleCapture(model,row=0,batch_size=-1) as capture:
            capture.save("input.input_ids",input_batch["input_ids"])
            capture.save("input.attention_mask",input_batch["attention_mask"])
            result=model(**input_batch,gold=labels)
            capture.save("functional.first_token",result["first_token"])
            capture.save("functional.candidate_scores",result["scores"])
            for i,p in enumerate(result["probabilities"]): capture.save(f"functional.sentence{i}.probabilities",p)
            capture.save("functional.per_sentence_loss",result["per_sentence_loss"])
            if phase=="train":
                result["loss"].backward()
                for name,param in model.named_parameters():
                    if param.grad is not None: capture.save("parameter_gradient."+name,param.grad)
                optimizer.step()
                capture.save("post_update.head.weight",model.head.weight)
            tensors={k:v.contiguous() for k,v in capture.values.items()}
        tensor_path=output/(phase+"-trace.pt")
        torch.save(tensors,tensor_path)
        loaded=torch.load(tensor_path,weights_only=True)
        assert set(loaded)==set(tensors) and all(torch.equal(loaded[k],tensors[k]) for k in tensors)
        metadata={"phase":phase,"status":"complete_synthetic_diagnostic_scope_not_formal_case",
                  "samples":data,"groups":input_batch["candidate_groups"],"gold":labels,
                  "predictions":result["predictions"],"loss":float(result["loss"].detach()),
                  "tensor_artifact":tensor_path.name,"sha256":digest(tensor_path),
                  "tensor_values":"complete_numeric_values_in_pt_not_shapes_or_hashes_only",
                  "nodes":capture.nodes,
                  "tensors":{k:{"shape":list(v.shape),"dtype":str(v.dtype),"numel":v.numel()} for k,v in tensors.items()}}
        atomic_json(output/(phase+"-trace.json"),metadata)
        traces.append({"phase":phase,"pt":tensor_path.name,"sha256":metadata["sha256"],"tensor_count":len(tensors)})
        assert any("word_embeddings" in k for k in tensors)
        assert any("query" in k for k in tensors)
        assert any("attention_scores_to_probabilities" in k for k in tensors)
        return result

    first=trace("train",groups,batch,gold)
    checkpoint=output/"checkpoint.pt"
    save_checkpoint(checkpoint,model,optimizer,contract=contract,cursor=1)
    checkpoint_hash=digest(checkpoint)
    model.train();optimizer.zero_grad(set_to_none=True)
    second=model(**batch,gold=gold);second["loss"].backward();optimizer.step()
    expected={k:v.detach().clone() for k,v in model.state_dict().items()}
    replay=tiny_model(seed=99,dropout=.1)
    replay_optimizer=torch.optim.AdamW(replay.parameters(),lr=1e-5,weight_decay=0.,eps=1e-8)
    assert load_checkpoint(checkpoint,replay,replay_optimizer,contract=contract)==1
    replay.train();replay_optimizer.zero_grad(set_to_none=True)
    repeated=replay(**batch,gold=gold);repeated["loss"].backward();replay_optimizer.step()
    assert torch.equal(second["loss"],repeated["loss"])
    assert all(torch.equal(v,replay.state_dict()[k]) for k,v in expected.items())
    def equal(a,b):
        if torch.is_tensor(a): return torch.equal(a,b)
        if isinstance(a,dict): return set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
        if isinstance(a,(tuple,list)): return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        return a==b
    assert equal(optimizer.state_dict(),replay_optimizer.state_dict())
    testing=copy.deepcopy(records)
    testing[0]["token"][0]="Carol";testing[1]["token"][0]="Dave"
    test_groups=encode_groups(testing,table,tokenizer,descriptions,split="test")
    with torch.no_grad(): trace("test",test_groups,collate_groups(test_groups),[r["relation"] for r in testing])
    result={**describe(),"status":"tiny_neural_trace_resume_verified",
            "first_loss":float(first["loss"].detach()),"second_loss":float(second["loss"].detach()),
            "checks":{"actual_joint_encoder":True,"actual_word_embedding_QKV_attention_traces":True,
                      "actual_encoder_and_head_gradients":True,"exact_dropout_post_update_resume":True,
                      "exact_optimizer_state_resume":True,"actual_train_and_test_numeric_roundtrip":True},
            "checkpoint":{"path":checkpoint.name,"sha256":checkpoint_hash},"traces":traces}
    atomic_json(output/"neural-evidence.json",result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe",action="store_true")
    parser.add_argument("--cache",type=Path)
    parser.add_argument("--paper-source",type=Path)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.describe: print(json.dumps(describe(),ensure_ascii=False));return
    if not all((args.cache,args.paper_source,args.output)):
        parser.error("--cache, --paper-source and --output required; synthetic diagnostic only")
    print(json.dumps(run(args.output,args.cache,args.paper_source),ensure_ascii=False))

if __name__=="__main__":main()
