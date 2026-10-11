"""Read-only dependency/data/device/exchange checks; no model training or download."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import ssl
from urllib.request import Request, urlopen

from .common import ROOT, PACKAGE, cohort
from .assets import CA


def check(data_dir: Path, gpu: str = "auto") -> dict:
    from .orchestrate import gpu_ids
    versions = {}
    for name, expected in (("transformers","4.44.0"),("huggingface-hub","0.36.2"),("numpy","1.26.4"),("ujson","5.11.0")):
        version = importlib.metadata.version(name)
        if version != expected:
            raise ValueError(f"{name} {version} differs from locked {expected}; install baselines/retacred/requirements.txt")
        versions[name] = version
    import torch
    if not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; formal reference suite is GPU-only")
    versions["torch"] = torch.__version__
    devices = gpu_ids(gpu)
    for split in ("train","valid","test"):
        path = data_dir/(split+".jsonl")
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing or symlinked canonical split: {path}")
    url, token = os.environ.get("PROJECT_EXCHANGE_URL",""), os.environ.get("PROJECT_EXCHANGE_TOKEN","")
    if not url.startswith("https://") or not token:
        raise ValueError("PROJECT_EXCHANGE_URL (HTTPS) and PROJECT_EXCHANGE_TOKEN required before training")
    context = ssl.create_default_context(cafile=CA)
    with urlopen(url.rstrip("/")+"/healthz",context=context,timeout=15) as response:
        health = json.load(response)
    if health.get("status") != "ok" or health.get("service") != "q-attention-artifact-exchange-v1":
        raise ValueError("not a healthy project artifact exchange")
    spec = json.loads((PACKAGE/"assets.lock.json").read_text())
    part = spec["files"][0]["parts"][0]
    request = Request(url.rstrip("/")+"/files/"+spec["exchange_namespace"]+"/"+part["name"],
                      method="HEAD",headers={"Authorization":"Bearer "+token})
    with urlopen(request,context=context,timeout=15) as response:
        if int(response.headers["Content-Length"]) != part["size_bytes"]:
            raise ValueError("asset download/authentication preflight failed")
    return {"status":"passed", "versions":versions, "devices":devices,
            "exchange":"healthy_authenticated_CA_verified", "large_assets":"hash_lock_present"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory",action="store_true")
    parser.add_argument("--gpu",default="auto")
    parser.add_argument("--data-dir",type=Path,default=ROOT/"data/relation/retacred")
    args = parser.parse_args()
    if args.inventory:
        for row in cohort()["models"]:
            print(f"{row['id']}: {row['status']}" + (" — " + row["reason"] if "reason" in row else ""))
        return
    print(json.dumps(check(args.data_dir,args.gpu),indent=2))


if __name__ == "__main__":
    main()
