"""Offline paper-table binding for MetaEntail primitives; no neural certification."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess

from .assets import reject_symlinks
from .common import digest
from .metaentail import NLI_LABELS, RELATIONS, TEMPLATES, nli_target, paper_spec, toy_report


def paper_tables(text):
    """Parse the two bounded pdftotext table regions, not infer missing cells."""
    lines = text.splitlines()
    start = next(i for i,line in enumerate(lines) if re.match(r"\s*ReTACRED\s+No relation\s+",line))
    stop = next(i for i in range(start+1,len(lines)) if re.match(r"\s*SemEval 2010\s+Other\s+",lines[i]))
    templates = {}
    for i,line in enumerate(lines[start:stop]):
        label, tail = line.strip().split("\u201c",1)
        if i==0:
            label = label.removeprefix("ReTACRED").strip()
        else:
            label = label.strip()
        value = tail.removesuffix("\u201d").replace("\u2019","'")
        value = re.sub(r"\b(subj|obj)\b",lambda match:"{subject}" if match[0]=="subj" else "{object}",value)
        if label.lower() in templates:
            raise ValueError("duplicate paper template label")
        templates[label.lower()] = value
    matrix, order = [], []
    for line in lines:
        match = re.match(r"\s*(No relation|org:[a-z/_ ]+|per:[a-z/_ ]+)\s+([012](?:\s+[012]){39})\s*$",line)
        if match:
            label = match[1].strip().lower()
            if label in order:
                raise ValueError("duplicate paper matrix row")
            order.append(label)
            matrix.append([int(value) for value in match[2].split()])
    if len(templates)!=40 or len(order)!=40 or set(templates)!=set(order):
        raise ValueError("required Table4 templates or Table13 cells missing")
    return templates,order,matrix


def verify_paper(cache):
    reject_symlinks(cache)
    path = cache/"paper.pdf"
    reject_symlinks(path)
    spec = paper_spec()
    if not path.is_file() or path.stat().st_size!=spec["paper"]["size_bytes"] or digest(path)!=spec["paper"]["sha256"]:
        raise ValueError("official MetaEntail paper hash/size mismatch")
    # Regenerate from the locked PDF so a changed cached .txt is never evidence.
    done = subprocess.run(["pdftotext","-layout",str(path),"-"],text=True,capture_output=True,check=True,timeout=30)
    templates,order,matrix = paper_tables(done.stdout)
    rows = spec["templates"]
    by_paper = {row["paper_label"].lower():row["id"] for row in rows}
    if set(by_paper)!=set(order):
        raise AssertionError("paper/canonical relation binding differs")
    for label,pattern in templates.items():
        if TEMPLATES[by_paper[label]]!=pattern:
            raise AssertionError("paper template mismatch: "+label)
    code = {"contradiction":0,"neutral":1,"entailment":2}
    expected = [[code[nli_target(by_paper[gold],by_paper[hypothesis])] for hypothesis in order] for gold in order]
    if expected!=matrix:
        raise AssertionError("translated meta-class rules differ from actual Table13 cells")
    counts = Counter(value for row in matrix for value in row)
    return {"status":"paper_tables_verified","paper_sha256":digest(path),"templates_checked":len(templates),
            "meta_class_cells_checked":1600,"paper_cell_codes":{"contradiction":counts[0],"neutral":counts[1],"entailment":counts[2]},
            "matrix_column_binding":"same_canonicalized_paper_relation_order_as_rows",
            "typography_adapted":spec["typographic_normalization"],"model_runtime_verified":False,"network_requests":0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-cache",type=Path,required=True)
    args = parser.parse_args(argv)
    result = {"paper_checks":verify_paper(args.paper_cache),"toy_checks":toy_report()["status"],
              "status":"paper_primitives_verified","model_implemented":False,"strict_comparison_eligible":False,
              "pending":paper_spec()["unresolved"]}
    print(json.dumps(result,indent=2,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
