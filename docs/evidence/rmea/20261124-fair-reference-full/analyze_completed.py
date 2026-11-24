"""Recompute all frozen endpoints from the compact completed report only.

No training, inference, checkpoint loading, cloud writes or grading promotion.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

MODES = ("uniform", "quantum", "dephased", "softmax12", "real12")
FILES = ("summary.json", "manifest.json", "analysis-readiness.json", "receipt.json",
         "run_summary.json", "data.sha256", "data_counts.txt")


def analyze(folder):
    folder = Path(folder)
    assert {p.name for p in folder.iterdir()} == set(FILES), "Seven-file report required"
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    ready = json.loads((folder / "analysis-readiness.json").read_text(encoding="utf-8"))
    assert summary["seed"] == 13 and summary["full_data_training"] is True
    assert summary["source_sha256"] == "584f61837ed746795c0d027d459b6cf050e1b55045f0ea9c04ae64b57981296a"
    assert summary["config_sha256"] == "e4b2ca0c1dd8b80bac9d560ab174e6b825d54bea3cda6a3e27fb86fb996558b1"
    assert set(summary["results"]) == set(MODES)
    assert ready["trace_materialized"] is False
    rows = {}
    max_error = 0.0
    for mode in MODES:
        result = summary["results"][mode]
        assert result["updates"] == 9140 and 1 <= result["selected_epoch"] <= 20
        assert result["evaluation_checkpoint"]["epoch"] == result["selected_epoch"]
        assert ready["checkpoints"][mode]["sha256"] == result["evaluation_checkpoint"]["sha256"]
        row = {"selected_epoch": result["selected_epoch"], "updates": result["updates"]}
        for split, count in (("valid", 19584), ("test", 13418)):
            reported = result[split]
            cm = reported["confusion_matrix"]
            assert len(cm) == 40 and all(len(r) == 40 for r in cm)
            assert all(type(n) is int and n >= 0 for r in cm for n in r)
            assert sum(map(sum, cm)) == reported["items"] == count
            assert len(reported["per_class"]) == 40
            assert reported["per_class"][0]["label"] == "no_relation"
            f1s = []
            for i in range(40):
                support = sum(cm[i])
                predicted = sum(r[i] for r in cm)
                f1 = 2 * cm[i][i] / (support + predicted) if support + predicted else 0.0
                f1s.append(f1)
                assert reported["per_class"][i]["support"] == support
                max_error = max(max_error, abs(f1 - reported["per_class"][i]["f1"]))
            macro = sum(f1s) / 40
            positive_macro = sum(f1s[1:]) / 39
            tp = sum(cm[i][i] for i in range(1, 40))
            gold_positive = sum(sum(r) for r in cm[1:])
            predicted_positive = sum(sum(r[1:]) for r in cm)
            micro = 2 * tp / (gold_positive + predicted_positive)
            for actual, expected in ((macro, reported["macro_f1"]),
                                     (positive_macro, reported["positive39_macro_f1"]),
                                     (micro, reported["micro_f1"])):
                max_error = max(max_error, abs(actual - expected))
            row[split] = {"macro_f1_all40": macro, "macro_f1_positive39": positive_macro,
                          "micro_f1_positive39": micro, "ce": reported["loss"],
                          "positive_to_null": sum(cm[i][0] for i in range(1, 40)),
                          "null_to_positive": sum(cm[0][1:])}
        rows[mode] = row
    assert max_error <= 1e-12, "Reported metrics differ from confusion matrices"
    contrasts = {mode: rows["quantum"]["test"]["macro_f1_all40"] - rows[mode]["test"]["macro_f1_all40"]
                 for mode in MODES if mode != "quantum"}
    for mode, value in contrasts.items():
        assert abs(value - summary["contrasts"][mode]["test_all40_macro_delta"]) <= 1e-12
    passed = all(value > 0 for value in contrasts.values())
    assert passed is summary["qualification_gate"]
    return {"at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "scope": "Completed seed13 frozen full-budget five-condition qualification; descriptive only",
            "report_sha256": {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in FILES},
            "rows": rows, "quantum_test_all40_deltas": contrasts,
            "qualification_gate": passed, "max_metric_recomputation_error": max_error,
            "disposition": "ELIGIBLE_FOR_SEPARATELY_FROZEN_REPLICATION" if passed else "CLOSE_EXACT_QUALIFICATION_PROFILE",
            "L3": "NOT_ACCEPTED", "L4-T": "NOT_ACCEPTED",
            "limitations": ["Single seed on a previously exposed benchmark; not independent replication.",
                            "Registered parameter matching is not equality of effective capacity.",
                            "All endpoints retained; no seed, budget, metric or checkpoint rescue."]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    record = analyze(args.report_dir)
    args.out.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
