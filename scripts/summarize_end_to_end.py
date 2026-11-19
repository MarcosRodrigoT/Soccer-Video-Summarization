#!/usr/bin/env python3
"""Single source for the end-to-end numbers reported in the paper.

Writes results_hr_10s/aggregate/end_to_end_summary.json with, for the
primary (Delta_tail = 10 s, results_hr_10s) and full-shot (results_hr)
protocols: corpus summary, bootstrap CIs (10^4 resamples of the 25
matches) of the mean per-match F1 and IoU, and the per-match rows of the
per-match table.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from soccer_summarization.metrics_summary import (  # noqa: E402
    bootstrap_ci, per_match_tp_mean, summarize,
)


def load(results_dir: str) -> list[dict]:
    files = sorted(Path(results_dir, "per_match").glob("*/evaluation.json"),
                   key=lambda p: int(p.parent.name.replace("Match", "")))
    return [json.load(open(f)) for f in files]


def main():
    out = {}
    for name, d in (("tail_10s", "results_hr_10s"), ("full_shot", "results_hr")):
        evals = load(d)
        s = summarize(evals)
        s["total_f1"] = 2 * s["tp"] / (2 * s["tp"] + s["fp"] + s["fn"])
        s["f1_ci"] = bootstrap_ci([e["clip_aware_f1"] for e in evals])
        s["iou_ci"] = bootstrap_ci([per_match_tp_mean(e, "clip_iou_per_gt") for e in evals])
        s["per_match"] = [{
            "match": e["match_id"], "gt": e["num_gt_events"],
            "tp": e["clip_aware_tp"], "fp": e["clip_aware_fp"], "fn": e["clip_aware_fn"],
            "f1": e["clip_aware_f1"],
            "iou": per_match_tp_mean(e, "clip_iou_per_gt"),
            "mae_start": per_match_tp_mean(e, "clip_start_mae_per_gt"),
            "mae_end": per_match_tp_mean(e, "clip_end_mae_per_gt"),
        } for e in evals]
        out[name] = s
    path = Path("results_hr_10s/aggregate/end_to_end_summary.json")
    json.dump(out, open(path, "w"), indent=2)
    p = out["tail_10s"]
    print(f"10 s: {p['tp']}/{p['fp']}/{p['fn']} F1 {p['f1_mean']:.4f}±{p['f1_std']:.4f} "
          f"CI [{p['f1_ci']['lower']:.4f}, {p['f1_ci']['upper']:.4f}] "
          f"IoU CI [{p['iou_ci']['lower']:.4f}, {p['iou_ci']['upper']:.4f}] total F1 {p['total_f1']:.4f}")
    print("wrote", path)


if __name__ == "__main__":
    main()
