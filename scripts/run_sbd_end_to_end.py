#!/usr/bin/env python3
"""End-to-end results with alternative shot boundary detectors (reviewer R1.2).

For each detector in results_sbd/<method>/per_match/<match>/shot_boundaries.csv,
the shot list of the pipeline is replaced by the detector's one and only the
assembly and evaluation stages are re-run; people counts and goal events are
the cached ones of results_hr/per_match. Both protocols are evaluated
(Delta_tail = 10 s and full shot). The method "ours" uses the cached shot list
and must reproduce the paper exactly; the script checks it.

    python scripts/run_sbd_end_to_end.py --methods ours transnetv2 pyscenedetect
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_single_match import load_match_info, run_pipeline  # noqa: E402
from soccer_summarization.config import PipelineConfig  # noqa: E402
from soccer_summarization.metrics_summary import bootstrap_ci, summarize  # noqa: E402

CACHED = ["player_counts.csv", "frame_classification.csv", "score_events.csv", "scoreboard_region.json"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["ours", "transnetv2", "pyscenedetect"])
    ap.add_argument("--config", default="config/default_hr_10s.yaml")
    ap.add_argument("--output", default="results_sbd/end_to_end_summary.json")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    matches_yaml = str(Path(args.config).parent / "matches_hr.yaml")
    match_ids = sorted(yaml.safe_load(open(matches_yaml))["matches"], key=lambda m: int(m[5:]))
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for method in args.methods:
            shots_dir = Path("results_hr/per_match") if method == "ours" else Path("results_sbd") / method / "per_match"
            missing = [m for m in match_ids if not (shots_dir / m / "shot_boundaries.csv").exists()]
            if missing:
                print(f"{method}: skipped, missing {missing}")
                continue
            out[method] = {}
            for tail, name in ((10.0, "tail_10s"), (None, "full_shot")):
                cfg = PipelineConfig.from_yaml(args.config)
                cfg.highlight_assembler.goal_clip_max_duration = tail
                evals = []
                for m in match_ids:
                    d = Path(tmp) / method / name / m
                    d.mkdir(parents=True)
                    for f in CACHED:
                        src = Path("results_hr/per_match") / m / f
                        if src.exists():
                            os.link(src, d / f)
                    os.link(shots_dir / m / "shot_boundaries.csv", d / "shot_boundaries.csv")
                    info = load_match_info(m, cfg, matches_yaml=matches_yaml)
                    run_pipeline(video_path=info["video_path"], gt_path=info["gt_path"], output_dir=str(d),
                                 config=cfg, match_id=m, skip_video_export=True, goals_path=info.get("goals_path"))
                    evals.append(json.load(open(d / "evaluation.json")))
                s = summarize(evals)
                s["f1_ci"] = bootstrap_ci([e["clip_aware_f1"] for e in evals])
                s["per_match_f1"] = {e["match_id"]: e["clip_aware_f1"] for e in evals}
                s["num_shots_total"] = sum(
                    sum(1 for _ in open(shots_dir / m / "shot_boundaries.csv")) - 1 for m in match_ids)
                out[method][name] = s
                print(f"{method:14s} {name:9s} TP/FP/FN {s['tp']}/{s['fp']}/{s['fn']}  "
                      f"F1 {s['f1_mean']:.4f}±{s['f1_std']:.4f}  IoU {s['iou_mean']:.4f}  "
                      f"MAE {s['mae_start_mean']:.3f}/{s['mae_end_mean']:.3f}  shots {s['num_shots_total']}",
                      flush=True)
    if "ours" in out:  # must reproduce the paper
        ref = json.load(open("results_hr_10s/aggregate/end_to_end_summary.json"))
        for name in ("tail_10s", "full_shot"):
            for k in ("tp", "fp", "fn", "f1_mean", "iou_mean", "mae_start_mean"):
                assert abs(out["ours"][name][k] - ref[name][k]) < 1e-9, (name, k)
        print("ours reproduces the paper exactly")
    prev = json.load(open(args.output)) if Path(args.output).exists() else {}
    prev.update(out)
    json.dump(prev, open(args.output, "w"), indent=2)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
