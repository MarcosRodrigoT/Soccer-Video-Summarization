#!/usr/bin/env python3
"""Check that sequential decoding reproduces the cached player counts.

The per-match results in results_hr/ were produced with the original
detector loop, which seeked to every sampled frame. This script re-runs only
the detector with sequential decoding and compares its per-frame people
counts with the cached ones.

Usage:
    python scripts/verify_sequential_decode.py --matches Match2 Match3 ... \\
        --output results_hr/runtime_analysis_rev/sequential_check/<name>.json
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_single_match import load_match_info  # noqa: E402
from soccer_summarization.config import PipelineConfig  # noqa: E402
from soccer_summarization.object_detection import PlayerDetector  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default_hr.yaml")
    ap.add_argument("--matches", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg = PipelineConfig.from_yaml(args.config)
    cfg.object_detection.sequential_decode = True
    detector = PlayerDetector(cfg.object_detection)
    out = {}
    for match_id in args.matches:
        info = load_match_info(match_id, cfg, matches_yaml=str(Path(args.config).parent / "matches_hr.yaml"))
        new = detector.run(info["video_path"])
        ref = pd.read_csv(Path(cfg.results_dir) / "per_match" / match_id / "player_counts.csv")
        m = new.merge(ref, on="frame", suffixes=("", "_ref"), how="outer", indicator=True)
        both = m[m["_merge"] == "both"]
        diff = (both["player_count"] - both["player_count_ref"]).abs()
        out[match_id] = {
            "frames_new": int(len(new)), "frames_ref": int(len(ref)),
            "frames_compared": int(len(both)),
            "identical_fraction": float((diff == 0).mean()),
            "max_abs_diff": int(diff.max()),
            "detection_sec": detector.last_timing,
        }
        print(match_id, out[match_id], flush=True)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.output, "w"), indent=2)


if __name__ == "__main__":
    main()
