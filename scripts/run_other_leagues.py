#!/usr/bin/env python3
"""Run the unchanged pipeline on the cross-league matches (reviewer R1.5).

Same configuration and parameters as the LaLiga corpus (default_hr_10s.yaml);
no evaluation against frame-level ground truth (there is none for these
matches): results are reviewed with scripts/annotations/prepare_goals_task.py.

    python scripts/run_other_leagues.py --matches PL1 SA1 BL1 L1
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_single_match import run_pipeline  # noqa: E402
from soccer_summarization.config import PipelineConfig  # noqa: E402
from soccer_summarization.utils import setup_logging  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", nargs="+", required=True)
    ap.add_argument("--config", default="config/default_hr_10s.yaml")
    ap.add_argument("--matches-yaml", default="config/matches_other_leagues.yaml")
    ap.add_argument("--out-root", default="results_other_leagues/per_match")
    args = ap.parse_args()
    setup_logging("INFO")
    cfg = PipelineConfig.from_yaml(args.config)
    info_all = yaml.safe_load(open(args.matches_yaml))["matches"]
    for match_id in args.matches:
        info = info_all[match_id]
        out = Path(args.out_root) / match_id
        out.mkdir(parents=True, exist_ok=True)
        video = next((Path(cfg.data_dir) / info["video_dir"]).glob("*.mp4"))
        dummy_gt = out / "no_ground_truth.csv"  # evaluation stage needs a file; metrics are ignored
        dummy_gt.write_text("start,end,text,GoalGT1,GoalGT2\n")
        run_pipeline(video_path=str(video), gt_path=str(dummy_gt), output_dir=str(out), config=cfg,
                     match_id=match_id, skip_video_export=True, goals_path=None)


if __name__ == "__main__":
    main()
