#!/usr/bin/env python3
"""Batch-process all matches defined in matches.yaml.

Usage:
    python scripts/run_all_matches.py [--config PATH] [--matches Match1,Match5]
"""

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from soccer_summarization.config import PipelineConfig
from soccer_summarization.evaluation import Evaluator, EvaluationResult
from soccer_summarization.utils import setup_logging
from scripts.run_single_match import load_match_info, run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Process all matches")
    parser.add_argument("--config", default="config/default.yaml", help="Pipeline config YAML")
    parser.add_argument("--matches", help="Comma-separated match IDs to process (default: all)")
    parser.add_argument("--matches-yaml", default=None, help="Path to matches YAML (default: config/matches.yaml)")
    parser.add_argument("--skip-ocr", action="store_true", help="Skip OCR for all matches")
    parser.add_argument("--skip-video-export", action="store_true", help="Skip video export")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    config = PipelineConfig.from_yaml(args.config)

    # Load match list
    matches_yaml = args.matches_yaml or (Path(__file__).parent.parent / "config" / "matches.yaml")
    with open(matches_yaml) as f:
        all_matches = yaml.safe_load(f)["matches"]

    if args.matches:
        match_ids = [m.strip() for m in args.matches.split(",")]
    else:
        match_ids = list(all_matches.keys())

    logger.info("Processing %d matches: %s", len(match_ids), ", ".join(match_ids))

    results = []
    errors = {}

    for match_id in match_ids:
        logger.info("=" * 60)
        logger.info("Processing %s", match_id)
        logger.info("=" * 60)

        try:
            info = load_match_info(match_id, config, matches_yaml=matches_yaml)
            output_dir = f"{config.results_dir}/per_match/{match_id}"

            result = run_pipeline(
                video_path=info["video_path"],
                gt_path=info["gt_path"],
                output_dir=output_dir,
                config=config,
                match_id=match_id,
                skip_ocr=args.skip_ocr,
                skip_video_export=args.skip_video_export,
            )
            results.append(result)

        except Exception as e:
            logger.error("FAILED %s: %s", match_id, e)
            traceback.print_exc()
            errors[match_id] = str(e)

    # Aggregate results
    if results:
        evaluator = Evaluator(config.evaluation)
        aggregate = evaluator.aggregate_results(results)

        agg_dir = Path(config.results_dir) / "aggregate"
        agg_dir.mkdir(parents=True, exist_ok=True)

        with open(agg_dir / "aggregate_metrics.json", "w") as f:
            json.dump(aggregate, f, indent=2, default=float)

        logger.info("=" * 60)
        logger.info("AGGREGATE RESULTS (%d matches)", len(results))
        logger.info("=" * 60)
        logger.info("Goal Detection (macro): P=%.3f R=%.3f F1=%.3f",
                     aggregate["precision_mean"], aggregate["recall_mean"], aggregate["f1_mean"])
        logger.info("Goal Detection (micro): P=%.3f R=%.3f F1=%.3f",
                     aggregate["micro_precision"], aggregate["micro_recall"], aggregate["micro_f1"])
        logger.info("Compression: %.1f%% +/- %.1f%%",
                     aggregate["compression_mean"] * 100, aggregate["compression_std"] * 100)
        logger.info("Goal Coverage: %.1f%% +/- %.1f%%",
                     aggregate["goal_coverage_mean"] * 100, aggregate["goal_coverage_std"] * 100)

    if errors:
        logger.warning("Failed matches: %s", list(errors.keys()))

    logger.info("Done. %d succeeded, %d failed.", len(results), len(errors))


if __name__ == "__main__":
    main()
