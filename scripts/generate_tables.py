#!/usr/bin/env python3
"""Generate LaTeX-ready result tables from evaluation outputs.

Usage:
    python scripts/generate_tables.py [--results-dir results/per_match] [--output-dir results/tables]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from soccer_summarization.utils import setup_logging


def load_all_results(results_dir: Path, matches_yaml: Path) -> list[dict]:
    """Load evaluation.json from all processed matches."""
    with open(matches_yaml) as f:
        all_matches = yaml.safe_load(f)["matches"]

    results = []
    for match_dir in sorted(results_dir.iterdir()):
        if not match_dir.is_dir():
            continue
        eval_file = match_dir / "evaluation.json"
        if eval_file.exists():
            with open(eval_file) as f:
                data = json.load(f)
            # Add match metadata
            match_id = match_dir.name
            if match_id in all_matches:
                data["home_team"] = all_matches[match_id].get("home_team", "")
                data["away_team"] = all_matches[match_id].get("away_team", "")
                data["score"] = all_matches[match_id].get("score", "")
                data["season"] = all_matches[match_id].get("season", "")
            results.append(data)

    return results


def generate_per_match_table(results: list[dict]) -> str:
    """Generate per-match results table in LaTeX."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Per-match goal detection and highlight quality results.}",
        r"\label{tab:per_match}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llccccccc}",
        r"\toprule",
        r"Match & Score & GT Goals & Det. Goals & P & R & F1 & Comp. (\%) & Cov. (\%) \\",
        r"\midrule",
    ]

    for r in results:
        match_id = r.get("match_id", "?")
        teams = f"{r.get('home_team', '')} vs {r.get('away_team', '')}"
        score = r.get("score", "?")
        gt_goals = r.get("num_gt_events", 0)
        pred_goals = r.get("num_predicted_goals", 0)
        p = r.get("precision", 0)
        rec = r.get("recall", 0)
        f1 = r.get("f1", 0)
        comp = r.get("compression_ratio", 0) * 100
        cov = r.get("goal_coverage", 0) * 100

        lines.append(
            f"{match_id} & {score} & {gt_goals} & {pred_goals} & "
            f"{p:.2f} & {rec:.2f} & {f1:.2f} & {comp:.1f} & {cov:.1f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def generate_aggregate_table(agg_path: Path) -> str:
    """Generate aggregate results table in LaTeX."""
    with open(agg_path) as f:
        agg = json.load(f)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Aggregate goal detection and highlight quality metrics.}",
        r"\label{tab:aggregate}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Metric & Mean & Std \\",
        r"\midrule",
        f"Precision (macro) & {agg['precision_mean']:.3f} & {agg['precision_std']:.3f} \\\\",
        f"Recall (macro) & {agg['recall_mean']:.3f} & {agg['recall_std']:.3f} \\\\",
        f"F1 (macro) & {agg['f1_mean']:.3f} & {agg['f1_std']:.3f} \\\\",
        r"\midrule",
        f"Precision (micro) & {agg['micro_precision']:.3f} & --- \\\\",
        f"Recall (micro) & {agg['micro_recall']:.3f} & --- \\\\",
        f"F1 (micro) & {agg['micro_f1']:.3f} & --- \\\\",
        r"\midrule",
        f"Compression (\\%) & {agg['compression_mean']*100:.1f} & {agg['compression_std']*100:.1f} \\\\",
        f"Goal Coverage (\\%) & {agg['goal_coverage_mean']*100:.1f} & {agg['goal_coverage_std']*100:.1f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX result tables")
    parser.add_argument("--results-dir", default="results/per_match")
    parser.add_argument("--output-dir", default="results/tables")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matches_yaml = Path(__file__).parent.parent / "config" / "matches.yaml"

    # Per-match table
    results = load_all_results(results_dir, matches_yaml)
    if not results:
        logger.error("No evaluation results found in %s", results_dir)
        return

    per_match_tex = generate_per_match_table(results)
    with open(output_dir / "per_match_table.tex", "w") as f:
        f.write(per_match_tex)
    logger.info("Per-match table: %s (%d matches)", output_dir / "per_match_table.tex", len(results))
    print(per_match_tex)

    # Aggregate table
    agg_path = Path("results/aggregate/aggregate_metrics.json")
    if agg_path.exists():
        agg_tex = generate_aggregate_table(agg_path)
        with open(output_dir / "aggregate_table.tex", "w") as f:
            f.write(agg_tex)
        logger.info("Aggregate table: %s", output_dir / "aggregate_table.tex")
        print("\n")
        print(agg_tex)
    else:
        logger.warning("No aggregate metrics found at %s", agg_path)


if __name__ == "__main__":
    main()
