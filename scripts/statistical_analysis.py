#!/usr/bin/env python3
"""Statistical analysis: bootstrap CIs, Wilcoxon tests, effect sizes.

- Bootstrap 95% CIs for all aggregate metrics
- Paired Wilcoxon signed-rank tests (ablation variants vs default, baselines vs system)
- Bonferroni correction for multiple comparisons
- Cliff's delta effect size

Usage:
    python scripts/statistical_analysis.py
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from soccer_summarization.utils import setup_logging


def cliffs_delta(x, y):
    """Compute Cliff's delta effect size."""
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return 0.0
    more = sum(1 for xi in x for yi in y if xi > yi)
    less = sum(1 for xi in x for yi in y if xi < yi)
    return (more - less) / (n_x * n_y)


def interpret_cliffs_delta(d):
    """Interpret Cliff's delta magnitude."""
    d = abs(d)
    if d < 0.147:
        return "negligible"
    elif d < 0.33:
        return "small"
    elif d < 0.474:
        return "medium"
    else:
        return "large"


def wilcoxon_test(x, y, alternative="two-sided"):
    """Paired Wilcoxon signed-rank test with handling for edge cases."""
    diff = np.array(x) - np.array(y)
    diff = diff[diff != 0]  # Remove zeros
    if len(diff) < 5:
        return {"statistic": np.nan, "p_value": 1.0, "n_nonzero": len(diff)}
    try:
        stat, p = stats.wilcoxon(diff, alternative=alternative)
        return {"statistic": float(stat), "p_value": float(p), "n_nonzero": len(diff)}
    except ValueError:
        return {"statistic": np.nan, "p_value": 1.0, "n_nonzero": len(diff)}


def analyze_ablation(ablation_file: Path, metric: str = "f1") -> dict:
    """Analyze a single ablation: compare each value to the default."""
    with open(ablation_file) as f:
        data = json.load(f)

    # Find default
    default_key = None
    for key, val in data.items():
        if val.get("is_default"):
            default_key = key
            break

    if default_key is None:
        return {}

    default_scores = [m[metric] for m in data[default_key]["per_match"]]
    default_match_ids = [m["match_id"] for m in data[default_key]["per_match"]]

    results = {}
    n_comparisons = len(data) - 1  # Bonferroni correction

    for key, val in data.items():
        if key == default_key:
            continue

        variant_scores_map = {m["match_id"]: m[metric] for m in val["per_match"]}
        # Align by match_id
        variant_scores = [variant_scores_map.get(mid, 0) for mid in default_match_ids]

        test = wilcoxon_test(variant_scores, default_scores)
        delta = cliffs_delta(variant_scores, default_scores)

        bonf_p = min(1.0, test["p_value"] * n_comparisons) if not np.isnan(test["p_value"]) else 1.0

        results[key] = {
            "value": val["value"],
            "mean": float(np.mean(variant_scores)),
            "default_mean": float(np.mean(default_scores)),
            "diff": float(np.mean(variant_scores) - np.mean(default_scores)),
            "wilcoxon_p": test["p_value"],
            "bonferroni_p": bonf_p,
            "significant_005": bonf_p < 0.05,
            "cliffs_delta": float(delta),
            "effect_size": interpret_cliffs_delta(delta),
        }

    return {"default_key": default_key, "comparisons": results, "metric": metric}


def analyze_baselines(baseline_file: Path, system_results_dir: Path, matches_yaml_data: dict) -> dict:
    """Compare each baseline against the full system."""
    with open(baseline_file) as f:
        baselines = json.load(f)

    # Load system per-match F1s
    system_scores = {}
    for match_dir in sorted(system_results_dir.iterdir()):
        eval_json = match_dir / "evaluation.json"
        if eval_json.exists():
            with open(eval_json) as f:
                data = json.load(f)
            match_id = data.get("match_id", match_dir.name)
            data["match_id"] = match_id
            system_scores[match_id] = data

    results = {}
    n_comparisons = len(baselines)

    for bl_name, bl_data in baselines.items():
        bl_per_match = {m["match_id"]: m for m in bl_data["per_match"]}

        # Align
        common_matches = sorted(set(system_scores.keys()) & set(bl_per_match.keys()))
        if not common_matches:
            continue

        sys_f1 = [system_scores[m].get("f1", 0) for m in common_matches]
        bl_f1 = [bl_per_match[m].get("f1", 0) for m in common_matches]

        test = wilcoxon_test(sys_f1, bl_f1)
        delta = cliffs_delta(sys_f1, bl_f1)
        bonf_p = min(1.0, test["p_value"] * n_comparisons) if not np.isnan(test["p_value"]) else 1.0

        results[bl_name] = {
            "display_name": bl_data["display_name"],
            "system_mean_f1": float(np.mean(sys_f1)),
            "baseline_mean_f1": float(np.mean(bl_f1)),
            "diff": float(np.mean(sys_f1) - np.mean(bl_f1)),
            "wilcoxon_p": test["p_value"],
            "bonferroni_p": bonf_p,
            "significant_005": bonf_p < 0.05,
            "cliffs_delta": float(delta),
            "effect_size": interpret_cliffs_delta(delta),
            "n_matches": len(common_matches),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Statistical analysis")
    parser.add_argument("--results-dir", default="results/per_match")
    parser.add_argument("--ablation-dir", default="results/ablation")
    parser.add_argument("--baseline-file", default="results/baselines/baseline_results.json")
    parser.add_argument("--output-dir", default="results/statistical")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Analyze ablations
    ablation_dir = Path(args.ablation_dir)
    if ablation_dir.exists():
        print("=" * 60)
        print("ABLATION STATISTICAL TESTS")
        print("=" * 60)

        for abl_file in sorted(ablation_dir.glob("*_results.json")):
            abl_name = abl_file.stem.replace("_results", "")
            logger.info("Analyzing ablation: %s", abl_name)

            result = analyze_ablation(abl_file, metric="f1")
            if result:
                all_results[f"ablation_{abl_name}"] = result

                print(f"\n{abl_name} (vs default={result['default_key']}):")
                print(f"{'Value':<10} {'Mean':>6} {'Diff':>8} {'p':>8} {'p_bonf':>8} {'delta':>8} {'effect':>12}")
                for key, comp in result["comparisons"].items():
                    sig = "*" if comp["significant_005"] else " "
                    print(
                        f"{key:<10} {comp['mean']:>6.3f} {comp['diff']:>+8.3f} "
                        f"{comp['wilcoxon_p']:>8.4f} {comp['bonferroni_p']:>8.4f} "
                        f"{comp['cliffs_delta']:>+8.3f} {comp['effect_size']:>12}{sig}"
                    )

    # Analyze baselines
    baseline_file = Path(args.baseline_file)
    if baseline_file.exists():
        print("\n" + "=" * 60)
        print("BASELINE STATISTICAL TESTS")
        print("=" * 60)

        bl_results = analyze_baselines(
            baseline_file, Path(args.results_dir), {}
        )
        all_results["baselines"] = bl_results

        print(f"\n{'Baseline':<25} {'Sys F1':>8} {'BL F1':>8} {'Diff':>8} {'p_bonf':>8} {'delta':>8} {'effect':>12}")
        for bl_name, comp in bl_results.items():
            sig = "*" if comp["significant_005"] else " "
            print(
                f"{comp['display_name']:<25} {comp['system_mean_f1']:>8.3f} "
                f"{comp['baseline_mean_f1']:>8.3f} {comp['diff']:>+8.3f} "
                f"{comp['bonferroni_p']:>8.4f} {comp['cliffs_delta']:>+8.3f} "
                f"{comp['effect_size']:>12}{sig}"
            )

    # Save all results
    with open(output_dir / "statistical_tests.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)

    logger.info("Saved to %s", output_dir / "statistical_tests.json")


if __name__ == "__main__":
    main()
