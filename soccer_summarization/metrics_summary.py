"""Corpus-level summary of per-match evaluation.json files, as reported in
the paper (headline table and per-protocol comparison).

* TP/FP/FN: end-to-end counts (OCR timestamp within tolerance AND clip IoU
  >= 0.5), summed over matches.
* F1: mean and sample standard deviation of the per-match end-to-end F1.
* IoU / MAE: averaged within each match over its end-to-end true positives,
  then averaged across matches.
* Compression: mean of the per-match compression ratio.
"""

import statistics as st


def summarize(evals: list[dict]) -> dict:
    f1 = [e["clip_aware_f1"] for e in evals]
    iou, mae_s, mae_e = [], [], []
    for e in evals:
        tp = [i for i, x in enumerate(e["clip_iou_per_gt"]) if x is not None and x >= 0.5]
        if tp:
            iou.append(st.mean(e["clip_iou_per_gt"][i] for i in tp))
            mae_s.append(st.mean(e["clip_start_mae_per_gt"][i] for i in tp))
            mae_e.append(st.mean(e["clip_end_mae_per_gt"][i] for i in tp))
    return {
        "num_matches": len(evals),
        "tp": sum(e["clip_aware_tp"] for e in evals),
        "fp": sum(e["clip_aware_fp"] for e in evals),
        "fn": sum(e["clip_aware_fn"] for e in evals),
        "f1_mean": st.mean(f1),
        "f1_std": st.stdev(f1),
        "num_perfect_matches": sum(x == 1.0 for x in f1),
        "iou_mean": st.mean(iou),
        "mae_start_mean": st.mean(mae_s),
        "mae_end_mean": st.mean(mae_e),
        "compression_mean": st.mean(e["compression_ratio"] for e in evals),
        "ocr_tp": sum(e["true_positives"] for e in evals),
        "ocr_fp": sum(e["false_positives"] for e in evals),
        "ocr_fn": sum(e["false_negatives"] for e in evals),
    }


def per_match_tp_mean(e: dict, key: str):
    """Mean of a per-GT metric over the end-to-end true positives of a match."""
    tp = [i for i, x in enumerate(e["clip_iou_per_gt"]) if x is not None and x >= 0.5]
    return st.mean(e[key][i] for i in tp) if tp else None


def bootstrap_ci(values, n_resamples: int = 10_000, seed: int = 0, level: float = 0.95):
    """Percentile bootstrap CI of the mean, resampling matches with replacement."""
    import numpy as np

    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n_resamples, len(v)))].mean(axis=1)
    a = (1 - level) / 2 * 100
    lo, hi = np.percentile(means, [a, 100 - a])
    return {"lower": float(lo), "upper": float(hi), "n_resamples": n_resamples, "seed": seed}
