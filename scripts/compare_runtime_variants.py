#!/usr/bin/env python3
"""Compare detector inference setups on Match1 (reviewer comment R1.4).

Reads the runtime_substages.json of each variant and writes
results_hr/runtime_analysis_rev/variants_summary.json with, per variant:
detection-loop time split into decoding and inference, total pipeline time,
real-time factor, agreement of the per-frame people counts with the FP32
reference, and the Match1 end-to-end result.

Variants:
  seek_fp32       original submission (seek to every sampled frame), June 2026
  sequential_fp32 sequential decoding, PyTorch FP32 (default configuration)
  sequential_fp16 sequential decoding, PyTorch FP16
  tensorrt_fp16   sequential decoding, TensorRT FP16 engine (384x640 input)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("results_hr")
VARIANTS = {
    "seek_fp32": BASE / "runtime_analysis" / "Match1",
    "sequential_fp32": BASE / "runtime_analysis_rev" / "Match1_sequential_fp32",
    "sequential_fp16": BASE / "runtime_analysis_rev" / "Match1_sequential_fp16",
    "tensorrt_fp16": BASE / "runtime_analysis_rev" / "Match1_sequential_tensorrt_fp16",
}
REFERENCE = BASE / "per_match" / "Match1" / "player_counts.csv"  # counts used in the paper


def main():
    ref = pd.read_csv(REFERENCE)
    out = {}
    for name, d in VARIANTS.items():
        data = json.load(open(d / "runtime_substages.json"))
        sub = {(r["stage"], r["substage"]): r["duration_sec"] for r in data["substages"]}
        n_sampled = int(np.ceil(data["total_frames"] / data["yolo_frame_step"]))
        if ("object_detection", "inference_loop") in sub:  # June run: not split
            loop = sub[("object_detection", "inference_loop")]
            decode = inference = None
        else:
            decode = sub[("object_detection", "frame_decode")]
            inference = sub[("object_detection", "yolo_inference")]
            loop = decode + inference
        rec = {
            "detection_loop_sec": loop,
            "decode_sec": decode,
            "inference_sec": inference,
            "inference_ms_per_sampled_frame": None if inference is None else 1000 * inference / n_sampled,
            "loop_ms_per_sampled_frame": 1000 * loop / n_sampled,
            "total_sec": data["total_compute_sec"],
            "realtime_factor": data["realtime_factor"],
            "speedup_vs_realtime": 1 / data["realtime_factor"],
            "detection_share": loop / data["total_compute_sec"],
            "f1_ocr": data["f1"],
            "f1_end_to_end": data["clip_aware_f1"],
            "num_shots": data["num_shots"],
            "num_score_events": data["num_score_events"],
            "num_play_long_shots": data["num_play_long_shots"],
            "timestamp": data["timestamp"],
        }
        counts = d / "player_counts.csv"
        if counts.exists():
            m = pd.read_csv(counts).merge(ref, on="frame", suffixes=("", "_ref"))
            diff = (m["player_count"] - m["player_count_ref"]).abs()
            rec["count_frames_compared"] = int(len(m))
            rec["count_identical_fraction"] = float((diff == 0).mean())
            rec["count_mean_abs_diff"] = float(diff.mean())
            rec["count_max_abs_diff"] = int(diff.max())
        out[name] = rec
        print(f"{name:16s} loop {loop:7.1f}s  total {rec['total_sec']:7.1f}s  RTF {rec['realtime_factor']:.3f}  "
              f"F1 {rec['f1_end_to_end']:.2f}  identical counts "
              f"{rec.get('count_identical_fraction', float('nan')):.4f}")
    path = BASE / "runtime_analysis_rev" / "variants_summary.json"
    json.dump(out, open(path, "w"), indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
