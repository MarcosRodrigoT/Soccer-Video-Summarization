"""Render every figure under scripts/figures/ into docs/figures/ as SVG + PDF.

Usage:
    python scripts/figures/render_all.py
"""
from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _style import save_figure  # noqa: E402

FIGURES = [
    ("fig01_system_architecture",            "system_architecture"),
    ("fig01_system_architecture_horizontal", "system_architecture_horizontal"),
    ("fig02_data_flow",                      "data_flow"),
    ("fig03_shot_segmentation_detail",   "shot_segmentation_detail"),
    ("fig04_player_detection_detail",    "player_detection_detail"),
    ("fig05_play_classification_detail", "play_classification_detail"),
    ("fig06_ocr_scoreboard_pipeline",    "ocr_scoreboard_pipeline"),
    ("fig07_goal_event_state_machine",   "goal_event_state_machine"),
    ("fig08_highlight_assembly_detail",  "highlight_assembly_detail"),
    ("fig09_evaluation_framework",       "evaluation_framework"),
    ("fig10_dataset_overview",           "dataset_overview_v2"),
    ("fig11_experiment_suite",           "experiment_suite"),
    ("fig12_qualitative_scoreboard",     "qualitative_scoreboard"),
    ("fig13_qualitative_player_detection","qualitative_player_detection"),
    ("fig14_qualitative_timeline",       "qualitative_timeline"),
    ("fig15_error_taxonomy",             "error_taxonomy"),
]


def main() -> int:
    failures = []
    for module_name, output_stem in FIGURES:
        t0 = time.perf_counter()
        try:
            mod = importlib.import_module(module_name)
            fig = mod.build_figure()
            save_figure(fig, output_stem)
            dt = time.perf_counter() - t0
            print(f"  [{dt:5.1f}s]  {module_name}")
        except Exception as e:  # noqa: BLE001
            print(f"  [ FAIL ]  {module_name}: {e}")
            failures.append((module_name, e))
    print()
    print(f"Done. {len(FIGURES) - len(failures)}/{len(FIGURES)} succeeded.")
    if failures:
        print("Failures:")
        for name, err in failures:
            print(f"  {name}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
