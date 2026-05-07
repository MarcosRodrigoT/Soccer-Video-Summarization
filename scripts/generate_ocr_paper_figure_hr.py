#!/usr/bin/env python3
"""Generate OCR paper figure for Match25 using high-resolution (1080p) video.

Temporarily scales the scoreboard region from 480p to 1080p coordinates,
generates the figure using the HR video with scaled OCR annotations,
then restores the original region file.
"""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from soccer_summarization.config import PipelineConfig
from generate_ocr_paper_figure import generate_paper_figure
from visualize_ocr_diagnostics import load_match_info

MATCH_ID = "Match25"
HR_VIDEO = "/tmp/Match25_hr.mp4"
OUTPUT_DIR = Path("results/ocr_paper_figures")

# Resolution scaling: 854x480 -> 1920x1080
SCALE_X = 1920 / 854
SCALE_Y = 1080 / 480
# Text/line scale factor (proportional to resolution increase)
TEXT_SCALE = (SCALE_X + SCALE_Y) / 2  # ~2.25


def main():
    config = PipelineConfig.from_yaml("config/default.yaml")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load match info and override video path
    info = load_match_info(MATCH_ID, config)
    info["video_path"] = HR_VIDEO

    # Scale scoreboard region for 1080p
    region_path = Path(f"results/per_match/{MATCH_ID}/scoreboard_region.json")
    backup_path = region_path.with_suffix(".json.bak")

    with open(region_path) as f:
        original_region = json.load(f)

    scaled_region = {
        "x": round(original_region["x"] * SCALE_X),
        "y": round(original_region["y"] * SCALE_Y),
        "w": round(original_region["w"] * SCALE_X),
        "h": round(original_region["h"] * SCALE_Y),
    }
    print(f"Original region: {original_region}")
    print(f"Scaled region:   {scaled_region}")

    # Backup and write scaled region
    shutil.copy2(region_path, backup_path)
    try:
        with open(region_path, "w") as f:
            json.dump(scaled_region, f)

        generate_paper_figure(MATCH_ID, info, config, OUTPUT_DIR,
                              text_scale=TEXT_SCALE, output_suffix="_hr")
    finally:
        # Restore original region
        shutil.move(str(backup_path), str(region_path))
        print("Restored original scoreboard_region.json")


if __name__ == "__main__":
    main()
