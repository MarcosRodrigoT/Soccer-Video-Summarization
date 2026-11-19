"""Figure 12 — Qualitative scoreboard pipeline on Match1 (rev. Jun 2026).

Compact horizontal layout: the three localization steps are arranged
side by side rather than stacked vertically, and the rasterized
broadcast-frame / IQR-map / mask images are rendered at high dpi so
that small overlay details remain legible in the final PDF.

  (a) broadcast frame with the auto-localized bounding box overlaid,
  (b) per-pixel IQR map used as the temporal-stability statistic,
  (c) binary mask after the morphological cascade, with the selected
      scoreboard region highlighted.

The colorbar of panel (b) is docked on its right side as before; (a)
and (c) reserve the same invisible right gutter so the three image
axes share an identical visible width.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from _style import FIGURES_DIR, set_paper_style

PROJECT = Path(__file__).resolve().parents[2]
MATCH_DIR = PROJECT / "results_hr" / "per_match" / "Match1"
VIDEO = Path("/mnt/Data-fast/mrt/lf/SoccerNetClip10Videos/Laliga_HR/Match1/"
             "Valencia CF - FC Barcelona (2-3) LALIGA 2013⧸2014 FULL MATCH "
             "[Qzrg3QDmQ4k].mp4")

PANEL_TITLE = dict(loc="left", fontsize=8.5, fontweight="bold")
BBOX_COLOR = "#FF1744"
SAVE_DPI = 300


def _read_frame(idx: int = 12000) -> np.ndarray | None:
    if not VIDEO.exists():
        return None
    cap = cv2.VideoCapture(str(VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)


def _binary_mask(iqr: np.ndarray) -> np.ndarray:
    thr = np.percentile(iqr, 8.0)
    raw = (iqr < thr).astype(np.uint8) * 255
    closed = cv2.morphologyEx(raw, cv2.MORPH_CLOSE,
                              np.ones((51, 21), np.uint8))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN,
                              np.ones((9, 9), np.uint8))
    return opened


def build_figure():
    set_paper_style()

    with open(MATCH_DIR / "scoreboard_region.json") as f:
        bbox = json.load(f)
    bx, by = int(bbox["x"]), int(bbox["y"])
    bw, bh = int(bbox["w"]), int(bbox["h"])

    iqr = np.load(MATCH_DIR / "iqr_map.npy")
    mask = _binary_mask(iqr)
    frame = _read_frame()

    # 1 row x 3 columns; equal panel widths and a single colorbar to the
    # right of panel (b). Panels (a) and (c) reserve an invisible right
    # gutter of the same width so the three image axes line up.
    fig = plt.figure(figsize=(10.6, 2.55))
    gs = fig.add_gridspec(1, 3, wspace=0.10)

    CBAR_SIZE = "3.0%"
    CBAR_PAD = 0.04

    def _reserve_cbar_space(ax):
        cax_dummy = make_axes_locatable(ax).append_axes(
            "right", size=CBAR_SIZE, pad=CBAR_PAD)
        cax_dummy.axis("off")

    # ---- Panel (a): broadcast frame + bbox ---------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    if frame is not None:
        ax_a.imshow(frame, interpolation="bilinear")
        ax_a.add_patch(Rectangle((bx, by), bw, bh, fill=False,
                                 edgecolor=BBOX_COLOR, linewidth=1.6))
    else:
        ax_a.text(0.5, 0.5, "frame unavailable",
                  ha="center", transform=ax_a.transAxes)
    ax_a.set_title("(a) broadcast frame + bbox", **PANEL_TITLE)
    ax_a.set_xticks([]); ax_a.set_yticks([])
    _reserve_cbar_space(ax_a)

    # ---- Panel (b): IQR map + side colorbar (same height as image) ---
    ax_b = fig.add_subplot(gs[0, 1])
    im = ax_b.imshow(iqr, cmap="magma", vmin=0, vmax=80,
                     interpolation="bilinear")
    ax_b.add_patch(Rectangle((bx, by), bw, bh, fill=False,
                             edgecolor=BBOX_COLOR, linewidth=1.2))
    ax_b.set_title("(b) per-pixel IQR map", **PANEL_TITLE)
    ax_b.set_xticks([]); ax_b.set_yticks([])

    cax_b = make_axes_locatable(ax_b).append_axes(
        "right", size=CBAR_SIZE, pad=CBAR_PAD)
    cbar = fig.colorbar(im, cax=cax_b)
    cbar.ax.tick_params(labelsize=6)

    # ---- Panel (c): binary mask after morphology ---------------------
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.imshow(mask, cmap="gray_r", interpolation="nearest")
    ax_c.add_patch(Rectangle((bx, by), bw, bh, fill=False,
                             edgecolor=BBOX_COLOR, linewidth=1.2))
    ax_c.set_title("(c) binary mask after cleaning", **PANEL_TITLE)
    ax_c.set_xticks([]); ax_c.set_yticks([])
    _reserve_cbar_space(ax_c)

    return fig


if __name__ == "__main__":
    fig = build_figure()
    name = "qualitative_scoreboard"
    fig.savefig(FIGURES_DIR / f"{name}.svg", dpi=SAVE_DPI,
                bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{name}.pdf", dpi=SAVE_DPI,
                bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=SAVE_DPI,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.svg + {name}.pdf + {name}.png at dpi={SAVE_DPI}")
