"""Figure 12 — Qualitative scoreboard pipeline on Match1 (rev. May 2026).

The figure now stacks the three localisation steps vertically so that
each panel has the same image dimensions:

  (a) broadcast frame with the auto-localised bounding box overlaid,
  (b) per-pixel IQR map used as the temporal-stability statistic,
  (c) binary mask after the morphological cascade, with the selected
      scoreboard region highlighted.

The colorbar of panel (b) sits in a dedicated narrow column to the
right of the map.  Its height is forced to match the height of the
image, so the IQR map keeps the same width and dimensions as panels
(a) and (c).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from _style import save_figure, set_paper_style

PROJECT = Path(__file__).resolve().parents[2]
MATCH_DIR = PROJECT / "results_hr" / "per_match" / "Match1"
VIDEO = Path("/mnt/Data-fast/mrt/lf/SoccerNetClip10Videos/Laliga_HR/Match1/"
             "Valencia CF - FC Barcelona (2-3) LALIGA 2013⧸2014 FULL MATCH "
             "[Qzrg3QDmQ4k].mp4")

PANEL_TITLE = dict(loc="left", fontsize=9.0, fontweight="bold")
BBOX_COLOR = "#FF1744"


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

    # Three rows of identical image width.  The colorbar of panel (b)
    # is docked to the IQR axes with ``make_axes_locatable``; the same
    # divider trick is used on panels (a) and (c) to append an
    # *invisible* side axes of the same width, so the visible image
    # widths of the three panels match exactly.
    fig = plt.figure(figsize=(7.4, 7.8))
    gs = fig.add_gridspec(
        3, 1,
        height_ratios=[1.0, 1.0, 1.0],
        hspace=0.16,
    )

    CBAR_SIZE = "2.0%"
    CBAR_PAD = 0.04

    def _reserve_cbar_space(ax):
        """Append an invisible axes of the same width as the colorbar of
        panel (b), so the image axes of (a) and (c) keep the same
        visible width as the IQR map in (b)."""
        cax_dummy = make_axes_locatable(ax).append_axes(
            "right", size=CBAR_SIZE, pad=CBAR_PAD)
        cax_dummy.axis("off")

    # ---- Panel (a): broadcast frame + bbox ---------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    if frame is not None:
        ax_a.imshow(frame)
        ax_a.add_patch(Rectangle((bx, by), bw, bh, fill=False,
                                 edgecolor=BBOX_COLOR, linewidth=1.8))
    else:
        ax_a.text(0.5, 0.5, "frame unavailable",
                  ha="center", transform=ax_a.transAxes)
    ax_a.set_title("(a) broadcast frame with localised bbox", **PANEL_TITLE)
    ax_a.set_xticks([]); ax_a.set_yticks([])
    _reserve_cbar_space(ax_a)

    # ---- Panel (b): IQR map + side colorbar (same height as image) ---
    ax_b = fig.add_subplot(gs[1, 0])
    im = ax_b.imshow(iqr, cmap="magma", vmin=0, vmax=80)
    ax_b.add_patch(Rectangle((bx, by), bw, bh, fill=False,
                             edgecolor=BBOX_COLOR, linewidth=1.2))
    ax_b.set_title("(b) per-pixel IQR map", **PANEL_TITLE)
    ax_b.set_xticks([]); ax_b.set_yticks([])

    cax_b = make_axes_locatable(ax_b).append_axes(
        "right", size=CBAR_SIZE, pad=CBAR_PAD)
    cbar = fig.colorbar(im, cax=cax_b)
    cbar.ax.tick_params(labelsize=7)

    # ---- Panel (c): binary mask after morphology ---------------------
    ax_c = fig.add_subplot(gs[2, 0])
    ax_c.imshow(mask, cmap="gray_r")
    ax_c.add_patch(Rectangle((bx, by), bw, bh, fill=False,
                             edgecolor=BBOX_COLOR, linewidth=1.2))
    ax_c.set_title("(c) binary mask after morphological cleaning",
                   **PANEL_TITLE)
    ax_c.set_xticks([]); ax_c.set_yticks([])
    _reserve_cbar_space(ax_c)

    return fig


if __name__ == "__main__":
    fig = build_figure()
    save_figure(fig, "qualitative_scoreboard", also_png=True)
