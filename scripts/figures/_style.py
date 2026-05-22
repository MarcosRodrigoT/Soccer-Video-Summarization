"""Shared styling for paper architecture figures.

All figure scripts under scripts/figures/ import from this module so the
visual identity is consistent. Output is always written as both SVG and PDF
to docs/figures/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Color palette — pastel academic with light borders
# ---------------------------------------------------------------------------
# Each entry is (fill, edge, text). Inspired by the existing pipeline_architecture.png
COLORS = {
    "input":        ("#BBDEFB", "#1976D2", "#0D47A1"),  # blue
    "visual":       ("#C8E6C9", "#388E3C", "#1B5E20"),  # green
    "scoreboard":   ("#FFE0B2", "#F57C00", "#E65100"),  # orange / yellow
    "intersection": ("#FFCDD2", "#C62828", "#B71C1C"),  # red
    "output":       ("#E1BEE7", "#7B1FA2", "#4A148C"),  # purple
    "evaluation":   ("#B0BEC5", "#37474F", "#263238"),  # slate
    "ground_truth": ("#CFD8DC", "#546E7A", "#37474F"),  # gray
    "neutral":      ("#F5F5F5", "#9E9E9E", "#424242"),  # light gray
    "accent":       ("#FFF59D", "#F9A825", "#F57F17"),  # amber accent
}

# ---------------------------------------------------------------------------
# rcParams — applied once per script
# ---------------------------------------------------------------------------
def set_paper_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.titleweight": "bold",
        "axes.edgecolor": "#212121",
        "axes.linewidth": 0.8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "savefig.transparent": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,   # embed TrueType (editable in Illustrator)
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

# ---------------------------------------------------------------------------
# Save helper — always writes both SVG and PDF
# ---------------------------------------------------------------------------
def save_figure(fig, name: str, *, also_png: bool = False) -> None:
    """Save figure as <name>.svg and <name>.pdf into docs/figures/."""
    svg_path = FIGURES_DIR / f"{name}.svg"
    pdf_path = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(svg_path)
    fig.savefig(pdf_path)
    if also_png:
        fig.savefig(FIGURES_DIR / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"  wrote {svg_path.name} + {pdf_path.name}")

# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------
def rounded_box(
    ax,
    x: float, y: float, w: float, h: float,
    label: str, *,
    color: str = "neutral",
    fontsize: float = 9,
    fontweight: str = "bold",
    rounding: float = 0.04,
    linewidth: float = 1.2,
    text_color: str | None = None,
    subtitle: str | None = None,
):
    """Draw a rounded rectangle with centered text. (x, y) is the lower-left corner."""
    fill, edge, fg = COLORS[color]
    text_color = text_color or fg
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.005,rounding_size={rounding}",
        linewidth=linewidth,
        edgecolor=edge,
        facecolor=fill,
        zorder=2,
    )
    ax.add_patch(patch)
    if subtitle:
        ax.text(
            x + w / 2, y + h / 2 + h * 0.13,
            label,
            ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight,
            color=text_color, zorder=3,
        )
        ax.text(
            x + w / 2, y + h / 2 - h * 0.20,
            subtitle,
            ha="center", va="center",
            fontsize=fontsize - 1.5, fontweight="normal",
            color=text_color, alpha=0.85, zorder=3,
        )
    else:
        ax.text(
            x + w / 2, y + h / 2,
            label,
            ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight,
            color=text_color, zorder=3,
        )
    return patch


def arrow(
    ax, start, end, *,
    color: str = "#424242",
    linewidth: float = 1.1,
    style: str = "-|>",
    linestyle: str = "-",
    rad: float = 0.0,
    label: str | None = None,
    label_pos: float = 0.5,
    label_offset: tuple[float, float] = (0.0, 0.02),
    label_fontsize: float = 7.5,
    label_color: str | None = None,
):
    """Draw a directed arrow from `start` to `end`. Optional curve via `rad`."""
    arr = FancyArrowPatch(
        start, end,
        arrowstyle=style,
        mutation_scale=10,
        linewidth=linewidth,
        linestyle=linestyle,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        zorder=1,
    )
    ax.add_patch(arr)
    if label:
        midx = start[0] + (end[0] - start[0]) * label_pos + label_offset[0]
        midy = start[1] + (end[1] - start[1]) * label_pos + label_offset[1]
        ax.text(
            midx, midy, label,
            ha="center", va="center",
            fontsize=label_fontsize,
            color=label_color or color,
            style="italic",
            zorder=3,
        )
    return arr


def clean_axes(ax):
    """Strip an axis to a bare canvas — no ticks, no spines, equal aspect, no margins."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal")


def page_size(width_in: float = 6.5, height_in: float = 4.0) -> tuple[float, float]:
    """Default to single-column journal width (6.5in usable)."""
    return (width_in, height_in)
