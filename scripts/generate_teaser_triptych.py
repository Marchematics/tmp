#!/usr/bin/env python3
"""Generate the opening teaser triptych for the NeurIPS draft.

Panels:
  left: one real hallucination case from COCO/GDino;
  middle: a compressed method chain;
  right: mini full-COCO alpha curves.
"""
from __future__ import annotations

import pathlib
from pathlib import Path

import os, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
try:
    from ovd_hallucination_fdr.paths import DATASETS as _DS
    _COCO_ROOT = _DS.get('coco_root', pathlib.Path('./data/coco2017'))
    _LVIS_ROOT = _DS.get('lvis_root', pathlib.Path('./data/lvis'))
    _OBJECTS365_ROOT = _DS.get('objects365_root', pathlib.Path('./data/objects365'))
    _OPEN_IMAGES_ROOT = _DS.get('open_images_root', pathlib.Path('./data/open_images_v7'))
except Exception:
    _COCO_ROOT = pathlib.Path(os.environ.get('COCO_ROOT', './data/coco2017'))
    _LVIS_ROOT = pathlib.Path(os.environ.get('LVIS_ROOT', './data/lvis'))
    _OBJECTS365_ROOT = pathlib.Path(os.environ.get('OBJECTS365_ROOT', './data/objects365'))
    _OPEN_IMAGES_ROOT = pathlib.Path(os.environ.get('OPEN_IMAGES_ROOT', './data/open_images_v7'))

import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs" / "paper_figures"
COCO_IMG_DIR = Path(str(_COCO_ROOT / "val2017"))
CASE_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "coco_gdino_fullscale_20seed_formal_hist"
    / "seed_0"
    / "test_candidates_with_evalues.csv"
)
RESULT_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "coco_gdino_fullscale_20seed_formal_hist"
    / "mean_results_by_alpha.csv"
)


CASE = {
    "file_name": "000000227686.jpg",
    "present_prompt": "horse",
    "absent_prompt": "dog",
}


COLORS = {
    "ink": "#1f2933",
    "muted": "#52606d",
    "panel_edge": "#cbd2d9",
    "present": "#0b7f55",
    "absent": "#c92a2a",
    "method": "#255f85",
    "method_light": "#e7f1f7",
    "result": "#1f77b4",
    "target": "#7b8794",
}


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 6.2,
            "axes.titlesize": 6.6,
            "axes.labelsize": 5.6,
            "xtick.labelsize": 5.0,
            "ytick.labelsize": 5.0,
            "legend.fontsize": 4.8,
            "figure.dpi": 180,
            "savefig.dpi": 320,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _load_case_rows() -> tuple[Path, pd.Series, pd.Series]:
    df = pd.read_csv(CASE_CSV)
    for col in ("is_tp", "is_null", "is_prompt_absent"):
        df[col] = df[col].astype(str).str.lower().eq("true")

    mask_file = df["file_name"].eq(CASE["file_name"])
    present = df[
        mask_file
        & (~df["is_prompt_absent"])
        & df["is_tp"]
        & df["category_name"].eq(CASE["present_prompt"])
    ].sort_values("score", ascending=False)
    absent = df[
        mask_file
        & df["is_prompt_absent"]
        & df["is_null"]
        & df["category_name"].eq(CASE["absent_prompt"])
    ].sort_values("score", ascending=False)

    if present.empty or absent.empty:
        raise RuntimeError("Could not find the configured teaser case rows.")

    return COCO_IMG_DIR / CASE["file_name"], present.iloc[0], absent.iloc[0]


def _draw_rounded_panel(ax: plt.Axes) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor="white",
            edgecolor=COLORS["panel_edge"],
            linewidth=0.8,
            zorder=-20,
        )
    )


def _add_badge(
    ax: plt.Axes,
    xy_axes: tuple[float, float],
    title: str,
    subtitle: str,
    color: str,
    width: float,
) -> None:
    x, y = xy_axes
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            0.115,
            transform=ax.transAxes,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            facecolor="white",
            edgecolor=color,
            linewidth=1.0,
            zorder=10,
        )
    )
    ax.text(
        x + 0.018,
        y + 0.074,
        title,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=4.8,
        weight="bold",
        color=color,
        zorder=11,
        clip_on=True,
    )
    ax.text(
        x + 0.018,
        y + 0.030,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=4.6,
        color=COLORS["ink"],
        zorder=11,
        clip_on=True,
    )


def draw_left_panel(ax: plt.Axes) -> None:
    _draw_rounded_panel(ax)
    img_path, present, absent = _load_case_rows()
    image = Image.open(img_path).convert("RGB")
    ax.imshow(image)
    ax.set_axis_off()

    # The two boxes are intentionally shown with a small visual offset because
    # the detector fires on almost the exact same horse region for both prompts.
    p_box = [float(present[c]) for c in ("x1", "y1", "x2", "y2")]
    a_box = [float(absent[c]) for c in ("x1", "y1", "x2", "y2")]
    px1, py1, px2, py2 = p_box
    ax.add_patch(
        Rectangle(
            (px1, py1),
            px2 - px1,
            py2 - py1,
            fill=False,
            edgecolor=COLORS["present"],
            linewidth=2.0,
            zorder=3,
        )
    )
    ax.add_patch(
        Rectangle(
            (a_box[0] + 4.0, a_box[1] - 4.0),
            a_box[2] - a_box[0],
            a_box[3] - a_box[1],
            fill=False,
            edgecolor=COLORS["absent"],
            linewidth=2.0,
            linestyle=(0, (4, 2)),
            zorder=4,
        )
    )

    _add_badge(
        ax,
        (0.045, 0.875),
        "present prompt: horse",
        f"correct box, score {float(present['score']):.2f}",
        COLORS["present"],
        0.72,
    )
    _add_badge(
        ax,
        (0.045, 0.735),
        "absent prompt: dog",
        f"false box, score {float(absent['score']):.2f}",
        COLORS["absent"],
        0.68,
    )
    ax.text(
        0.045,
        0.040,
        "Top-scoring absent-query\ndetections can be\nall hallucinations.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.3,
        weight="bold",
        color="white",
        bbox=dict(facecolor="black", edgecolor="none", alpha=0.66, pad=4.0),
        zorder=12,
    )


def _method_box(ax: plt.Axes, x: float, y: float, w: float, h: float, label: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.014,rounding_size=0.018",
            facecolor=COLORS["method_light"],
            edgecolor=COLORS["method"],
            linewidth=0.9,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=5.5,
        weight="bold",
        color=COLORS["ink"],
        linespacing=1.05,
    )


def draw_middle_panel(ax: plt.Axes) -> None:
    _draw_rounded_panel(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.04,
        0.90,
        "Calibration wrapper",
        ha="left",
        va="center",
        fontsize=7.0,
        weight="bold",
        color=COLORS["ink"],
    )

    labels = [
        "OVD\nscores",
        "fit split\nlearns phi(s)",
        "null-only\ncalibration",
        "e-values",
        "per-family\ne-BH",
    ]
    xs = [0.045, 0.225, 0.425, 0.625, 0.805]
    ws = [0.120, 0.145, 0.150, 0.115, 0.130]
    y, h = 0.50, 0.18
    for x, w, label in zip(xs, ws, labels):
        _method_box(ax, x, y, w, h, label)
    for x0, w0, x1 in zip(xs[:-1], ws[:-1], xs[1:]):
        ax.add_patch(
            FancyArrowPatch(
                (x0 + w0 + 0.012, y + h / 2),
                (x1 - 0.014, y + h / 2),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.0,
                color=COLORS["muted"],
            )
        )

    ax.text(
        0.50,
        0.285,
        "Dependent boxes stay inside one image-query family;\nper-family e-BH still controls FDR.",
        ha="center",
        va="center",
        fontsize=5.8,
        color=COLORS["ink"],
    )
    ax.text(
        0.50,
        0.160,
        "No independence assumption among boxes from the same prompt.",
        ha="center",
        va="center",
        fontsize=5.4,
        color=COLORS["muted"],
    )


def draw_right_panel(ax: plt.Axes) -> None:
    _draw_rounded_panel(ax)
    ax.axis("off")
    ax.text(
        0.04,
        0.90,
        "Full COCO val / GDino",
        ha="left",
        va="center",
        fontsize=7.0,
        weight="bold",
        color=COLORS["ink"],
        transform=ax.transAxes,
    )

    df = pd.read_csv(RESULT_CSV)
    betting = df[df["method"].eq("betting")].sort_values("alpha")

    fdp_ax = ax.inset_axes([0.10, 0.49, 0.36, 0.30])
    rec_ax = ax.inset_axes([0.57, 0.49, 0.34, 0.30])

    alphas = betting["alpha"].to_numpy()
    fdp = betting["fdp"].to_numpy()
    recall = betting["recall"].to_numpy()

    fdp_ax.plot(
        alphas,
        alphas,
        linestyle=(0, (3, 2)),
        color=COLORS["target"],
        linewidth=1.0,
        label="target",
    )
    fdp_ax.plot(
        alphas,
        fdp,
        color=COLORS["result"],
        marker="o",
        markersize=2.4,
        linewidth=1.2,
        label="ours",
    )
    fdp_ax.set_title("FDP vs alpha", pad=1.5)
    fdp_ax.set_xlabel("alpha", labelpad=0.5)
    fdp_ax.set_ylabel("FDP", labelpad=0.5)
    fdp_ax.set_xticks([0.05, 0.10, 0.20])
    fdp_ax.set_ylim(0, 0.22)
    fdp_ax.grid(True, alpha=0.25, linewidth=0.45)
    fdp_ax.legend(frameon=False, loc="upper left", handlelength=0.9, borderpad=0.05)

    rec_ax.plot(
        alphas,
        recall,
        color=COLORS["present"],
        marker="o",
        markersize=2.4,
        linewidth=1.2,
    )
    rec_ax.set_title("Recall vs alpha", pad=1.5)
    rec_ax.set_xlabel("alpha", labelpad=0.5)
    rec_ax.set_ylabel("recall", labelpad=0.5)
    rec_ax.set_xticks([0.05, 0.10, 0.20])
    rec_ax.set_ylim(0, 0.24)
    rec_ax.grid(True, alpha=0.25, linewidth=0.45)

    alpha10 = betting[betting["alpha"].sub(0.10).abs() < 1e-9].iloc[0]
    summary = (
        f"At alpha=0.10: FDP {100 * float(alpha10['fdp']):.1f}% "
        f"and recall {100 * float(alpha10['recall']):.1f}%."
    )
    ax.text(
        0.08,
        0.205,
        textwrap.fill(summary, width=45),
        ha="left",
        va="center",
        fontsize=5.9,
        color=COLORS["ink"],
        transform=ax.transAxes,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _setup_style()

    fig = plt.figure(figsize=(7.25, 2.22), constrained_layout=False)
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.05, 1.22, 1.08],
        left=0.012,
        right=0.992,
        top=0.965,
        bottom=0.055,
        wspace=0.055,
    )
    ax_left = fig.add_subplot(gs[0, 0])
    ax_mid = fig.add_subplot(gs[0, 1])
    ax_right = fig.add_subplot(gs[0, 2])

    draw_left_panel(ax_left)
    draw_middle_panel(ax_mid)
    draw_right_panel(ax_right)

    pdf_path = OUT_DIR / "fig_teaser_triptych.pdf"
    png_path = OUT_DIR / "fig_teaser_triptych.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.015)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
