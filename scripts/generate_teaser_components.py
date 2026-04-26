#!/usr/bin/env python3
"""Generate only the teaser components that need computation.

Outputs:
  - left true-vs-hallucinated detection case;
  - right mini FDP/recall result panel.
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


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image

from generate_teaser_triptych import OUT_DIR, _setup_style


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COCO_IMG_DIR = Path(str(_COCO_ROOT / "val2017"))
CASE_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "coco_gdino_fullscale_20seed_formal_hist"
    / "seed_5"
    / "test_candidates_with_evalues.csv"
)
RESULT_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "coco_gdino_fullscale_20seed_formal_hist"
    / "mean_results_by_alpha.csv"
)
CASE = {
    "file_name": "000000048153.jpg",
    "present_prompt": "person",
    "absent_prompt": "toothbrush",
}

INK = "#1f2933"
MUTED = "#52606d"
PRESENT = "#0b7f55"
ABSENT = "#c92a2a"


def _load_detection_case() -> tuple[Path, pd.Series, pd.Series]:
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
        raise RuntimeError("Could not find the configured true/hallucinated detection case.")
    return COCO_IMG_DIR / CASE["file_name"], present.iloc[0], absent.iloc[0]


def _draw_box(ax: plt.Axes, row: pd.Series, color: str, linestyle: str = "-") -> None:
    x1, y1, x2, y2 = [float(row[c]) for c in ("x1", "y1", "x2", "y2")]
    ax.add_patch(
        Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            edgecolor=color,
            linewidth=2.0,
            linestyle=linestyle,
        )
    )


def _draw_prompt_label(
    ax: plt.Axes,
    prompt: str,
    verdict: str,
    score: float,
    color: str,
) -> None:
    ax.text(
        0.04,
        0.96,
        f'prompt: "{prompt}"',
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.3,
        weight="bold",
        color=color,
        bbox=dict(facecolor="white", edgecolor=color, linewidth=1.0, boxstyle="round,pad=0.28"),
    )
    ax.text(
        0.04,
        0.835,
        f"{verdict}, score {score:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.3,
        color=INK,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.92, boxstyle="round,pad=0.25"),
    )


def draw_left_detection_pair(fig: plt.Figure) -> None:
    img_path, present, absent = _load_detection_case()
    image = Image.open(img_path).convert("RGB")
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 0.18],
        left=0.012,
        right=0.988,
        top=0.98,
        bottom=0.055,
        wspace=0.035,
        hspace=0.04,
    )
    ax_true = fig.add_subplot(gs[0, 0])
    ax_hall = fig.add_subplot(gs[0, 1])
    ax_note = fig.add_subplot(gs[1, :])

    for ax in (ax_true, ax_hall):
        ax.imshow(image)
        ax.set_axis_off()

    _draw_box(ax_true, present, PRESENT, "-")
    _draw_prompt_label(
        ax_true,
        CASE["present_prompt"],
        "true object",
        float(present["score"]),
        PRESENT,
    )

    _draw_box(ax_hall, absent, ABSENT, (0, (4, 2)))
    _draw_prompt_label(
        ax_hall,
        CASE["absent_prompt"],
        "hallucination",
        float(absent["score"]),
        ABSENT,
    )

    ax_note.axis("off")
    ax_note.text(
        0.5,
        0.58,
        "Same image, same visual evidence: the absent-query hallucination scores higher than the true detection.",
        ha="center",
        va="center",
        fontsize=8.3,
        weight="bold",
        color=INK,
    )
    ax_note.text(
        0.5,
        0.13,
        "Top-scoring absent-query detections can be all hallucinations.",
        ha="center",
        va="center",
        fontsize=7.0,
        color=MUTED,
    )


def render_left_detection() -> None:
    fig = plt.figure(figsize=(4.35, 2.45))
    draw_left_detection_pair(fig)
    fig.savefig(OUT_DIR / "fig_teaser_left_detection.pdf", bbox_inches="tight", pad_inches=0.012)
    fig.savefig(OUT_DIR / "fig_teaser_left_detection.png", bbox_inches="tight", pad_inches=0.012)
    plt.close(fig)


def render_right_effect() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Nimbus Roman", "Times New Roman", "Liberation Serif", "Times"],
            "mathtext.fontset": "stix",
            "font.size": 7.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    df = pd.read_csv(RESULT_CSV)
    betting = df[df["method"].eq("betting")].sort_values("alpha")
    alphas = betting["alpha"].to_numpy()
    fdp = betting["fdp"].to_numpy()
    recall = betting["recall"].to_numpy()
    alpha10 = betting[betting["alpha"].sub(0.10).abs() < 1e-9].iloc[0]
    highlight = abs(alphas - 0.10) < 1e-9

    fig = plt.figure(figsize=(2.15, 3.05))
    gs = fig.add_gridspec(
        2,
        1,
        left=0.205,
        right=0.970,
        bottom=0.105,
        top=0.875,
        hspace=0.58,
    )
    ax_fdp = fig.add_subplot(gs[0, 0])
    ax_rec = fig.add_subplot(gs[1, 0])

    fig.text(
        0.5,
        0.975,
        "Full COCO val / GroundingDINO",
        ha="center",
        va="top",
        fontsize=8.2,
        fontweight="bold",
    )

    ax_fdp.plot(
        alphas,
        alphas,
        color="#8a96a3",
        linestyle=(0, (3, 2)),
        linewidth=1.0,
        label="target",
        zorder=1,
    )
    ax_fdp.plot(
        alphas,
        fdp,
        color="#1f77b4",
        linewidth=1.35,
        zorder=2,
    )
    ax_fdp.scatter(
        alphas[~highlight],
        fdp[~highlight],
        color="#1f77b4",
        marker="o",
        s=21,
        label="ours",
        zorder=3,
    )
    ax_fdp.scatter(
        [0.10],
        [float(alpha10["fdp"])],
        color="#1f77b4",
        marker="D",
        s=34,
        edgecolor="white",
        linewidth=0.45,
        zorder=4,
    )
    ax_fdp.text(0.108, float(alpha10["fdp"]) + 0.018, "6.4%", fontsize=6.0, color="#1f77b4")
    ax_fdp.set_title("FDP", pad=2.0)
    ax_fdp.set_xlabel(r"target $\alpha$", labelpad=1.5)
    ax_fdp.set_ylabel("empirical FDP", labelpad=1.5)
    ax_fdp.set_xlim(0.042, 0.208)
    ax_fdp.set_ylim(0.0, 0.22)
    ax_fdp.set_xticks([0.05, 0.10, 0.15, 0.20])
    ax_fdp.set_yticks([0.0, 0.1, 0.2])
    ax_fdp.grid(True, color="#d8dde3", linewidth=0.45, alpha=0.8)
    ax_fdp.legend(frameon=False, loc="upper left", handlelength=1.1, borderpad=0.1)

    ax_rec.plot(
        alphas,
        recall,
        color="#0b7f55",
        linewidth=1.35,
        zorder=2,
    )
    ax_rec.scatter(
        alphas[~highlight],
        recall[~highlight],
        color="#0b7f55",
        marker="o",
        s=21,
        zorder=3,
    )
    ax_rec.scatter(
        [0.10],
        [float(alpha10["recall"])],
        color="#0b7f55",
        marker="D",
        s=34,
        edgecolor="white",
        linewidth=0.45,
        zorder=4,
    )
    ax_rec.text(0.108, float(alpha10["recall"]) + 0.018, "16.4%", fontsize=6.0, color="#0b7f55")
    ax_rec.set_title("Recall", pad=2.0)
    ax_rec.set_xlabel(r"target $\alpha$", labelpad=1.5)
    ax_rec.set_ylabel("recall", labelpad=1.5)
    ax_rec.set_xlim(0.042, 0.208)
    ax_rec.set_ylim(0.0, 0.24)
    ax_rec.set_xticks([0.05, 0.10, 0.15, 0.20])
    ax_rec.set_yticks([0.0, 0.1, 0.2])
    ax_rec.grid(True, color="#d8dde3", linewidth=0.45, alpha=0.8)

    for ax in (ax_fdp, ax_rec):
        for spine in ax.spines.values():
            spine.set_linewidth(0.75)

    fig.savefig(OUT_DIR / "fig_teaser_right_effect.pdf", bbox_inches="tight", pad_inches=0.012)
    fig.savefig(OUT_DIR / "fig_teaser_right_effect.png", bbox_inches="tight", pad_inches=0.012, dpi=600)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _setup_style()
    render_left_detection()
    render_right_effect()
    print(f"Saved {OUT_DIR / 'fig_teaser_left_detection.pdf'}")
    print(f"Saved {OUT_DIR / 'fig_teaser_left_detection.png'}")
    print(f"Saved {OUT_DIR / 'fig_teaser_right_effect.pdf'}")
    print(f"Saved {OUT_DIR / 'fig_teaser_right_effect.png'}")


if __name__ == "__main__":
    main()
