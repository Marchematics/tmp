"""
Generate a 2x2 figure for A1 selection-bias defense:
  [LVIS, OI] x [candidates/family CDF (present-only), category share log-log]
Output PDF/PNG to outputs/a1_selection_bias_check/.
"""
import json
import numpy as np
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data_raw/a_project_2/outputs/a1_selection_bias_check"


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f]


def cdf(x):
    x = np.sort(np.asarray(x))
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def cat_shares(fams):
    c = Counter(f["category_id"] for f in fams)
    total = sum(c.values()) or 1
    shares = sorted((v / total for v in c.values()), reverse=True)
    return np.array(shares)


def make(name, full_p, clean_p, ax_cdf, ax_share):
    full = load_jsonl(full_p)
    clean = load_jsonl(clean_p)
    full_pres = [f for f in full if not f["is_prompt_absent"]]

    # CDF of candidates/family (present-only on both sides)
    x1, y1 = cdf([len(f["candidates"]) for f in full_pres])
    x2, y2 = cdf([len(f["candidates"]) for f in clean if not f["is_prompt_absent"]])
    ax_cdf.plot(x1, y1, label=f"full present (n={len(full_pres)})", lw=2)
    ax_cdf.plot(x2, y2, label=f"clean present (n={sum(1 for f in clean if not f['is_prompt_absent'])})",
                lw=2, ls="--")
    ax_cdf.set_xlabel("candidates per family")
    ax_cdf.set_ylabel("CDF")
    ax_cdf.set_title(f"{name}: candidates/family CDF\n(present-only, KS p>0.05 = indistinguishable)")
    ax_cdf.set_xscale("symlog")
    ax_cdf.grid(alpha=0.3)
    ax_cdf.legend(fontsize=8)

    # Category-share rank plot (log-log)
    s_full = cat_shares(full)
    s_clean = cat_shares(clean)
    ax_share.loglog(np.arange(1, len(s_full) + 1), s_full, label="full", lw=2)
    ax_share.loglog(np.arange(1, len(s_clean) + 1), s_clean, label="clean", lw=2, ls="--")
    ax_share.set_xlabel("category rank")
    ax_share.set_ylabel("share of families")
    ax_share.set_title(f"{name}: category share by rank\n(parallel curves = same shape, no truncation bias)")
    ax_share.grid(alpha=0.3, which="both")
    ax_share.legend(fontsize=8)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    root = ROOT / "data_raw/a_project_2/outputs"

    make(
        "LVIS / GDino",
        root / "lvis_groundingdino_1000_presentall_absentraremix.jsonl",
        root / "lvis_groundingdino_aux_clean_negposexh.jsonl",
        axes[0, 0],
        axes[0, 1],
    )
    make(
        "OpenImages V7 / GDino",
        root / "oi_1000_gdino.jsonl",
        root / "oi_1000_gdino_present_only.jsonl",
        axes[1, 0],
        axes[1, 1],
    )

    fig.suptitle(
        "A1 selection-bias defense: clean subset is a structural restriction, not a cherry-pick",
        fontsize=12, y=1.00,
    )
    fig.tight_layout()
    pdf = OUT / "fig_selection_bias.pdf"
    png = OUT / "fig_selection_bias.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=150)
    print(f"written: {pdf}")
    print(f"written: {png}")


if __name__ == "__main__":
    main()
