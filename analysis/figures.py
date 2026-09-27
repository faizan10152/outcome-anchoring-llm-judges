"""Poster figures.

Print target: DIN A1. Figures are rendered at 300 DPI (the brief requires >=150
PPI) and sized so that a point in the figure is roughly a point on the poster,
which keeps every label above the 24 pt body-text floor.

Colours come from the validated reference palette: categorical blue/orange for
the two-series charts (adjacent-pair CVD dE 24.7, all six checks pass), a single
blue ramp for sequential magnitude, and the blue<->red diverging pair where a
value has a sign. Identity is never carried by colour alone - every chart is
directly labelled.

    python analysis/figures.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))

import matplotlib as mpl  # noqa: E402

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from metrics import bootstrap_gap_ci, build_pairs, load_judgments, rate  # noqa: E402

# --- validated palette -------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8a84"
BLUE = "#2a78d6"      # categorical slot 1 / sequential hue
ORANGE = "#eb6834"    # categorical slot 2
RED = "#e34948"       # diverging warm pole
GRID = "#e4e3df"
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
       "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
       "#184f95", "#104281", "#0d366b"]

JUDGE_LABEL = {"qwen2.5-7b": "Qwen2.5 7B", "llama3.1-8b": "Llama 3.1 8B",
               "qwen3-14b": "Qwen3 14B"}
# stacked form for cramped x axes
JUDGE_STACK = {"qwen2.5-7b": "Qwen2.5\n7B", "llama3.1-8b": "Llama 3.1\n8B",
               "qwen3-14b": "Qwen3\n14B"}
READOUT_LABEL = {"direct": "direct", "reason_then_verdict": "reasoning"}
ORDER = [("qwen2.5-7b", "direct"), ("qwen2.5-7b", "reason_then_verdict"),
         ("llama3.1-8b", "direct"), ("llama3.1-8b", "reason_then_verdict"),
         ("qwen3-14b", "direct"), ("qwen3-14b", "reason_then_verdict")]

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 17,
    "font.family": "DejaVu Sans", "text.color": INK,
    "axes.labelcolor": INK, "axes.edgecolor": GRID,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.titlesize": 21, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})


def seq_color(value: float) -> str:
    """Map a rate in [0,1] onto the sequential blue ramp."""
    return SEQ[min(len(SEQ) - 1, int(round(value * (len(SEQ) - 1))))]


def cell_stats(rows):
    """detection on faulty traces and false-positive rate on clean, per cell."""
    d = defaultdict(lambda: {"faulty": [], "clean": []})
    for r in rows:
        if r["detected"] is None:
            continue
        key = (r["judge_id"], r["readout"])
        d[key]["clean" if r["fault_type"] is None else "faulty"].append(r["detected"])
    return {k: (rate(v["faulty"]), rate(v["clean"])) for k, v in d.items()}


# --------------------------------------------------------------------------
def fig_discrimination(rows, out: Path):
    """The headline: detection is meaningless without the false-positive rate.

    The discrimination row lives in its own thin axes below the chart rather
    than as text under the ticks, which collided with the tick labels.
    """
    stats = cell_stats(rows)
    labels = [f"{JUDGE_STACK[j]}\n{READOUT_LABEL[r]}" for j, r in ORDER]
    det = [stats[k][0] * 100 for k in ORDER]
    fpr = [stats[k][1] * 100 for k in ORDER]

    fig, (ax, axd) = plt.subplots(
        2, 1, figsize=(10.2, 6.15), height_ratios=[1, 0.19])
    x = np.arange(len(ORDER))
    w = 0.38
    ax.bar(x - w / 2 - 0.012, det, w, label="FAULTY traces",
           color=BLUE, zorder=3)
    ax.bar(x + w / 2 + 0.012, fpr, w, label="CLEAN traces",
           color=ORANGE, zorder=3)
    for xi, (d, f) in enumerate(zip(det, fpr)):
        ax.text(xi - w / 2 - 0.012, d + 2.4, f"{d:.0f}%", ha="center", fontsize=15,
                color=INK, fontweight="bold")
        ax.text(xi + w / 2 + 0.012, f + 2.4, f"{f:.0f}%", ha="center", fontsize=15,
                color=INK, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=15)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Reported as mishandled", fontsize=17)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100%"])
    ax.yaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)
    ax.set_title("No judge separates faulty from clean traces", pad=38)
    ax.legend(loc="lower center", frameon=False, fontsize=16, ncols=2,
              bbox_to_anchor=(0.5, 1.005), handlelength=1.3, columnspacing=2.2)

    # discrimination strip
    axd.set_xlim(ax.get_xlim()); axd.set_ylim(0, 1)
    for s_ in axd.spines.values():
        s_.set_visible(False)
    axd.set_xticks([]); axd.set_yticks([])
    # label sits in the figure's left margin: inside the strip it collided
    # with the first column's value
    fig.text(0.010, 0.125, "discrimination\n(faulty − clean)", ha="left",
             va="center", fontsize=14, color=INK_2)
    for xi, (d, f) in enumerate(zip(det, fpr)):
        gapv = d - f
        axd.text(xi, 0.5, f"{gapv:+.0f} pts", ha="center", va="center", fontsize=18,
                 fontweight="bold", color=INK if gapv >= 10 else INK_MUTED)
    fig.subplots_adjust(bottom=0.10, left=0.155, right=0.985, top=0.855, hspace=0.46)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  wrote {out.name}")


def fig_fault_outcome_heatmap(rows, out: Path):
    """Detection by fault type and outcome: the matched-pair comparison."""
    cols = [(f, o) for f in ("F1", "F2", "F3") for o in ("correct", "wrong")]
    grid = []
    for judge, readout in ORDER:
        row = []
        for fault, outcome in cols:
            sel = [r["detected"] for r in rows
                   if r["judge_id"] == judge and r["readout"] == readout
                   and r["fault_type"] == fault
                   and r["outcome_version"] == outcome and r["detected"] is not None]
            row.append(rate(sel) or 0.0)
        grid.append(row)
    grid = np.array(grid)

    fig, ax = plt.subplots(figsize=(10.2, 6.3))
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            ax.add_patch(Rectangle((j + 0.02, i + 0.02), 0.96, 0.96,
                                   facecolor=seq_color(v), edgecolor=SURFACE, lw=2))
            ax.text(j + 0.5, i + 0.5, f"{v * 100:.0f}", ha="center", va="center",
                    fontsize=18, fontweight="bold",
                    color="#ffffff" if v > 0.45 else INK)
    ax.set_xlim(0, grid.shape[1]); ax.set_ylim(grid.shape[0], 0)
    ax.set_xticks([j + 0.5 for j in range(len(cols))])
    ax.set_xticklabels([f"{f}\n{o}" for f, o in cols], fontsize=17)
    ax.set_yticks([i + 0.5 for i in range(len(ORDER))])
    ax.set_yticklabels([f"{JUDGE_LABEL[j]} · {READOUT_LABEL[r]}" for j, r in ORDER],
                       fontsize=17)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.tick_params(length=0)
    for j in (2, 4):
        ax.axvline(j, color=INK_MUTED, lw=2)
    ax.set_title("Detection (%) by fault type and outcome", pad=14)
    fig.text(0.015, 0.035,
             "F1 fabricated value   ·   F2 phantom action   ·   F3 skipped check",
             fontsize=16, color=INK_2)
    fig.subplots_adjust(left=0.27, right=0.99, top=0.89, bottom=0.155)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  wrote {out.name}")


def fig_anchoring_gap(rows, out: Path):
    """The anchoring gap with bootstrap CIs over pairs - the H1 test."""
    pairs = build_pairs(rows)
    labels, gaps, los, his, ns = [], [], [], [], []
    for judge, readout in ORDER:
        sel = [v for k, v in pairs.items() if k[0] == judge and k[1] == readout]
        if not sel:
            continue
        g = (rate([w for w, _ in sel]) - rate([c for _, c in sel])) * 100
        lo, hi = bootstrap_gap_ci(sel)
        labels.append(f"{JUDGE_LABEL[judge]} · {READOUT_LABEL[readout]}")
        gaps.append(g); los.append(lo * 100); his.append(hi * 100); ns.append(len(sel))

    fig, ax = plt.subplots(figsize=(10.2, 5.7))
    y = np.arange(len(labels))[::-1]
    for yi, g, lo, hi in zip(y, gaps, los, his):
        col = BLUE if g >= 0 else RED
        ax.plot([lo, hi], [yi, yi], color=col, lw=6, solid_capstyle="round",
                alpha=0.35, zorder=2)
        ax.plot([g], [yi], "o", color=col, markersize=15, zorder=3)
    ax.axvline(0, color=INK_2, lw=2.2, zorder=1)
    # numbers in a fixed column so nothing runs off the edge
    for yi, g, lo, hi, n in zip(y, gaps, los, his, ns):
        ax.text(46, yi, f"{g:+.0f}%", va="center", ha="right", fontsize=17,
                color=INK, fontweight="bold")
        ax.text(52, yi, f"[{lo:+.0f}, {hi:+.0f}]", va="center", ha="left",
                fontsize=17, color=INK_2)
        ax.text(92, yi, f"n={n}", va="center", ha="right", fontsize=17, color=INK_2)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=17)
    ax.set_xlim(-46, 95)
    ax.set_xlabel("Anchoring gap  (percentage points)", fontsize=17, labelpad=12)
    ax.set_xticks([-40, -30, -20, -10, 0, 10, 20, 30])
    ax.set_xticklabels(["−40", "−30", "−20", "−10", "0", "+10", "+20", "+30%"])
    ax.xaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("H1 not supported: small, inconsistent gaps",
                 pad=14, loc="left", x=0.0)
    fig.text(0.015, 0.055, "gap = detection(wrong) − detection(correct)   ·   "
             "dot = gap   ·   bar = 95% bootstrap CI", fontsize=15, color=INK_2)
    fig.text(0.015, 0.018, "5 of 6 intervals include zero; 4 cells positive, "
             "1 negative, 1 exactly zero.", fontsize=15, color=INK_2)
    fig.subplots_adjust(left=0.265, right=0.985, top=0.88, bottom=0.235)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  wrote {out.name}")


def fig_prompt_effect(primed: Path, neutral: Path, out: Path):
    """The instrument-validation figure: one prompt sentence breaks the judge."""
    def stats(path):
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        # Same judge, same readout, same 101 traces in both arms: the only thing
        # that differs is the shared framing of the judge prompt.
        rows = [r for r in rows if r["readout"] == "direct"
                and r["judge_id"] == "qwen2.5-7b" and r["detected"] is not None]
        faulty = [r["detected"] for r in rows if r["fault_type"]]
        clean = [r["detected"] for r in rows if r["fault_type"] is None]
        det, fpr = rate(faulty), rate(clean)
        if det is None or fpr is None:
            raise SystemExit(f"{path.name}: needs both faulty and clean traces "
                             f"(faulty={len(faulty)}, clean={len(clean)})")
        return det * 100, fpr * 100

    p_det, p_fpr = stats(primed)
    n_det, n_fpr = stats(neutral)

    fig, ax = plt.subplots(figsize=(9.0, 6.2))
    x = np.arange(2); w = 0.34
    ax.bar(x - w / 2 - 0.012, [p_det, n_det], w, color=BLUE, zorder=3,
           label="FAULTY traces")
    ax.bar(x + w / 2 + 0.012, [p_fpr, n_fpr], w, color=ORANGE, zorder=3,
           label="CLEAN traces")
    for xi, (d, f) in enumerate([(p_det, p_fpr), (n_det, n_fpr)]):
        ax.text(xi - w / 2 - 0.012, d + 2.2, f"{d:.0f}%", ha="center",
                fontsize=18, fontweight="bold")
        ax.text(xi + w / 2 + 0.012, f + 2.2, f"{f:.0f}%", ha="center",
                fontsize=18, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Judge prompt names\nwhat to check for",
                        "Neutral judge prompt"], fontsize=17)
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100%"])
    ax.yaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)
    ax.set_ylabel("% of traces reported as mishandled", fontsize=17)
    ax.set_title("A primed judge looks accurate and measures nothing", pad=34)
    ax.legend(loc="lower center", frameon=False, fontsize=16, ncols=2,
              bbox_to_anchor=(0.5, 1.005), handlelength=1.3, columnspacing=2.2)
    fig.text(0.06, 0.028, "All 101 traces (84 faulty, 17 clean), Qwen2.5 7B, "
             "direct readout.", fontsize=15, color=INK_2)
    fig.subplots_adjust(left=0.145, right=0.985, top=0.855, bottom=0.175)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  wrote {out.name}")


def main() -> int:
    cfg = yaml.safe_load((ROOT / "config" / "experiment.yaml").read_text())
    out_dir = ROOT / cfg["paths"]["figures"]
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_judgments(ROOT / cfg["paths"]["judgments"])
    print(f"figures from {len(rows)} judgements -> {out_dir.relative_to(ROOT)}")

    fig_discrimination(rows, out_dir / "fig1_discrimination.png")
    fig_fault_outcome_heatmap(rows, out_dir / "fig2_fault_outcome.png")
    fig_anchoring_gap(rows, out_dir / "fig3_anchoring_gap.png")
    primed = ROOT / "results" / "primed_control.jsonl"
    neutral = ROOT / cfg["paths"]["judgments"]
    if primed.exists() and neutral.exists():
        fig_prompt_effect(primed, neutral, out_dir / "fig4_prompt_effect.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
