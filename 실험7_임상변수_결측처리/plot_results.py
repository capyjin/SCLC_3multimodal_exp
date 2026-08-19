# -*- coding: utf-8 -*-
"""Figures for the Step 3 model comparison.

fig1 -- forest plot of paired delta-C-index with patient-level bootstrap 95% CI.
        Effect estimates against a zero reference are what a forest plot is for;
        the zero line is the whole point of the figure, so it is drawn as the
        only emphasized rule on the panel.
fig2 -- absolute C-index per variant, so the reader can see the level, not just
        the delta (a +0.01 delta reads very differently at 0.64 than at 0.71).

Encoding decisions (matching the project's existing figures in
plot_all_figures.py, whose BLUE/ORANGE pair was validated for CVD separation:
adjacent dE 24.7 protan / 33.6 normal, both well above the >=8 threshold):
  - colour carries **target identity only** (OS vs PFS) -- two categorical
    slots, fixed, never cycled.
  - "CI excludes zero" is carried by **marker fill (solid vs hollow)** plus the
    printed numeric CI, never by colour. Significance is not a colour job here,
    and encoding it in colour would both collide with the target hues and make
    the distinction invisible to a CVD reader.
  - every row is directly labelled with its delta, so the figure is readable
    without the axis.
  - one x-axis. Deltas and absolute C-index live in separate figures rather
    than a dual axis.

Run:  python clinical/plot_results.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from core import plotstyle as ps
from core.plotstyle import BLUE, GRID, INK, INK2, MUTED, ORANGE, SURFACE, TARGET_COLOR

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "EXP_20260805_clinical_missing_handling")
PLOT_DIR = os.path.join(OUT_DIR, "plots")

ps.apply(font_size=10.5)   # 프로젝트 공통 팔레트/폰트 (core/plotstyle.py)

CONFIG_LABEL = {"clin_only": "Clinical-only (secondary)", "clin_report": "Clinical+Report (primary)"}


def _parse_ci(text: str) -> tuple[float, float]:
    lo, hi = text.strip("[]").split(",")
    return float(lo), float(hi)


def fig_forest(deltas: pd.DataFrame, out_path: str) -> None:
    rows = []
    for config in ("clin_report", "clin_only"):          # primary panel first
        for comparison in deltas["comparison"].unique():
            for target in ("os", "pfs"):
                sel = deltas[(deltas["config"] == config) & (deltas["comparison"] == comparison)
                             & (deltas["target"] == target)]
                if len(sel):
                    rows.append(sel.iloc[0])
    if not rows:
        return
    plot_df = pd.DataFrame(rows).reset_index(drop=True)

    height = 0.42 * len(plot_df) + 2.6
    fig, ax = plt.subplots(figsize=(10.5, height))

    # Layout: each config group gets a labelled header row so the panel identity
    # (primary vs secondary) is readable inside the figure, not only in the title.
    ypos, ylabels, group_breaks, group_headers = [], [], [], []
    y = 0.0
    last_key = None
    for _, r in plot_df.iterrows():
        key = (r["config"], r["comparison"])
        if last_key is None or r["config"] != last_key[0]:
            if last_key is not None:
                y += 0.5
                group_breaks.append(y)
                y += 0.9
            group_headers.append((y - 0.75, CONFIG_LABEL[r["config"]]))
        elif key != last_key:
            y += 0.55
        ypos.append(y)
        ylabels.append(f"{r['comparison']}   [{r['target'].upper()}]")
        last_key = key
        y += 1.0

    # zero reference -- the only emphasized rule on the panel
    ax.axvline(0.0, color=INK, linewidth=1.4, zorder=2)

    for yi, (_, r) in zip(ypos, plot_df.iterrows()):
        lo, hi = _parse_ci(r["ci95_fold_level"])
        color = TARGET_COLOR[r["target"]]
        excludes_zero = bool(r["ci_excludes_zero_fold_level"])
        ax.plot([lo, hi], [yi, yi], color=color, linewidth=2.0, solid_capstyle="round", zorder=3)
        ax.plot([r["delta_fold_mean"]], [yi],
                marker="o", markersize=9, zorder=4,
                color=color if excludes_zero else SURFACE,
                markeredgecolor=color, markeredgewidth=2.0)
        ax.text(hi + 0.004, yi, f"{r['delta_fold_mean']:+.4f}  [{lo:+.3f}, {hi:+.3f}]",
                va="center", ha="left", fontsize=8.5, color=INK2)

    for gy in group_breaks:
        ax.axhline(gy, color=GRID, linewidth=1.0, zorder=1)

    ax.set_yticks(ypos)
    ax.set_yticklabels(ylabels, fontsize=9, color=INK)
    # Rows are built top-to-bottom in reading order (primary config first), so
    # the axis is flipped by setting limits directly. invert_yaxis() combined
    # with an explicit set_ylim double-flipped this and put the secondary panel
    # on top, which is why it is not used here.
    ax.set_ylim(max(ypos) + 0.9, min(h[0] for h in group_headers) - 0.9)
    ax.set_xlabel("delta fold-mean C-index (avg over seeds x 5 folds) vs. its reference model", color=INK2)
    ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    xmax = max(_parse_ci(r["ci95_fold_level"])[1] for _, r in plot_df.iterrows())
    xmin = min(_parse_ci(r["ci95_fold_level"])[0] for _, r in plot_df.iterrows())
    span = xmax - xmin
    ax.set_xlim(xmin - 0.05 * span, xmax + 0.46 * span)

    for hy, label in group_headers:
        ax.text(ax.get_xlim()[0], hy, label, fontsize=10.5, color=INK,
                fontweight="bold", va="center", ha="left")

    handles = [
        Line2D([], [], color=BLUE, marker="o", markersize=9, linewidth=2.0, label="OS"),
        Line2D([], [], color=ORANGE, marker="o", markersize=9, linewidth=2.0, label="PFS"),
        Line2D([], [], color=MUTED, marker="o", markersize=9, linewidth=0,
               markerfacecolor=SURFACE, markeredgecolor=MUTED, markeredgewidth=2.0,
               label="95% CI includes 0 (hollow)"),
        Line2D([], [], color=MUTED, marker="o", markersize=9, linewidth=0,
               markerfacecolor=MUTED, markeredgecolor=MUTED, label="95% CI excludes 0 (solid)"),
    ]
    # Legend lives above the data area -- placing it inside overlapped the
    # bottom rows once the row count grew.
    ax.legend(handles=handles, frameon=False, fontsize=9, ncol=4,
              loc="lower left", bbox_to_anchor=(0.0, 1.005))

    ax.set_title("Paired delta fold-mean C-index, fold-stratified bootstrap 95% CI",
                 fontsize=12.5, color=INK, loc="left", pad=34, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig_levels(summary: pd.DataFrame, out_path: str) -> None:
    configs = [c for c in ("clin_report", "clin_only") if c in set(summary["config"])]
    fig, axes = plt.subplots(1, len(configs), figsize=(5.6 * len(configs), 4.4), sharey=True)
    if len(configs) == 1:
        axes = [axes]

    variants = ["original", "A", "B", "C"]
    for ax, config in zip(axes, configs):
        series = {}
        for target in ("os", "pfs"):
            sub = summary[(summary["config"] == config) & (summary["target"] == target)]
            series[target] = sub.set_index("variant").reindex(variants)

        lo_all, hi_all = [], []
        for target in ("os", "pfs"):
            sub = series[target].dropna(subset=["fold_mean_avg"])
            xs = np.arange(len(sub))
            err = sub["fold_mean_sd_across_seeds"]
            ax.errorbar(xs, sub["fold_mean_avg"], yerr=err,
                        color=TARGET_COLOR[target], marker="o", markersize=8, linewidth=2.0,
                        capsize=3, label=target.upper())
            lo_all += list(sub["fold_mean_avg"] - err)
            hi_all += list(sub["fold_mean_avg"] + err)

        # Label side is decided per x by which series is higher there -- a fixed
        # "OS above / PFS below" rule collides wherever PFS overtakes OS.
        for xi, variant in enumerate(variants):
            pair = [(t, series[t].loc[variant, "fold_mean_avg"],
                     series[t].loc[variant, "fold_mean_sd_across_seeds"])
                    for t in ("os", "pfs") if variant in series[t].index
                    and not pd.isna(series[t].loc[variant, "fold_mean_avg"])]
            if not pair:
                continue
            pair.sort(key=lambda p: p[1], reverse=True)      # highest first
            for (target, value, err), (offset, va) in zip(pair, ((+1, "bottom"), (-1, "top"))):
                ax.text(xi, value + offset * (err + 0.007), f"{value:.3f}",
                        ha="center", va=va, fontsize=8.5, color=INK2)

        # Only draw the 0.50 reference when it is near the data; with real
        # C-indices (~0.62-0.72) it sits far below and would only compress the
        # axis. Label on the left, where no marker is ever placed.
        lo, hi = min(lo_all), max(hi_all)
        if lo - 0.5 < 0.12:
            ax.axhline(0.5, color=MUTED, linestyle=":", linewidth=1.0)
            ax.text(-0.45, 0.502, "random 0.50", fontsize=8, color=MUTED, va="bottom", ha="left")
            lo = min(lo, 0.49)
        pad = 0.06 * (hi - lo)
        ax.set_ylim(lo - pad - 0.012, hi + pad + 0.012)
        ax.set_xlim(-0.5, len(variants) - 0.5)
        ax.set_xticks(np.arange(len(variants)))
        ax.set_xticklabels(variants, fontsize=10, color=INK)
        ax.set_title(CONFIG_LABEL[config], fontsize=11, color=INK, loc="left", fontweight="bold")
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("model variant", color=INK2)

    axes[0].set_ylabel("fold-mean C-index (avg over 3 seeds)", color=INK2)
    axes[0].legend(frameon=False, fontsize=9.5, loc="lower left")
    fig.suptitle("C-index level by variant  (error bar = SD across seeds 42/142/242)",
                 fontsize=12, color=INK, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    summary = pd.read_csv(os.path.join(OUT_DIR, "summary_by_variant.csv"))
    deltas = pd.read_csv(os.path.join(OUT_DIR, "paired_deltas.csv"))

    fig_forest(deltas, os.path.join(PLOT_DIR, "fig1_delta_forest.png"))
    fig_levels(summary, os.path.join(PLOT_DIR, "fig2_cindex_levels.png"))
    print(f"wrote fig1_delta_forest.png / fig2_cindex_levels.png to {PLOT_DIR}")


if __name__ == "__main__":
    main()
