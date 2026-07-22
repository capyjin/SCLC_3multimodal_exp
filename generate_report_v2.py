# -*- coding: utf-8 -*-
"""Regime-comparison figure (original bs16/ep30 vs improved bs32/ep60) and the
updated RESULTS.md with the full investigation + conclusion."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(ROOT, "report_assets")
os.makedirs(ASSETS, exist_ok=True)

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#0ca30c"
SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASE, "text.color": INK, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False, "ytick.color": MUTED, "xtick.color": MUTED,
})

# OS results: (label, orig_mean, orig_std, improved_mean, improved_std)
CONFIGS_OS = [
    ("Report only",          0.6233, 0.0200, 0.6268, 0.0474),
    ("Clinical only",        0.6389, 0.0373, 0.6466, 0.0407),
    ("Image only",           0.6469, 0.0320, 0.6388, 0.0468),
    ("Clinical + Image",     0.6786, 0.0446, 0.6774, 0.0418),
    ("Clinical + Report",    0.6870, 0.0525, 0.7076, 0.0472),
    ("Tri-modal (all 3)",    0.6949, 0.0464, 0.6775, 0.0372),
]

# ---- FIG 5: original vs improved regime (OS) ----
labels = [c[0] for c in CONFIGS_OS]
orig_m = [c[1] for c in CONFIGS_OS]
orig_s = [c[2] for c in CONFIGS_OS]
imp_m = [c[3] for c in CONFIGS_OS]
imp_s = [c[4] for c in CONFIGS_OS]
x = np.arange(len(labels))
w = 0.38

fig, ax = plt.subplots(figsize=(11, 5.4))
b1 = ax.bar(x - w/2, orig_m, w, yerr=orig_s, label="Original  (batch 16, 30 ep)",
            color=BLUE, capsize=3, ecolor=MUTED, error_kw={"elinewidth": 1, "alpha": 0.8})
b2 = ax.bar(x + w/2, imp_m, w, yerr=imp_s, label="Improved  (batch 32, 60 ep)",
            color=ORANGE, capsize=3, ecolor=MUTED, error_kw={"elinewidth": 1, "alpha": 0.8})
for xi, m in zip(x - w/2, orig_m):
    ax.text(xi, m + 0.012, f"{m:.3f}", ha="center", va="bottom", fontsize=8, color=INK2)
for xi, m in zip(x + w/2, imp_m):
    ax.text(xi, m + 0.012, f"{m:.3f}", ha="center", va="bottom", fontsize=8, color=INK, fontweight="bold")
ax.axhline(0.7083, color=GREEN, lw=1.6, ls=(0, (4, 3)), zorder=1)
ax.text(len(labels)-0.5, 0.7083, "  prior clin+report 0.708", va="bottom", ha="right", fontsize=8.5, color=GREEN)
ax.axhline(0.5, color=BASE, lw=1.3, ls=(0, (5, 4)), zorder=0)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10, color=INK)
ax.set_ylabel("OS C-index (5-fold mean ± SD)")
ax.set_ylim(0.5, 0.80)
ax.yaxis.grid(True, color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=10, loc="upper left")
ax.set_title("Fixing the batch=16 Cox handicap: tabular jumps, but image drags the fusion down",
             fontsize=12.5, color=INK, loc="left", pad=10, fontweight="bold")
# annotate the two key moves
ax.annotate("", xy=(4+w/2, 0.7076+0.055), xytext=(4-w/2, 0.6870+0.06),
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.8))
ax.text(4, 0.775, "+0.021", color=GREEN, fontsize=9, ha="center", fontweight="bold")
ax.annotate("", xy=(5+w/2, 0.6775+0.048), xytext=(5-w/2, 0.6949+0.055),
            arrowprops=dict(arrowstyle="->", color="#d03b3b", lw=1.8))
ax.text(5, 0.775, "-0.017", color="#d03b3b", fontsize=9, ha="center", fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "fig5_regime_compare_os.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote fig5_regime_compare_os.png")
