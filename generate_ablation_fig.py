# -*- coding: utf-8 -*-
"""Ablation figure: OS C-index vs modality set on the SAME 238 split / loop.
Shows adding image does NOT hurt (monotonic climb 1->2->3 modalities)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_assets")
os.makedirs(ASSETS, exist_ok=True)

BLUE, ORANGE, MAGENTA = "#2a78d6", "#eb6834", "#e87ba4"
SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASE, "text.color": INK, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False, "ytick.color": MUTED, "xtick.color": MUTED,
})

# (label, mean, std, n_modalities)
rows = [
    ("Report only",            0.6233, 0.0200, 1),
    ("Clinical only",          0.6389, 0.0373, 1),
    ("Image only",             0.6469, 0.0320, 1),
    ("Clinical + Image",       0.6786, 0.0446, 2),
    ("Clinical + Report",      0.6870, 0.0525, 2),
    ("Image+Clinical+Report",  0.6949, 0.0464, 3),
]
cmap = {1: BLUE, 2: ORANGE, 3: MAGENTA}
labels = [r[0] for r in rows]
means = [r[1] for r in rows]
stds = [r[2] for r in rows]
colors = [cmap[r[3]] for r in rows]
y = range(len(rows))

fig, ax = plt.subplots(figsize=(9.5, 5.0))
ax.barh(list(y), means, xerr=stds, color=colors, height=0.68,
        error_kw={"elinewidth": 1.2, "ecolor": MUTED, "capsize": 3}, zorder=3)
for yi, m, s in zip(y, means, stds):
    ax.text(m + s + 0.004, yi, f"{m:.3f}", va="center", ha="left", fontsize=9.5,
            color=INK, fontweight="bold")
# reference lines
ax.axvline(0.5, color=BASE, lw=1.4, ls=(0, (5, 4)), zorder=1)
ax.text(0.5, len(rows) - 0.35, " random 0.50", color=MUTED, fontsize=8.5, va="bottom", ha="left")
ax.axvline(0.7083, color="#0ca30c", lw=1.6, ls=(0, (4, 3)), zorder=1)
ax.text(0.7083, -0.75, "prior 2-modal\n0.708 (diff pipeline)", color="#0ca30c", fontsize=8.5,
        va="bottom", ha="center")

ax.set_yticks(list(y))
ax.set_yticklabels(labels, fontsize=10.5, color=INK)
ax.set_xlim(0.5, 0.80)
ax.set_xlabel("OS C-index (5-fold mean ± SD) — same 238 split, same training loop")
ax.xaxis.grid(True, color=GRID, lw=0.8)
ax.set_axisbelow(True)

# legend for modality count
from matplotlib.patches import Patch
handles = [Patch(color=BLUE, label="1 modality"), Patch(color=ORANGE, label="2 modalities"),
           Patch(color=MAGENTA, label="3 modalities (tri-modal)")]
ax.legend(handles=handles, frameon=False, fontsize=9.5, loc="lower right")
ax.set_title("Ablation: does adding a modality hurt?  (No — C-index climbs monotonically)",
             fontsize=12.5, color=INK, loc="left", pad=10, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "fig4_ablation_os.png"), dpi=150, bbox_inches="tight")
print("wrote report_assets/fig4_ablation_os.png")
