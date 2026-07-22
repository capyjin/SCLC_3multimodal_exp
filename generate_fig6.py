# -*- coding: utf-8 -*-
"""fig6: method-B late fusion. Left = late fusion vs tabular baseline (does the
image arm help?). Right = image encoder alone (SimpleCNN vs pretrained ResNet18)."""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(ROOT, "report_assets")
os.makedirs(ASSETS, exist_ok=True)
with open(os.path.join(ROOT, "outputs", "late_fusion_B", "results.json")) as f:
    R = json.load(f)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, MUTED, GRID, BASE, GREEN, RED = \
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#0ca30c", "#d03b3b"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASE, "text.color": INK, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False, "ytick.color": MUTED, "xtick.color": MUTED,
})

def g(t, k):
    d = R[t][k]
    return d["mean"], d["std"]

targets = ["os", "pfs"]
tlabels = ["OS", "PFS"]
x = np.arange(2)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.3), gridspec_kw={"width_ratios": [1.35, 1]})

# ---- Left: tabular vs late+SimpleCNN vs late+ResNet ----
series = [
    ("Tabular only (clin+report)", "tabular_only", BLUE),
    ("Late + SimpleCNN",           "late_simplecnn", ORANGE),
    ("Late + ResNet18",            "late_resnet18", AQUA),
]
w = 0.26
for i, (name, key, color) in enumerate(series):
    means = [g(t, key)[0] for t in targets]
    stds = [g(t, key)[1] for t in targets]
    offs = x - w + w * i
    axL.bar(offs, means, w * 0.92, label=name, color=color, yerr=stds, capsize=3,
            ecolor=MUTED, error_kw={"elinewidth": 1, "alpha": 0.8})
    for off, m in zip(offs, means):
        axL.text(off, m + 0.008, f"{m:.3f}", ha="center", va="bottom", fontsize=8.3,
                 color=INK, fontweight="bold" if key != "tabular_only" else "normal")
# delta annotations vs tabular baseline
for j, t in enumerate(targets):
    tb = g(t, "tabular_only")[0]
    best = g(t, "late_simplecnn")[0]
    d = best - tb
    col = GREEN if d > 0 else RED
    axL.text(x[j], 0.775, f"{'+' if d>=0 else ''}{d:.3f}", ha="center", color=col, fontsize=9.5, fontweight="bold")
axL.axhline(0.5, color=BASE, lw=1.3, ls=(0, (5, 4)), zorder=0)
axL.set_xticks(x); axL.set_xticklabels(tlabels, fontsize=12, color=INK)
axL.set_ylabel("C-index (5-fold mean ± SD)")
axL.set_ylim(0.5, 0.82)
axL.yaxis.grid(True, color=GRID, lw=0.8); axL.set_axisbelow(True)
axL.legend(frameon=False, fontsize=9.3, loc="lower left", ncol=1)
axL.set_title("Late fusion (method B): does the image arm help?", fontsize=12.5,
              color=INK, loc="left", pad=10, fontweight="bold")
axL.text(0.5, 0.792, "Δ vs tabular →", transform=axL.get_xaxis_transform() if False else axL.transData,
         ha="center", fontsize=8, color=MUTED)

# ---- Right: image encoder alone ----
series2 = [
    ("SimpleCNN", "image_simplecnn_only", ORANGE),
    ("ResNet18 (pretrained)", "image_resnet18_only", AQUA),
]
w2 = 0.32
for i, (name, key, color) in enumerate(series2):
    means = [g(t, key)[0] for t in targets]
    stds = [g(t, key)[1] for t in targets]
    offs = x - w2/2 + w2 * i
    axR.bar(offs, means, w2 * 0.92, label=name, color=color, yerr=stds, capsize=3,
            ecolor=MUTED, error_kw={"elinewidth": 1, "alpha": 0.8})
    for off, m in zip(offs, means):
        axR.text(off, m + 0.008, f"{m:.3f}", ha="center", va="bottom", fontsize=8.5, color=INK, fontweight="bold")
axR.axhline(0.5, color=BASE, lw=1.3, ls=(0, (5, 4)), zorder=0)
axR.set_xticks(x); axR.set_xticklabels(tlabels, fontsize=12, color=INK)
axR.set_ylabel("Image-only C-index")
axR.set_ylim(0.5, 0.72)
axR.yaxis.grid(True, color=GRID, lw=0.8); axR.set_axisbelow(True)
axR.legend(frameon=False, fontsize=9.3, loc="upper right")
axR.set_title("Image encoder alone: bigger ≠ better", fontsize=12.5, color=INK, loc="left", pad=10, fontweight="bold")

fig.suptitle("Method B: tabular(clin+report) + image via CoxPH stack  —  image helps OS (SimpleCNN), not PFS",
             fontsize=13, color=INK, x=0.01, ha="left", fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(os.path.join(ASSETS, "fig6_late_fusion_B.png"), dpi=150, bbox_inches="tight")
print("wrote fig6_late_fusion_B.png")
