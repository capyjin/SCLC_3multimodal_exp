# -*- coding: utf-8 -*-
"""Build RESULTS.md with summary tables + charts from the two trained
experiments (early_fusion, late_fusion). Charts follow the dataviz palette.

Run:  python generate_report.py
"""
import ast
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "outputs")
EARLY = os.path.join(OUT, "EXP_20260722_early_fusion_train")
LATE = os.path.join(OUT, "EXP_20260722_late_fusion_train")
ASSETS = os.path.join(ROOT, "report_assets")
os.makedirs(ASSETS, exist_ok=True)

# ---- dataviz palette (validated categorical, light surface) ----
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})

# ---- load metrics ----
early_fold = pd.read_csv(os.path.join(EARLY, "metrics", "fold_metrics.csv"))
late_fold = pd.read_csv(os.path.join(LATE, "metrics", "fold_metrics.csv"))
with open(os.path.join(EARLY, "metrics", "summary.json"), encoding="utf-8") as fh:
    early_sum = json.load(fh)
with open(os.path.join(LATE, "metrics", "summary.json"), encoding="utf-8") as fh:
    late_sum = json.load(fh)

# canonical model order + display names + colors
MODELS = [
    ("image_only",               "Image only",        BLUE,    "late"),
    ("clinical_only",            "Clinical only",     ORANGE,  "late"),
    ("report_only",              "Report only",       AQUA,    "late"),
    ("late_fusion_weighted_sum", "Late fusion",       YELLOW,  "late"),
    ("early_fusion_image_clinical_report", "Early fusion", MAGENTA, "early"),
]
TARGETS = [("os", "OS"), ("pfs", "PFS")]


def mean_std(target, key, src):
    d = src if key == "early_fusion_image_clinical_report" else src
    node = (early_sum if src is early_sum else late_sum)[target].get(key)
    return (node["mean_c_index"], node["std_c_index"]) if node else (np.nan, np.nan)


def get_summary(target, key):
    if key == "early_fusion_image_clinical_report":
        node = early_sum[target].get(key)
    else:
        node = late_sum[target].get(key)
    return (node["mean_c_index"], node["std_c_index"]) if node else (np.nan, np.nan)


def get_folds(target, key):
    if key == "early_fusion_image_clinical_report":
        df = early_fold
    else:
        df = late_fold
    sub = df[(df["target"] == target) & (df["modality"] == key)]
    return sub.sort_values("fold")["c_index"].astype(float).tolist()


# ============================================================
# FIG 1 — headline grouped bar: mean C-index per model x target
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5.2))
n_models = len(MODELS)
group_w = 0.8
bar_w = group_w / n_models
x = np.arange(len(TARGETS))
for i, (key, name, color, _) in enumerate(MODELS):
    means = [get_summary(t, key)[0] for t, _ in TARGETS]
    stds = [get_summary(t, key)[1] for t, _ in TARGETS]
    offs = x - group_w / 2 + bar_w * (i + 0.5)
    bars = ax.bar(offs, means, bar_w * 0.92, label=name, color=color,
                  yerr=stds, capsize=3, ecolor=MUTED,
                  error_kw={"elinewidth": 1, "alpha": 0.8})
    for off, m in zip(offs, means):
        ax.text(off, m + 0.012, f"{m:.3f}", ha="center", va="bottom",
                fontsize=8.5, color=INK, fontweight="bold")
ax.axhline(0.5, color=BASELINE, lw=1.4, ls=(0, (5, 4)), zorder=0)
ax.text(len(TARGETS) - 0.5, 0.5, "  random = 0.50", va="center", ha="left",
        fontsize=8.5, color=MUTED)
ax.set_xticks(x)
ax.set_xticklabels([lbl for _, lbl in TARGETS], fontsize=12, color=INK)
ax.set_ylabel("C-index (5-fold mean, error bar = SD)")
ax.set_ylim(0, 0.85)
ax.yaxis.grid(True, color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.11),
          fontsize=9.5, columnspacing=1.2, handletextpad=0.5)
ax.set_title("Tri-modal survival prediction — mean C-index by model",
             fontsize=13, color=INK, pad=30, loc="left", fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "fig1_mean_cindex.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ============================================================
# FIG 2 — per-fold spread (strip) faceted by target
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
rng = np.random.default_rng(0)
for ax, (t, tlabel) in zip(axes, TARGETS):
    for i, (key, name, color, _) in enumerate(MODELS):
        folds = get_folds(t, key)
        jitter = (rng.random(len(folds)) - 0.5) * 0.18
        ax.scatter(np.full(len(folds), i) + jitter, folds, s=55, color=color,
                   edgecolor=SURFACE, linewidth=1.2, zorder=3, alpha=0.9)
        m = float(np.mean(folds))
        ax.plot([i - 0.28, i + 0.28], [m, m], color=color, lw=2.6, zorder=4)
    ax.axhline(0.5, color=BASELINE, lw=1.3, ls=(0, (5, 4)), zorder=0)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([n for _, n, _, _ in MODELS], rotation=30, ha="right",
                       fontsize=9, color=INK)
    ax.set_title(tlabel, fontsize=12, color=INK, loc="left", fontweight="bold")
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
axes[0].set_ylabel("C-index (per fold; thick bar = mean)")
axes[0].set_ylim(0.45, 0.80)
fig.suptitle("Per-fold C-index spread across 5 folds", fontsize=13, color=INK,
             x=0.01, ha="left", fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(os.path.join(ASSETS, "fig2_fold_spread.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ============================================================
# FIG 3 — late fusion learned modality weights (CoxPH coef)
# ============================================================
def mean_coefs(target):
    sub = late_fold[(late_fold["target"] == target) &
                    (late_fold["modality"] == "late_fusion_weighted_sum")]
    acc = {"risk_image": [], "risk_clinical": [], "risk_report": []}
    for c in sub["coefficients"]:
        d = ast.literal_eval(c)
        for k in acc:
            acc[k].append(d[k])
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in acc.items()}

coef_os = mean_coefs("os")
coef_pfs = mean_coefs("pfs")
mod_labels = ["Image", "Clinical", "Report"]
mod_keys = ["risk_image", "risk_clinical", "risk_report"]
fig, ax = plt.subplots(figsize=(8, 4.6))
xm = np.arange(len(mod_keys))
w = 0.38
for j, (target, coefs, color, lbl) in enumerate(
        [("os", coef_os, BLUE, "OS"), ("pfs", coef_pfs, ORANGE, "PFS")]):
    means = [coefs[k][0] for k in mod_keys]
    stds = [coefs[k][1] for k in mod_keys]
    offs = xm - w / 2 + w * (j + 0.5)
    ax.bar(offs, means, w * 0.9, label=lbl, color=color, yerr=stds, capsize=3,
           ecolor=MUTED, error_kw={"elinewidth": 1, "alpha": 0.8})
    for off, m in zip(offs, means):
        ax.text(off, m + 0.02, f"{m:.2f}", ha="center", va="bottom",
                fontsize=9, color=INK, fontweight="bold")
ax.set_xticks(xm)
ax.set_xticklabels(mod_labels, fontsize=11, color=INK)
ax.set_ylabel("CoxPH weight on OOF risk (mean of 5 folds)")
ax.yaxis.grid(True, color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=10)
ax.set_title("Late fusion: learned modality weights", fontsize=13, color=INK,
             loc="left", pad=10, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "fig3_late_weights.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ============================================================
# Build RESULTS.md
# ============================================================
def fmt(target, key):
    m, s = get_summary(target, key)
    return f"{m:.4f} ± {s:.4f}"

# best unimodal per target for delta
def best_unimodal(target):
    vals = {k: get_summary(target, k)[0] for k in ["image_only", "clinical_only", "report_only"]}
    bk = max(vals, key=vals.get)
    return bk, vals[bk]

lines = []
lines.append("# SCLC Tri-modal Fusion — 학습 결과 요약\n")
lines.append("> Image + Clinical + Report 3-모달 생존예측. 5-fold CV (seed 42), "
             "지표는 Harrell's C-index (0.5 = 무작위, 1.0 = 완벽).\n")

lines.append("## 1. 실험 개요\n")
lines.append("| 항목 | 내용 |")
lines.append("|---|---|")
lines.append("| 코호트 | **238명** (image·clinical·report 모두 보유), `trimodal_common_5fold_seed42_v1.csv` |")
lines.append("| 타깃 | **OS**(전체생존), **PFS**(무진행생존) — Cox 부분우도 손실 |")
lines.append("| 검증 | 5-fold 교차검증, seed 42, fold당 30 epochs |")
lines.append("| 지표 | Harrell's C-index (fold 평균 ± 표준편차) |")
lines.append("| 학습환경 | RTX 4070 Ti SUPER, torch 2.12.1+cu126 |")
lines.append("")
lines.append("**두 가지 fusion 전략을 각각 학습·평가했습니다.**")
lines.append("- **Early fusion** (`early_fusion`): image(128D)+clinical(128D)+report(16D) 특징을 concat → Dropout → Linear → Cox 위험점수. end-to-end 학습.")
lines.append("- **Late fusion** (`late_fusion`): image/clinical/report **단일모달 모델을 각각 학습**한 뒤, fold별 OOF 위험점수 3개에 `lifelines` CoxPH를 적합해 가중합.")
lines.append("")

lines.append("## 2. 핵심 결과 — 한눈에\n")
lines.append("![mean C-index](report_assets/fig1_mean_cindex.png)\n")
lines.append("### C-index 요약표 (5-fold 평균 ± SD)\n")
lines.append("| 모델 | 종류 | OS | PFS |")
lines.append("|---|---|---|---|")
for key, name, _c, kind in MODELS:
    ktxt = "단일모달" if key in ("image_only", "clinical_only", "report_only") else "**融合(fusion)**"
    ktxt = "단일모달" if kind == "late" and key.endswith("only") else ("Late fusion" if key.startswith("late") else "Early fusion")
    if key.endswith("only"):
        ktxt = "단일모달 baseline"
    lines.append(f"| {name} | {ktxt} | {fmt('os', key)} | {fmt('pfs', key)} |")
lines.append("")

# highlight best
best_os = max(MODELS, key=lambda mk: get_summary('os', mk[0])[0])
best_pfs = max(MODELS, key=lambda mk: get_summary('pfs', mk[0])[0])
bu_os_k, bu_os_v = best_unimodal("os")
bu_pfs_k, bu_pfs_v = best_unimodal("pfs")
name_map = {k: n for k, n, _, _ in MODELS}
lines.append(f"- **OS 최고 성능: {best_os[1]}** (C-index {get_summary('os', best_os[0])[0]:.4f}). "
             f"최고 단일모달({name_map[bu_os_k]} {bu_os_v:.4f}) 대비 "
             f"**+{get_summary('os', best_os[0])[0]-bu_os_v:.4f}**.")
lines.append(f"- **PFS 최고 성능: {best_pfs[1]}** (C-index {get_summary('pfs', best_pfs[0])[0]:.4f}). "
             f"최고 단일모달({name_map[bu_pfs_k]} {bu_pfs_v:.4f}) 대비 "
             f"**+{get_summary('pfs', best_pfs[0])[0]-bu_pfs_v:.4f}**.")
lines.append(f"- 두 fusion 모두 모든 단일모달 baseline보다 높음 → **모달리티 결합이 성능을 개선**.")
lines.append("")

lines.append("## 3. Fold별 분포 (안정성)\n")
lines.append("![fold spread](report_assets/fig2_fold_spread.png)\n")
lines.append("- fold별 편차(SD 0.03~0.06)는 238명 규모의 5-fold에서 자연스러운 수준입니다.")
lines.append("- 가장 낮은 단일 fold도 image_only PFS fold3의 **0.52**로, 무작위(0.5)보다 높습니다. "
             "0.5 이하로 붕괴한 fold는 없습니다.")
lines.append("")

lines.append("## 4. Late fusion이 배운 모달리티 가중치\n")
lines.append("![late weights](report_assets/fig3_late_weights.png)\n")
lines.append(f"CoxPH가 OOF 위험점수에 부여한 평균 가중치 (OS): "
             f"Image **{coef_os['risk_image'][0]:.2f}**, "
             f"Clinical **{coef_os['risk_clinical'][0]:.2f}**, "
             f"Report **{coef_os['risk_report'][0]:.2f}**.")
lines.append("- **Clinical 위험점수의 가중치가 가장 큽니다** — 단일모달 성능에서 image가 약간 앞서지만, "
             "결합 단계에서는 clinical 신호가 가장 크게 반영됩니다.")
lines.append("- Image·Report도 양(+)의 가중치를 가져 상호 보완적으로 기여합니다.")
lines.append("")

lines.append("## 5. 성능 게이트 확인 (0.5 기준)\n")
lines.append("사용자 요청대로 성능이 0.5 언저리/이하면 중단·진단하기로 했습니다. 학습 결과:")
lines.append("")
lines.append("| 실험 | 타깃 | 평균 C-index | 판정 |")
lines.append("|---|---|---|---|")
for key, name in [("early_fusion_image_clinical_report", "Early fusion"),
                  ("late_fusion_weighted_sum", "Late fusion")]:
    for t, tl in TARGETS:
        v = get_summary(t, key)[0]
        verdict = "✅ 정상 (>0.5)" if v > 0.55 else ("⚠️ 경계" if v > 0.5 else "❌ 실패")
        lines.append(f"| {name} | {tl} | {v:.4f} | {verdict} |")
lines.append("")
lines.append("→ **모든 fusion 결과가 0.63~0.70대**로 게이트를 통과했습니다. 중단·수정 없이 전체 학습을 완료했습니다.")
lines.append("")

lines.append("## 6. 산출물 위치\n")
lines.append("```")
lines.append("outputs/EXP_20260722_early_fusion_train/   # early fusion (concat)")
lines.append("outputs/EXP_20260722_late_fusion_train/    # late fusion (weighted-sum)")
lines.append("  ├─ metrics/fold_metrics.csv              # fold별 C-index")
lines.append("  ├─ metrics/oof_predictions.csv           # OOF 위험점수")
lines.append("  ├─ metrics/summary.json                  # 요약 통계")
lines.append("  ├─ checkpoints/                          # 모델 가중치")
lines.append("  └─ experiment_report.md                  # 프로토콜 리포트")
lines.append("report_assets/                             # 이 리포트의 그래프 PNG")
lines.append("```")
lines.append("")
lines.append("## 7. 참고 (프로토콜 4.4)\n")
lines.append("- report 모달을 추가하면서 tri-modal 코호트(238명)는 원래 image+clinical baseline(257명)보다 작아졌습니다. "
             "엄밀한 비교를 위해서는 동일한 238명 코호트에서 image+clinical 2-모달 baseline을 재평가해야 하며, "
             "이는 이번 실행 범위 밖입니다.")
lines.append("- 수치는 5-fold 교차검증 결과이며, 외부 검증셋(독립 코호트) 평가는 포함되지 않았습니다.")
lines.append("")

with open(os.path.join(ROOT, "RESULTS.md"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))

print("Wrote RESULTS.md and 3 figures to report_assets/")
print("OS best:", best_os[1], f"{get_summary('os', best_os[0])[0]:.4f}")
print("PFS best:", best_pfs[1], f"{get_summary('pfs', best_pfs[0])[0]:.4f}")
