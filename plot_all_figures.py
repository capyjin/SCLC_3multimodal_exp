# -*- coding: utf-8 -*-
"""All report figures (fig4, fig5, fig6) in one place. Each figure is its own
function; color palette, output paths, and the shared OS ablation numbers
(fig4 and fig5 both plot the same "original regime" results) are defined once
at module level instead of being copy-pasted per script.

Run:  python plot_all_figures.py                 # draw all figures
      python plot_all_figures.py --only fig4,fig6 # draw a subset
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ── 공통 경로 ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(ROOT, "report_assets")
os.makedirs(ASSETS, exist_ok=True)

# ── 공통 색상 팔레트 (예전엔 파일마다 따로 정의돼 있던 것을 하나로 합침) ──────
BLUE, ORANGE, MAGENTA, AQUA = "#2a78d6", "#eb6834", "#e87ba4", "#1baf7a"
GREEN, RED = "#0ca30c", "#d03b3b"
SURFACE, INK, INK2, MUTED, GRID, BASE = \
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASE, "text.color": INK, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False, "ytick.color": MUTED, "xtick.color": MUTED,
})

# 한글 라벨이 네모(□)로 깨지지 않도록 CJK 폰트를 '대체 후보'로 뒤에 붙인다.
# 기본 폰트(DejaVu Sans)를 맨 앞에 두므로 영문 그림(fig4~6)의 모양은 그대로 유지되고,
# DejaVu에 없는 한글 글자만 뒤 폰트로 넘어간다(matplotlib 3.6+ font fallback).
_installed = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
_ko_fonts = [n for n in ("Noto Sans CJK KR", "NanumGothic", "Malgun Gothic", "AppleGothic")
             if n in _installed]
plt.rcParams["font.family"] = ["DejaVu Sans"] + _ko_fonts
plt.rcParams["axes.unicode_minus"] = False  # 한글 폰트에서 마이너스 기호가 깨지는 것 방지

# ── 공통 데이터: OS 5-fold ablation 결과 ────────────────────────────────────
# fig4(모달리티 조합별 성능)와 fig5(원래 레짐 vs 개선 레짐)가 "원래 레짐"
# 수치를 그대로 공유하므로, 예전처럼 두 파일에 똑같은 숫자를 따로 박아두지
# 않고 여기 한 곳에서만 정의한다. 키는 ablation.py의 CONFIGS 이름과 맞춤.
ABLATION_OS = {
    "report_only": dict(n_modalities=1, orig=(0.6233, 0.0200), improved=(0.6268, 0.0474)),
    "clin_only":   dict(n_modalities=1, orig=(0.6389, 0.0373), improved=(0.6466, 0.0407)),
    "image_only":  dict(n_modalities=1, orig=(0.6469, 0.0320), improved=(0.6388, 0.0468)),
    "clin_image":  dict(n_modalities=2, orig=(0.6786, 0.0446), improved=(0.6774, 0.0418)),
    "clin_report": dict(n_modalities=2, orig=(0.6870, 0.0525), improved=(0.7076, 0.0472)),
    "all":         dict(n_modalities=3, orig=(0.6949, 0.0464), improved=(0.6775, 0.0372)),
}
ABLATION_ORDER = ["report_only", "clin_only", "image_only", "clin_image", "clin_report", "all"]


def _save(fig, filename):
    """그림을 report_assets/에 저장하고 로그를 남긴 뒤 닫는다 (세 그림이 공통으로 씀)."""
    fig.savefig(os.path.join(ASSETS, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote report_assets/{filename}")


# ════════════════════════════════════════════════════════════════════════════
# fig4: ablation — OS C-index vs 모달리티 조합 (같은 238-split, 같은 학습 루프)
# 모달리티를 추가해도 성능이 떨어지지 않는지(단조 증가) 보여준다.
# ════════════════════════════════════════════════════════════════════════════
FIG4_LABELS = {
    "report_only": "Report only",
    "clin_only": "Clinical only",
    "image_only": "Image only",
    "clin_image": "Clinical + Image",
    "clin_report": "Clinical + Report",
    "all": "Image+Clinical+Report",
}


def plot_fig4_ablation():
    cmap = {1: BLUE, 2: ORANGE, 3: MAGENTA}
    keys = ABLATION_ORDER
    labels = [FIG4_LABELS[k] for k in keys]
    means = [ABLATION_OS[k]["orig"][0] for k in keys]
    stds = [ABLATION_OS[k]["orig"][1] for k in keys]
    colors = [cmap[ABLATION_OS[k]["n_modalities"]] for k in keys]
    y = range(len(keys))

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.barh(list(y), means, xerr=stds, color=colors, height=0.68,
            error_kw={"elinewidth": 1.2, "ecolor": MUTED, "capsize": 3}, zorder=3)
    for yi, m, s in zip(y, means, stds):
        ax.text(m + s + 0.004, yi, f"{m:.3f}", va="center", ha="left", fontsize=9.5,
                color=INK, fontweight="bold")
    # 기준선: 랜덤(0.5)과 예전 2-modal 파이프라인 점수(0.708)
    ax.axvline(0.5, color=BASE, lw=1.4, ls=(0, (5, 4)), zorder=1)
    ax.text(0.5, len(keys) - 0.35, " random 0.50", color=MUTED, fontsize=8.5, va="bottom", ha="left")
    ax.axvline(0.7083, color=GREEN, lw=1.6, ls=(0, (4, 3)), zorder=1)
    ax.text(0.7083, -0.75, "prior 2-modal\n0.708 (diff pipeline)", color=GREEN, fontsize=8.5,
            va="bottom", ha="center")

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=10.5, color=INK)
    ax.set_xlim(0.5, 0.80)
    ax.set_xlabel("OS C-index (5-fold mean ± SD) — same 238 split, same training loop")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)

    handles = [Patch(color=BLUE, label="1 modality"), Patch(color=ORANGE, label="2 modalities"),
               Patch(color=MAGENTA, label="3 modalities (tri-modal)")]
    ax.legend(handles=handles, frameon=False, fontsize=9.5, loc="lower right")
    ax.set_title("Ablation: does adding a modality hurt?  (No — C-index climbs monotonically)",
                 fontsize=12.5, color=INK, loc="left", pad=10, fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig4_ablation_os.png")


# ════════════════════════════════════════════════════════════════════════════
# fig5: 레짐 비교 — 원래(batch16/30ep) vs 개선(batch32/60ep) 학습 설정
# batch=16일 때 Cox loss가 불리했던 문제를 고치면 tabular는 오르지만,
# tri-modal(영상 포함)은 오히려 내려가는 걸 보여준다.
# ════════════════════════════════════════════════════════════════════════════
FIG5_LABELS = {
    "report_only": "Report only",
    "clin_only": "Clinical only",
    "image_only": "Image only",
    "clin_image": "Clinical + Image",
    "clin_report": "Clinical + Report",
    "all": "Tri-modal (all 3)",
}


def plot_fig5_regime_comparison():
    keys = ABLATION_ORDER
    labels = [FIG5_LABELS[k] for k in keys]
    orig_m = [ABLATION_OS[k]["orig"][0] for k in keys]
    orig_s = [ABLATION_OS[k]["orig"][1] for k in keys]
    imp_m = [ABLATION_OS[k]["improved"][0] for k in keys]
    imp_s = [ABLATION_OS[k]["improved"][1] for k in keys]
    x = np.arange(len(labels))
    w = 0.38

    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.bar(x - w / 2, orig_m, w, yerr=orig_s, label="Original  (batch 16, 30 ep)",
           color=BLUE, capsize=3, ecolor=MUTED, error_kw={"elinewidth": 1, "alpha": 0.8})
    ax.bar(x + w / 2, imp_m, w, yerr=imp_s, label="Improved  (batch 32, 60 ep)",
           color=ORANGE, capsize=3, ecolor=MUTED, error_kw={"elinewidth": 1, "alpha": 0.8})
    for xi, m in zip(x - w / 2, orig_m):
        ax.text(xi, m + 0.012, f"{m:.3f}", ha="center", va="bottom", fontsize=8, color=INK2)
    for xi, m in zip(x + w / 2, imp_m):
        ax.text(xi, m + 0.012, f"{m:.3f}", ha="center", va="bottom", fontsize=8, color=INK, fontweight="bold")
    ax.axhline(0.7083, color=GREEN, lw=1.6, ls=(0, (4, 3)), zorder=1)
    ax.text(len(labels) - 0.5, 0.7083, "  prior clin+report 0.708", va="bottom", ha="right", fontsize=8.5, color=GREEN)
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

    # 핵심 변화 두 가지(clin+report 상승, tri-modal 하락)를 화살표로 강조.
    # 델타 값은 하드코딩하지 않고 위 데이터에서 직접 계산한다.
    i_cr, i_all = keys.index("clin_report"), keys.index("all")
    ax.annotate("", xy=(i_cr + w / 2, imp_m[i_cr] + 0.055), xytext=(i_cr - w / 2, orig_m[i_cr] + 0.06),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.8))
    ax.text(i_cr, 0.775, f"{imp_m[i_cr] - orig_m[i_cr]:+.3f}", color=GREEN, fontsize=9, ha="center", fontweight="bold")
    ax.annotate("", xy=(i_all + w / 2, imp_m[i_all] + 0.048), xytext=(i_all - w / 2, orig_m[i_all] + 0.055),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.8))
    ax.text(i_all, 0.775, f"{imp_m[i_all] - orig_m[i_all]:+.3f}", color=RED, fontsize=9, ha="center", fontweight="bold")

    fig.tight_layout()
    _save(fig, "fig5_regime_compare_os.png")


# ════════════════════════════════════════════════════════════════════════════
# fig6: method-B late fusion. 왼쪽 = late fusion vs tabular baseline (영상
# 팔이 실제로 도움이 되는가). 오른쪽 = 이미지 인코더 단독 비교
# (SimpleCNN vs 사전학습 ResNet18).
# ════════════════════════════════════════════════════════════════════════════
def plot_fig6_late_fusion_b():
    with open(os.path.join(ROOT, "outputs", "late_fusion_B", "results.json")) as f:
        R = json.load(f)

    def g(t, k):
        d = R[t][k]
        return d["mean"], d["std"]

    targets = ["os", "pfs"]
    tlabels = ["OS", "PFS"]
    x = np.arange(2)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.3), gridspec_kw={"width_ratios": [1.35, 1]})

    # ---- 왼쪽: tabular vs late+SimpleCNN vs late+ResNet ----
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
    # tabular 대비 델타 표시
    for j, t in enumerate(targets):
        tb = g(t, "tabular_only")[0]
        best = g(t, "late_simplecnn")[0]
        d = best - tb
        col = GREEN if d > 0 else RED
        axL.text(x[j], 0.775, f"{'+' if d >= 0 else ''}{d:.3f}", ha="center", color=col, fontsize=9.5, fontweight="bold")
    axL.axhline(0.5, color=BASE, lw=1.3, ls=(0, (5, 4)), zorder=0)
    axL.set_xticks(x)
    axL.set_xticklabels(tlabels, fontsize=12, color=INK)
    axL.set_ylabel("C-index (5-fold mean ± SD)")
    axL.set_ylim(0.5, 0.82)
    axL.yaxis.grid(True, color=GRID, lw=0.8)
    axL.set_axisbelow(True)
    axL.legend(frameon=False, fontsize=9.3, loc="lower left", ncol=1)
    axL.set_title("Late fusion (method B): does the image arm help?", fontsize=12.5,
                  color=INK, loc="left", pad=10, fontweight="bold")
    axL.text(0.5, 0.792, "Δ vs tabular →", transform=axL.transData, ha="center", fontsize=8, color=MUTED)

    # ---- 오른쪽: 이미지 인코더 단독 비교 ----
    series2 = [
        ("SimpleCNN", "image_simplecnn_only", ORANGE),
        ("ResNet18 (pretrained)", "image_resnet18_only", AQUA),
    ]
    w2 = 0.32
    for i, (name, key, color) in enumerate(series2):
        means = [g(t, key)[0] for t in targets]
        stds = [g(t, key)[1] for t in targets]
        offs = x - w2 / 2 + w2 * i
        axR.bar(offs, means, w2 * 0.92, label=name, color=color, yerr=stds, capsize=3,
                ecolor=MUTED, error_kw={"elinewidth": 1, "alpha": 0.8})
        for off, m in zip(offs, means):
            axR.text(off, m + 0.008, f"{m:.3f}", ha="center", va="bottom", fontsize=8.5, color=INK, fontweight="bold")
    axR.axhline(0.5, color=BASE, lw=1.3, ls=(0, (5, 4)), zorder=0)
    axR.set_xticks(x)
    axR.set_xticklabels(tlabels, fontsize=12, color=INK)
    axR.set_ylabel("Image-only C-index")
    axR.set_ylim(0.5, 0.72)
    axR.yaxis.grid(True, color=GRID, lw=0.8)
    axR.set_axisbelow(True)
    axR.legend(frameon=False, fontsize=9.3, loc="upper right")
    axR.set_title("Image encoder alone: bigger ≠ better", fontsize=12.5, color=INK, loc="left", pad=10, fontweight="bold")

    fig.suptitle("Method B: tabular(clin+report) + image via CoxPH stack  —  image helps OS (SimpleCNN), not PFS",
                 fontsize=13, color=INK, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "fig6_late_fusion_B.png")


# ════════════════════════════════════════════════════════════════════════════
# fig7: OS vs PFS 모달리티 사다리 (같은 조건 bs32/ep60, ablation.py 결과 JSON)
# "모달리티를 1개 → 2개 → 3개로 늘릴 때 OS는 오르는데 PFS는 어디서 꺾이나?"
# ════════════════════════════════════════════════════════════════════════════
WHY_PFS_DIR = os.path.join(ROOT, "outputs", "ablation_why_pfs")

# 사다리 순서: 모달리티 1개 → 2개 → 3개
LADDER = ["report_only", "clin_only", "image_only", "clin_image", "clin_report", "all"]
LADDER_LABELS = {
    "report_only": "Report",
    "clin_only": "Clinical",
    "image_only": "Image",
    "clin_image": "Clin+Image",
    "clin_report": "Clin+Report",
    "all": "Clin+Report+Image",
}
N_MODALITIES = {"report_only": 1, "clin_only": 1, "image_only": 1,
                "clin_image": 2, "clin_report": 2, "all": 3}


def _load_why_pfs(target):
    """ablation.py가 저장한 results_{target}.json을 읽어온다."""
    with open(os.path.join(WHY_PFS_DIR, f"results_{target}.json")) as f:
        return json.load(f)["configs"]


def plot_fig7_os_vs_pfs_ladder():
    os_cfg, pfs_cfg = _load_why_pfs("os"), _load_why_pfs("pfs")
    keys = [k for k in LADDER if k in os_cfg and k in pfs_cfg]
    labels = [LADDER_LABELS[k] for k in keys]
    x = np.arange(len(keys))
    w = 0.38

    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    for i, (cfg, name, color) in enumerate([(os_cfg, "OS (전체생존)", BLUE),
                                            (pfs_cfg, "PFS (무진행생존)", ORANGE)]):
        means = [cfg[k]["mean"] for k in keys]
        stds = [cfg[k]["std"] for k in keys]
        offs = x - w / 2 + w * i
        ax.bar(offs, means, w, yerr=stds, label=name, color=color, capsize=3,
               ecolor=MUTED, error_kw={"elinewidth": 1, "alpha": 0.8})
        for off, m in zip(offs, means):
            ax.text(off, m + 0.010, f"{m:.3f}", ha="center", va="bottom", fontsize=8.2, color=INK)

    # 2모달(clin+report) → 3모달(all)로 갈 때의 변화를 화살표로 강조:
    # OS는 오르고 PFS는 내려가는 것이 이 그림의 핵심 메시지.
    if "clin_report" in keys and "all" in keys:
        i_cr, i_all = keys.index("clin_report"), keys.index("all")
        for i, (cfg, sign_color_up) in enumerate([(os_cfg, True), (pfs_cfg, False)]):
            d = cfg["all"]["mean"] - cfg["clin_report"]["mean"]
            xpos = (i_cr + i_all) / 2 - w / 2 + w * i
            ax.text(xpos, 0.80, f"{d:+.3f}", ha="center", fontsize=10,
                    color=GREEN if d > 0 else RED, fontweight="bold")
    ax.text((len(keys) - 1.5), 0.822, "이미지 추가 효과 (Clin+Report → 3-modal)",
            ha="center", fontsize=8.5, color=MUTED)

    ax.axhline(0.5, color=BASE, lw=1.3, ls=(0, (5, 4)), zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, color=INK)
    ax.set_ylabel("C-index (5-fold mean ± SD)")
    ax.set_ylim(0.55, 0.84)   # 0.5 기준선은 화면 밖 — 막대 차이를 크게 보여주기 위함
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.set_title("모달리티 사다리: concat fusion에서는 이미지를 더하면 OS·PFS 모두 떨어진다"
                 "  (같은 split·같은 학습조건 bs32/ep60)",
                 fontsize=12.5, color=INK, loc="left", pad=10, fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig7_os_vs_pfs_ladder.png")


# ════════════════════════════════════════════════════════════════════════════
# fig8: 원인 진단 — "발언권 vs 실력"의 불일치
# 왼쪽 = 각 모달리티가 최종 위험점수를 흔드는 비중(analyze_contribution.py 결과).
# 오른쪽 = 그 모달리티를 단독으로 썼을 때의 실제 성능.
# 이미지는 발언권이 70% 넘는데 단독 실력은 꼴찌 → 좋은 모달리티가 묻힌다.
# ════════════════════════════════════════════════════════════════════════════
MODAL_COLOR = {"image": MAGENTA, "clinical": BLUE, "report": ORANGE}
MODAL_KO = {"image": "Image", "clinical": "Clinical", "report": "Report"}


def _load_contribution(target):
    with open(os.path.join(WHY_PFS_DIR, f"contribution_{target}.json")) as f:
        return json.load(f)


def plot_fig8_contribution_mismatch():
    contrib = {t: _load_contribution(t) for t in ("os", "pfs")}
    cfgs = {t: _load_why_pfs(t) for t in ("os", "pfs")}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.4), gridspec_kw={"width_ratios": [1.3, 1]})

    # ---- 왼쪽: 기여도 비중 누적 막대 ----
    # 막대 4개: (PFS, clin_report) (PFS, all) (OS, clin_report) (OS, all)
    bars = [("pfs", "clin_report"), ("pfs", "all"), ("os", "clin_report"), ("os", "all")]
    bar_labels = ["PFS\nClin+Report", "PFS\n+Image", "OS\nClin+Report", "OS\n+Image"]
    x = np.arange(len(bars))
    for xi, (t, cfg) in zip(x, bars):
        d = contrib[t][cfg]
        total = sum(d.values())
        bottom = 0.0
        for mod in ("clinical", "report", "image"):   # 아래부터 쌓는 순서
            if mod not in d:
                continue
            share = d[mod] / total * 100
            axL.bar(xi, share, 0.6, bottom=bottom, color=MODAL_COLOR[mod],
                    label=MODAL_KO[mod] if xi == 0 or (mod == "image" and xi == 1) else None)
            if share > 6:
                axL.text(xi, bottom + share / 2, f"{share:.0f}%", ha="center", va="center",
                         fontsize=10, color="white", fontweight="bold")
            bottom += share
    axL.set_xticks(x)
    axL.set_xticklabels(bar_labels, fontsize=9.8, color=INK)
    axL.set_ylabel("최종 위험점수를 흔드는 비중 (%)")
    axL.set_ylim(0, 100)
    axL.legend(frameon=False, fontsize=9.5, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, -0.10))
    axL.set_title("① 이미지를 넣는 순간 발언권의 70% 이상을 가져간다",
                  fontsize=12, color=INK, loc="left", pad=10, fontweight="bold")
    axL.yaxis.grid(True, color=GRID, lw=0.8)
    axL.set_axisbelow(True)

    # ---- 오른쪽: 단독 성능(실력) ----
    singles = [("image_only", "image"), ("clin_only", "clinical"), ("report_only", "report")]
    x2 = np.arange(len(singles))
    w = 0.38
    for i, t in enumerate(("pfs", "os")):
        means = [cfgs[t][k]["mean"] for k, _ in singles]
        stds = [cfgs[t][k]["std"] for k, _ in singles]
        offs = x2 - w / 2 + w * i
        axR.bar(offs, means, w, yerr=stds, capsize=3, ecolor=MUTED,
                color=ORANGE if t == "pfs" else BLUE, label=t.upper(),
                error_kw={"elinewidth": 1, "alpha": 0.8})
        for off, m in zip(offs, means):
            axR.text(off, m + 0.008, f"{m:.3f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    axR.axhline(0.5, color=BASE, lw=1.3, ls=(0, (5, 4)), zorder=0)
    axR.set_xticks(x2)
    axR.set_xticklabels([MODAL_KO[m] for _, m in singles], fontsize=10.5, color=INK)
    axR.set_ylabel("단독 사용 시 C-index")
    axR.set_ylim(0.5, 0.72)
    axR.yaxis.grid(True, color=GRID, lw=0.8)
    axR.set_axisbelow(True)
    axR.legend(frameon=False, fontsize=10, loc="upper right")
    axR.set_title("② 그런데 단독 실력은 셋 다 비슷하다 (0.61~0.65)",
                  fontsize=12, color=INK, loc="left", pad=10, fontweight="bold")

    fig.suptitle("원인: '발언권'과 '실력'의 불일치 — 실력이 비슷한데 이미지만 결정을 지배한다",
                 fontsize=13, color=INK, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, "fig8_contribution_mismatch.png")


# ════════════════════════════════════════════════════════════════════════════
# fig9: 같은 이미지 행동, 다른 결과 (§9.4.1)
# 이미지는 clinical만 있을 때도(2-modal) clin+report 있을 때(3-modal)도
# 발언권을 80%→70%대로 똑같이 독점한다. 다른 건 "누구를 밀어냈는지"뿐.
# 왼쪽 = 발언권 비교, 오른쪽 = 그 결과로 성능이 오르는지/내리는지.
# ════════════════════════════════════════════════════════════════════════════
def plot_fig9_same_image_different_outcome():
    contrib = {t: _load_contribution(t) for t in ("os", "pfs")}
    cfgs = {t: _load_why_pfs(t) for t in ("os", "pfs")}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.4), gridspec_kw={"width_ratios": [1.15, 1]})

    # ---- 왼쪽: 이미지가 가져가는 발언권 (2-modal vs 3-modal, 항상 지배적) ----
    pairs = [("pfs", "clin_image", "PFS\nClin+Image"), ("pfs", "all", "PFS\nClin+Report+Image"),
             ("os", "clin_image", "OS\nClin+Image"), ("os", "all", "OS\nClin+Report+Image")]
    x = np.arange(len(pairs))
    shares = []
    for t, cfg, _ in pairs:
        d = contrib[t][cfg]
        shares.append(d["image"] / sum(d.values()) * 100)
    axL.bar(x, shares, 0.55, color=MAGENTA)
    for xi, s in zip(x, shares):
        axL.text(xi, s + 1.5, f"{s:.0f}%", ha="center", va="bottom", fontsize=11, color=INK, fontweight="bold")
    axL.set_ylim(0, 95)
    axL.set_xticks(x)
    axL.set_xticklabels([lbl for _, _, lbl in pairs], fontsize=9.5, color=INK)
    axL.set_ylabel("이미지가 가져가는 발언권 (%)")
    axL.yaxis.grid(True, color=GRID, lw=0.8)
    axL.set_axisbelow(True)
    axL.set_title("① 이미지의 '행동'은 파트너와 무관하게 항상 같다 (70~80% 독점)",
                  fontsize=12, color=INK, loc="left", pad=10, fontweight="bold")

    # ---- 오른쪽: 그런데 결과(순변화)는 정반대 ----
    deltas = [("PFS", cfgs["pfs"]["clin_image"]["mean"] - cfgs["pfs"]["clin_only"]["mean"], "Clinical\n+Image"),
              ("PFS", cfgs["pfs"]["all"]["mean"] - cfgs["pfs"]["clin_report"]["mean"], "Clin+Report\n+Image"),
              ("OS", cfgs["os"]["clin_image"]["mean"] - cfgs["os"]["clin_only"]["mean"], "Clinical\n+Image"),
              ("OS", cfgs["os"]["all"]["mean"] - cfgs["os"]["clin_report"]["mean"], "Clin+Report\n+Image")]
    x2 = np.arange(len(deltas))
    colors2 = [GREEN if d > 0 else RED for _, d, _ in deltas]
    axR.bar(x2, [d for _, d, _ in deltas], 0.55, color=colors2)
    for xi, (_, d, _) in zip(x2, deltas):
        axR.text(xi, d + (0.002 if d >= 0 else -0.004), f"{d:+.3f}", ha="center",
                 va="bottom" if d >= 0 else "top", fontsize=10.5, color=INK, fontweight="bold")
    axR.axhline(0, color=BASE, lw=1.3)
    axR.set_xticks(x2)
    axR.set_xticklabels([f"{t}\n{lbl}" for t, _, lbl in deltas], fontsize=9.5, color=INK)
    axR.set_ylabel("이미지 추가로 인한 C-index 변화")
    axR.yaxis.grid(True, color=GRID, lw=0.8)
    axR.set_axisbelow(True)
    axR.set_title("② 그런데 결과는 정반대 — 밀려난 파트너가 원래 셌는지가 갈랐다",
                  fontsize=12, color=INK, loc="left", pad=10, fontweight="bold")

    fig.suptitle("같은 이미지, 다른 결과: 평범한 clinical을 밀어내면 이득, 잘하던 clin+report 팀을 밀어내면 손해",
                 fontsize=12.8, color=INK, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, "fig9_same_image_different_outcome.png")


# ════════════════════════════════════════════════════════════════════════════
# fig10: late fusion에서 왜 OS만 이득인가 (§10, 교수님 질문)
# 왼쪽 = CoxPH가 이미지에 준 가중치(beta_img)를 fold별로 신뢰구간과 함께 표시.
#        OS는 5개 fold 전부 0보다 위, PFS는 0을 걸치며 부호까지 뒤집힘.
# 오른쪽 = 이미지 점수를 '난수'로 바꾼 대조군과 비교.
#        OS는 난수가 절대 못 따라오고, PFS는 난수와 구분이 안 됨.
# ════════════════════════════════════════════════════════════════════════════
def _load_pfs_diagnosis():
    with open(os.path.join(ROOT, "outputs", "late_fusion_B", "pfs_diagnosis.json")) as f:
        return json.load(f)


def plot_fig10_late_fusion_why():
    d = _load_pfs_diagnosis()

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.4), gridspec_kw={"width_ratios": [1.15, 1]})

    # ---- 왼쪽: fold별 beta_img (forest plot) ----
    # 위쪽 5줄 = OS, 아래쪽 5줄 = PFS
    ypos, labels_y, colors_y = [], [], []
    y = 0
    for t, color in (("os", BLUE), ("pfs", ORANGE)):
        for row in d[t]["beta_img_per_fold"]:
            ypos.append(y); labels_y.append(f"{t.upper()} fold{row['fold']}")
            colors_y.append(color); y += 1
        y += 0.8   # 그룹 사이 간격

    i = 0
    for t, color in (("os", BLUE), ("pfs", ORANGE)):
        for row in d[t]["beta_img_per_fold"]:
            axL.plot([row["lo"], row["hi"]], [ypos[i], ypos[i]], color=color, lw=2.2, alpha=0.75)
            axL.plot(row["coef"], ypos[i], "o", color=color, ms=7,
                     markeredgecolor="white", markeredgewidth=1.2)
            i += 1
    axL.axvline(0, color=RED, lw=1.6, ls=(0, (4, 3)), zorder=0)
    # y축을 뒤집으므로(invert_yaxis) 화면 맨 위는 min(ypos)에 해당한다.
    axL.text(0.02, min(ypos) - 0.75, " ← β=0 : 이미지가 기여 없음", color=RED,
             fontsize=9.5, va="center", ha="left", fontweight="bold")
    axL.set_yticks(ypos)
    axL.set_yticklabels(labels_y, fontsize=9.5, color=INK)
    axL.invert_yaxis()
    axL.set_xlabel("CoxPH가 이미지에 준 가중치 β  (95% 신뢰구간)")
    axL.xaxis.grid(True, color=GRID, lw=0.8)
    axL.set_axisbelow(True)
    axL.set_title("① OS는 5/5 fold 모두 0보다 위 · PFS는 0을 걸치고 부호까지 뒤집힘",
                  fontsize=11.5, color=INK, loc="left", pad=10, fontweight="bold")

    # ---- 오른쪽: 난수 대조군 ----
    x = np.arange(2)
    w = 0.26
    for j, t in enumerate(("os", "pfs")):
        r = d[t]
        sh = r["stack_shuffled_image"]
        # 막대 3개: tabular 단독 / 진짜 이미지 / 난수 이미지
        vals = [r["stack_tabular_only"], r["stack_real_image"], sh["mean"]]
        cols = [BASE, BLUE if t == "os" else ORANGE, MUTED]
        names = ["tabular 단독", "＋진짜 이미지", "＋난수 이미지"]
        for k, (v, c) in enumerate(zip(vals, cols)):
            off = x[j] - w + w * k
            axR.bar(off, v, w * 0.9, color=c,
                    label=names[k] if j == 0 else None)
            axR.text(off, v + 0.003, f"{v:.3f}", ha="center", va="bottom",
                     fontsize=8.3, color=INK)
        # 난수 분포의 95% 범위를 세로선으로
        off = x[j] - w + w * 2
        axR.plot([off, off], [sh["p2_5"], sh["p97_5"]], color=INK2, lw=1.8, zorder=5)
    axR.set_xticks(x)
    axR.set_xticklabels(["OS", "PFS"], fontsize=12, color=INK)
    axR.set_ylabel("late fusion C-index")
    axR.set_ylim(0.60, 0.75)
    axR.yaxis.grid(True, color=GRID, lw=0.8)
    axR.set_axisbelow(True)
    axR.legend(frameon=False, fontsize=9, loc="upper right", ncol=1)
    axR.set_title("② 이미지를 난수로 바꿔보면 — OS만 진짜가 이긴다",
                  fontsize=11.5, color=INK, loc="left", pad=10, fontweight="bold")

    fig.suptitle("late fusion에서 이미지가 OS만 올리는 이유: PFS에서는 이미지의 기여가 '0'과 구분되지 않는다",
                 fontsize=12.8, color=INK, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, "fig10_late_fusion_why_os_only.png")


# ════════════════════════════════════════════════════════════════════════════
# fig11: 난수 대조 실험의 분포 (PPT ④번 상자를 그림으로)
# 이미지 위험점수를 무작위로 섞어 200번 돌린 결과의 '분포'를 그리고,
# 진짜 이미지를 썼을 때의 값을 세로선으로 표시한다.
#   OS  = 진짜 값이 분포 오른쪽 바깥 → 이미지에 진짜 정보가 있다
#   PFS = 진짜 값이 분포 한가운데   → 이미지가 난수와 구분되지 않는다
# ════════════════════════════════════════════════════════════════════════════
def plot_fig11_random_control_distribution():
    d = _load_pfs_diagnosis()

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    for ax, (t, name, color) in zip(axes, [("os", "OS (전체생존)", BLUE),
                                           ("pfs", "PFS (무진행생존)", ORANGE)]):
        r = d[t]
        sh = r["stack_shuffled_image"]
        draws = np.array(sh["draws"])

        ax.hist(draws, bins=28, color=MUTED, alpha=0.55,
                label=f"난수 이미지 {sh['n_repeat']}회")
        ax.axvline(r["stack_tabular_only"], color=BASE, lw=2.2, ls=(0, (5, 3)),
                   label=f"tabular 단독 {r['stack_tabular_only']:.3f}")
        ax.axvline(r["stack_real_image"], color=color, lw=3.0,
                   label=f"진짜 이미지 {r['stack_real_image']:.3f}")

        # 진짜 값이 난수 분포의 어디쯤인지 화살표로 강조
        frac = sh["frac_random_beats_real"]
        verdict = ("난수가 한 번도 못 이김\n→ 진짜 신호" if frac == 0
                   else f"난수가 {frac:.0%} 확률로 이김\n→ 난수와 구분 불가")
        ax.annotate(verdict, xy=(r["stack_real_image"], ax.get_ylim()[1] * 0.72),
                    xytext=(0.03 if t == "os" else 0.62, 0.80),
                    textcoords="axes fraction", fontsize=10, color=INK, fontweight="bold",
                    ha="left", va="top",
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.8))

        ax.set_xlabel("late fusion C-index")
        ax.set_ylabel("횟수")
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=9.2, loc="upper left" if t == "pfs" else "upper right")
        ax.set_title(name, fontsize=12.5, color=INK, loc="left", pad=10, fontweight="bold")

    fig.suptitle("난수 대조 실험: 이미지 점수를 무작위로 섞어 200번 돌린 분포와 진짜 값의 위치",
                 fontsize=13, color=INK, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "fig11_random_control_distribution.png")


# ════════════════════════════════════════════════════════════════════════════
# fig12: "왜 PFS는 떨어지나" — 성능 변화를 단계로 분해 (waterfall)
# tabular 단독 → 난수를 넣었을 때 → 진짜 이미지를 넣었을 때
# 난수만 넣어도 깎이는 폭이 '가중치 추정 비용'이다.
# ════════════════════════════════════════════════════════════════════════════
def plot_fig12_cost_waterfall():
    d = _load_pfs_diagnosis()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), sharey=False)
    for ax, (t, name, color) in zip(axes, [("os", "OS — 비용을 내고도 남는 장사", BLUE),
                                           ("pfs", "PFS — 비용만 내고 얻은 것 없음", ORANGE)]):
        r = d[t]
        sh = r["stack_shuffled_image"]
        steps = [("tabular\n단독", r["stack_tabular_only"], BASE),
                 ("＋난수\n이미지", sh["mean"], MUTED),
                 ("＋진짜\n이미지", r["stack_real_image"], color)]
        x = np.arange(len(steps))
        ax.bar(x, [v for _, v, _ in steps], 0.55, color=[c for _, _, c in steps])
        for xi, (_, v, _) in zip(x, steps):
            ax.text(xi, v + 0.0015, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=10.5, color=INK, fontweight="bold")

        # 단계별 변화량을 라벨로. 단, '난수 → 진짜' 구간에서 진짜 값이 난수 분포의
        # 95% 구간 안에 있으면 그 차이는 우연과 구분되지 않으므로 그렇게 표기한다
        # (과장 방지 — RESULTS.md 10.5의 경고와 동일한 취지).
        for i in range(len(steps) - 1):
            delta = steps[i + 1][1] - steps[i][1]
            note = ""
            if i == 1 and sh["p2_5"] <= r["stack_real_image"] <= sh["p97_5"]:
                note = "\n(우연과 구분 불가)"
            ax.annotate(f"{delta:+.3f}{note}",
                        xy=(i + 0.5, max(steps[i][1], steps[i + 1][1])),
                        xytext=(i + 0.5, max(steps[i][1], steps[i + 1][1]) + 0.011),
                        ha="center", fontsize=10,
                        color=GREEN if delta > 0 else (MUTED if note else RED),
                        fontweight="bold")

        lo = min(v for _, v, _ in steps)
        hi = max(v for _, v, _ in steps)
        ax.set_ylim(lo - 0.02, hi + 0.025)
        ax.set_xticks(x)
        ax.set_xticklabels([s for s, _, _ in steps], fontsize=10.5, color=INK)
        ax.set_ylabel("late fusion C-index")
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_title(name, fontsize=12.2, color=INK, loc="left", pad=10, fontweight="bold")

    fig.suptitle("난수만 넣어도 성능이 깎인다 = '가중치 추정 비용' · OS는 그 비용을 넘는 이득이 있었다",
                 fontsize=12.8, color=INK, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "fig12_cost_waterfall.png")


FIGURES = {
    "fig4": plot_fig4_ablation,
    "fig5": plot_fig5_regime_comparison,
    "fig6": plot_fig6_late_fusion_b,
    "fig7": plot_fig7_os_vs_pfs_ladder,
    "fig8": plot_fig8_contribution_mismatch,
    "fig9": plot_fig9_same_image_different_outcome,
    "fig10": plot_fig10_late_fusion_why,
    "fig11": plot_fig11_random_control_distribution,
    "fig12": plot_fig12_cost_waterfall,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                     help="comma-separated subset of {%s} (default: all)" % ",".join(FIGURES))
    args = ap.parse_args()
    names = [n.strip() for n in args.only.split(",")] if args.only else list(FIGURES)
    for name in names:
        FIGURES[name]()


if __name__ == "__main__":
    main()
