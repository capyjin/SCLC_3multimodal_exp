# -*- coding: utf-8 -*-
"""RESULTS.md 의 모든 그림(fig1, fig4~fig12)을 한 곳에서 그린다.

그림 하나당 함수 하나. 색 팔레트/rcParams 는 ``sclc.plotstyle`` 에서 가져오고
(예전엔 그림 스크립트마다 복붙돼 있었다), 출력 경로와 fig4·fig5 가 공유하는
ablation 수치는 이 모듈 최상단에서 한 번만 정의한다.

fig1 은 예전 ``generate_report.py`` 에 있던 것을 옮겨온 것이다. 그 파일은 그림
외에 **RESULTS.md 를 통째로 덮어쓰는** 기능이 있었는데, RESULTS.md 는 그 뒤
1000줄 넘게 손으로 쓴 문서가 되어 실행하면 내용이 날아가는 함정이었다. 그래서
쓰이는 그림(fig1)만 여기로 옮기고 파일은 지웠다. 같이 있던 fig2(fold 산포)·
fig3(late fusion 가중치)는 어느 문서에서도 참조하지 않아 함께 정리했다.

Run:  python 도구/plot_all_figures.py                  # 전부 그리기
      python 도구/plot_all_figures.py --only fig4,fig6  # 일부만
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Patch

from sclc import plotstyle as ps
from sclc.plotstyle import (AQUA, BASE, BLUE, GREEN, GRID, INK, INK2, MAGENTA,
                            MUTED, ORANGE, RED, SURFACE)

ps.apply()

# ── 공통 경로 ──────────────────────────────────────────────────────────────
ROOT = PROJECT_ROOT
ASSETS = os.path.join(ROOT, "report_assets")
os.makedirs(ASSETS, exist_ok=True)

# ── 공통 데이터: OS 5-fold ablation 결과 ────────────────────────────────────
# fig4(모달리티 조합별 성능)와 fig5(원래 레짐 vs 개선 레짐)가 "원래 레짐"
# 수치를 그대로 공유하므로, 예전처럼 두 파일에 똑같은 숫자를 따로 박아두지
# 않고 여기 한 곳에서만 정의한다. 키는 sclc.model.MODALITY_CONFIGS 이름과 맞춤.
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
    """실험3 ablation.py 가 저장한 results_{target}.json 을 읽어온다."""
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

    # 세 막대 모두 '임상+판독지(tabular)'가 깔린 상태에서의 비교라는 점을 라벨에 명시.
    # (막대 하나가 이미지 단독 성적으로 오해되는 것을 막기 위함)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), sharey=False)
    for ax, (t, name, color) in zip(axes, [("os", "OS — 비용을 내고도 남는 장사", BLUE),
                                           ("pfs", "PFS — 비용만 내고 얻은 것 없음", ORANGE)]):
        r = d[t]
        sh = r["stack_shuffled_image"]
        steps = [("임상+판독지\n단독", r["stack_tabular_only"], BASE),
                 ("＋무작위로 섞은\n이미지", sh["mean"], MUTED),
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
        ax.set_ylim(lo - 0.032, hi + 0.025)
        ax.set_xticks(x)
        ax.set_xticklabels([s for s, _, _ in steps], fontsize=10, color=INK)
        ax.set_ylabel("late fusion C-index")
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_title(name, fontsize=12.2, color=INK, loc="left", pad=10, fontweight="bold")

        # ⚠ 흔한 오해 방지용 각주: 세 막대는 모두 임상+판독지가 깔린 상태의 값이며,
        #   이미지 '단독' 성적은 따로 있다. 특히 이미지 단독이 0.5보다 확실히 높다는
        #   사실이 "이미지는 노이즈"라는 잘못된 결론을 막아준다. (verify_shuffle_sanity.py)
        solo_img = {"os": 0.657, "pfs": 0.615}[t]
        ax.text(0.5, 0.035,
                f"※ 세 막대 모두 '임상+판독지'가 포함된 값 ｜ 참고: 이미지 단독 {solo_img:.3f}, "
                f"섞은 이미지 단독 0.50",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=8.6, color=INK2)

    fig.suptitle("정보가 0인 값을 넣어도 성능이 깎인다 = '가중치 추정 비용' · OS는 그 비용을 넘는 이득이 있었다",
                 fontsize=12.6, color=INK, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "fig12_cost_waterfall.png")


# ════════════════════════════════════════════════════════════════════════════
# fig1: 초기 조건(bs16/ep30) 모델별 평균 C-index — RESULTS.md §2
# 예전 generate_report.py 에 있던 그림. 2026-07-22 실행의 metrics/summary.json 을 읽는다.
# ════════════════════════════════════════════════════════════════════════════
EARLY_DIR = os.path.join(ROOT, "outputs", "EXP_20260722_early_fusion_train")
LATE_DIR = os.path.join(ROOT, "outputs", "EXP_20260722_late_fusion_train")

FIG1_MODELS = [
    ("image_only",                          "Image only",    BLUE,    "late"),
    ("clinical_only",                       "Clinical only", ORANGE,  "late"),
    ("report_only",                         "Report only",   AQUA,    "late"),
    ("late_fusion_weighted_sum",            "Late fusion",   "#eda100", "late"),
    ("early_fusion_image_clinical_report",  "Early fusion",  MAGENTA, "early"),
]
FIG1_TARGETS = [("os", "OS"), ("pfs", "PFS")]


def plot_fig1_mean_cindex():
    def _summary(path):
        with open(os.path.join(path, "metrics", "summary.json"), encoding="utf-8") as fh:
            return json.load(fh)

    early_sum, late_sum = _summary(EARLY_DIR), _summary(LATE_DIR)

    def get(target, key):
        src = early_sum if key == "early_fusion_image_clinical_report" else late_sum
        node = src[target].get(key)
        return (node["mean_c_index"], node["std_c_index"]) if node else (np.nan, np.nan)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    group_w = 0.8
    bar_w = group_w / len(FIG1_MODELS)
    x = np.arange(len(FIG1_TARGETS))
    for i, (key, name, color, _kind) in enumerate(FIG1_MODELS):
        means = [get(t, key)[0] for t, _ in FIG1_TARGETS]
        stds = [get(t, key)[1] for t, _ in FIG1_TARGETS]
        offs = x - group_w / 2 + bar_w * (i + 0.5)
        ax.bar(offs, means, bar_w * 0.92, label=name, color=color, yerr=stds, capsize=3,
               ecolor=MUTED, error_kw={"elinewidth": 1, "alpha": 0.8})
        for off, m in zip(offs, means):
            ax.text(off, m + 0.012, f"{m:.3f}", ha="center", va="bottom",
                    fontsize=8.5, color=INK, fontweight="bold")

    ax.axhline(0.5, color=BASE, lw=1.4, ls=(0, (5, 4)), zorder=0)
    ax.text(len(FIG1_TARGETS) - 0.5, 0.5, "  random = 0.50", va="center", ha="left",
            fontsize=8.5, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in FIG1_TARGETS], fontsize=12, color=INK)
    ax.set_ylabel("C-index (5-fold mean, error bar = SD)")
    ax.set_ylim(0, 0.85)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.11),
              fontsize=9.5, columnspacing=1.2, handletextpad=0.5)
    ax.set_title("Tri-modal survival prediction — mean C-index by model",
                 fontsize=13, color=INK, pad=30, loc="left", fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig1_mean_cindex.png")


# ── fig13 · fig14: MoE 게이트(M0) ─────────────────────────────────────────
# 이 두 그림만 RESULTS.md 가 아니라 RESULTS_TABLE_final.md 표 6 에서 참조한다.
# 수치는 전부 결과 JSON 에서 읽는다(손으로 옮겨 적으면 재실행 때 조용히 어긋난다).
MOE_NOISE_BAND = 0.015          # 이 문서의 판정선 (fold별 차이의 표준오차 0.008~0.014)
# ⚠️ 아래는 MoE 실험 당시의 기준이다. 최종 채택 모델은 두 타깃 모두 RadBERT (README.md).
# 당시 판독지 인코더는 타깃마다 채택이 달랐다(표 3): OS 는 RadBERT 가 전 방식에서 우세해
# 채택, PFS 는 concat/late 둘 다 RadBERT 가 오히려 나빠 TF-IDF 를 유지 채택했다
# (예: concat 임상+판독지 PFS 는 TF-IDF 0.6696 vs RadBERT 0.6456). 그래서 게이트
# 성능도 이 표 6 그림에서는 **타깃별 채택 인코더**로 봐야 late fusion(표 1)과
# 앞뒤가 맞는다 -- 두 타깃을 전부 RadBERT로 보면 PFS 기준선 자체가 채택되지 않은
# 조합이 된다.
MOE_M0_DIR = {"os": os.path.join(ROOT, "outputs", "moe_gate_m0_radbert"),
              "pfs": os.path.join(ROOT, "outputs", "moe_gate_m0")}
MOE_TUNED_DIR = {"os": os.path.join(ROOT, "outputs", "moe_gate_m0_tuned_radbert"),
                 "pfs": os.path.join(ROOT, "outputs", "moe_gate_m0_tuned")}
MOE_ENCODER_LABEL = {"os": "RadBERT", "pfs": "TF-IDF"}

# 다이어그램 박스용 연한 배경. 데이터 마크가 아니라 '용기'라서 채도를 낮게 두고,
# 정체성은 테두리(BLUE/ORANGE)가 진다.
_TINT_FROZEN, _TINT_GATE, _TINT_SUM = "#f1f0ea", "#fdece3", "#e7f0fb"


def _moe_box(ax, x0, y0, x1, y1, text, *, fc, ec, lw=1.5, fontsize=9.5,
             weight="normal", ls="-"):
    """라운드 박스 + 가운데 정렬 텍스트 (fig13 전용)."""
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0,rounding_size=1.8",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=2))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center",
            fontsize=fontsize, color=INK, fontweight=weight, zorder=3,
            linespacing=1.55)


def _moe_arrow(ax, xy_from, xy_to, *, color=BASE, lw=1.8):
    ax.annotate("", xy=xy_to, xytext=xy_from, zorder=1,
                arrowprops=dict(arrowstyle="-|>,head_width=0.28,head_length=0.6",
                                color=color, linewidth=lw,
                                shrinkA=0, shrinkB=0))


def plot_fig13_moe_m0_architecture():
    """M0 게이트 구조도 — 무엇이 얼려져 있고 무엇이 새로 학습되는지가 요점."""
    fig, ax = plt.subplots(figsize=(11.6, 8.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # ── 1층: 동결된 두 전문가 백본 ──
    ax.text(50, 99.2, "① 동결된 전문가 백본  —  재학습 없음, 위험점수·임베딩만 꺼내 씀",
            ha="center", va="top", fontsize=9.6, color=INK2, fontweight="bold")
    # 한글과 mathtext($...$)를 한 문자열에 섞으면 mathtext 는 폰트 fallback 을 하지
    # 않아 한글이 □ 로 깨진다. 그래서 수식도 일반 텍스트로 쓴다.
    _moe_box(ax, 3, 79, 47, 95.5,
             "영상 전문가  (SimpleCNN)\n"
             "임베딩 512차원  ·  위험점수 r_img",
             fc=_TINT_FROZEN, ec=BASE, ls=(0, (4, 2.5)))
    _moe_box(ax, 53, 79, 97, 95.5,
             "Tabular 전문가  (임상 + 판독지)\n"
             "임베딩 128 · 16차원  ·  위험점수 r_tab",
             fc=_TINT_FROZEN, ec=BASE, ls=(0, (4, 2.5)))

    # ── 2층: 게이트 입력 ──
    _moe_arrow(ax, (25, 79), (33, 68.5))
    _moe_arrow(ax, (75, 79), (67, 68.5))
    ax.text(50, 74.6, "임베딩 3개", ha="center", va="center", fontsize=8.6, color=MUTED)
    _moe_box(ax, 19, 56, 81, 68.5,
             "② 게이트 입력   24차원\n"
             "영상 512 · 임상 128 · 판독지 16 을 각각 PCA-8 → 이어붙임\n"
             "(PCA·표준화는 train fold 171명으로만 fit)",
             fc=_TINT_FROZEN, ec=BASE, fontsize=9.2)

    # ── 3층: 게이트 네트워크 (유일한 학습 대상) ──
    _moe_arrow(ax, (50, 56), (50, 45.5), color=ORANGE)
    _moe_box(ax, 19, 33, 81, 45.5,
             "③ 게이트 네트워크   ★ 이 실험에서 새로 학습되는 유일한 부분\n"
             "softmax( Linear(24 → 2) )   ·   학습 파라미터 51개",
             fc=_TINT_GATE, ec=ORANGE, lw=2.0, fontsize=9.8, weight="bold")

    # ── 4층: 가중합 ──
    _moe_arrow(ax, (50, 33), (50, 22.5), color=ORANGE)
    ax.text(51.4, 27.7, "g(x) = [ g_img , g_tab ],   합 = 1,   환자마다 다른 값",
            ha="left", va="center", fontsize=9.0, color=ORANGE, fontweight="bold")
    # 위험점수는 게이트를 거치지 않고 바깥쪽으로 내려온다(구조상 중요).
    for x_side, lbl in ((8, "r_img"), (92, "r_tab")):
        _moe_arrow(ax, (x_side, 79), (x_side, 22.5), color=BLUE)
        ax.text(x_side + (1.8 if x_side < 50 else -1.8), 51, lbl,
                ha="left" if x_side < 50 else "right",
                va="center", fontsize=10, color=BLUE, fontweight="bold")
    _moe_box(ax, 4, 10, 96, 22.5,
             "④ 가중합\n"
             "risk  =  α · ( g_img(x) · r_img   +   g_tab(x) · r_tab )",
             fc=_TINT_SUM, ec=BLUE, lw=2.0, fontsize=12.5, weight="bold")

    # ── 하단: 학습 방식과 대조군 ──
    ax.text(4, 5.6, "학습:", fontsize=9.2, color=INK2, fontweight="bold", ha="left", va="center")
    ax.text(12, 5.6, "Cox 부분우도 · full-batch(fold당 171명) · 5-fold × 5 seed",
            fontsize=9.2, color=INK2, ha="left", va="center")
    ax.text(4, 1.6, "대조군:", fontsize=9.2, color=INK2, fontweight="bold", ha="left", va="center")
    ax.text(12, 1.6, "g 를 상수로 고정하면 late fusion(전 환자 동일 비율)과 수학적으로 동치 "
                     "→ 게이트가 실제로 도움이 되는지 바로 비교 가능",
            fontsize=9.2, color=INK2, ha="left", va="center")

    ax.set_title("MoE 게이트 M0 — 결합 비율을 환자마다 학습시키는 구조",
                 fontsize=13.5, color=INK, loc="left", pad=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig13_moe_m0_architecture.png")


def plot_fig14_moe_m0_performance():
    """게이트 2종 vs 고정 비율 — 판정선(±0.015) 안에 들어가는지가 핵심.

    MoE 실험 당시 타깃별 인코더로 읽는다(표 3): OS=RadBERT, PFS=TF-IDF.
    두 타깃을 같은 인코더로 맞추면 PFS 기준선이 채택되지 않은 조합이 된다.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.9))
    for ax, target, tlabel in zip(axes, ["os", "pfs"],
                                  ["OS — 전체생존", "PFS — 무진행생존"]):
        with open(os.path.join(MOE_M0_DIR[target], "results.json"), encoding="utf-8") as fh:
            M0 = json.load(fh)
        with open(os.path.join(MOE_TUNED_DIR[target], "results.json"), encoding="utf-8") as fh:
            TUNED = json.load(fh)
        tlabel = f"{tlabel}  ({MOE_ENCODER_LABEL[target]})"
        base = M0[target]["coxph_reproduction_mean"]
        # 기준선은 '이 실험 내부 대조군'이다 — 결합 계수를 tabular 축의 in-sample
        # (train-fold 자기채점) 점수로 학습해서, 표 1의 공식 late fusion(진짜 OOF로
        # 학습)과는 값이 다르다(같은 방법론이 아님, 재구성 오차가 아니다). 게이트도
        # 같은 in-sample 이점을 받으므로 이 대조군과 비교하는 것 자체는 공정하다.
        bars = [
            ("고정 비율\n(이 실험 대조군)", base, ps.TARGET_COLOR[target]),
            ("M0 게이트\n(임베딩 PCA-8)", M0[target]["learned_gate_mean"], MUTED),
            ("M0-tuned\n(inner-CV 선택)", TUNED[target]["tuned_gate_mean"], MUTED),
        ]
        x = np.arange(len(bars))

        # 판정선 띠 — "이 안이면 우연과 구분 불가". 결론을 그림이 직접 보여준다.
        ax.axhspan(base - MOE_NOISE_BAND, base + MOE_NOISE_BAND,
                   color=ps.TARGET_COLOR[target], alpha=0.10, zorder=0, lw=0)
        ax.axhline(base, color=ps.TARGET_COLOR[target], lw=1.2, zorder=1)
        ax.text(2.46, base + MOE_NOISE_BAND, f"판정선 ±{MOE_NOISE_BAND:g}", ha="right",
                va="bottom", fontsize=8.4, color=ps.TARGET_COLOR[target])

        ax.bar(x, [v for _, v, _ in bars], 0.52,
               color=[c for _, _, c in bars], zorder=2)
        for xi, (_, val, _) in zip(x, bars):
            ax.text(xi, val + 0.006, f"{val:.4f}", ha="center", va="bottom",
                    fontsize=10.2, color=INK, fontweight="bold", zorder=3)
        # Δ 와 판정은 축 아래에 둔다. 막대 안에 넣으면 회색 막대 위의 회색 글자가
        # 되어 안 보이고, 막대 위에 두면 판정선 라벨과 부딪힌다.
        for xi, (_, val, _) in zip(x, bars):
            if val == base:
                note, color, weight = "기준선", MUTED, "normal"
            else:
                d = val - base
                inside = abs(d) < MOE_NOISE_BAND
                note = f"Δ {d:+.4f}  ·  {'판정선 안' if inside else '판정선 밖'}"
                color = MUTED if inside else RED
                weight = "normal" if inside else "bold"
            ax.annotate(note, xy=(xi, -0.175), xycoords=("data", "axes fraction"),
                        ha="center", va="top", fontsize=9.2, color=color,
                        fontweight=weight, annotation_clip=False)

        ax.set_xticks(x)
        ax.set_xticklabels([lbl for lbl, _, _ in bars], fontsize=9.8, color=INK)
        ax.set_xlim(-0.6, 2.55)
        ax.set_ylim(0.5, 0.775)
        ax.set_ylabel("C-index (5-fold 평균 · 0.50 = 무작위)")
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_title(tlabel, fontsize=12, color=INK, loc="left", pad=8,
                     fontweight="bold")

    fig.suptitle("환자별로 학습한 게이트는 고정 비율을 넘지 못했다  (타깃별 채택 인코더 기준)",
                 fontsize=13.5, color=INK, x=0.008, ha="left", y=0.985,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0.055, 1, 0.945))   # 아래 여백 = 축 밑 Δ 줄 자리
    _save(fig, "fig14_moe_m0_performance.png")


# ── fig15: 발표용 파이프라인 그림 (사용자 원본 도식의 수정본) ──────────────
# fig13(문서용 M0 구조도)과 같은 내용을 발표용 가로 흐름으로 다시 그린 것.
# 원본 도식에서 고친 것: (1) 게이트 입력은 tabular 통짜가 아니라 영상·임상·판독지
# 3개 전문가를 각각 PCA -> concat, (2) 백본 동결 표시, (3) PCA 차원은 d 로 일반화
# (M0=8, M0-tuned=2/4 등 조건마다 다르므로 특정 숫자를 박으면 틀린다).
_GF_TINT_FROZEN, _GF_TINT_GATE, _GF_TINT_SUM = "#f1f0ea", "#fdece3", "#e7f0fb"
_GF_TINT_EMB, _GF_TINT_RISK = "#eef1f5", "#e7f0fb"


def _gf_box(ax, x0, y0, x1, y1, title, body="", *, fc, ec, lw=1.5, ls="-",
            title_size=9.4, body_size=8.0, title_color=INK):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.4",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=2))
    cx, h = (x0 + x1) / 2, y1 - y0
    if body:
        ax.text(cx, y1 - h * 0.32, title, ha="center", va="center", zorder=3,
                fontsize=title_size, color=title_color, fontweight="bold")
        ax.text(cx, y0 + h * 0.34, body, ha="center", va="center", zorder=3,
                fontsize=body_size, color=INK2, linespacing=1.5)
    else:
        ax.text(cx, (y0 + y1) / 2, title, ha="center", va="center", zorder=3,
                fontsize=title_size, color=title_color, fontweight="bold")


def _gf_arrow(ax, p0, p1, *, color=BASE, lw=1.7, rad=0.0):
    ax.annotate("", xy=p1, xytext=p0, zorder=1, arrowprops=dict(
        arrowstyle="-|>,head_width=0.26,head_length=0.55", color=color,
        linewidth=lw, shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}"))


def plot_fig15_gated_fusion_pipeline():
    fig, ax = plt.subplots(figsize=(17.5, 8.8))
    ax.set_xlim(0, 108)
    ax.set_ylim(-10, 102)
    ax.axis("off")

    # ── 입력 ──────────────────────────────────────────────────────────────
    for i, dx in enumerate((0, 1.6, 3.2)):          # 스캔 여러 장 느낌
        ax.add_patch(FancyBboxPatch(
            (2.2 + dx, 68.5 - dx * 0.55), 6.4, 11.5,
            boxstyle="round,pad=0,rounding_size=0.8", facecolor="#dcdcd6" if i < 2 else "#c9c9c2",
            edgecolor=BASE, linewidth=1.1, zorder=2 + i))
    ax.text(6.7, 65.5, "PET-CT 영상", ha="center", va="top", fontsize=9, color=INK, fontweight="bold")
    ax.text(6.7, 62.2, "환자 N명 · 2D MIP", ha="center", va="top", fontsize=7.6, color=MUTED)

    clin_rows = [("GENDER", "#e8eef7"), ("STAGE", "#fbeceb"), ("SMOKING_STATUS", "#eef6ee"),
                 ("MMRC", "#f4f0e4")]
    for i, (name, fill) in enumerate(clin_rows):
        y = 37.5 - i * 5.6
        _gf_box(ax, 1.2, y, 12.2, y + 4.6, name, fc=fill, ec=BASE, lw=1.0, title_size=7.4)
    ax.text(6.7, 43.2, "임상 21변수", ha="center", va="bottom", fontsize=8.6,
            color=INK, fontweight="bold")
    ax.text(6.7, 13.6, "…", ha="center", va="center", fontsize=10, color=MUTED)
    _gf_box(ax, 1.2, 5.5, 12.2, 11.5, "판독지 텍스트",
            "TF-IDF / RadBERT", fc="#f3edf7", ec=BASE, lw=1.0,
            title_size=7.8, body_size=6.8)

    # ── 백본 (동결) ───────────────────────────────────────────────────────
    ax.text(22.7, 82.5, "① 이미 학습되어 얼린 백본 (재학습 없음)",
            ha="center", va="bottom", fontsize=8.8, color=INK2, fontweight="bold")
    _gf_box(ax, 15.0, 66.5, 30.5, 81.0, "영상 모델  ❄ 동결",
            "SimpleCNN + DeepSurv", fc=_GF_TINT_FROZEN, ec=BASE, ls=(0, (4, 2.5)))
    # concat 은 여기 있으면 안 된다 -- 두 브랜치는 각자 임베딩을 내고(게이트는 그걸
    # concat 이전에 따로 가져간다), concat 은 r_tab 을 만드는 head 경로에만 있다.
    _gf_box(ax, 15.0, 14.0, 30.5, 42.0, "Tabular 모델  ❄ 동결",
            "임상 MLP (21 → 128)\n판독지 MLP (400/768 → 16)\n두 브랜치는 서로 독립",
            fc=_GF_TINT_FROZEN, ec=BASE, ls=(0, (4, 2.5)))
    _gf_arrow(ax, (12.6, 74.0), (15.0, 74.0))
    _gf_arrow(ax, (12.6, 28.0), (15.0, 28.0))
    _gf_arrow(ax, (12.6, 8.5), (15.0, 20.0))

    # ── 임베딩 3개 + 위험점수 2개 ────────────────────────────────────────
    # 핵심 수정: 게이트가 보는 전문가는 tabular 통짜가 아니라 임상/판독지 따로 = 3개.
    _gf_box(ax, 34.0, 88.0, 47.0, 96.5, "위험점수  r_img",
            fc=_GF_TINT_RISK, ec=BLUE, lw=1.8, title_size=9.6, title_color=BLUE)
    _gf_box(ax, 34.0, 66.0, 47.0, 80.5, "영상 임베딩", "512차원 × N명",
            fc=_GF_TINT_EMB, ec=BASE)
    _gf_box(ax, 34.0, 31.0, 47.0, 43.0, "임상 임베딩", "128차원 × N명 · L2 정규화",
            fc=_GF_TINT_EMB, ec=BASE, body_size=7.6)
    _gf_box(ax, 34.0, 16.0, 47.0, 28.0, "판독지 임베딩", "16차원 × N명 · L2 정규화",
            fc=_GF_TINT_EMB, ec=BASE, body_size=7.6)
    _gf_box(ax, 34.0, 2.0, 47.0, 10.5, "위험점수  r_tab",
            fc=_GF_TINT_RISK, ec=BLUE, lw=1.8, title_size=9.6, title_color=BLUE)

    _gf_arrow(ax, (30.5, 74.0), (34.0, 73.5))
    _gf_arrow(ax, (30.5, 32.0), (34.0, 37.0))
    _gf_arrow(ax, (30.5, 24.0), (34.0, 22.0))
    # 백본 자신의 head 가 내는 위험점수 (게이트와 무관하게 이미 정해져 있음)
    _gf_arrow(ax, (40.5, 80.5), (40.5, 88.0), color=BLUE)
    ax.text(41.4, 84.2, "백본 head", ha="left", va="center", fontsize=7.4, color=BLUE)
    # r_tab 은 두 브랜치 임베딩을 concat 한 뒤 head 하나가 낸다 -- 화살표도 둘.
    _gf_arrow(ax, (34.0, 34.0), (35.8, 10.5), color=BLUE, rad=0.32)
    _gf_arrow(ax, (41.5, 16.0), (41.5, 10.5), color=BLUE)
    ax.text(33.0, 6.2, "concat(128+16) → head", ha="right", va="center",
            fontsize=7.6, color=BLUE)

    # ── 게이트 입력 (전문가별 PCA → concat) ──────────────────────────────
    _gf_box(ax, 50.5, 24.0, 64.5, 58.0, "② 게이트 입력",
            "영상 · 임상 · 판독지 임베딩을\n각각 따로 PCA → 이어붙임\n\nd = PCA 차원 × 전문가 3개",
            fc=_GF_TINT_FROZEN, ec=BASE, body_size=8.2)
    _gf_arrow(ax, (47.0, 70.0), (50.5, 51.0), rad=-0.12)
    _gf_arrow(ax, (47.0, 37.0), (50.5, 42.0), rad=0.0)
    _gf_arrow(ax, (47.0, 22.0), (50.5, 32.0), rad=0.12)
    ax.text(57.5, 59.0, "전문가 3개가 각각 들어간다", ha="center", va="bottom",
            fontsize=8.0, color=MUTED)

    # ── 게이트 네트워크 (유일한 학습 대상) ───────────────────────────────
    _gf_box(ax, 68.0, 28.0, 81.5, 54.0, "③ 게이트 네트워크",
            "softmax( Linear(d → 2) )\n\n★ 새로 학습되는\n유일한 부분\n학습 파라미터 2d+3개",
            fc=_GF_TINT_GATE, ec=ORANGE, lw=2.0, body_size=8.2)
    _gf_arrow(ax, (64.5, 41.0), (68.0, 41.0), color=ORANGE)

    # ── 최종 가중합 ──────────────────────────────────────────────────────
    _gf_box(ax, 84.5, 26.0, 101.0, 56.0, "④ 가중합",
            "risk =\nα · ( g_tab(x)·r_tab\n     + g_img(x)·r_img )\n\n"
            "g_tab, g_img 는 환자마다\n다른 값 (합 = 1)",
            fc=_GF_TINT_SUM, ec=BLUE, lw=2.0, body_size=8.4)
    _gf_arrow(ax, (81.5, 41.0), (84.5, 41.0), color=ORANGE)
    ax.text(83.0, 43.0, "g(x)", ha="center", va="bottom", fontsize=8.4,
            color=ORANGE, fontweight="bold")
    # 위험점수는 게이트를 거치지 않고 곧장 가중합으로 (구조상 중요)
    _gf_arrow(ax, (47.0, 92.0), (90.0, 56.0), color=BLUE, rad=-0.18)
    _gf_arrow(ax, (47.0, 6.0), (90.0, 26.0), color=BLUE, rad=0.18)

    _gf_arrow(ax, (92.7, 26.0), (92.7, 18.0), color=BLUE, lw=2.0)
    ax.text(92.7, 16.5, "최종 RISK", ha="center", va="top", fontsize=10.5,
            color=INK, fontweight="bold")

    # ── 하단 주석 ────────────────────────────────────────────────────────
    ax.text(0.5, -3.0, "학습:", fontsize=8.8, color=INK2, fontweight="bold", ha="left", va="center")
    ax.text(7.0, -3.0, "게이트만 Cox 부분우도로 학습 (fold당 171명, full-batch) · "
                       "PCA·표준화는 train fold 안에서만 fit", fontsize=8.8, color=INK2,
            ha="left", va="center")
    ax.text(0.5, -7.5, "대조군:", fontsize=8.8, color=INK2, fontweight="bold", ha="left", va="center")
    ax.text(7.0, -7.5, "g 를 상수로 고정하면 전 환자 동일 비율(고정 비율 결합)과 수학적으로 동치 "
                       "→ 게이트가 실제로 도움이 되는지 직접 비교 가능", fontsize=8.8,
            color=INK2, ha="left", va="center")

    ax.set_title("Gated Multimodal Fusion 실험 — 파이프라인",
                 fontsize=15, color=INK, loc="left", pad=16, fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig15_gated_fusion_pipeline.png")


FIGURES = {
    "fig1": plot_fig1_mean_cindex,
    "fig4": plot_fig4_ablation,
    "fig5": plot_fig5_regime_comparison,
    "fig6": plot_fig6_late_fusion_b,
    "fig7": plot_fig7_os_vs_pfs_ladder,
    "fig8": plot_fig8_contribution_mismatch,
    "fig9": plot_fig9_same_image_different_outcome,
    "fig10": plot_fig10_late_fusion_why,
    "fig11": plot_fig11_random_control_distribution,
    "fig12": plot_fig12_cost_waterfall,
    "fig13": plot_fig13_moe_m0_architecture,
    "fig14": plot_fig14_moe_m0_performance,
    "fig15": plot_fig15_gated_fusion_pipeline,
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
