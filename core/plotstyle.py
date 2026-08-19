# -*- coding: utf-8 -*-
"""그림 공통 스타일 — 팔레트 · rcParams · 한글 폰트 fallback.

이 팔레트/설정 블록은 예전에 4개 파일(plot_all_figures / generate_report /
km_cox_analysis / plot_results)에 각각 복사돼 있었다. 한 곳만 고치면 문서 안의
그림들이 서로 다른 시각 언어를 갖게 되므로 여기로 합쳤다.

BLUE/ORANGE 쌍은 색각이상(CVD) 분리도가 검증된 조합이다
(인접 dE 24.7 protan / 33.6 normal, 둘 다 기준치 8 이상).

사용법::

    from core import plotstyle as ps
    ps.apply()                      # rcParams 적용 (import 시점이 아니라 명시 호출)
    ax.plot(..., color=ps.BLUE)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (Agg 백엔드 설정 뒤에 import 해야 한다)

# ── 범주형 색 (계열 정체성 전용. 유의성/강조에는 절대 쓰지 않는다) ──────────
BLUE, ORANGE, MAGENTA, AQUA = "#2a78d6", "#eb6834", "#e87ba4", "#1baf7a"
YELLOW, GREEN, RED = "#eda100", "#0ca30c", "#d03b3b"

# ── 무채색 (배경/글자/격자) ────────────────────────────────────────────────
SURFACE, INK, INK2, MUTED, GRID, BASE = \
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BASELINE = BASE   # 예전 이름 (기준선 색으로 쓰던 별칭)

# 타깃(OS/PFS)은 2슬롯 고정 범주 — 순환시키지 않는다.
TARGET_COLOR = {"os": BLUE, "pfs": ORANGE}

_KOREAN_FONT_CANDIDATES = ("Noto Sans CJK KR", "NanumGothic", "Malgun Gothic", "AppleGothic")


def korean_font_family() -> list[str]:
    """기본 폰트 뒤에 설치된 CJK 폰트를 '대체 후보'로 붙인 목록.

    기본(DejaVu Sans)을 맨 앞에 두므로 영문 그림의 모양은 그대로 유지되고,
    DejaVu 에 없는 한글 글자만 뒤 폰트로 넘어간다(matplotlib 3.6+ font fallback).
    이게 없으면 한글 라벨이 네모(□)로 깨진다.
    """
    installed = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    return ["DejaVu Sans"] + [n for n in _KOREAN_FONT_CANDIDATES if n in installed]


def apply(font_size: float = 11) -> None:
    """프로젝트 공통 rcParams 를 적용한다."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASE, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "font.size": font_size,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    plt.rcParams["font.family"] = korean_font_family()
    plt.rcParams["axes.unicode_minus"] = False   # 한글 폰트에서 마이너스 기호 깨짐 방지
