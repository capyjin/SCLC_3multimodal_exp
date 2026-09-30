# -*- coding: utf-8 -*-
"""라벨을 보지 않는 순수 이미지 함수 -- 1st-order + GLCM 텍스처 radiomics 특징.

[왜 src/sclc 에 있나]
  실험11(MoE 융합)이 CNN/radiomics/텍스트/임상을 별도 expert 로 쓰려면 radiomics
  추출기가 필요하고, 그 전에 "CNN 딥피처와 radiomics 가 중복되는가"를 검증하는
  실험(exp_cnn_radiomics_corr.py)도 같은 추출기를 쓴다. 두 곳 이상이 쓰므로
  실험 폴더가 아니라 여기에 둔다 (코드_구조.md 원칙 1).

[분할 마스크가 없다는 것의 의미]
  이 코호트는 종양 segmentation mask 가 없는 2D MIP 이므로, 여기서 계산하는
  1차 통계량과 GLCM 텍스처는 전부 **이미지 전체**에 대한 것이다 (병변 국한
  radiomics 가 아니라 "이 이미지 전체가 어떻게 생겼나"). pyradiomics 같은 표준
  라이브러리도, scikit-image 도 이 환경에 없어 GLCM 은 직접 구현했다
  (distance=1, 4방향 0/45/90/135도 평균, 32 gray level 양자화).

[실험10 과의 관계]
  실험10/exp_trivial_stats.py 는 w/h/mean/std/frac_hot/frac_dark 6개 "아무나
  뽑을 수 있는" 전역 통계만으로 CNN 성능에 근접한다는 것을 보였다. 여기 특징은
  그 6개를 **포함하지 않고** 그것들이 놓친 고차 모멘트(왜도/첨도)와 텍스처
  (GLCM)만 추가한다 -- "그 6개를 넘어서는 정보가 있는가"를 깨끗하게 보기 위해서다.

[누수 없음]  라벨을 전혀 보지 않는 순수 이미지 함수라서 fold 밖에서 한 번만
  계산해도 안전하다 (실험10과 동일한 논리).
"""
import os

import numpy as np
from PIL import Image, ImageOps

GLCM_LEVELS = 32
GLCM_OFFSETS = ((1, 0), (0, 1), (1, 1), (1, -1))  # (dx, dy): 0, 90, 45, 135도, distance=1
EPS = 1e-12

FIRST_ORDER_NAMES = (
    # mean/std/variance/energy 는 일부러 뺐다 -- 실험10 trivial_6stats 의
    # mean/std 와 정의가 같거나(mean, std) 그 결정론적 함수(variance=std^2,
    # energy~=mean^2+variance)라서 넣으면 "radiomics vs trivial_6stats" 비교가
    # 순환논리가 된다(실측: 넣었더니 held-out canonical r=1.0000, fold간
    # 분산 0 -- 중복 정의 때문이었지 진짜 상관이 아니었다).
    "skewness", "kurtosis", "entropy", "min", "max", "range",
    "p10", "p25", "median", "p75", "p90", "iqr", "mad",
    "grad_mean", "grad_std",
)
GLCM_NAMES = ("glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity",
              "glcm_energy", "glcm_correlation", "glcm_entropy")
FEATURE_NAMES = FIRST_ORDER_NAMES + GLCM_NAMES


def load_gray_array(image_path: str, research_id: int, inverted_ids) -> np.ndarray:
    """sclc.dataset.preprocess_data 와 동일한 전처리(그레이스케일 + 반전 보정),
    [0, 1] 범위로 정규화. 리사이즈는 하지 않는다 (원본 해상도에서 텍스처 계산)."""
    img = Image.open(image_path).convert("L")
    if int(research_id) in inverted_ids:
        img = ImageOps.invert(img)
    return np.asarray(img, dtype=np.float32) / 255.0


def _first_order(a: np.ndarray) -> dict[str, float]:
    flat = a.ravel()
    mean = float(flat.mean())
    std = float(flat.std())
    std_eps = max(std, EPS)
    centered = flat - mean
    hist, _ = np.histogram(flat, bins=32, range=(0.0, 1.0))
    probs = hist / max(hist.sum(), 1)
    probs = probs[probs > 0]
    grad_y, grad_x = np.gradient(a)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    p10, p25, median, p75, p90 = np.percentile(flat, [10, 25, 50, 75, 90])
    return {
        "mean": mean,
        "std": std,
        "variance": float(std**2),
        "skewness": float(np.mean(centered**3) / std_eps**3),
        "kurtosis": float(np.mean(centered**4) / std_eps**4),
        "entropy": float(-(probs * np.log(probs)).sum()),
        "energy": float(np.mean(flat**2)),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "range": float(flat.max() - flat.min()),
        "p10": float(p10), "p25": float(p25), "median": float(median),
        "p75": float(p75), "p90": float(p90),
        "iqr": float(p75 - p25),
        "mad": float(np.median(np.abs(flat - median))),
        "grad_mean": float(grad_mag.mean()),
        "grad_std": float(grad_mag.std()),
    }


def _glcm_direction(quantized: np.ndarray, dx: int, dy: int, levels: int) -> np.ndarray:
    h, w = quantized.shape
    if dy >= 0:
        src = quantized[: h - dy if dy else h, : w - dx if dx else w]
        dst = quantized[dy: h, dx: w]
    else:
        src = quantized[-dy: h, : w - dx if dx else w]
        dst = quantized[: h + dy, dx: w]
    pairs = src.ravel() * levels + dst.ravel()
    counts = np.bincount(pairs, minlength=levels * levels).reshape(levels, levels).astype(np.float64)
    counts = counts + counts.T  # symmetric GLCM
    total = counts.sum()
    return counts / total if total > 0 else counts


def _glcm_features(a: np.ndarray, levels: int = GLCM_LEVELS,
                    offsets=GLCM_OFFSETS) -> dict[str, float]:
    quantized = np.clip((a * levels).astype(np.int64), 0, levels - 1)
    idx = np.arange(levels)
    i_grid, j_grid = np.meshgrid(idx, idx, indexing="ij")

    per_dir = {name: [] for name in GLCM_NAMES}
    for dx, dy in offsets:
        p = _glcm_direction(quantized, dx, dy, levels)
        mu_i = float((p.sum(axis=1) * idx).sum())
        mu_j = float((p.sum(axis=0) * idx).sum())
        sigma_i = float(np.sqrt((p.sum(axis=1) * (idx - mu_i) ** 2).sum()))
        sigma_j = float(np.sqrt((p.sum(axis=0) * (idx - mu_j) ** 2).sum()))
        diff = i_grid - j_grid
        contrast = float((p * diff**2).sum())
        dissimilarity = float((p * np.abs(diff)).sum())
        homogeneity = float((p / (1.0 + diff**2)).sum())
        energy = float((p**2).sum())
        denom = max(sigma_i * sigma_j, EPS)
        correlation = float(((i_grid - mu_i) * (j_grid - mu_j) * p).sum() / denom)
        nz = p[p > 0]
        entropy = float(-(nz * np.log(nz)).sum())
        per_dir["glcm_contrast"].append(contrast)
        per_dir["glcm_dissimilarity"].append(dissimilarity)
        per_dir["glcm_homogeneity"].append(homogeneity)
        per_dir["glcm_energy"].append(energy)
        per_dir["glcm_correlation"].append(correlation)
        per_dir["glcm_entropy"].append(entropy)
    return {name: float(np.mean(vals)) for name, vals in per_dir.items()}


def extract_one(image_path: str, research_id: int, inverted_ids) -> dict[str, float]:
    a = load_gray_array(image_path, research_id, inverted_ids)
    feats = _first_order(a)
    feats.update(_glcm_features(a))
    return feats


# ---------------------------------------------------------------------------
# 교란(nuisance) 전역 통계 6개 -- 실험10 exp_trivial_stats.py 가 도입한 그 정의.
# 세 번째 사용처(실험14 영상 교란보정 검정)가 생겨 여기로 승격했다. 정의를 바꾸면
# outputs/image_permutation/trivial_stats.json 과 비교가 깨지므로 고정한다.
#   w, h      원본 PNG 폭/높이 -- 크롭 범위(체격 - FOV) 대리
#   mean,std  전역 밝기 - 대비
#   frac_hot  밝은 화소 비율(>0.6). MIP 은 고섭취가 어둡게 찍히므로 사실상 배경 면적
#   frac_dark 어두운 화소 비율(<0.1) -- 최고섭취 영역 면적 대리
# ---------------------------------------------------------------------------
TRIVIAL_NAMES = ("w", "h", "mean", "std", "frac_hot", "frac_dark")


def extract_trivial_one(image_path: str, research_id: int, inverted_ids) -> dict[str, float]:
    img = Image.open(image_path).convert("L")
    if int(research_id) in inverted_ids:
        img = ImageOps.invert(img)
    w, h = img.size
    a = np.asarray(img, dtype=np.float32) / 255.0
    return {"w": float(w), "h": float(h), "mean": float(a.mean()), "std": float(a.std()),
            "frac_hot": float((a > 0.6).mean()), "frac_dark": float((a < 0.1).mean())}


def extract_trivial_stats(image_dir: str, research_ids, inverted_ids=frozenset()) -> "pd.DataFrame":
    """환자별 전역 통계 6개. 라벨 미사용이라 fold 밖에서 한 번만 호출한다."""
    import pandas as pd

    rows = {int(rid): extract_trivial_one(os.path.join(image_dir, f"{int(rid)}.png"),
                                          int(rid), inverted_ids)
            for rid in research_ids}
    return pd.DataFrame.from_dict(rows, orient="index")[list(TRIVIAL_NAMES)]


def extract_features(image_dir: str, research_ids, inverted_ids=frozenset()) -> "pd.DataFrame":
    """환자별 radiomics 특징표. 라벨 미사용이므로 fold 밖에서 한 번만 호출한다."""
    import pandas as pd

    rows = {}
    for rid in research_ids:
        path = os.path.join(image_dir, f"{int(rid)}.png")
        rows[int(rid)] = extract_one(path, int(rid), inverted_ids)
    return pd.DataFrame.from_dict(rows, orient="index")[list(FEATURE_NAMES)]
