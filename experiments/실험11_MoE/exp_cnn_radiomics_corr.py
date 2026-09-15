# -*- coding: utf-8 -*-
"""CNN 딥피처와 radiomics 특징이 서로 중복되는지 검정 (MoE 4-expert 후보 설계용).

동기:
  4-expert MoE 후보(SimpleCNN + Radiomics + RadBERT/TF-IDF + clinical MLP)에서
  CNN 과 radiomics 를 별도 expert 로 두는 게 의미가 있으려면 두 특징이 서로 다른
  정보를 담고 있어야 한다. 실험10(E팔, exp_trivial_stats.py)은 이미 "CNN
  딥피처(512차원)가 전역 통계 6개(w/h/mean/std/frac_hot/frac_dark)와 거의 같은
  성능을 낸다"는 것을 보였다 -- CNN 이 배운 것 자체가 저차 정보에 가깝다는
  정황이었다. 이 실험은 그 정황을 라벨 없이 **직접** 검정한다: radiomics
  특징(1차 통계+GLCM 텍스처, sclc.radiomics)이 CNN 임베딩만으로 얼마나
  설명되는지를 측정한다.

측정 3가지 (전부 라벨 미사용, 순수 특징 간 비교, out-of-fold):
  1) ridge R^2 (5-fold CV, RidgeCV 로 알파도 train fold 안에서만 선택) :
     radiomics 각 특징을 예측변수로 예측했을 때 out-of-fold R^2. 높으면 "그
     radiomics 특징은 예측변수 안에 이미 있다"(중복). 예측변수 셋 3개를 비교:
       trained_cnn_512d  : 학습된 SimpleCNN 백본 임베딩
       random_cnn_512d   : 랜덤초기화 CNN (실험8/10과 동일한 "구조만 같고
                            학습 안 한" 대조군) -- 이게 trained 와 비슷하면
                            "CNN 이 학습으로 얻은 것"이 아니라 "그 구조가
                            어차피 포착하는 저차 정보"라는 뜻.
       trivial_6stats    : 실험10 식 전역통계 6개 -- 이게 위 둘과 비슷하면
                           radiomics 도 결국 그 6개 수준의 정보라는 뜻.
  2) rank-1 CCA 의 held-out canonical correlation (5-fold CV) : 두 특징 블록
     사이 첫 정준상관을 학습에 안 쓴 fold 에 투영해 계산한다. in-sample CCA 는
     차원 수가 크면 인위적으로 1에 가까워지므로 반드시 CV 로 봐야 한다.
  3) RV 계수 (in-sample, 참고용) : 두 블록(PCA 로 같은 차원 20으로 축소) 사이
     전체 상관구조를 요약하는 단일 지표(정준상관의 다변량 일반화).

CNN 임베딩 출처: outputs/image_cph/emb_os_fold{1..5}.npz -- 실험8이 이미 만들어
둔 fold-safe OOF 캐시(읽기 전용). os/pfs 의 fold 배정이 완전히 동일함을 직접
확인했으므로(각 fold test_rid 집합 일치) os 캐시 하나로 238명 전원을 커버한다.

radiomics 특징: sclc.radiomics.extract_features -- 라벨 미사용 순수 이미지
함수라서 fold 밖에서 한 번만 계산해도 누수 없음(실험10과 같은 논리).

Run: python 실험11_MoE/exp_cnn_radiomics_corr.py
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import json

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from sclc import cohort, radiomics
from sclc.dataset import INVERTED_IMAGE_IDS

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "cnn_radiomics_corr")
EMB_DIR = os.path.join(PROJECT_ROOT, "outputs", "image_cph")
SEED = 42
N_SPLITS = 5
PCA_K = 20  # CNN 512차원 -> CCA/RV 전에 축소할 차원
RIDGE_ALPHAS = np.logspace(-3, 4, 15)


def trivial_six_stats(image_dir: str, research_ids) -> pd.DataFrame:
    """실험10/exp_trivial_stats.py::image_stats 와 동일한 정의를 독립적으로
    재구현 (코드_구조.md 원칙 1: 실험 폴더끼리 import 금지. 세 번째 사용처가
    생기면 core 로 승격할 후보)."""
    rows = {}
    for rid in research_ids:
        img = Image.open(os.path.join(image_dir, f"{int(rid)}.png")).convert("L")
        if int(rid) in INVERTED_IMAGE_IDS:
            img = ImageOps.invert(img)
        w, h = img.size
        a = np.asarray(img, dtype=np.float32) / 255.0
        rows[int(rid)] = {"w": float(w), "h": float(h), "mean": float(a.mean()),
                          "std": float(a.std()), "frac_hot": float((a > 0.6).mean()),
                          "frac_dark": float((a < 0.1).mean())}
    return pd.DataFrame.from_dict(rows, orient="index")


def load_cnn_oof(target: str = "os") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """outputs/image_cph 의 fold-safe OOF 캐시에서 (trained_emb, random_emb, rid)
    를 238명 전원에 대해 모은다 (읽기 전용, 실험8의 산출물)."""
    trained, random_, rids = [], [], []
    for fold in range(1, 6):
        path = os.path.join(EMB_DIR, f"emb_{target}_fold{fold}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} 없음 -- 먼저 실험8/exp_image_cph.py 를 돌려야 함")
        z = np.load(path)
        trained.append(z["test_emb"])
        random_.append(z["test_emb_random"])
        rids.append(z["test_rid"])
    return np.concatenate(trained), np.concatenate(random_), np.concatenate(rids)


def cv_ridge_r2(X: np.ndarray, Y: np.ndarray, n_splits=N_SPLITS, seed=SEED) -> np.ndarray:
    """각 Y 열(radiomics 특징)을 X 로 예측한 out-of-fold R^2. 표준화와 알파
    선택 모두 매 fold 의 train 행에서만 한다(RidgeCV 내부 LOOCV도 train 행 안).
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_pred = np.zeros_like(Y, dtype=np.float64)
    for tr_idx, te_idx in kf.split(X):
        x_scaler = StandardScaler().fit(X[tr_idx])
        y_scaler = StandardScaler().fit(Y[tr_idx])
        x_tr, x_te = x_scaler.transform(X[tr_idx]), x_scaler.transform(X[te_idx])
        y_tr = y_scaler.transform(Y[tr_idx])
        model = RidgeCV(alphas=RIDGE_ALPHAS).fit(x_tr, y_tr)
        pred_std = model.predict(x_te)
        oof_pred[te_idx] = y_scaler.inverse_transform(pred_std)
    ss_res = ((Y - oof_pred) ** 2).sum(axis=0)
    ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / np.clip(ss_tot, 1e-12, None)


def cv_rank1_cca(X: np.ndarray, Y: np.ndarray, n_splits=N_SPLITS, seed=SEED) -> tuple[float, list[float]]:
    """첫 정준상관을 held-out fold 에 투영해 계산 (in-sample CCA 는 차원이
    크면 인위적으로 1에 가까워지므로 CV 필수)."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    corrs = []
    for tr_idx, te_idx in kf.split(X):
        x_scaler = StandardScaler().fit(X[tr_idx])
        y_scaler = StandardScaler().fit(Y[tr_idx])
        x_tr, x_te = x_scaler.transform(X[tr_idx]), x_scaler.transform(X[te_idx])
        y_tr, y_te = y_scaler.transform(Y[tr_idx]), y_scaler.transform(Y[te_idx])
        cca = CCA(n_components=1).fit(x_tr, y_tr)
        u_te, v_te = cca.transform(x_te, y_te)
        corrs.append(float(np.corrcoef(u_te[:, 0], v_te[:, 0])[0, 1]))
    return float(np.mean(corrs)), corrs


def rv_coefficient(X: np.ndarray, Y: np.ndarray) -> float:
    """RV 계수 -- 두 특징 블록 사이 전체 상관구조의 다변량 일반화(0~1)."""
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    Sxy = Xc.T @ Yc
    Sxx = Xc.T @ Xc
    Syy = Yc.T @ Yc
    num = np.trace(Sxy @ Sxy.T)
    den = np.sqrt(np.trace(Sxx @ Sxx.T) * np.trace(Syy @ Syy.T))
    return float(num / den)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    cohort_df = cohort.load_trimodal_cohort()
    research_ids = sorted(cohort_df["research_id"].unique().tolist())
    print(f"코호트 {len(research_ids)}명")

    print("radiomics 특징 추출 중 (1차 통계 + GLCM, 라벨 미사용)...")
    rad_df = radiomics.extract_features(cohort.DEFAULT_IMAGE_DIR, research_ids, set(INVERTED_IMAGE_IDS))
    trivial_df = trivial_six_stats(cohort.DEFAULT_IMAGE_DIR, research_ids)

    trained_emb, random_emb, rids = load_cnn_oof("os")
    order = np.argsort(rids)
    trained_emb, random_emb, rids = trained_emb[order], random_emb[order], rids[order]
    assert list(map(int, rids)) == research_ids, "CNN OOF 캐시의 rid 순서가 코호트와 다름"

    Y = rad_df.loc[research_ids, list(radiomics.FEATURE_NAMES)].to_numpy(dtype=np.float64)
    trivial_X = trivial_df.loc[research_ids, ["w", "h", "mean", "std", "frac_hot", "frac_dark"]].to_numpy(dtype=np.float64)

    results = {"n_patients": len(research_ids), "radiomics_features": list(radiomics.FEATURE_NAMES)}

    print("\n=== (1) ridge R^2 (5-fold CV): radiomics <- 예측변수 ===")
    for name, X in (("trained_cnn_512d", trained_emb),
                    ("random_cnn_512d", random_emb),
                    ("trivial_6stats", trivial_X)):
        r2 = cv_ridge_r2(X, Y)
        results.setdefault("ridge_r2", {})[name] = {
            "per_feature": {f: round(float(v), 4) for f, v in zip(radiomics.FEATURE_NAMES, r2)},
            "mean": float(np.mean(r2)), "median": float(np.median(r2)),
        }
        print(f"  {name:<18} mean R^2={np.mean(r2):.4f}  median={np.median(r2):.4f}  "
              f"(min={r2.min():.4f}, max={r2.max():.4f})")

    print("\n=== (2) rank-1 CCA, held-out canonical correlation (5-fold CV) ===")
    trained_pca = PCA(n_components=PCA_K, random_state=SEED).fit_transform(trained_emb)
    random_pca = PCA(n_components=PCA_K, random_state=SEED).fit_transform(random_emb)
    for name, X in (("trained_cnn_pca20", trained_pca),
                    ("random_cnn_pca20", random_pca),
                    ("trivial_6stats", trivial_X)):
        mean_r, fold_r = cv_rank1_cca(X, Y)
        results.setdefault("cca_rank1_heldout", {})[name] = {
            "mean": mean_r, "per_fold": [round(r, 4) for r in fold_r],
        }
        print(f"  {name:<18} held-out canonical r = {mean_r:.4f}  (folds={['%.3f' % r for r in fold_r]})")

    print("\n=== (3) RV 계수 (전체 표본 적합, in-sample -- 참고용) ===")
    y_std = StandardScaler().fit_transform(Y)
    for name, X in (("trained_cnn_pca20", trained_pca),
                    ("random_cnn_pca20", random_pca),
                    ("trivial_6stats", trivial_X)):
        rv = rv_coefficient(StandardScaler().fit_transform(X), y_std)
        results.setdefault("rv_coefficient", {})[name] = rv
        print(f"  {name:<18} RV = {rv:.4f}")

    out_path = os.path.join(OUT_DIR, "results.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
