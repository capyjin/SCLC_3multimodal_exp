# -*- coding: utf-8 -*-
"""E팔 — CNN 없이 전역 통계량만으로 생존을 맞춰본다 (신호의 '정체' 규명).

순열검정(C팔)은 "영상에 신호가 있나"만 답한다. 있다고 나와도 그게 **종양
표현형**인지 **촬영/체격 아티팩트**인지는 구분하지 못한다. 그래서 CNN 을 아예
빼고, PNG 한 장에서 눈감고 뽑을 수 있는 스칼라 6개만으로 같은 fold 에서
CoxPH 를 적합한다. 이것이 CNN(0.6570/0.6154)에 근접하면, 영상 브랜치가 배운
것의 상당 부분은 종양이 아니라 전역 밝기·대비·크롭 기하로 설명된다.

    w, h        원본 PNG 폭/높이 — 크롭 범위(체격·FOV) 대리. 512x512 로 리사이즈
                될 때 종횡비 왜곡으로 신경망에도 그대로 들어간다.
    mean, std   픽셀 평균/표준편차 — 전역 밝기와 대비
    frac_hot    밝은 화소 비율(>0.6) — 고섭취 영역의 면적 대리
    frac_dark   어두운 화소 비율(<0.1) — 배경 면적 대리

누수 방지: 특징은 라벨을 전혀 보지 않는 순수 이미지 함수이고, 표준화 통계와
CoxPH 계수는 **fold 의 학습 환자만**으로 적합한다. penalizer 는 0.1 로 고정해
test 를 보고 고르는 일이 없게 한다(탐색 금지).

공정한 비교를 위해 두 가지를 같이 낸다.
  · ``train`` (171명) — 이 저장소의 기본 관례
  · ``train+val`` (191명) — CNN 은 val 을 체크포인트 선택에 쓰므로 데이터 예산이
    같아진다. 이걸 안 내면 E 가 불리한 조건에서 잰 값이 된다.
그리고 E 에도 **자기 순열 귀무**를 붙인다(학습이 없어 1000회가 몇 십 초). 귀무 없이
"E 가 CNN 과 구분 안 된다"고 말할 수는 없기 때문이다.

Run:  python 실험10_영상단독_난수대조검정/exp_trivial_stats.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from PIL import Image, ImageOps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nulls  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core import cohort  # noqa: E402
from core.dataset import INVERTED_IMAGE_IDS  # noqa: E402
from core.metrics import cindex  # noqa: E402
from core.train import fold_plan  # noqa: E402

FEATURES = ("w", "h", "mean", "std", "frac_hot", "frac_dark")
PENALIZER = 0.1
N_PERM = 1000
PERM_SCOPES = ("global", "stratified")   # 앞이 주력 (C팔과 동일)
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "outputs", "image_permutation")


def image_stats(image_dir: str, research_ids) -> pd.DataFrame:
    """환자별 전역 통계량 표. 라벨을 보지 않으므로 fold 밖에서 한 번만 계산한다."""
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


def fold_cindices(stats: pd.DataFrame, labels: pd.DataFrame, plan, target: str,
                  feats: list[str], fit_on: tuple[str, ...] = ("train",)) -> list[float]:
    """fold 별 C-index. 표준화·CoxPH 모두 ``fit_on`` 분할에서만 적합한다(test 제외)."""
    out = []
    for _fold, ids in plan:
        fit_ids = [i for key in fit_on for i in ids[key]]
        train = stats.loc[fit_ids, feats]
        mu, sd = train.mean(), train.std().replace(0.0, 1.0)
        design = ((train - mu) / sd).copy()
        design["T"] = labels.loc[fit_ids, f"{target}_days"].to_numpy(float)
        design["E"] = labels.loc[fit_ids, f"{target}_event"].to_numpy(float)
        beta = CoxPHFitter(penalizer=PENALIZER).fit(design, "T", "E").params_[feats].to_numpy()
        risk = ((stats.loc[ids["test"], feats] - mu) / sd).to_numpy() @ beta
        out.append(cindex(labels.loc[ids["test"], f"{target}_days"].to_numpy(float), risk,
                          labels.loc[ids["test"], f"{target}_event"].to_numpy(float)))
    return out


def permutation_null(stats, cohort_df, plan, target, feats, scope, n_perm=N_PERM):
    """E 팔의 자기 귀무분포 — 라벨을 섞고 같은 CoxPH 절차를 반복한다."""
    draws = []
    for r in range(1, n_perm + 1):
        permuted = nulls.permute_labels(cohort_df, target, 42 + 100 * r, scope)
        plabels = permuted.drop_duplicates("research_id").set_index("research_id")
        draws.append(float(np.mean(fold_cindices(stats, plabels, plan, target, feats))))
    return np.array(draws)


def censoring_auc(stats, labels, plan, target, feats) -> float:
    """전역통계가 **중도절단 여부 자체**를 얼마나 맞히는가.

    이 값이 0.5 를 넘으면 층화 순열(=환자의 event 비트를 고정)의 귀무 중심이
    0.5 위로 밀린다. 두 순열 범위의 귀무가 왜 다른지를 설명하는 수치다.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    aucs = []
    for _fold, ids in plan:
        scaler = StandardScaler().fit(stats.loc[ids["train"], feats])
        model = LogisticRegression(max_iter=2000).fit(
            scaler.transform(stats.loc[ids["train"], feats]),
            labels.loc[ids["train"], f"{target}_event"].to_numpy(int))
        prob = model.predict_proba(scaler.transform(stats.loc[ids["test"], feats]))[:, 1]
        aucs.append(roc_auc_score(labels.loc[ids["test"], f"{target}_event"].to_numpy(int), prob))
    return float(np.mean(aucs))


def main() -> None:
    cohort_df = cohort.load_trimodal_cohort()
    labels = cohort_df.drop_duplicates("research_id").set_index("research_id")
    plan = fold_plan(cohort_df)
    stats = image_stats(cohort.DEFAULT_IMAGE_DIR, labels.index)
    all6 = list(FEATURES)

    results = {}
    arms = [(f, [f], ("train",)) for f in FEATURES]
    arms += [("ALL6", all6, ("train",)), ("ALL6_train_val", all6, ("train", "val"))]
    for name, feats, fit_on in arms:
        results[name] = {"fit_on": "+".join(fit_on)}
        for target in ("os", "pfs"):
            folds = fold_cindices(stats, labels, plan, target, feats, fit_on)
            results[name][target] = {"folds": [round(c, 4) for c in folds],
                                     "mean": float(np.mean(folds)), "std": float(np.std(folds))}

    for target in ("os", "pfs"):
        observed = results["ALL6"][target]["mean"]
        results["ALL6"][target]["nulls"], results["ALL6"][target]["permutation_p"] = {}, {}
        for scope in PERM_SCOPES:
            null = permutation_null(stats, cohort_df, plan, target, all6, scope)
            results["ALL6"][target]["nulls"][scope] = {
                "n": int(len(null)), "mean": float(null.mean()), "std": float(null.std()),
                "q95": float(np.percentile(null, 95))}
            results["ALL6"][target]["permutation_p"][scope] = float(
                (1 + int((null >= observed).sum())) / (len(null) + 1))
        results["ALL6"][target]["censoring_auc"] = censoring_auc(stats, labels, plan, target, all6)

    os.makedirs(DEFAULT_OUT, exist_ok=True)
    path = os.path.join(DEFAULT_OUT, "trivial_stats.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"features": all6, "penalizer": PENALIZER, "n_perm": N_PERM,
                   "perm_scopes": list(PERM_SCOPES), "results": results}, fh, ensure_ascii=False, indent=2)

    print(f"{'특징':<16}{'OS':>9}{'PFS':>9}    (SimpleCNN 영상단독 0.6570 / 0.6154)")
    for name in results:
        print(f"{name:<16}{results[name]['os']['mean']:9.4f}{results[name]['pfs']['mean']:9.4f}")
    for target in ("os", "pfs"):
        a = results["ALL6"][target]
        for scope in PERM_SCOPES:
            n = a["nulls"][scope]
            print(f"  ALL6 {target.upper()} 귀무[{scope:<10}] {n['mean']:.4f} ± {n['std']:.4f} "
                  f"(95분위 {n['q95']:.4f}) · 순열 p = {a['permutation_p'][scope]:.4f}")
        print(f"  ALL6 {target.upper()} 전역통계 -> 중도절단 예측 AUC = {a['censoring_auc']:.4f}")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
