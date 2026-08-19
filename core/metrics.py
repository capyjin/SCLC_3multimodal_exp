# -*- coding: utf-8 -*-
"""평가 보조 — C-index 관례, 과적합 격차, fold 쌍대 검정.

여기 모인 함수들은 전부 **여러 실험 스크립트에 같은 내용이 복붙돼 있던 것**이다.
특히 ``train_val_gap`` 은 4개 파일(ablation / bert_text / suv_features /
text_source)에 글자 단위로 같은 4줄 루프로 들어 있었다. 정의가 한 곳에만 있어야
"이 표의 gap 과 저 표의 gap 이 같은 정의인가"를 매번 확인하지 않아도 된다.
"""
import numpy as np
from lifelines.utils import concordance_index
from scipy import stats


def cindex(durations, risks, events) -> float:
    """이 저장소의 부호 관례를 고정한 Harrell C-index.

    학습 루프(``train.cox_ph_loss``)는 **위험점수가 클수록 빨리 사건**이 나도록
    학습한다. lifelines 는 반대로 "점수가 클수록 오래 산다"를 가정하므로 항상
    부호를 뒤집어야 한다. 그 부호를 빼먹는 실수가 조용한 0.5-대칭 오류로
    이어지기 때문에 함수로 못 박아 둔다.
    """
    return float(concordance_index(np.asarray(durations, dtype=float),
                                   -np.asarray(risks, dtype=float),
                                   np.asarray(events, dtype=float)))


def fold_mean_cindex(score_map: dict, labels, plan, target: str) -> float:
    """fold 별 test 환자에서 C-index 를 구해 평균 (late fusion 평가와 동일 방식).

    fold 마다 모델이 달라 위험점수의 척도가 다르므로, 238명을 한 덩어리로
    이어붙여 순위를 매기면 fold 간 척도 차이가 신호로 섞인다. 그래서 항상
    fold 안에서만 비교한다.
    """
    cis = []
    for _fold, ids in plan:
        te = ids["test"]
        cis.append(cindex(labels.loc[te, f"{target}_days"].to_numpy(float),
                          [score_map[i] for i in te],
                          labels.loc[te, f"{target}_event"].to_numpy(float)))
    return float(np.mean(cis))


def train_val_gap(training_history: list[dict]) -> list[float]:
    """fold 별 (train C-index − val C-index) 격차 = 과적합 정도.

    각 fold 에서 **val 이 가장 좋았던 epoch**(= 체크포인트로 선택되는 그 시점)의
    격차를 쓴다. 마지막 epoch 이 아니라 선택된 시점을 재야 실제로 평가에 쓰인
    모델의 과적합을 재는 것이 된다.
    """
    gaps = []
    for fold in sorted({row["fold"] for row in training_history}):
        rows = [r for r in training_history if r["fold"] == fold]
        best = max(rows, key=lambda r: r["val_cindex"])
        gaps.append(float(best["train_cindex"] - best["val_cindex"]))
    return gaps


def mean_train_val_gap(training_history: list[dict]) -> float | None:
    gaps = train_val_gap(training_history)
    return float(np.mean(gaps)) if gaps else None


def paired_pvalues(arm_folds, base_folds) -> dict:
    """fold 쌍대 비교 (paired t-test + Wilcoxon).

    ⚠️ n=5 에서 wilcoxon 양측 p 의 최소값은 0.0625 라 **원리적으로 0.05 를 못
    넘는다.** 그래서 paired t-test 를 같이 낸다. 둘 다 fold 5개짜리라 검정력이
    매우 낮으니 참고용으로만 읽어야 한다.
    """
    a, b = np.asarray(arm_folds, dtype=float), np.asarray(base_folds, dtype=float)
    out = {"n_folds": int(len(a)), "n_improved": int((a > b).sum())}
    if len(a) != len(b) or len(a) < 2 or np.allclose(a, b):
        out["ttest_p"], out["wilcoxon_p"] = None, None
        return out
    out["ttest_p"] = float(stats.ttest_rel(a, b).pvalue)
    try:
        out["wilcoxon_p"] = float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        out["wilcoxon_p"] = None
    return out
