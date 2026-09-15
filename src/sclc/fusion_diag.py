# -*- coding: utf-8 -*-
"""late fusion **진단** 원자 — 저장된 OOF 위험점수만으로 도는 계산들.

[왜 이 모듈이 따로 있나]
  late fusion 을 검증하는 코드는 학습 코드와 성질이 다르다. 재학습이 없고
  (캐시된 OOF 만 읽는다), 수백 번 반복되며(순열검정 200회), 그래서 **조용해야
  한다**. ``fusion_stack.combine_risk_scores`` 는 fold 마다 계수를 출력하는데,
  그걸 200번 부르면 로그가 1000줄 늘어난다.

  정리 전에는 그 조용한 버전이 실험 폴더마다 한 벌씩 있었다:

      실험5/analyze_late_fusion_pfs.py :  stack_cindex() / tab_only_cindex()
      실험5/verify_shuffle_sanity.py   :  fold_mean_cindex()
      실험1/seed_ensemble.py           :  fold_cindex() / normalize_within_folds()

  세 벌이 같은 절차(fold 안에서만 비교, 위험점수 부호 반전)를 각자 구현하고
  있었고, 하나라도 부호나 fold 경계를 틀리면 조용히 0.5 근처 값이 나온다.

[이 모듈이 지키는 두 가지 규율]
  1. **fold 안에서만 비교한다.** fold 마다 모델이 달라 위험점수의 척도가
     다르므로, 238명을 한 덩어리로 이어붙여 순위를 매기면 fold 간 척도 drift
     가 신호로 섞인다 (실험9 에서 실측으로 확인된 함정).
  2. **메타학습기는 train fold 로만 적합한다.** ``iter_fold_stack`` 이 train
     환자의 OOF 점수로 CoxPH 를 적합하고 test 에만 적용한다 —
     ``fusion_stack.combine_risk_scores`` 와 정확히 같은 루프이며, 실제로 그
     함수가 이 생성기를 쓴다(정의가 두 곳에 있으면 언젠가 갈라진다).
"""
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy import stats

from sclc.metrics import cindex


# ---------------------------------------------------------------------------
# 1) fold별 CoxPH stack — 결합기와 진단이 공유하는 루프
# ---------------------------------------------------------------------------
def risk_frame(risks: dict[str, dict], ids, labels, target: str, names=None) -> pd.DataFrame:
    """``{공변량이름: {research_id: 위험점수}}`` + 환자 명단 -> CoxPH 설계행렬.

    컬럼 순서는 ``names`` 순서를 따른다 (CoxPH 계수 딕셔너리의 키 순서가 곧
    이 순서다). ``duration``/``event`` 는 마지막에 붙는다.
    """
    names = list(names or risks)
    data = {name: [risks[name][i] for i in ids] for name in names}
    data["duration"] = labels.loc[ids, f"{target}_days"].to_numpy(dtype=float)
    data["event"] = labels.loc[ids, f"{target}_event"].to_numpy(dtype=float)
    return pd.DataFrame(data)


def iter_fold_stack(risks: dict[str, dict], labels, plan, target: str, names=None):
    """fold 마다 ``(fold, ids, cph, test_df, names)`` 를 내놓는 생성기.

    CoxPH 는 **train 환자의 OOF 점수로만** 적합된다. 이 한 줄이 late fusion
    전체의 누수 방어선이라, 정의가 한 곳에만 있어야 한다.
    """
    names = list(names or risks)
    for fold, ids in plan:
        train_df = risk_frame(risks, ids["train"], labels, target, names)
        test_df = risk_frame(risks, ids["test"], labels, target, names)
        cph = CoxPHFitter()
        cph.fit(train_df, duration_col="duration", event_col="event")
        yield fold, ids, cph, test_df, names


def stack_fold_cindices(risks: dict[str, dict], labels, plan, target: str,
                        names=None) -> list[float]:
    """CoxPH 로 묶은 위험점수의 fold별 C-index (조용한 버전).

    ``names`` 로 공변량 부분집합을 고를 수 있다 — ``["risk_tabular"]`` 만 주면
    "영상을 뺀 같은 파이프라인"이 되므로, 영상 기여도를 **같은 절차 안에서**
    비교할 수 있다 (다른 코드로 잰 tabular 수치와 대조하면 절차 차이가 섞인다).
    """
    cis = []
    for _fold, _ids, cph, test_df, cols in iter_fold_stack(risks, labels, plan, target, names):
        pred = cph.predict_partial_hazard(test_df[cols]).to_numpy()
        cis.append(cindex(test_df["duration"], pred, test_df["event"]))
    return cis


def stack_mean_cindex(risks, labels, plan, target, names=None) -> float:
    return float(np.mean(stack_fold_cindices(risks, labels, plan, target, names)))


# ---------------------------------------------------------------------------
# 2) 계수 검정 — "영상 축의 beta 가 0과 다른가"
# ---------------------------------------------------------------------------
def coef_per_fold(risks, labels, plan, target, covariate: str = "risk_image") -> list[dict]:
    """fold 마다 train 환자로 적합한 CoxPH 에서 ``covariate`` 의 계수·95%CI·p."""
    rows = []
    for fold, ids, cph, _test_df, _names in iter_fold_stack(risks, labels, plan, target):
        s = cph.summary.loc[covariate]
        rows.append({"fold": fold, "coef": float(s["coef"]),
                     "lo": float(s["coef lower 95%"]), "hi": float(s["coef upper 95%"]),
                     "p": float(s["p"])})
    return rows


def coef_summary(frame: pd.DataFrame, covariate: str) -> dict:
    """전체 환자로 적합한 CoxPH 에서 한 공변량의 계수 블록."""
    cph = CoxPHFitter().fit(frame, duration_col="duration", event_col="event")
    s = cph.summary.loc[covariate]
    return {"model": cph, "coef": float(s["coef"]), "lo": float(s["coef lower 95%"]),
            "hi": float(s["coef upper 95%"]), "p": float(s["p"])}


def likelihood_ratio_test(frame: pd.DataFrame, reduced_cols: list[str],
                          full_cols: list[str]) -> dict:
    """중첩된 두 Cox 모형의 우도비 검정 — "공변량을 더하면 설명력이 느는가".

    ⚠️ 여기 들어가는 위험점수는 전체 코호트의 OOF 라, 이 검정은 fold 를
    가로질러 pooled 로 계산된다(원래 실험5 구현 그대로). 절대적 크기보다
    OS/PFS 사이의 **대비**를 읽기 위한 값이다.
    """
    keep = ["duration", "event"]
    m_reduced = CoxPHFitter().fit(frame[reduced_cols + keep], duration_col="duration", event_col="event")
    m_full = CoxPHFitter().fit(frame[full_cols + keep], duration_col="duration", event_col="event")
    stat = 2 * (m_full.log_likelihood_ - m_reduced.log_likelihood_)
    df = len(full_cols) - len(reduced_cols)
    return {"stat": float(stat), "df": df, "p": float(stats.chi2.sf(stat, df)),
            "ll_reduced": float(m_reduced.log_likelihood_),
            "ll_full": float(m_full.log_likelihood_)}


# ---------------------------------------------------------------------------
# 3) 보완성 — "주 모달리티가 틀린 쌍을 보조 모달리티가 맞히는가"
# ---------------------------------------------------------------------------
def rescue_rate(durations, events, primary_risk, backup_risk) -> dict:
    """비교가능한 모든 환자쌍 중 **primary 가 틀린 쌍**에서 backup 의 정답률.

    C-index 는 "전체 쌍 중 몇 %를 맞혔나"라서 두 모달리티가 같은 쌍을 맞히는지
    다른 쌍을 맞히는지 구분하지 못한다. 융합이 이득을 보려면 backup 이
    primary 가 놓친 쌍을 맞혀야 하므로, 그 부분집합에서만 정답률을 잰다.
    0.5 근처면 "그 쌍들에 대해서는 동전던지기" = 보완성 없음.
    """
    d = np.asarray(durations, dtype=float)
    e = np.asarray(events, dtype=float)
    rp = np.asarray(primary_risk, dtype=float)
    rb = np.asarray(backup_risk, dtype=float)
    ok = tot = 0
    n = len(d)
    for a in range(n):
        for b in range(a + 1, n):
            # 비교가능한 쌍인지: 먼저 일어난 쪽이 event 여야 한다
            if d[a] < d[b] and e[a] == 1:
                first, second = a, b
            elif d[b] < d[a] and e[b] == 1:
                first, second = b, a
            else:
                continue
            if rp[first] > rp[second]:
                continue                       # primary 가 맞힌 쌍은 건너뛴다
            tot += 1
            if rb[first] > rb[second]:
                ok += 1
    return {"correct": ok, "total": tot, "rate": float(ok / tot) if tot else None}


# ---------------------------------------------------------------------------
# 4) 난수 대조군 — "그 축을 난수로 바꿔도 결과가 같은가"
# ---------------------------------------------------------------------------
def permutation_draws(ids, values, score_fn, n_repeat: int = 200, seed: int = 42) -> np.ndarray:
    """환자-점수 대응을 무작위로 섞어 ``score_fn(섞인 map)`` 을 ``n_repeat`` 번.

    분포 자체(값들의 히스토그램)는 그대로 두고 **누구의 점수인지만** 섞으므로,
    "이 축이 진짜 정보를 담고 있나"를 축의 스케일과 무관하게 검정한다.

    ⚠️ 난수 소비 순서가 곧 재현성이다. ``ids`` 순서와 ``seed`` 를 바꾸면 기존
    산출물(``pfs_diagnosis.json`` 의 ``draws``)이 재현되지 않는다.
    """
    ids = list(ids)
    vals = np.asarray([values[i] for i in ids], dtype=float)
    rng = np.random.default_rng(seed)
    return np.array([score_fn(dict(zip(ids, rng.permutation(vals)))) for _ in range(n_repeat)])


def null_block(draws: np.ndarray, observed: float) -> dict:
    """난수 대조 분포 요약. ``frac_random_beats_real`` 이 0 에 가까워야 정보가 있다."""
    return {"mean": float(draws.mean()), "std": float(draws.std()),
            "p2_5": float(np.percentile(draws, 2.5)),
            "p97_5": float(np.percentile(draws, 97.5)),
            "frac_random_beats_real": float((draws >= observed).mean()),
            "n_repeat": len(draws),
            "draws": [float(v) for v in draws]}


# ---------------------------------------------------------------------------
# 5) 시드 앙상블 — 위험점수를 fold 안에서 정규화한 뒤 평균
# ---------------------------------------------------------------------------
NORMALIZE_MODES = ("raw", "z", "rank")


def normalize_within_folds(risk_by_id: dict, plan, mode: str) -> dict:
    """(이 시드, 각 fold) 안에서만 정규화한다. mode: ``raw`` | ``z`` | ``rank``

    fold 를 가로질러 정규화하면 fold 간 스케일 drift 가 섞여 들어간다.
    """
    if mode == "raw":
        return dict(risk_by_id)
    if mode not in NORMALIZE_MODES:
        raise ValueError(f"알 수 없는 mode: {mode} (가능: {NORMALIZE_MODES})")
    out = {}
    for _fold, ids in plan:
        te = ids["test"]
        r = np.array([risk_by_id[i] for i in te], dtype=float)
        if mode == "z":
            sd = r.std()
            v = (r - r.mean()) / sd if sd > 0 else np.zeros_like(r)
        else:  # rank
            v = r.argsort().argsort().astype(float) / max(len(r) - 1, 1)
        out.update(dict(zip(te, v)))
    return out


def ensemble_risks(risks_per_seed: list[dict], plan, mode: str) -> dict:
    """여러 시드의 위험점수를 fold 안 정규화 후 평균 — 학습 파라미터 추가 없음.

    각 시드의 OOF 점수는 그 환자를 학습에 쓰지 않은 체크포인트가 낸 값이므로,
    시드끼리 평균내도 "그 환자 라벨을 본 적 없다"는 성질이 유지된다.
    """
    normed = [normalize_within_folds(r, plan, mode) for r in risks_per_seed]
    return {i: float(np.mean([n[i] for n in normed])) for i in normed[0]}
