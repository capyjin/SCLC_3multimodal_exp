# -*- coding: utf-8 -*-
"""[실험] Stage-aware late fusion — LS/ES가 서로 다른 융합 가중치를 배워야 하는가?

[연구 질문]
  SCLC는 LS(제한기, n=72)와 ES(확장기, n=166)의 치료법·진행 양상이 다르다.
  현재 최고 모델은 C+R concat 위험도와 영상 위험도를 CoxPH로 결합하는데,
  이때 **모든 환자에게 같은 가중치**를 쓴다. Stage별로 가중치를 다르게 학습시키면
  PFS 예측이 나아지는가?

[⚠️ 설계상 가장 중요한 점 — 왜 M1이 기준선인가]
  `stage`는 이미 clinical 범주형 변수 13개 중 하나라 C+R 모델의 입력이다.
  그런데 결합 단계에 stage를 **그냥 변수로 하나 더** 넣기만 해도 전체 C-index가
  0.6531 → 0.6671 로 오른다(LRT p=5.4e-5). 즉 "stage별 가중치" 효과와
  "stage 정보를 결합 단계에 추가" 효과는 **서로 다른 것**이며 섞이면 안 된다.
    M0_shared      : zT, zI                      (현재 모델)
    M1_stage_main  : zT, zI, sc                  ← **주 비교 기준선**
    M2_stage_aware : zT, zI, sc, sc*zT, sc*zI    ← 제안 모델
  **M2 vs M1 = 순수한 stage-aware fusion 효과** (둘 다 stage 직접효과를 포함).
  M2 vs M0 은 참고용으로만 보고한다(개선폭이 커 보이지만 원인 구분 불가).

[⚠️ 평가 지표 — pooled-OOF를 쓰면 안 되는 이유]
  fold별 모델이 내는 위험점수의 스케일이 크게 다르다(영상 fold별 평균 0.02~1.59).
  238명 OOF를 전부 이어붙여 순위를 매기면 이 fold 간 drift가 신호로 섞인다
  (실측: 융합모델 pooled-OOF 0.5779 vs fold별 평균 0.6531, 즉 -0.075 인공물).
  따라서 **fold 안에서만, stage 안에서만 비교쌍을 세어 합산**하는 방식을 주 지표로 쓴다.
  이 지표에서 LS 비교쌍은 fold별 187/176/85/54/18개로 극히 불균등하므로
  단순 fold 평균이 아니라 **쌍 개수로 가중**해야 한다(fold5가 3.5%인데 20% 영향 방지).

[누수 방지]
  - 표준화(μ,σ): 각 outer fold의 **train 171명으로만** 적합, test에 그대로 적용.
  - CoxPH 계수: fold별 train 환자의 OOF 위험점수로만 적합.
  - test set으로 λ를 고르지 않는다(inner-CV는 train 안에서만).
  - 영상 OOF 재사용의 정당성: ImageOnlyEvaluator는 clinical_frame에서
    research_id/기간/사건만 읽고 brain_meta를 전혀 보지 않으므로, brain_meta 수정
    전후로 영상 위험점수가 불변이다. 이를 verify_image_oof_invariance()로 검증한다.

[무엇을 재학습하고 무엇을 재사용하나]
  tabular(C+R): **재학습**. 저장된 OOF는 brain_meta 수정 이전 값이라 무효.
  image       : **재사용**(위 불변성 근거). GPU 20분 절약 + 3-way 실험과 bit-exact 일치.

Run:  python fusion/exp_stage_aware_fusion.py --targets pfs
      python fusion/exp_stage_aware_fusion.py --targets pfs,os --bootstrap 2000
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import hashlib
import json
import itertools

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

from core import cohort
from core.train import fold_plan

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "stage_aware_fusion")
LEGACY_OOF_TMPL = os.path.join(PROJECT_ROOT, "outputs", "late_fusion_B", "oof_{target}.json")
STAGE_LS, STAGE_ES = 1, 2

# cols: 설계행렬 컬럼, strata: 층화 변수, pen: ridge를 걸 컬럼(상호작용만)
MODEL_SPECS = {
    "M0_shared":       dict(cols=["zT", "zI"],                         strata=None,      pen=[]),
    "M1_stage_main":   dict(cols=["zT", "zI", "sc"],                   strata=None,      pen=[]),
    "M2_stage_aware":  dict(cols=["zT", "zI", "sc", "scT", "scI"],     strata=None,      pen=["scT", "scI"]),
    "M2p_no_main":     dict(cols=["zT", "zI", "scT", "scI"],           strata=None,      pen=["scT", "scI"]),
    "M3_strat_shared": dict(cols=["zT", "zI"],                         strata=["group"], pen=[]),
    "M4_strat_aware":  dict(cols=["zT", "zI", "scT", "scI"],           strata=["group"], pen=["scT", "scI"]),
}
PRIMARY_CONTRAST = ("M2_stage_aware", "M1_stage_main")   # 순수 stage-aware 효과
LEGACY_CONTRAST = ("M2_stage_aware", "M0_shared")        # 참고(원 요청)
LAMBDA_GRID = (0.0, 0.01, 0.05, 0.1, 0.5, 1.0)
DECISION_THRESHOLD = 0.03      # within-stage 부트스트랩 SD(LS 0.032)에 맞춘 값


# ═══════════════════════════════════════════════════════════════════════════
# 1. OOF 위험점수 확보
# ═══════════════════════════════════════════════════════════════════════════
def verify_image_oof_invariance(target: str) -> dict:
    """[게이트 A] 영상 OOF 재사용의 정당성 검증.

    brain_meta 수정이 영상 브랜치에 영향을 줄 수 없음을 데이터로 증명한다:
    두 조건의 코호트가 brain_meta 컬럼 하나만 다르고, fold/split/기간/사건/stage 가
    전부 동일해야 한다. (ImageOnlyEvaluator 는 이 중 research_id/기간/사건만 읽는다.)
    """
    a = cohort.load_trimodal_cohort(fix_brain_meta=True).sort_values("research_id").reset_index(drop=True)
    b = cohort.load_trimodal_cohort(fix_brain_meta=False).sort_values("research_id").reset_index(drop=True)
    diff_cols = [c for c in a.columns if not a[c].equals(b[c])]
    must_match = ["research_id", "fold", "split", f"{target}_days", f"{target}_event", "stage"]
    ok = (diff_cols == ["brain_meta"]) and all(a[c].equals(b[c]) for c in must_match)
    return {"differing_columns": diff_cols, "invariant_columns_checked": must_match, "passed": bool(ok)}


def _sha(d: dict) -> str:
    s = json.dumps({str(k): round(float(v), 10) for k, v in sorted(d.items())}, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def get_oof_scores(target: str, *, tab_epochs=60, tab_batch=32, seed=42,
                   fix_brain_meta=True, reuse_image=True, cache=None,
                   force_retrain=False) -> dict:
    """{'tabular': {rid: risk}, 'image': {rid: risk}, 'provenance': {...}}"""
    if cache and os.path.exists(cache) and not force_retrain:
        d = json.load(open(cache))
        print(f"[oof] cache hit {os.path.basename(cache)}")
        return {"tabular": {int(k): v for k, v in d["tabular"].items()},
                "image": {int(k): v for k, v in d["image"].items()},
                "provenance": d["provenance"]}

    from core.fusion_stack import get_tabular_oof, oof_dict

    print(f"\n########## C+R(tabular) 재학습  target={target} "
          f"bs={tab_batch} ep={tab_epochs} fix_brain_meta={fix_brain_meta} ##########")
    ev = get_tabular_oof(target, batch_size=tab_batch, epochs=tab_epochs,
                         seed=seed, fix_brain_meta=fix_brain_meta)
    tab = oof_dict(ev.oof_predictions)

    if reuse_image:
        src = LEGACY_OOF_TMPL.format(target=target)
        img = {int(k): v for k, v in json.load(open(src))["image"].items()}
        img_src = src
        print(f"[oof] image 재사용 (재학습 없음): {os.path.basename(src)}  n={len(img)}")
    else:
        from core.fusion_stack import get_image_oof_simplecnn
        iev = get_image_oof_simplecnn(target, epochs=30)
        img = oof_dict(iev.oof_predictions)
        img_src = "retrained"

    prov = {"target": target, "fix_brain_meta": fix_brain_meta, "tab_epochs": tab_epochs,
            "tab_batch": tab_batch, "seed": seed, "image_source": img_src,
            "sha_tabular": _sha(tab), "sha_image": _sha(img),
            "tabular_cindex_perfold": [round(float(c), 4) for c in ev.c_indices]}
    out = {"tabular": tab, "image": img, "provenance": prov}
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        json.dump({"tabular": {str(k): v for k, v in tab.items()},
                   "image": {str(k): v for k, v in img.items()},
                   "provenance": prov}, open(cache, "w"), indent=2)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 2. 설계행렬 (누수 차단 지점)
# ═══════════════════════════════════════════════════════════════════════════
def build_design(ids, tab_risk, img_risk, labels, group_by_id, target,
                 *, fit_stats=None, normalize="zscore"):
    """fit_stats=None -> 이 ids로 μ/σ를 적합해 함께 반환(train fold).
       fit_stats 전달 -> 그 값을 그대로 적용만 함(test fold). ← 누수 차단 지점

    sc = +0.5(ES) / -0.5(LS) 고정. train 평균중심화를 쓰지 않는 이유:
      fold별 LS 비율이 27~36%로 달라 중심화하면 β_main의 의미가 fold마다 달라지고,
      ridge가 fold마다 다른 목표로 수축한다. ±0.5 고정이면
        weight_LS = β_main - 0.5*β_int,  weight_ES = β_main + 0.5*β_int
      이 되어 β_int→0 수축이 정확히 '공통 가중치 모델'로 수렴한다.
    """
    ids = [int(i) for i in ids]
    t = np.array([tab_risk[i] for i in ids], dtype=float)
    m = np.array([img_risk[i] for i in ids], dtype=float)

    if normalize == "none":
        stats = {"muT": 0.0, "sdT": 1.0, "muI": 0.0, "sdI": 1.0}
    elif fit_stats is None:
        stats = {"muT": float(t.mean()), "sdT": float(t.std() or 1.0),
                 "muI": float(m.mean()), "sdI": float(m.std() or 1.0)}
    else:
        stats = fit_stats
    zT = (t - stats["muT"]) / stats["sdT"]
    zI = (m - stats["muI"]) / stats["sdI"]

    grp = np.array([group_by_id[i] for i in ids])
    sc = np.where(grp == STAGE_ES, 0.5, -0.5)

    df = pd.DataFrame({
        "zT": zT, "zI": zI, "sc": sc, "scT": sc * zT, "scI": sc * zI,
        "group": grp,
        "duration": labels.loc[ids, f"{target}_days"].to_numpy(float),
        "event": labels.loc[ids, f"{target}_event"].to_numpy(float),
    }, index=pd.Index(ids, name="research_id"))
    return df, stats


def fit_predict_fold(train_df, test_df, spec, penalizer=0.0):
    cols, strata, pen_cols = spec["cols"], spec["strata"], spec["pen"]
    use = cols + (strata or []) + ["duration", "event"]
    pen = np.array([penalizer if c in pen_cols else 0.0 for c in cols]) if penalizer else 0.0

    cph = CoxPHFitter(penalizer=pen)
    cph.fit(train_df[use], duration_col="duration", event_col="event",
            strata=strata, robust=False)
    lp = cph.predict_log_partial_hazard(test_df[use])

    coefs = {c: float(cph.params_[c]) for c in cols}
    out = {
        "lp_test": {int(i): float(v) for i, v in lp.items()},
        "coefs": coefs,
        "p": {c: float(cph.summary.loc[c, "p"]) for c in cols},
        "loglik": float(cph.log_likelihood_),
        "n_train": int(len(train_df)), "n_events_train": int(train_df["event"].sum()),
        "n_events_train_LS": int(train_df.loc[train_df.group == STAGE_LS, "event"].sum()),
        "n_events_train_ES": int(train_df.loc[train_df.group == STAGE_ES, "event"].sum()),
    }
    # stage별 실효 가중치 (핵심 산출물)
    for base, inter in (("zT", "scT"), ("zI", "scI")):
        if inter in coefs:
            b, bi = coefs.get(base, 0.0), coefs[inter]
            out[f"weight_LS_{base}"] = b - 0.5 * bi
            out[f"weight_ES_{base}"] = b + 0.5 * bi
        elif base in coefs:
            out[f"weight_LS_{base}"] = out[f"weight_ES_{base}"] = coefs[base]
    return out


def run_cv(cohort_df, target, tab_risk, img_risk, group_by_id, spec_name,
           *, penalizer=0.0, normalize="zscore", fold_plan=None):
    spec = MODEL_SPECS[spec_name]
    labels = cohort_df.drop_duplicates("research_id").set_index("research_id")
    plan = fold_plan if fold_plan is not None else fold_plan(cohort_df, None)

    lp_by_id, folds = {}, []
    for fold, ids in plan:
        tr, te = list(ids["train"]), list(ids["test"])
        assert not (set(tr) & set(te)), f"fold{fold}: train/test 중복"
        tr_df, stats = build_design(tr, tab_risk, img_risk, labels, group_by_id, target,
                                    fit_stats=None, normalize=normalize)
        te_df, _ = build_design(te, tab_risk, img_risk, labels, group_by_id, target,
                                fit_stats=stats, normalize=normalize)   # ← 같은 stats 객체
        r = fit_predict_fold(tr_df, te_df, spec, penalizer)
        r["fold"], r["norm_stats"] = fold, stats
        lp_by_id.update(r.pop("lp_test"))
        folds.append(r)

    assert len(lp_by_id) == cohort_df["research_id"].nunique(), "모든 환자가 정확히 1회 test여야 함"
    return {"model": spec_name, "penalizer": penalizer, "lp_by_id": lp_by_id, "folds": folds}


# ═══════════════════════════════════════════════════════════════════════════
# 3. 평가 — fold내·stage내 쌍만 세어 합산
# ═══════════════════════════════════════════════════════════════════════════
def _pair_counts(dur, evt, risk):
    """(일치쌍수, 비교가능쌍수). lifelines 동점 규약(0.5) 동일."""
    conc = comp = 0.0
    n = len(dur)
    for a, b in itertools.combinations(range(n), 2):
        if dur[a] < dur[b] and evt[a] == 1:
            s, l = a, b
        elif dur[b] < dur[a] and evt[b] == 1:
            s, l = b, a
        else:
            continue
        comp += 1
        conc += 1.0 if risk[s] > risk[l] else (0.5 if risk[s] == risk[l] else 0.0)
    return conc, comp


def evaluate(lp_by_id, cohort_df, target, group_by_id, *, fold_plan=None):
    labels = cohort_df.drop_duplicates("research_id").set_index("research_id")
    plan = fold_plan if fold_plan is not None else fold_plan(cohort_df, None)

    perfold, buckets = [], {}
    for fold, ids in plan:
        te = [int(i) for i in ids["test"]]
        d = labels.loc[te, f"{target}_days"].to_numpy(float)
        e = labels.loc[te, f"{target}_event"].to_numpy(float)
        r = np.array([lp_by_id[i] for i in te])
        perfold.append(concordance_index(d, -r, e))
        for key, mask in (("LS", np.array([group_by_id[i] == STAGE_LS for i in te])),
                          ("ES", np.array([group_by_id[i] == STAGE_ES for i in te]))):
            if mask.sum() >= 2:
                c, n = _pair_counts(d[mask], e[mask], r[mask])
                acc = buckets.setdefault(key, [0.0, 0.0])
                acc[0] += c; acc[1] += n

    within = {k: {"cindex": (v[0] / v[1]) if v[1] else float("nan"), "pairs": int(v[1])}
              for k, v in buckets.items()}
    tot_c = sum(v[0] for v in buckets.values()); tot_n = sum(v[1] for v in buckets.values())
    within["stratified_overall"] = {"cindex": tot_c / tot_n if tot_n else float("nan"),
                                    "pairs": int(tot_n)}
    return {"overall_perfold_mean": float(np.mean(perfold)),
            "overall_perfold_sd": float(np.std(perfold)),
            "overall_perfold": [round(float(c), 4) for c in perfold],
            "within": within}


def paired_bootstrap_delta(lp_a, lp_b, cohort_df, target, group_by_id, *, B=2000, seed=42):
    """환자 단위 Poisson(1) 가중 부트스트랩. 두 모델을 같은 리샘플에서 동시 계산해
    Δ의 신뢰구간을 직접 구한다(주변 CI를 빼는 것보다 훨씬 좁고 정확)."""
    labels = cohort_df.drop_duplicates("research_id").set_index("research_id")
    plan = fold_plan(cohort_df, None)
    rng = np.random.default_rng(seed)

    pre = []
    for fold, ids in plan:
        te = [int(i) for i in ids["test"]]
        d = labels.loc[te, f"{target}_days"].to_numpy(float)
        e = labels.loc[te, f"{target}_event"].to_numpy(float)
        g = np.array([group_by_id[i] for i in te])
        ra = np.array([lp_a[i] for i in te]); rb = np.array([lp_b[i] for i in te])
        pairs = [(a, b) for a, b in itertools.combinations(range(len(te)), 2)
                 if (d[a] < d[b] and e[a] == 1) or (d[b] < d[a] and e[b] == 1)]
        idx = [(a, b) if (d[a] < d[b] and e[a] == 1) else (b, a) for a, b in pairs]
        pre.append((np.array(te), g, ra, rb, idx))

    def stat(weights, sel):
        ca = cb = n = 0.0
        for (te, g, ra, rb, idx), w in zip(pre, weights):
            for s, l in idx:
                if sel is not None and not (g[s] == sel and g[l] == sel):
                    continue
                ww = w[s] * w[l]
                if ww == 0:
                    continue
                n += ww
                ca += ww * (1.0 if ra[s] > ra[l] else 0.5 if ra[s] == ra[l] else 0.0)
                cb += ww * (1.0 if rb[s] > rb[l] else 0.5 if rb[s] == rb[l] else 0.0)
        return (ca / n - cb / n) if n else np.nan

    out = {}
    for name, sel in (("LS", STAGE_LS), ("ES", STAGE_ES), ("stratified_overall", None)):
        ds = []
        for _ in range(B):
            w = [rng.poisson(1.0, size=len(p[0])).astype(float) for p in pre]
            v = stat(w, sel)
            if not np.isnan(v):
                ds.append(v)
        ds = np.array(ds)
        out[name] = {"delta_mean": float(ds.mean()), "sd": float(ds.std()),
                     "ci95": [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))],
                     "P_gt_0": float((ds > 0).mean()), "B": len(ds)}
    return out


def nested_lrt(cohort_df, target, tab_risk, img_risk, group_by_id, small, big,
               *, normalize="zscore"):
    """중첩모형 LRT를 **fold별로 따로** 계산한다.

    ⚠️ 로그우도를 5개 fold에 걸쳐 합산하면 안 된다. fold별 train set(171명)은
       서로 80% 이상 겹치므로 독립 표본이 아니다. 합산하면 정보량을 ~5배로
       착각해 p값이 심하게 부풀려진다(실측: 합산 p<0.0001 vs fold별 p=0.21~0.90).
       따라서 fold별 p를 모두 보고하고, 요약은 '5개 중 몇 개가 유의한가'로 한다.
    """
    from scipy import stats as sps
    labels = cohort_df.drop_duplicates("research_id").set_index("research_id")
    df = len(MODEL_SPECS[big]["cols"]) - len(MODEL_SPECS[small]["cols"])

    per_fold = []
    for fold, ids in fold_plan(cohort_df, None):
        tr_df, _ = build_design(list(ids["train"]), tab_risk, img_risk, labels,
                                group_by_id, target, normalize=normalize)
        ll = {}
        for name in (small, big):
            spec = MODEL_SPECS[name]
            use = spec["cols"] + (spec["strata"] or []) + ["duration", "event"]
            c = CoxPHFitter().fit(tr_df[use], "duration", "event", strata=spec["strata"])
            ll[name] = float(c.log_likelihood_)
        chi2 = 2 * (ll[big] - ll[small])
        per_fold.append({"fold": fold, "chi2": chi2, "df": df,
                         "p": float(sps.chi2.sf(max(chi2, 0), df))})

    ps = [f["p"] for f in per_fold]
    return {"small": small, "big": big, "df": df, "per_fold": per_fold,
            "n_folds_p_lt_05": int(sum(p < 0.05 for p in ps)),
            "median_p": float(np.median(ps)),
            "note": "fold별 개별 LRT. train set이 겹치므로 합산 금지."}


# ═══════════════════════════════════════════════════════════════════════════
# 4. 음성 대조군
# ═══════════════════════════════════════════════════════════════════════════
def negative_control_random(cohort_df, target, tab_risk, img_risk, *, n_draws=200,
                            seed=7, contrast=PRIMARY_CONTRAST, normalize="zscore"):
    """stage를 '같은 비율의 무작위 이분 라벨'로 바꿔 200회. 상호작용의 경험적 귀무분포.
    ※ 반드시 M2 vs M1 대비로 봐야 한다(둘 다 주효과 포함) — M2 vs M0 로 보면
      stage 주효과가 섞여 실제보다 유의해 보인다."""
    labels = cohort_df.drop_duplicates("research_id").set_index("research_id")
    ids = list(labels.index)
    n_pos = int((labels["stage"] == STAGE_ES).sum())
    rng = np.random.default_rng(seed)
    big, small = contrast
    deltas = {"LS": [], "ES": [], "stratified_overall": []}
    for _ in range(n_draws):
        pos = set(rng.choice(ids, size=n_pos, replace=False))
        g = {int(i): (STAGE_ES if i in pos else STAGE_LS) for i in ids}
        try:
            eb = evaluate(run_cv(cohort_df, target, tab_risk, img_risk, g, big,
                                 normalize=normalize)["lp_by_id"], cohort_df, target, g)
            es = evaluate(run_cv(cohort_df, target, tab_risk, img_risk, g, small,
                                 normalize=normalize)["lp_by_id"], cohort_df, target, g)
        except Exception:
            continue
        for k in deltas:
            deltas[k].append(eb["within"][k]["cindex"] - es["within"][k]["cindex"])
    return {k: {"mean": float(np.mean(v)), "sd": float(np.std(v)),
                "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                "n": len(v)} for k, v in deltas.items() if v}


# ═══════════════════════════════════════════════════════════════════════════
# 5. 판정 + 실행
# ═══════════════════════════════════════════════════════════════════════════
def apply_decision_rule(res):
    """사전등록 판정. within-stage 부트스트랩 SD(LS 0.032/ES 0.019)에 맞춰
    임계값 0.03. 저장소 관례(0.016)는 overall 기준이라 within-stage엔 부적절."""
    big, small = PRIMARY_CONTRAST
    d = res["bootstrap"][f"{big}_vs_{small}"]
    lrt = res["lrt"][f"{big}_vs_{small}"]
    dl, de = d["LS"]["delta_mean"], d["ES"]["delta_mean"]
    ci_ex = lambda c: c["ci95"][0] > 0 or c["ci95"][1] < 0
    sign_ok = res.get("interaction_sign_consistency", {}).get("scI_positive_folds", 0) >= 4
    # fold별 LRT: 과반(3/5) 이상이 유의해야 '계수 근거 있음'으로 본다
    lrt_ok = lrt["n_folds_p_lt_05"] >= 3

    improved = ((dl >= DECISION_THRESHOLD and ci_ex(d["LS"])) or
                (de >= DECISION_THRESHOLD and ci_ex(d["ES"])))
    harmed = ((dl <= -DECISION_THRESHOLD and ci_ex(d["LS"])) or
              (de <= -DECISION_THRESHOLD and ci_ex(d["ES"])))
    no_harm = dl > -DECISION_THRESHOLD and de > -DECISION_THRESHOLD

    if improved and no_harm and lrt_ok and sign_ok:
        v = "HELPS"
    elif harmed:
        v = "HARMFUL"
    elif not lrt_ok and abs(dl) < DECISION_THRESHOLD and abs(de) < DECISION_THRESHOLD:
        v = "NO_EVIDENCE"
    else:
        v = "INCONCLUSIVE"

    # 실격 조건: 전체만 오르고 stage 내부는 그대로 → stage 주효과일 뿐
    ov = res["metrics"][big]["within"]["stratified_overall"]["cindex"] - \
         res["metrics"]["M0_shared"]["within"]["stratified_overall"]["cindex"]
    disq = (ov > DECISION_THRESHOLD and abs(dl) < DECISION_THRESHOLD and abs(de) < DECISION_THRESHOLD)
    return {"verdict": v, "delta_LS": dl, "delta_ES": de,
            "lrt_n_folds_p_lt_05": lrt["n_folds_p_lt_05"], "lrt_median_p": lrt["median_p"],
            "sign_consistent": sign_ok, "threshold": DECISION_THRESHOLD,
            "disqualifier_stage_main_effect_only": bool(disq)}


def print_summary(res, target):
    print(f"\n{'=' * 78}\n  SUMMARY  target={target}\n{'=' * 78}")
    print(f"{'model':<18}{'overall(fold평균)':>18}{'LS':>10}{'ES':>10}{'strat.overall':>15}")
    for name in res["metrics"]:
        m = res["metrics"][name]; w = m["within"]
        print(f"{name:<18}{m['overall_perfold_mean']:>12.4f}±{m['overall_perfold_sd']:.3f}"
              f"{w['LS']['cindex']:>10.4f}{w['ES']['cindex']:>10.4f}"
              f"{w['stratified_overall']['cindex']:>15.4f}")
    print(f"\n  (LS 비교쌍 {res['metrics']['M0_shared']['within']['LS']['pairs']}개, "
          f"ES {res['metrics']['M0_shared']['within']['ES']['pairs']}개)")

    print(f"\n--- 주 비교: {PRIMARY_CONTRAST[0]} vs {PRIMARY_CONTRAST[1]} (순수 stage-aware 효과) ---")
    b = res["bootstrap"][f"{PRIMARY_CONTRAST[0]}_vs_{PRIMARY_CONTRAST[1]}"]
    for k in ("LS", "ES", "stratified_overall"):
        v = b[k]
        print(f"  {k:<20}Δ={v['delta_mean']:+.4f}  95%CI[{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]"
              f"  P(Δ>0)={v['P_gt_0']:.2f}")
    l = res["lrt"][f"{PRIMARY_CONTRAST[0]}_vs_{PRIMARY_CONTRAST[1]}"]
    print(f"  LRT(fold별, df={l['df']}): " +
          "  ".join(f"f{x['fold']} p={x['p']:.3f}" for x in l["per_fold"]))
    print(f"       -> p<0.05 인 fold: {l['n_folds_p_lt_05']}/5   (중앙값 p={l['median_p']:.3f})")
    print(f"       ※ fold별 train set이 80% 이상 겹치므로 로그우도 합산 LRT는 쓰지 않음")

    print(f"\n--- stage별 학습된 융합 가중치 (fold별) ---")
    for name in ("M2_stage_aware",):
        if name not in res["fold_detail"]:
            continue
        print(f"  {name}:")
        for f in res["fold_detail"][name]:
            print(f"    fold{f['fold']}  zI: LS={f.get('weight_LS_zI', float('nan')):+.3f} "
                  f"ES={f.get('weight_ES_zI', float('nan')):+.3f}  |  "
                  f"zT: LS={f.get('weight_LS_zT', float('nan')):+.3f} "
                  f"ES={f.get('weight_ES_zT', float('nan')):+.3f}  "
                  f"(train LS event={f['n_events_train_LS']})")
    print(f"\n>>> 판정: {res['decision']['verdict']}")
    if res["decision"]["disqualifier_stage_main_effect_only"]:
        print("    ⚠️ 전체만 개선되고 stage 내부는 그대로 → 'stage 주효과'이지 stage-aware fusion 아님")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="pfs")
    ap.add_argument("--models", default=",".join(MODEL_SPECS))
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--nc1_draws", type=int, default=0, help="0이면 건너뜀(느림)")
    ap.add_argument("--lambda_grid", default="")
    ap.add_argument("--normalize", default="zscore", choices=("zscore", "none"))
    ap.add_argument("--tab_epochs", type=int, default=60)
    ap.add_argument("--tab_batch", type=int, default=32)
    ap.add_argument("--no_reuse_image", action="store_true")
    ap.add_argument("--force_retrain", action="store_true")
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for target in [t.strip() for t in args.targets.split(",")]:
        print(f"\n{'#' * 78}\n#  target = {target}\n{'#' * 78}")

        gate_a = verify_image_oof_invariance(target)
        print(f"[게이트A] 영상 OOF 불변성: {'PASS' if gate_a['passed'] else 'FAIL'} "
              f"(다른 컬럼: {gate_a['differing_columns']})")
        assert gate_a["passed"], "영상 OOF 재사용 근거가 깨졌다 — --no_reuse_image 로 재학습할 것"

        oof = get_oof_scores(target, tab_epochs=args.tab_epochs, tab_batch=args.tab_batch,
                             reuse_image=not args.no_reuse_image,
                             cache=os.path.join(args.out, f"oof_{target}_bf.json"),
                             force_retrain=args.force_retrain)
        tab, img = oof["tabular"], oof["image"]
        cdf = cohort.load_trimodal_cohort()
        grp = {int(k): int(v) for k, v in
               cdf.drop_duplicates("research_id").set_index("research_id")["stage"].items()}

        res = {"target": target, "provenance": oof["provenance"], "gate_image_invariance": gate_a,
               "metrics": {}, "fold_detail": {}, "bootstrap": {}, "lrt": {}}
        lps = {}
        for name in [m.strip() for m in args.models.split(",")]:
            cv = run_cv(cdf, target, tab, img, grp, name, normalize=args.normalize)
            lps[name] = cv["lp_by_id"]
            res["metrics"][name] = evaluate(cv["lp_by_id"], cdf, target, grp)
            res["fold_detail"][name] = cv["folds"]
            m = res["metrics"][name]
            print(f"[CV] {name:<18} overall={m['overall_perfold_mean']:.4f}  "
                  f"LS={m['within']['LS']['cindex']:.4f}  ES={m['within']['ES']['cindex']:.4f}")
            json.dump(res, open(os.path.join(args.out, f"results_{target}.json"), "w"),
                      indent=2, default=str)

        # 상호작용 부호 일관성 (판정 기준의 일부)
        if "M2_stage_aware" in res["fold_detail"]:
            sc = sum(1 for f in res["fold_detail"]["M2_stage_aware"] if f["coefs"].get("scI", 0) > 0)
            res["interaction_sign_consistency"] = {"scI_positive_folds": sc, "n_folds": 5}

        for big, small in (PRIMARY_CONTRAST, LEGACY_CONTRAST):
            if big in lps and small in lps:
                res["lrt"][f"{big}_vs_{small}"] = nested_lrt(cdf, target, tab, img, grp,
                                                             small, big, normalize=args.normalize)
                if args.bootstrap:
                    res["bootstrap"][f"{big}_vs_{small}"] = paired_bootstrap_delta(
                        lps[big], lps[small], cdf, target, grp, B=args.bootstrap)

        if args.lambda_grid:
            grid = [float(x) for x in args.lambda_grid.split(",")]
            res["lambda_sweep"] = {}
            for lam in grid:
                cv = run_cv(cdf, target, tab, img, grp, "M2_stage_aware",
                            penalizer=lam, normalize=args.normalize)
                res["lambda_sweep"][str(lam)] = evaluate(cv["lp_by_id"], cdf, target, grp)
                print(f"[λ] {lam:<6} LS={res['lambda_sweep'][str(lam)]['within']['LS']['cindex']:.4f}"
                      f"  ES={res['lambda_sweep'][str(lam)]['within']['ES']['cindex']:.4f}")

        if args.nc1_draws:
            print(f"[NC1] 무작위 라벨 {args.nc1_draws}회 (M2 vs M1 대비)...")
            res["negative_control_random"] = negative_control_random(
                cdf, target, tab, img, n_draws=args.nc1_draws, normalize=args.normalize)

        res["decision"] = apply_decision_rule(res)
        json.dump(res, open(os.path.join(args.out, f"results_{target}.json"), "w"),
                  indent=2, default=str)
        print_summary(res, target)
        print(f"\nwrote {os.path.join(args.out, f'results_{target}.json')}")


if __name__ == "__main__":
    main()
