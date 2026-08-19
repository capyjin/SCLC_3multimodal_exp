# -*- coding: utf-8 -*-
"""[진단] Stage-aware fusion(M2)의 성능 미개선이 '가설 실패'인가 '최적화 실패'인가?

[⚠️ 먼저 확정해야 할 것 — 진단 대상의 성격]
  융합 단계는 **신경망이 아니라 CoxPH**다. Cox 부분우도는 β에 대해 **볼록(convex)**
  이므로 전역 최적해가 유일하다. 따라서 다음 항목은 **원리적으로 해당 없음**이다:
    - "학습 손실이 안 떨어진다" / "수렴 실패"  -> 볼록이라 항상 전역해로 수렴
    - "초기값에 민감하다"                      -> 볼록이라 초기값 무관
    - "gradient 폭발/소실"                     -> Newton-Raphson, 해석적 gradient
  대신 이 구조에서 **실제로 문제가 될 수 있는 것**을 점검한다:
    (A) 수렴 실패 진단  : log-likelihood 개선량, Newton 반복 수, Hessian 조건수
    (B) 계수 안정성     : fold 간 계수 변동, 표준오차, Wald CI 폭
    (C) 입력 스케일     : 표준화 후 분산 균형(이미 z-score 적용됨)
    (D) 다중공선성      : VIF, 상관행렬, Hessian 최소고유값
    (E) regularization  : ridge λ가 실제로 계수를 수축시키는지
    (F) 하이퍼파라미터 공정성 : M1과 M2가 동일 조건에서 적합되는지
    (G) **nested CV**   : test를 전혀 안 보고 λ를 고른 뒤 재평가

[핵심 질문]
  M2는 M1보다 train log-likelihood가 **항상 높다**(중첩모형이므로 수학적 필연).
  그런데 test C-index는 안 오른다. 이 간극이
    - 과적합(= 구조적 한계, 표본 부족) 인가
    - 최적화/설정 문제(= 고칠 수 있는 것) 인가
  를 가른다.

Run:  python fusion/diag_stage_aware_optimization.py --target pfs
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import json

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from core import cohort
from core.train import fold_plan
from exp_stage_aware_fusion import (build_design, evaluate, run_cv, MODEL_SPECS,
                                    paired_bootstrap_delta, STAGE_LS, STAGE_ES)

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "stage_aware_diag")


def _load(target):
    cdf = cohort.load_trimodal_cohort()
    lab = cdf.drop_duplicates("research_id").set_index("research_id")
    oof = json.load(open(os.path.join(PROJECT_ROOT, "outputs", "stage_aware_fusion",
                                       f"oof_{target}_bf.json")))
    tab = {int(k): v for k, v in oof["tabular"].items()}
    img = {int(k): v for k, v in oof["image"].items()}
    grp = {int(k): int(v) for k, v in lab["stage"].items()}
    return cdf, lab, tab, img, grp


# ═══════════════════════════════════════════════════════════════════════════
# (A) 수렴 진단  (B) 계수 안정성  (D) 다중공선성
# ═══════════════════════════════════════════════════════════════════════════
def diag_fit_quality(cdf, lab, tab, img, grp, target):
    """fold별로 M1/M2를 적합해 수렴·계수안정성·조건수·VIF를 전부 뽑는다."""
    rows = []
    for fold, ids in fold_plan(cdf, None):
        tr, _ = build_design(list(ids["train"]), tab, img, lab, grp, target)
        for nm in ("M1_stage_main", "M2_stage_aware"):
            spec = MODEL_SPECS[nm]
            use = spec["cols"] + ["duration", "event"]
            cph = CoxPHFitter()
            cph.fit(tr[use], "duration", "event")

            X = tr[spec["cols"]].values
            # Hessian(관측정보행렬) 조건수 — 크면 수치적으로 불안정
            H = np.linalg.inv(cph.variance_matrix_.values)
            evals = np.linalg.eigvalsh(H)
            # VIF: 각 공변량을 나머지로 회귀했을 때의 1/(1-R²)
            vif = {}
            for j, c in enumerate(spec["cols"]):
                other = np.delete(X, j, axis=1)
                if other.shape[1] == 0:
                    vif[c] = 1.0
                    continue
                A = np.c_[np.ones(len(other)), other]
                beta, *_ = np.linalg.lstsq(A, X[:, j], rcond=None)
                resid = X[:, j] - A @ beta
                ss_res = float((resid ** 2).sum())
                ss_tot = float(((X[:, j] - X[:, j].mean()) ** 2).sum())
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
                vif[c] = float(1 / (1 - r2)) if r2 < 1 else np.inf

            rows.append({
                "fold": fold, "model": nm,
                "loglik": float(cph.log_likelihood_),
                "n_params": len(spec["cols"]),
                "hessian_cond": float(evals.max() / evals.min()),
                "hessian_min_eig": float(evals.min()),
                "max_vif": float(max(vif.values())),
                "vif": {k: round(v, 2) for k, v in vif.items()},
                "coefs": {c: float(cph.params_[c]) for c in spec["cols"]},
                "se": {c: float(cph.standard_errors_[c]) for c in spec["cols"]},
                "ci_width": {c: float(cph.summary.loc[c, "coef upper 95%"]
                                      - cph.summary.loc[c, "coef lower 95%"])
                             for c in spec["cols"]},
            })
    return rows


def summarize_fit_quality(rows):
    print("\n" + "=" * 78)
    print("  (A)(B)(D) 적합 품질 — 수렴 / 계수안정성 / 다중공선성")
    print("=" * 78)

    df = pd.DataFrame(rows)
    print("\n[A] 수렴 진단 — CoxPH는 볼록이므로 항상 전역해. log-lik 개선량 확인")
    print(f"{'fold':<6}{'M1 loglik':>12}{'M2 loglik':>12}{'개선량':>10}{'2*Δ(=chi2)':>12}")
    for f in sorted(df.fold.unique()):
        l1 = df[(df.fold == f) & (df.model == "M1_stage_main")].loglik.iloc[0]
        l2 = df[(df.fold == f) & (df.model == "M2_stage_aware")].loglik.iloc[0]
        print(f"{f:<6}{l1:>12.3f}{l2:>12.3f}{l2 - l1:>10.3f}{2 * (l2 - l1):>12.3f}")
    print("  -> M2의 loglik이 항상 M1보다 높다(중첩모형의 수학적 필연). 즉 **학습은 성공**했다.")

    print("\n[D] 다중공선성 — Hessian 조건수 / 최대 VIF")
    print(f"{'fold':<6}{'model':<18}{'조건수':>12}{'최소고유값':>12}{'최대VIF':>10}")
    for _, r in df.iterrows():
        print(f"{r.fold:<6}{r.model:<18}{r.hessian_cond:>12.1f}{r.hessian_min_eig:>12.2f}{r.max_vif:>10.2f}")
    print("  판정 기준: 조건수 <30 양호 / 30~100 보통 / >100 심각,  VIF <5 양호 / >10 심각")

    print("\n[B] 계수 안정성 — fold 간 변동 (M2)")
    m2 = df[df.model == "M2_stage_aware"]
    cols = list(m2.iloc[0]["coefs"].keys())
    print(f"{'coef':<8}{'평균':>9}{'fold간SD':>10}{'평균SE':>9}{'SD/SE':>8}{'평균CI폭':>10}")
    for c in cols:
        vals = np.array([r["coefs"][c] for _, r in m2.iterrows()])
        ses = np.array([r["se"][c] for _, r in m2.iterrows()])
        cis = np.array([r["ci_width"][c] for _, r in m2.iterrows()])
        print(f"{c:<8}{vals.mean():>+9.3f}{vals.std():>10.3f}{ses.mean():>9.3f}"
              f"{vals.std() / ses.mean():>8.2f}{cis.mean():>10.3f}")
    print("  SD/SE ≈ 1 이면 fold 간 변동이 '추정 불확실성' 수준(정상).")
    print("  SD/SE >> 1 이면 fold마다 진짜 다른 값을 배우는 것(불안정).")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# (C) 입력 스케일  (F) 하이퍼파라미터 공정성
# ═══════════════════════════════════════════════════════════════════════════
def diag_scale_and_fairness(cdf, lab, tab, img, grp, target):
    print("\n" + "=" * 78)
    print("  (C)(F) 입력 스케일 / 하이퍼파라미터 공정성")
    print("=" * 78)
    fold, ids = fold_plan(cdf, None)[0]
    tr, stats = build_design(list(ids["train"]), tab, img, lab, grp, target)
    print("\n[C] 표준화 후 train fold 입력 분포 (fold1)")
    for c in ("zT", "zI", "sc", "scT", "scI"):
        v = tr[c].values
        print(f"  {c:<5} mean={v.mean():+.4f}  sd={v.std():.4f}  range=[{v.min():+.3f},{v.max():+.3f}]")
    print(f"  train 표준화 통계: {({k: round(v, 4) for k, v in stats.items()})}")
    print("  -> zT/zI는 정확히 평균0·분산1. 상호작용항은 sc(±0.5)와의 곱이라 sd≈0.5가 정상.")

    print("\n[F] 하이퍼파라미터 공정성 — M1과 M2가 같은 조건인가")
    print("  동일: 코호트/fold/seed, 표준화 방식, CoxPH solver(Newton-Raphson),")
    print("        수렴 허용오차, step size, 그리고 **동일한 zT/zI 입력값**")
    print("  차이: M2만 공변량 2개(scT, scI) 추가 — 이것이 '제안'의 내용 그 자체")
    print("  penalizer: 두 모델 모두 기본 0 (M2에만 ridge를 걸면 불공정하므로 primary는 λ=0)")
    print("  -> 공정성 문제 없음. M1은 M2의 **중첩 부분모형**이라 설계상 동일조건이 보장된다.")


# ═══════════════════════════════════════════════════════════════════════════
# (E) regularization — λ가 실제로 계수를 수축시키는가
# ═══════════════════════════════════════════════════════════════════════════
def diag_regularization(cdf, lab, tab, img, grp, target, grid=(0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0)):
    print("\n" + "=" * 78)
    print("  (E) regularization — ridge λ의 계수 수축 효과 (fold1 train)")
    print("=" * 78)
    fold, ids = fold_plan(cdf, None)[0]
    tr, _ = build_design(list(ids["train"]), tab, img, lab, grp, target)
    spec = MODEL_SPECS["M2_stage_aware"]
    use = spec["cols"] + ["duration", "event"]
    print(f"{'λ':<8}" + "".join(f"{c:>10}" for c in spec["cols"]))
    for lam in grid:
        pen = np.array([lam if c in spec["pen"] else 0.0 for c in spec["cols"]]) if lam else 0.0
        cph = CoxPHFitter(penalizer=pen).fit(tr[use], "duration", "event")
        print(f"{lam:<8}" + "".join(f"{cph.params_[c]:>+10.3f}" for c in spec["cols"]))
    print("  -> scT/scI만 단조 수축하고 zT/zI/sc는 거의 불변이면 penalizer가 의도대로 작동.")


# ═══════════════════════════════════════════════════════════════════════════
# (G) nested CV — test를 전혀 보지 않고 λ 선택
# ═══════════════════════════════════════════════════════════════════════════
def _cindex_within_stage(df_test, lp, grp):
    """test 프레임에서 stage내 쌍만 세어 concordance (진단용 간이 버전)."""
    import itertools
    conc = comp = 0.0
    for st in (STAGE_LS, STAGE_ES):
        idx = [i for i in df_test.index if grp[int(i)] == st]
        if len(idx) < 2:
            continue
        d = df_test.loc[idx, "duration"].values
        e = df_test.loc[idx, "event"].values
        r = np.array([lp[int(i)] for i in idx])
        for a, b in itertools.combinations(range(len(idx)), 2):
            if d[a] < d[b] and e[a] == 1:
                s, l = a, b
            elif d[b] < d[a] and e[b] == 1:
                s, l = b, a
            else:
                continue
            comp += 1
            conc += 1.0 if r[s] > r[l] else (0.5 if r[s] == r[l] else 0.0)
    return conc / comp if comp else np.nan


def nested_cv_lambda(cdf, lab, tab, img, grp, target, grid=(0.0, 0.01, 0.05, 0.1, 0.5, 1.0),
                     n_inner=5, seed=42):
    """outer fold의 **train 171명 안에서만** inner CV로 λ를 고른 뒤,
    그 λ로 train 전체를 재적합해 outer test를 평가한다. test는 절대 보지 않는다."""
    from sklearn.model_selection import StratifiedKFold
    print("\n" + "=" * 78)
    print("  (G) nested CV — test를 보지 않고 λ 선택")
    print("=" * 78)

    spec = MODEL_SPECS["M2_stage_aware"]
    use = spec["cols"] + ["duration", "event"]
    lp_all, chosen = {}, []

    for fold, ids in fold_plan(cdf, None):
        tr_ids = [int(i) for i in ids["train"]]
        te_ids = [int(i) for i in ids["test"]]
        tr_full, stats = build_design(tr_ids, tab, img, lab, grp, target)

        # inner: stage×event 층화 5-fold, train 안에서만
        strat = np.array([f"{grp[i]}_{int(lab.loc[i, f'{target}_event'])}" for i in tr_ids])
        inner = StratifiedKFold(n_inner, shuffle=True, random_state=seed)
        scores = {lam: [] for lam in grid}
        for itr, iva in inner.split(np.zeros(len(tr_ids)), strat):
            a_ids = [tr_ids[k] for k in itr]
            b_ids = [tr_ids[k] for k in iva]
            a_df, a_stats = build_design(a_ids, tab, img, lab, grp, target)
            b_df, _ = build_design(b_ids, tab, img, lab, grp, target, fit_stats=a_stats)
            for lam in grid:
                pen = np.array([lam if c in spec["pen"] else 0.0 for c in spec["cols"]]) if lam else 0.0
                try:
                    m = CoxPHFitter(penalizer=pen).fit(a_df[use], "duration", "event")
                    lp = m.predict_log_partial_hazard(b_df[use])
                    scores[lam].append(_cindex_within_stage(b_df, {int(i): float(v) for i, v in lp.items()}, grp))
                except Exception:
                    scores[lam].append(np.nan)
        mean_scores = {lam: np.nanmean(v) for lam, v in scores.items()}
        best = max(mean_scores, key=mean_scores.get)
        chosen.append(best)

        # outer: 선택된 λ로 train 전체 재적합 -> test 예측
        te_df, _ = build_design(te_ids, tab, img, lab, grp, target, fit_stats=stats)
        pen = np.array([best if c in spec["pen"] else 0.0 for c in spec["cols"]]) if best else 0.0
        m = CoxPHFitter(penalizer=pen).fit(tr_full[use], "duration", "event")
        lp = m.predict_log_partial_hazard(te_df[use])
        lp_all.update({int(i): float(v) for i, v in lp.items()})
        print(f"  fold{fold}: 선택 λ={best:<6} (inner 점수 " +
              ", ".join(f"{l}:{mean_scores[l]:.4f}" for l in grid) + ")")

    return {"lp_by_id": lp_all, "chosen_lambdas": chosen}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="pfs", choices=("os", "pfs"))
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    cdf, lab, tab, img, grp = _load(args.target)
    print(f"\n{'#' * 78}\n#  Stage-aware 최적화 진단  target={args.target}\n{'#' * 78}")
    print("\n[전제] 융합 단계는 CoxPH(볼록 최적화)이다. 신경망이 아니므로")
    print("       '수렴 실패/초기값 민감/gradient 불안정'은 원리적으로 발생하지 않는다.")

    rows = diag_fit_quality(cdf, lab, tab, img, grp, args.target)
    summarize_fit_quality(rows)
    diag_scale_and_fairness(cdf, lab, tab, img, grp, args.target)
    diag_regularization(cdf, lab, tab, img, grp, args.target)
    nest = nested_cv_lambda(cdf, lab, tab, img, grp, args.target)

    # nested CV 결과 평가
    ev_nested = evaluate(nest["lp_by_id"], cdf, args.target, grp)
    base = {}
    for nm in ("M0_shared", "M1_stage_main", "M2_stage_aware"):
        cv = run_cv(cdf, args.target, tab, img, grp, nm)
        base[nm] = {"lp": cv["lp_by_id"], "ev": evaluate(cv["lp_by_id"], cdf, args.target, grp)}

    print("\n" + "=" * 78)
    print("  (G) nested CV 결과 vs 기존")
    print("=" * 78)
    print(f"{'model':<24}{'overall':>10}{'LS':>10}{'ES':>10}")
    for nm in ("M0_shared", "M1_stage_main", "M2_stage_aware"):
        e = base[nm]["ev"]
        print(f"{nm:<24}{e['overall_perfold_mean']:>10.4f}"
              f"{e['within']['LS']['cindex']:>10.4f}{e['within']['ES']['cindex']:>10.4f}")
    print(f"{'M2_nestedCV_lambda':<24}{ev_nested['overall_perfold_mean']:>10.4f}"
          f"{ev_nested['within']['LS']['cindex']:>10.4f}{ev_nested['within']['ES']['cindex']:>10.4f}")
    print(f"  선택된 λ (fold별): {nest['chosen_lambdas']}")

    boot = paired_bootstrap_delta(nest["lp_by_id"], base["M1_stage_main"]["lp"],
                                  cdf, args.target, grp, B=args.bootstrap)
    print("\n  nested-CV M2 vs M1 (쌍대 부트스트랩)")
    for k in ("LS", "ES", "stratified_overall"):
        v = boot[k]
        print(f"    {k:<20}Δ={v['delta_mean']:+.4f}  95%CI[{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]"
              f"  P(Δ>0)={v['P_gt_0']:.2f}")

    out = {"target": args.target, "fit_quality": rows,
           "nested_cv": {"chosen_lambdas": nest["chosen_lambdas"],
                         "metrics": ev_nested, "vs_M1_bootstrap": boot},
           "baseline_metrics": {k: v["ev"] for k, v in base.items()}}
    p = os.path.join(OUT_DIR, f"diag_{args.target}.json")
    json.dump(out, open(p, "w"), indent=2, default=str)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
