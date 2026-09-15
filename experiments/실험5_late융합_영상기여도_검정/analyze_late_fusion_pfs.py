# -*- coding: utf-8 -*-
"""[실험5-a] ★교수님 질문 — late fusion 에서 왜 OS만 오르고 PFS는 떨어지는가?

late fusion 의 최종 결합은 CoxPH 2변수 회귀다:
    log h(t) = beta_tab * (tabular 위험점수) + beta_img * (영상 위험점수)

따라서 "영상이 도움이 되는가"는 **beta_img 가 0과 유의하게 다른가**로 직접 검정할
수 있다. 저장된 체크포인트에서 OOF 위험점수를 되살린 뒤 아래를 계산한다.

  ① beta_img 의 fold별 값 + 95% 신뢰구간 + p-value  -> 신호가 안정적인가?
  ② 우도비 검정 (tabular 단독 vs tabular+image)     -> 영상이 정보를 더하는가?
  ③ tabular ↔ image 위험점수 상관                    -> 중복인가?
  ④ tabular 가 틀린 환자쌍에서 영상의 구제율          -> 보완적인가?
  ⑤ 영상 점수를 난수로 바꿔 200회 반복               -> 난수와 구분되는가?

결론(RESULTS.md §10): OS 는 beta=+0.322 (p=0.008, 5/5 fold 양수)로 확실한 신호,
PFS 는 beta=+0.008 (p=0.948)로 0과 구분 불가. 난수 대조에서도 OS 는 난수가 0/200
승, PFS 는 난수가 146/200(73%) 승. 즉 **영상에 PFS 고유 정보가 없다.**

Run:  python 실험5_late융합_영상기여도_검정/analyze_late_fusion_pfs.py
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import json

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from scipy import stats

from core import cohort
from core.fusion_stack import get_image_oof_simplecnn, get_tabular_oof, oof_dict
from core.train import fold_plan

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "late_fusion_B")


# ---------------------------------------------------------------------------
# 1) 저장된 체크포인트로 OOF 위험점수 재생성 (재학습 없음: epochs=0)
# ---------------------------------------------------------------------------
def oof_scores(target):
    """tabular(clin+report) / image(SimpleCNN) 각각의 fold-out 위험점수를 되살린다.

    학습은 이미 끝나 체크포인트가 있으므로, 같은 설정으로 Evaluator를 다시 만들되
    가중치를 로드해 예측만 수행한다. (Evaluator가 fold별 best 체크포인트를 자동으로
    불러오는 구조라, 동일 save_dir를 주고 epochs=0으로 두면 재학습 없이 평가만 한다.)
    """
    cache = os.path.join(OUT_DIR, f"oof_{target}.json")
    if os.path.exists(cache):
        d = json.load(open(cache))
        return {k: {int(i): v for i, v in d[k].items()} for k in ("tabular", "image")}

    tab = get_tabular_oof(target, epochs=0, batch_size=32, seed=42, out_dir=OUT_DIR)
    img = get_image_oof_simplecnn(target, epochs=0, batch_size=16, seed=42, out_dir=OUT_DIR)
    out = {"tabular": oof_dict(tab.oof_predictions), "image": oof_dict(img.oof_predictions)}
    json.dump(out, open(cache, "w"))
    return out


# ---------------------------------------------------------------------------
# 2) 분석
# ---------------------------------------------------------------------------
def analyze(target):
    df_all = cohort.load_trimodal_cohort()
    labels = df_all.drop_duplicates("research_id").set_index("research_id")
    scores = oof_scores(target)

    def frame(ids):
        return pd.DataFrame({
            "risk_tabular": [scores["tabular"][i] for i in ids],
            "risk_image": [scores["image"][i] for i in ids],
            "duration": labels.loc[ids, f"{target}_days"].to_numpy(float),
            "event": labels.loc[ids, f"{target}_event"].to_numpy(float),
        })

    res = {"target": target}

    # ── ① fold별 beta_img: 값 · 95% CI · p-value ─────────────────────────
    rows = []
    for fold, ids in fold_plan(df_all, max_folds=None):
        tr = frame(ids["train"])
        cph = CoxPHFitter().fit(tr, duration_col="duration", event_col="event")
        s = cph.summary.loc["risk_image"]
        rows.append({
            "fold": fold, "coef": float(s["coef"]),
            "lo": float(s["coef lower 95%"]), "hi": float(s["coef upper 95%"]),
            "p": float(s["p"]),
        })
    res["beta_img_per_fold"] = rows
    coefs = np.array([r["coef"] for r in rows])
    res["beta_img_mean"] = float(coefs.mean())
    res["beta_img_positive_folds"] = int((coefs > 0).sum())
    # 0을 포함하지 않는 fold 수 = 유의한 fold 수
    res["beta_img_significant_folds"] = int(sum(1 for r in rows if r["p"] < 0.05))

    # ── ② 우도비 검정: tabular 단독 vs tabular+image (전체 238명 OOF) ────
    full = frame(labels.index.tolist())
    m1 = CoxPHFitter().fit(full[["risk_tabular", "duration", "event"]],
                           duration_col="duration", event_col="event")
    m2 = CoxPHFitter().fit(full, duration_col="duration", event_col="event")
    lr_stat = 2 * (m2.log_likelihood_ - m1.log_likelihood_)
    res["lrt"] = {
        "stat": float(lr_stat), "df": 1,
        "p": float(stats.chi2.sf(lr_stat, 1)),
        "ll_tabular": float(m1.log_likelihood_), "ll_both": float(m2.log_likelihood_),
    }
    s = m2.summary.loc["risk_image"]
    res["beta_img_pooled"] = {
        "coef": float(s["coef"]), "lo": float(s["coef lower 95%"]),
        "hi": float(s["coef upper 95%"]), "p": float(s["p"]),
    }

    # ── ③ 두 위험점수의 상관 (중복 여부) ────────────────────────────────
    # (a) pooled = 5개 fold의 OOF 점수를 한 덩어리로 합쳐서 계산.
    #     ⚠ fold마다 모델이 달라 위험점수의 척도(scale)가 다르므로, 그 차이가
    #     상관에 섞여 들어간다. 즉 fold 간 이질성이 만드는 가짜 성분이 포함된다.
    rho, p = stats.spearmanr(full["risk_tabular"], full["risk_image"])
    res["risk_correlation"] = {"spearman": float(rho), "p": float(p)}

    # (b) within-fold = fold별로 따로 상관을 구해 평균. 척도 문제가 없으므로
    #     이쪽이 "두 모달리티가 실제로 얼마나 겹치는가"의 올바른 추정치다.
    #     문서(RESULTS.md 10.5/10.6)에서 인용하는 값은 이 within-fold 값이다.
    per_fold_rho = []
    for fold, ids in fold_plan(df_all, max_folds=None):
        te = ids["test"]
        a = np.array([scores["tabular"][i] for i in te])
        b = np.array([scores["image"][i] for i in te])
        per_fold_rho.append({"fold": int(fold), "spearman": float(stats.spearmanr(a, b)[0])})
    res["risk_correlation_within_fold"] = {
        "per_fold": per_fold_rho,
        "mean": float(np.mean([r["spearman"] for r in per_fold_rho])),
    }

    # ── ④ 단독 C-index & 보완성: tabular가 '틀린' 쌍에서 이미지는 맞히나? ──
    d = full["duration"].to_numpy(); e = full["event"].to_numpy()
    rt = full["risk_tabular"].to_numpy(); ri = full["risk_image"].to_numpy()
    res["cindex_tabular"] = float(concordance_index(d, -rt, e))
    res["cindex_image"] = float(concordance_index(d, -ri, e))

    # 비교 가능한 모든 환자쌍을 tabular가 맞힌 쌍/틀린 쌍으로 나눠 이미지 정확도 계산
    ok = tot = 0
    n = len(d)
    for a in range(n):
        for b in range(a + 1, n):
            # (a,b)가 비교 가능한 쌍인지: 먼저 일어난 쪽이 event여야 함
            if d[a] < d[b] and e[a] == 1:
                first, second = a, b
            elif d[b] < d[a] and e[b] == 1:
                first, second = b, a
            else:
                continue
            tab_ok = rt[first] > rt[second]      # tabular가 먼저 죽는 쪽에 높은 위험 부여?
            if tab_ok:
                continue                          # tabular가 맞힌 쌍은 건너뜀
            tot += 1
            if ri[first] > ri[second]:            # 그 쌍을 이미지는 맞혔나?
                ok += 1
    res["image_rescue_rate"] = {"correct": ok, "total": tot,
                                "rate": float(ok / tot) if tot else None}

    # ── ⑤ 결정적 대조군: 이미지 점수를 '난수'로 바꿔도 결과가 같은가? ──────
    # late fusion 전체 파이프라인(fold별 CoxPH 적합 -> test 평가)을 그대로 돌리되,
    # 이미지 위험점수만 무작위로 섞는다(shuffle). 진짜 이미지가 난수와 성능이
    # 같다면, 이미지는 정보를 주는 게 아니라 '잡음'만 주고 있다는 직접 증거다.
    def stack_cindex(image_map):
        cis = []
        for fold, ids in fold_plan(df_all, max_folds=None):
            def fr(idl):
                return pd.DataFrame({
                    "risk_tabular": [scores["tabular"][i] for i in idl],
                    "risk_image": [image_map[i] for i in idl],
                    "duration": labels.loc[idl, f"{target}_days"].to_numpy(float),
                    "event": labels.loc[idl, f"{target}_event"].to_numpy(float),
                })
            tr, te = fr(ids["train"]), fr(ids["test"])
            cph = CoxPHFitter().fit(tr, duration_col="duration", event_col="event")
            pred = cph.predict_partial_hazard(te[["risk_tabular", "risk_image"]]).to_numpy()
            cis.append(concordance_index(te["duration"], -pred, te["event"]))
        return float(np.mean(cis))

    def tab_only_cindex():
        cis = []
        for fold, ids in fold_plan(df_all, max_folds=None):
            tr = frame(ids["train"])[["risk_tabular", "duration", "event"]]
            te = frame(ids["test"])[["risk_tabular", "duration", "event"]]
            cph = CoxPHFitter().fit(tr, duration_col="duration", event_col="event")
            pred = cph.predict_partial_hazard(te[["risk_tabular"]]).to_numpy()
            cis.append(concordance_index(te["duration"], -pred, te["event"]))
        return float(np.mean(cis))

    res["stack_real_image"] = stack_cindex(scores["image"])
    res["stack_tabular_only"] = tab_only_cindex()

    ids_all = list(labels.index)
    rng = np.random.default_rng(42)
    vals = np.array([scores["image"][i] for i in ids_all])
    shuffled = []
    for _ in range(200):
        perm = rng.permutation(vals)
        shuffled.append(stack_cindex(dict(zip(ids_all, perm))))
    shuffled = np.array(shuffled)
    res["stack_shuffled_image"] = {
        "mean": float(shuffled.mean()), "std": float(shuffled.std()),
        "p2_5": float(np.percentile(shuffled, 2.5)),
        "p97_5": float(np.percentile(shuffled, 97.5)),
        # 진짜 이미지가 난수보다 나은 비율 (1에 가까워야 '정보가 있다')
        "frac_random_beats_real": float((shuffled >= res["stack_real_image"]).mean()),
        "n_repeat": len(shuffled),
        # 히스토그램(fig11)을 그리려면 원본 200개 값이 필요하다.
        "draws": [float(v) for v in shuffled],
    }
    return res


def main():
    out = {}
    for t in ("os", "pfs"):
        print(f"\n{'='*66}\n  target = {t.upper()}\n{'='*66}")
        r = analyze(t)
        out[t] = r
        print("① beta_img (fold별)")
        for row in r["beta_img_per_fold"]:
            mark = "*" if row["p"] < 0.05 else " "
            print(f"   fold {row['fold']}: {row['coef']:+.3f}  "
                  f"[{row['lo']:+.3f}, {row['hi']:+.3f}]  p={row['p']:.3f} {mark}")
        print(f"   평균 {r['beta_img_mean']:+.3f} | 양수 {r['beta_img_positive_folds']}/5 "
              f"| 유의(p<.05) {r['beta_img_significant_folds']}/5")
        b = r["beta_img_pooled"]
        print(f"   전체 pooled: {b['coef']:+.3f} [{b['lo']:+.3f}, {b['hi']:+.3f}] p={b['p']:.4f}")
        print(f"② 우도비 검정 (이미지를 더하면 설명력이 느는가): "
              f"chi2={r['lrt']['stat']:.2f}, p={r['lrt']['p']:.4f}")
        wf = r["risk_correlation_within_fold"]
        print(f"③ tabular↔image 위험점수 상관:")
        print(f"     pooled       rho={r['risk_correlation']['spearman']:+.3f} "
              f"(p={r['risk_correlation']['p']:.3g})  ← fold 척도차가 섞여 과대평가됨")
        print(f"     within-fold  rho={wf['mean']:+.3f}  "
              f"(fold별 {[round(x['spearman'], 3) for x in wf['per_fold']]})  ← 이 값을 문서에 인용")
        print(f"④ 단독 C-index  tabular={r['cindex_tabular']:.4f}  image={r['cindex_image']:.4f}")
        rr = r["image_rescue_rate"]
        print(f"   tabular가 틀린 {rr['total']}쌍 중 이미지가 맞힌 비율 = {rr['rate']:.3f}")
        sh = r["stack_shuffled_image"]
        print(f"⑤ late fusion C-index:  tabular단독={r['stack_tabular_only']:.4f}  "
              f"진짜이미지={r['stack_real_image']:.4f}  "
              f"난수이미지={sh['mean']:.4f} [{sh['p2_5']:.4f}, {sh['p97_5']:.4f}]")
        print(f"   난수가 진짜를 이긴 비율 = {sh['frac_random_beats_real']:.3f} "
              f"({sh['n_repeat']}회 반복)")

    json.dump(out, open(os.path.join(OUT_DIR, "pfs_diagnosis.json"), "w"), indent=2)
    print(f"\nwrote {OUT_DIR}/pfs_diagnosis.json")


if __name__ == "__main__":
    main()
