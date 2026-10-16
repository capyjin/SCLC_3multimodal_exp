# -*- coding: utf-8 -*-
"""[실험16-1] 실제 238명 OS/PFS 레이블로 Efron Cox 손실을 외부 기준과 대조한다 — 신경망 학습 없음.

단위 테스트(tests/test_cox_loss.py)는 합성 데이터만 쓴다. 여기서는 실제 동점 구조
(OS 사건 199건 중 53건, PFS 212건 중 87건이 동점 시각)에서 같은 결과가 나오는지 본다.

  - 선형 예측자 = stage + ECOG + 표준화 LDH (예시 공변량일 뿐, 모델이 아니다)
  - lifelines CoxPHFitter(Efron) 적합 -> 그 beta 에서 우리 Efron NLL x N_event 와 -loglik 비교
  - 우리 손실을 LBFGS 로 최소화한 beta 와 lifelines / scikit-survival(Efron·Breslow) beta 비교
  - 같은 beta 에서 Efron·Breslow·기존 cox_ph_loss(입력 순서 5가지) 값 — 기존 손실의 순서 의존성 크기

Run:  python experiments/실험16_GMU_nestedCV/diagnostics/check_cox_loss.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))                 # diagnostics/
EXP_DIR = os.path.dirname(HERE)                                   # 실험16 폴더 (nested_fusion 패키지)
PROJECT_ROOT = os.path.dirname(os.path.dirname(EXP_DIR))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, HERE)

import json

import numpy as np
import pandas as pd
import torch
from lifelines import CoxPHFitter
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

from sclc import cohort, paths
from sclc.train import cox_partial_nll, cox_ph_loss

OUT = paths.outputs("gmu_nested", "check_cox_loss.json")
COVARIATES = ["stage", "ecog", "ldh_z"]


def fit_ours(x, t, e, ties):
    xt, tt, et = (torch.as_tensor(a, dtype=torch.float64) for a in (x, t, e))
    beta = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([beta], max_iter=500, tolerance_grad=1e-12, tolerance_change=1e-14,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = cox_partial_nll(xt @ beta, tt, et, ties=ties)
        loss.backward()
        return loss

    opt.step(closure)
    return beta.detach().numpy()


def main() -> None:
    df = cohort.load_trimodal_cohort().drop_duplicates("research_id").set_index("research_id")
    df = df.assign(ldh_z=(df["ldh"] - df["ldh"].mean()) / df["ldh"].std())
    x = df[COVARIATES].to_numpy(float)
    out = {"covariates": COVARIATES, "n": len(df)}
    for target in ("os", "pfs"):
        t = df[f"{target}_days"].to_numpy(float)
        e = df[f"{target}_event"].to_numpy(float)
        ev_t = pd.Series(t[e == 1]).value_counts()
        frame = pd.DataFrame(x, columns=COVARIATES).assign(T=t, E=e)
        lif = CoxPHFitter().fit(frame, duration_col="T", event_col="E")
        b_lif = lif.params_.to_numpy()
        eta = torch.as_tensor(x @ b_lif)
        tt, et = torch.as_tensor(t), torch.as_tensor(e)
        efron = cox_partial_nll(eta, tt, et, ties="efron").item()
        breslow = cox_partial_nll(eta, tt, et, ties="breslow").item()
        old = []
        for s in range(5):
            p = torch.as_tensor(np.random.default_rng(s).permutation(len(t)))
            old.append(cox_ph_loss(eta[p], tt[p], et[p]).item())
        y = Surv.from_arrays(event=e.astype(bool), time=t)
        rec = {
            "n_event": int(e.sum()), "events_in_tied_times": int(ev_t[ev_t > 1].sum()),
            "lifelines_neg_loglik": -lif.log_likelihood_,
            "ours_efron_total_at_lifelines_beta": efron * e.sum(),
            "abs_diff_loglik": abs(efron * e.sum() + lif.log_likelihood_),
            "beta_lifelines_efron": b_lif.round(6).tolist(),
        }
        for ties in ("efron", "breslow"):
            b_ours = fit_ours(x, t, e, ties)
            b_sk = CoxPHSurvivalAnalysis(alpha=0, ties=ties, n_iter=200, tol=1e-12).fit(x, y).coef_
            rec[f"beta_ours_{ties}"] = b_ours.round(6).tolist()
            rec[f"beta_sksurv_{ties}"] = b_sk.round(6).tolist()
            rec[f"max_abs_beta_diff_vs_sksurv_{ties}"] = float(np.abs(b_ours - b_sk).max())
        rec["max_abs_beta_diff_ours_efron_vs_lifelines"] = float(
            np.abs(np.array(rec["beta_ours_efron"]) - b_lif).max())
        rec["nll_per_event_at_lifelines_beta"] = {
            "efron": efron, "breslow": breslow,
            "old_cox_ph_loss_5_orders": [round(v, 6) for v in old],
            "old_range": max(old) - min(old)}
        out[target] = rec
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
