# -*- coding: utf-8 -*-
"""Gated fusion 을 채택 모델과 **같은 OOF 점수·같은 fold** 로 학습해 직접 비교할 수 있게 잰다.

예전 게이트 실험(실험11, moe_gate_*_radbert)은 학습 환자 점수로 "축 모델이 자기 학습 데이터를 채점한 값"
(in-sample)을 써서 같은 파이프라인의 고정 가중치 대조군부터 OS 0.730 으로 부풀어 있었고, 그래서 논문 융합표에는
Δ 만 실을 수 있었다. 여기서는 채택 모델의 CoxPH 결합기와 똑같이 train 환자의 OOF 점수
(outputs/late_fusion_B_radbert_tests/oof_<target>.json)로 게이트를 학습한다.

모델 (곱셈형 sigmoid 게이트, 학습 파라미터 10개):
    risk(x) = beta_tab * g_tab(x) * z_tab + beta_img * g_img(x) * z_img,   g_m(x) = sigmoid(c_m . x)
    z_*  = train fold 기준으로 표준화한 OOF 위험점수
    x    = [z_tab, z_img, 확장기(중심화), 나이(표준화)]   (--gate_inputs stage_age, 기본)
           = [z_tab, z_img]                                (--gate_inputs risk)
  2026-10-12: 지도교수 지적(병기는 이미 임상 입력에 있다 — 실험9 stage-aware 를 접은 이유)에 따라 논문 표에는
  임상변수를 다시 넣지 않는 risk 판(두 모델의 출력만 보고 게이트를 정함, GMU/MoE 게이트와 같은 형태)을 쓴다.
  c = 0 이면 g = 0.5 로 모든 환자에게 같아져 beta 가 그만큼 커지므로 고정 가중치 Cox(채택 모델)와 정확히 같다
  (2026-10-12: 예전 '2 * sigmoid' 의 2 는 beta 에 흡수되는 표기상의 상수라 뺐다 — 같은 모델이다) → 상수 게이트로 0.7224 재현을 먼저 확인한다.
  손실 = Cox 부분우도(Breslow) + lambda * ||c||^2. lambda 는 train 171명 안의 5-fold inner CV 로 고른다
  (후보 0.01–100, 동점이면 큰 쪽). 게이트가 도움이 안 되면 큰 lambda 가 뽑혀 고정 가중치로 돌아간다.

Run:  python experiments/실험12_융합방식_조건통일/exp_gate_oof.py --targets os,pfs
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import argparse
import json

import numpy as np
import torch
from sklearn.model_selection import KFold
from sclc import cohort, paths
from sclc.evaluation import cindex
from sclc.late_fusion import labels_by_id
from sclc.train import fold_plan
from sclc.utils.cli import comma_list

OOF_DIR = paths.outputs("late_fusion_B_radbert_tests")
OUT_DIR = paths.outputs("gate_oof")
LAMBDAS = (0.01, 0.1, 1.0, 10.0, 100.0)
OFFICIAL = {"os": [0.7102, 0.7866, 0.6912, 0.7073, 0.7167],      # radbert_full late_tab_radbert+img
            "pfs": [0.6435, 0.7014, 0.642, 0.6204, 0.6276]}
torch.set_default_dtype(torch.float64)


def cox_nll(risk, t, e):
    """Breslow 음의 부분우도 (동점 포함 정확한 위험집합; n≈171 이라 행렬로 계산)."""
    at_risk = (t[None, :] >= t[:, None]).to(risk.dtype)            # [i, j] = j 가 i 시점에 위험집합
    log_den = torch.logsumexp(risk[None, :] + torch.log(at_risk), dim=1)
    return -((risk - log_den) * e).sum() / e.sum()


def forward(params, Z, X, gated):
    beta = params["beta"]
    if not gated:
        return Z @ beta
    g = torch.sigmoid(X @ params["C"].T)                           # [n, 2], c = 0 이면 모두 0.5
    return (g * beta * Z).sum(dim=1)


def fit(Z, X, t, e, gated: bool, lam: float = 0.0):
    params = {"beta": torch.zeros(2, requires_grad=True)}
    if gated:
        params["C"] = torch.zeros(2, X.shape[1], requires_grad=True)
    opt = torch.optim.LBFGS(list(params.values()), lr=1.0, max_iter=500,
                            tolerance_grad=1e-10, tolerance_change=1e-12, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = cox_nll(forward(params, Z, X, gated), t, e)
        if gated:
            loss = loss + lam * (params["C"] ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return {k: v.detach() for k, v in params.items()}


def design(ids, risks, labels, clin, stats=None, target="os", gate_inputs="stage_age"):
    """표준화한 위험점수 Z, 게이트 입력 X, 기간·사건. 표준화 통계는 train 에서만 잡는다."""
    rt = np.array([risks["tabular"][i] for i in ids])
    ri = np.array([risks["image"][i] for i in ids])
    es = (clin.loc[ids, "stage"].to_numpy() == 2).astype(float)
    age = clin.loc[ids, "age_at_diagnosis"].to_numpy(dtype=float)
    if stats is None:
        stats = {"rt": (rt.mean(), rt.std()), "ri": (ri.mean(), ri.std()), "es": es.mean(),
                 "age": (age.mean(), age.std())}
    zt = (rt - stats["rt"][0]) / stats["rt"][1]
    zi = (ri - stats["ri"][0]) / stats["ri"][1]
    Z = np.stack([zt, zi], axis=1)
    if gate_inputs == "risk":
        X = np.stack([zt, zi], axis=1)
    else:
        X = np.stack([zt, zi, es - stats["es"], (age - stats["age"][0]) / stats["age"][1]], axis=1)
    t = labels.loc[ids, f"{target}_days"].to_numpy(dtype=float)
    e = labels.loc[ids, f"{target}_event"].to_numpy(dtype=float)
    to = lambda a: torch.tensor(a)
    return to(Z), to(X), to(t), to(e), stats


def pick_lambda(Z, X, t, e, seed=42) -> tuple[float, dict]:
    """train 환자 안의 5-fold inner CV 로 게이트 벌점 lambda 를 고른다 (fold 평균 C-index, 동점이면 큰 쪽)."""
    scores = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    for lam in LAMBDAS:
        cis = []
        for tr, va in kf.split(np.arange(len(t))):
            p = fit(Z[tr], X[tr], t[tr], e[tr], gated=True, lam=lam)
            cis.append(cindex(t[va].numpy(), forward(p, Z[va], X[va], True).numpy(), e[va].numpy()))
        scores[lam] = float(np.mean(cis))
    best = max(scores.values())
    chosen = max(lam for lam, s in scores.items() if s >= best - 1e-12)
    return chosen, scores


def run_target(target: str, gate_inputs: str = "stage_age") -> dict:
    coh = cohort.load_trimodal_cohort()
    labels = labels_by_id(coh, target)
    clin = coh.drop_duplicates("research_id").set_index("research_id")
    with open(os.path.join(OOF_DIR, f"oof_{target}.json"), encoding="utf-8") as fh:
        payload = json.load(fh)
    risks = {k: {int(i): float(v) for i, v in payload[k].items()} for k in ("tabular", "image")}

    const_ci, gate_ci, recs = [], [], []
    for fold, ids in fold_plan(coh):
        Ztr, Xtr, ttr, etr, stats = design(ids["train"], risks, labels, clin, target=target, gate_inputs=gate_inputs)
        Zte, Xte, tte, ete, _ = design(ids["test"], risks, labels, clin, stats, target=target, gate_inputs=gate_inputs)

        p0 = fit(Ztr, Xtr, ttr, etr, gated=False)                  # 상수 게이트 = 고정 가중치 Cox
        c0 = cindex(tte.numpy(), forward(p0, Zte, Xte, False).numpy(), ete.numpy())

        lam, inner = pick_lambda(Ztr, Xtr, ttr, etr)
        p = fit(Ztr, Xtr, ttr, etr, gated=True, lam=lam)
        risk_te = forward(p, Zte, Xte, True)
        c = cindex(tte.numpy(), risk_te.numpy(), ete.numpy())
        g = torch.sigmoid(Xte @ p["C"].T).numpy()
        g_img = g[:, 1]
        gate_dev = float(np.abs(g / 0.5 - 1.0).max())             # 꺼진 상태(0.5) 대비 최대 상대 변화

        const_ci.append(c0)
        gate_ci.append(c)
        recs.append({"fold": int(fold), "constant_gate_ci": c0, "gate_ci": c, "lambda": lam, "inner_cv": inner,
                     "beta": p["beta"].tolist(), "C": p["C"].tolist(),
                     "image_gate_mean": float(g_img.mean()), "image_gate_sd": float(g_img.std()),
                     "gate_max_rel_change": gate_dev})
        print(f"[{target}] fold {fold}: constant {c0:.4f} (official {OFFICIAL[target][fold - 1]:.4f})  "
              f"gate {c:.4f}  lambda={lam}  g_img {g_img.mean():.2f}±{g_img.std():.2f}")

    diff = np.abs(np.array(const_ci) - np.array(OFFICIAL[target]))
    out = {"target": target,
           "gate_inputs": ["z_tab", "z_img"] + ([] if gate_inputs == "risk" else ["extensive_stage", "age"]),
           "lambdas": list(LAMBDAS), "constant_gate_folds": [float(c) for c in const_ci],
           "official_folds": OFFICIAL[target], "reproduction_max_abs_diff": float(diff.max()),
           "mean": float(np.mean(gate_ci)), "std": float(np.std(gate_ci)),
           "folds": [float(c) for c in gate_ci], "fold_records": recs}
    print(f"[{target}] constant gate {np.mean(const_ci):.4f} (max |diff| vs official {diff.max():.4f})  "
          f"gated {out['mean']:.4f} ± {out['std']:.4f}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", type=comma_list, default=["os", "pfs"])
    ap.add_argument("--gate_inputs", choices=("stage_age", "risk"), default="stage_age")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    for t in args.targets:
        res = run_target(t, args.gate_inputs)
        path = os.path.join(OUT_DIR, f"results_{t}{'_riskonly' if args.gate_inputs == 'risk' else ''}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2, ensure_ascii=False)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
