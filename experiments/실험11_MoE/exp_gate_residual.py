# -*- coding: utf-8 -*-
"""잔차(residual) 게이트 -- CoxPH 2변수 해에서 정확히 출발해, 거기서 벗어나는
게 진짜 손실을 줄일 때만 벗어나도록 설계.

동기: M0(무작위 초기화)와 M1(무작위 초기화)이 둘 다 baseline 을 못 이겼는데,
M1 은 심지어 크게(-0.12) 무너졌다. "초기화가 나빠서 실패했나, 애초에 신호가
없어서 실패했나"를 분리하기 위해, correction 항의 마지막 layer 를 0으로 초기화한다
-- 그러면 학습 시작 시점의 예측이 **정확히** CoxPH 2변수 해와 같고, gradient descent
는 Cox loss 를 실제로 낮출 때만 그 지점에서 벗어난다. 즉 이 설계는 원리적으로
CoxPH 해보다 "운 나쁘게" 나빠질 이유가 없다(과적합으로 나빠질 수는 있지만, 무작위
초기화가 나쁜 지점에서 출발해 못 돌아오는 실패 모드는 없다).

  risk = coxph_risk(고정, 이미 fit 된 2변수 CoxPH) + weight_decay 로 0에 붙잡힌
         작은 MLP(gate_features -> 1) 보정항

Run: python 실험11_MoE/exp_gate_residual.py --targets os,pfs
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import itertools
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from sclc import expert_embeddings as ee
from exp_gate_m0 import cox_partial_log_likelihood, BASELINE, EPOCHS, LR

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "moe_gate_residual")
INNER_FOLDS = 5
SELECTION_SEED = 42
FINAL_SEEDS = [42, 43, 44, 45, 46]

GATE_INPUT_OPTIONS = ("risk_only", "pca2", "pca4")
WEIGHT_DECAY_OPTIONS = (1.0, 3e-1, 1e-1)  # correction 항을 0 근처에 강하게 붙잡음
CANDIDATES = list(itertools.product(GATE_INPUT_OPTIONS, WEIGHT_DECAY_OPTIONS))


class ResidualGate(nn.Module):
    """마지막 layer 를 0으로 초기화 -- 학습 시작 시점 예측은 coxph_risk 와 정확히 같다."""

    def __init__(self, gate_input_dim: int, hidden: int = 8):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(gate_input_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, coxph_risk: torch.Tensor, gate_features: torch.Tensor):
        correction = self.net(gate_features).squeeze(-1)
        return coxph_risk + correction, correction


def fit_coxph_risk(risk_train_raw, dur_train, evt_train, risk_test_raw, dur_test, evt_test):
    """2변수 CoxPH 를 outer-train 에 fit 하고, log partial hazard 를 train/test 모두에 대해 낸다.
    (predict_log_partial_hazard 를 써서 스케일이 로그-위험 그대로 유지되게 한다 -- Cox loss 가
    바로 이 스케일을 기대함.)"""
    cols = ["image", "tabular"]
    df = pd.DataFrame(risk_train_raw, columns=cols)
    df["duration"], df["event"] = dur_train, evt_train
    cph = CoxPHFitter(penalizer=0.0).fit(df, "duration", "event")
    risk_tr = cph.predict_log_partial_hazard(pd.DataFrame(risk_train_raw, columns=cols)).to_numpy()
    risk_te = cph.predict_log_partial_hazard(pd.DataFrame(risk_test_raw, columns=cols)).to_numpy()
    ci_repro = float(concordance_index(dur_test, -risk_te, evt_test))
    return risk_tr.astype(np.float32), risk_te.astype(np.float32), ci_repro, dict(cph.params_)


def build_gate_features(gate_input: str, train: dict, test: dict):
    if gate_input == "risk_only":
        raw_tr = np.stack([train["image_risk"], train["tabular_risk"]], axis=1)
        raw_te = np.stack([test["image_risk"], test["tabular_risk"]], axis=1)
    else:
        k = {"pca2": 2, "pca4": 4}[gate_input]
        parts_tr, parts_te = [], []
        for name in ("image_emb", "clinical_emb", "report_emb"):
            kk = min(k, train[name].shape[1], train[name].shape[0] - 1)
            pca = PCA(n_components=kk, random_state=0).fit(train[name])
            parts_tr.append(pca.transform(train[name]))
            parts_te.append(pca.transform(test[name]))
        raw_tr, raw_te = np.concatenate(parts_tr, axis=1), np.concatenate(parts_te, axis=1)
    scaler = StandardScaler().fit(raw_tr)
    return scaler.transform(raw_tr).astype(np.float32), scaler.transform(raw_te).astype(np.float32)


def train_and_eval(train: dict, test: dict, gate_input: str, weight_decay: float, seed: int, device):
    raw_risk_tr = np.stack([train["image_risk"], train["tabular_risk"]], axis=1)
    raw_risk_te = np.stack([test["image_risk"], test["tabular_risk"]], axis=1)
    coxph_tr, coxph_te, coxph_ci, coef = fit_coxph_risk(
        raw_risk_tr, train["duration"], train["event"], raw_risk_te, test["duration"], test["event"])
    gate_tr, gate_te = build_gate_features(gate_input, train, test)

    torch.manual_seed(seed)
    model = ResidualGate(gate_tr.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)

    coxph_tr_t = torch.from_numpy(coxph_tr).to(device)
    gate_tr_t = torch.from_numpy(gate_tr).to(device)
    dur_tr_t = torch.from_numpy(train["duration"]).float().to(device)
    evt_tr_t = torch.from_numpy(train["event"]).float().to(device)

    model.train()
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        pred, _ = model(coxph_tr_t, gate_tr_t)
        loss = cox_partial_log_likelihood(pred, dur_tr_t, evt_tr_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        coxph_te_t = torch.from_numpy(coxph_te).to(device)
        gate_te_t = torch.from_numpy(gate_te).to(device)
        pred_te, corr_te = model(coxph_te_t, gate_te_t)
    ci = float(concordance_index(test["duration"], -pred_te.cpu().numpy(), test["event"]))
    return ci, coxph_ci, corr_te.cpu().numpy()


def inner_cv_select(train: dict, device):
    n = len(train["duration"])
    kf = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SELECTION_SEED)
    scores = {c: [] for c in CANDIDATES}
    for tr_idx, va_idx in kf.split(np.arange(n)):
        inner_tr = {k: v[tr_idx] for k, v in train.items() if k != "rid"}
        inner_va = {k: v[va_idx] for k, v in train.items() if k != "rid"}
        for gate_input, wd in CANDIDATES:
            ci, _, _ = train_and_eval(inner_tr, inner_va, gate_input, wd, SELECTION_SEED, device)
            scores[(gate_input, wd)].append(ci)
    mean_scores = {c: float(np.mean(v)) for c, v in scores.items()}
    best = max(mean_scores, key=mean_scores.get)
    return best, mean_scores


def run_target(target: str, max_folds=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = ee.load_gate_inputs(target, max_folds=max_folds)
    fold_results = []
    for fold, fold_data in data.items():
        train, test = fold_data["train"], fold_data["test"]
        print(f"\n[gate_residual/{target}] fold {fold}: inner-CV 로 후보 {len(CANDIDATES)}개 탐색 중...")
        best_cand, inner_scores = inner_cv_select(train, device)
        print(f"  선택됨: gate_input={best_cand[0]} weight_decay={best_cand[1]:g} "
              f"(inner-CV C-index={inner_scores[best_cand]:.4f})")

        results = [train_and_eval(train, test, *best_cand, s, device) for s in FINAL_SEEDS]
        cis = [r[0] for r in results]
        coxph_ci = results[0][1]
        correction_scale = [float(np.abs(r[2]).mean()) for r in results]  # 보정항이 실제로 얼마나 벗어났나

        fold_results.append({
            "fold": fold, "selected_candidate": {"gate_input": best_cand[0], "weight_decay": best_cand[1]},
            "coxph_reproduction_ci": coxph_ci,
            "residual_gate_ci": {"mean": float(np.mean(cis)), "std": float(np.std(cis)), "per_seed": cis},
            "mean_abs_correction": float(np.mean(correction_scale)),
        })
        print(f"  outer test: coxph_repro={coxph_ci:.4f}  residual_gate={np.mean(cis):.4f}±{np.std(cis):.4f}  "
              f"|correction|_mean={np.mean(correction_scale):.4f}")

    summary = {
        "target": target, "baseline": BASELINE[target],
        "coxph_reproduction_mean": float(np.mean([f["coxph_reproduction_ci"] for f in fold_results])),
        "residual_gate_mean": float(np.mean([f["residual_gate_ci"]["mean"] for f in fold_results])),
        "folds": fold_results,
    }
    summary["delta_vs_coxph_reproduction"] = summary["residual_gate_mean"] - summary["coxph_reproduction_mean"]
    summary["delta_vs_existing_late_fusion"] = summary["residual_gate_mean"] - BASELINE[target]["late_coxph"]
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="os,pfs")
    ap.add_argument("--max_folds", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    all_results = {}
    for target in [t.strip() for t in args.targets.split(",") if t.strip()]:
        all_results[target] = run_target(target, max_folds=args.max_folds)
    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 70}\n요약 (잔차게이트, CoxPH 해에서 출발)\n{'=' * 70}")
    for target, r in all_results.items():
        print(f"\n[{target.upper()}] 기존 late fusion = {r['baseline']['late_coxph']:.4f}")
        print(f"  CoxPH 재현      = {r['coxph_reproduction_mean']:.4f}")
        print(f"  잔차게이트      = {r['residual_gate_mean']:.4f}")
        print(f"  Δ(잔차-CoxPH재현) = {r['delta_vs_coxph_reproduction']:+.4f}")
    print(f"\n-> {os.path.join(OUT_DIR, 'results.json')}")


if __name__ == "__main__":
    main()
