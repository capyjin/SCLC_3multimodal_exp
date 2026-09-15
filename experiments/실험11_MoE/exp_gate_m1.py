# -*- coding: utf-8 -*-
"""M1 -- 원본 임베딩(image 512 / clinical 128 / report 16)에 cross-attention +
새 head 를 얹는 유연설계. Opus 설계안(2026-08-31)의 M1 사양을 따른다.

M0(risk 스칼라 재조합)와의 구조적 차이: M0는 각 expert 가 이미 내놓은 위험점수
"의견"을 재가중합만 할 수 있지만, M1은 원본 임베딩에 **새 head**를 얹으므로
원래 head가 놓쳤을 조합을 원칙적으로 학습할 수 있다. 다만 M0(tuned)가 이미
"outer-train 171명 규모에서 patient-adaptive 신호를 못 찾는다"를 inner-CV 선택
불안정성으로 보여줬으므로, M1 은 파라미터가 더 많아 과적합 위험이 더 크다 --
기대치는 낮게 잡고, 같은 inner-CV 안전장치로 검증한다.

architecture:
  tok_k  = LayerNorm(Linear(d_k -> proj_dim)) + expert별 학습 토큰임베딩(3, proj_dim)
  tok'   = tok + Dropout(MultiheadAttention(tok, tok, tok, nhead=2))   # 1 layer, FFN 없음
  s_k    = shared Linear(proj_dim -> 1) 적용 (토큰마다 같은 head, 파라미터 최소화)
  g      = softmax(Linear(flatten(tok') -> 3))
  risk   = alpha * sum_k g_k * s_k

하이퍼파라미터(proj_dim, weight_decay, dropout)는 exp_gate_m0_tuned.py 와 동일한
outer-train 안 inner-5fold 로만 선택한다.

Run: python 실험11_MoE/exp_gate_m1.py --targets os,pfs
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
import torch
import torch.nn as nn
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold

from sclc import expert_embeddings as ee
from exp_gate_m0 import cox_partial_log_likelihood, coxph_reproduction, BASELINE, EPOCHS, LR

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "moe_gate_m1")
INNER_FOLDS = 5
SELECTION_SEED = 42
FINAL_SEEDS = [42, 43, 44, 45, 46]
EXPERT_DIMS = (512, 128, 16)  # image, clinical, report

PROJ_DIM_OPTIONS = (8, 16)
WEIGHT_DECAY_OPTIONS = (1e-1, 3e-2, 1e-2)
DROPOUT_OPTIONS = (0.5,)  # 파라미터 대비 표본이 작아 강한 dropout만 후보로
CANDIDATES = list(itertools.product(PROJ_DIM_OPTIONS, WEIGHT_DECAY_OPTIONS, DROPOUT_OPTIONS))


class GateM1(nn.Module):
    def __init__(self, dims=EXPERT_DIMS, proj_dim=16, dropout=0.5, nhead=2):
        super().__init__()
        self.proj = nn.ModuleList([nn.Sequential(nn.Linear(d, proj_dim), nn.LayerNorm(proj_dim)) for d in dims])
        self.tok_embed = nn.Parameter(torch.zeros(len(dims), proj_dim))
        self.attn = nn.MultiheadAttention(proj_dim, nhead, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(proj_dim, 1)  # 토큰 간 공유(파라미터 최소화)
        self.gate = nn.Linear(proj_dim * len(dims), len(dims))
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, embs: list[torch.Tensor]):
        toks = torch.stack([p(e) for p, e in zip(self.proj, embs)], dim=1)  # [B,K,proj_dim]
        toks = toks + self.tok_embed.unsqueeze(0)
        attn_out, attn_w = self.attn(toks, toks, toks, need_weights=True)
        tok2 = toks + self.dropout(attn_out)
        s = self.head(tok2).squeeze(-1)  # [B,K]
        g = torch.softmax(self.gate(tok2.flatten(1)), dim=-1)  # [B,K]
        risk = self.alpha * (g * s).sum(dim=-1)
        return risk, g


def to_tensors(split: dict, device):
    return [
        torch.from_numpy(split["image_emb"]).float().to(device),
        torch.from_numpy(split["clinical_emb"]).float().to(device),
        torch.from_numpy(split["report_emb"]).float().to(device),
    ]


def train_and_eval(train_split: dict, test_split: dict, proj_dim: int, weight_decay: float,
                   dropout: float, seed: int, device) -> tuple[float, np.ndarray]:
    torch.manual_seed(seed)
    model = GateM1(proj_dim=proj_dim, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)

    embs_tr = to_tensors(train_split, device)
    dur_tr = torch.from_numpy(train_split["duration"]).float().to(device)
    evt_tr = torch.from_numpy(train_split["event"]).float().to(device)

    model.train()
    for _ in range(EPOCHS):
        optimizer.zero_grad()
        pred, _ = model(embs_tr)
        loss = cox_partial_log_likelihood(pred, dur_tr, evt_tr)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        embs_te = to_tensors(test_split, device)
        pred_te, g_te = model(embs_te)
    ci = float(concordance_index(test_split["duration"], -pred_te.cpu().numpy(), test_split["event"]))
    return ci, g_te.cpu().numpy()


def inner_cv_select(train_split: dict, device) -> tuple[tuple, float]:
    n = len(train_split["duration"])
    kf = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SELECTION_SEED)
    scores = {cand: [] for cand in CANDIDATES}
    for tr_idx, va_idx in kf.split(np.arange(n)):
        inner_tr = {k: v[tr_idx] for k, v in train_split.items() if k != "rid"}
        inner_va = {k: v[va_idx] for k, v in train_split.items() if k != "rid"}
        for proj_dim, wd, dropout in CANDIDATES:
            ci, _ = train_and_eval(inner_tr, inner_va, proj_dim, wd, dropout, SELECTION_SEED, device)
            scores[(proj_dim, wd, dropout)].append(ci)
    mean_scores = {cand: float(np.mean(v)) for cand, v in scores.items()}
    best = max(mean_scores, key=mean_scores.get)
    return best, mean_scores


def run_target(target: str, max_folds: int | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = ee.load_gate_inputs(target, max_folds=max_folds)

    fold_results = []
    for fold, fold_data in data.items():
        train, test = fold_data["train"], fold_data["test"]
        print(f"\n[gate_m1/{target}] fold {fold}: inner-CV 로 후보 {len(CANDIDATES)}개 탐색 중...")
        best_cand, inner_scores = inner_cv_select(train, device)
        print(f"  선택됨: proj_dim={best_cand[0]} weight_decay={best_cand[1]:g} dropout={best_cand[2]} "
              f"(inner-CV C-index={inner_scores[best_cand]:.4f})")

        coxph_ci, _ = coxph_reproduction({
            "raw_risk_train": np.stack([train["image_risk"], train["tabular_risk"]], axis=1),
            "raw_risk_test": np.stack([test["image_risk"], test["tabular_risk"]], axis=1),
            "duration_train": train["duration"], "event_train": train["event"],
            "duration_test": test["duration"], "event_test": test["event"],
        })

        test_results = [train_and_eval(train, test, *best_cand, s, device) for s in FINAL_SEEDS]
        test_cis = [r[0] for r in test_results]
        gate_stds = [float(r[1].std(axis=0).mean()) for r in test_results]  # G1: 환자별 게이트 분산

        fold_results.append({
            "fold": fold, "selected_candidate": {"proj_dim": best_cand[0], "weight_decay": best_cand[1], "dropout": best_cand[2]},
            "coxph_reproduction_ci": coxph_ci,
            "m1_gate_ci": {"mean": float(np.mean(test_cis)), "std": float(np.std(test_cis)), "per_seed": test_cis},
            "g1_gate_weight_std_mean": float(np.mean(gate_stds)),
        })
        print(f"  outer test: coxph_repro={coxph_ci:.4f}  M1_gate={np.mean(test_cis):.4f}±{np.std(test_cis):.4f}  "
              f"gate_std={np.mean(gate_stds):.4f}")

    summary = {
        "target": target, "baseline": BASELINE[target],
        "coxph_reproduction_mean": float(np.mean([f["coxph_reproduction_ci"] for f in fold_results])),
        "m1_gate_mean": float(np.mean([f["m1_gate_ci"]["mean"] for f in fold_results])),
        "folds": fold_results,
    }
    summary["delta_vs_coxph_reproduction"] = summary["m1_gate_mean"] - summary["coxph_reproduction_mean"]
    summary["delta_vs_existing_late_fusion"] = summary["m1_gate_mean"] - BASELINE[target]["late_coxph"]
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

    print(f"\n{'=' * 70}\n요약 (M1, inner-CV 튜닝, outer-test 는 최종 1회만)\n{'=' * 70}")
    for target, r in all_results.items():
        print(f"\n[{target.upper()}] 기존 late fusion = {r['baseline']['late_coxph']:.4f}")
        print(f"  CoxPH 재현    = {r['coxph_reproduction_mean']:.4f}")
        print(f"  M1 게이트     = {r['m1_gate_mean']:.4f}")
        print(f"  Δ(M1-CoxPH재현) = {r['delta_vs_coxph_reproduction']:+.4f}")
        print(f"  fold별 선택: {[f['selected_candidate'] for f in r['folds']]}")

    print(f"\n-> {os.path.join(OUT_DIR, 'results.json')}")


if __name__ == "__main__":
    main()
