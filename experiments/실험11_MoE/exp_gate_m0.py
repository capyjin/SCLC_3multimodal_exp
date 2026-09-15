# -*- coding: utf-8 -*-
"""M0 -- 얼린 image/tabular expert 위에 최소 gate 층만 새로 학습.

설계는 Opus 검토안(2026-08-31)을 따른다:
  s_k        = 고정. expert 자신의 head 출력 risk (image_risk, tabular_risk).
               fold-train 통계로 z-score 표준화.
  gate_input = expert별 PCA-8(train 171로만 fit) 을 이어붙인 것.
  g          = softmax(Linear(gate_input -> K=2))
  risk       = alpha * (g_image*image_risk + g_tabular*tabular_risk)

이 구조는 g 가 상수로 붕괴하면 late fusion 의 CoxPH 2변수 결합과 수학적으로
동치다(alpha 가 전역 스케일, g 의 비율이 CoxPH 계수비 역할) -- 그래서 G2(상수게이트
대조군)와의 차이가 "gate 가 실제로 뭔가 하는지"를 바로 드러낸다.

백본은 전부 sclc.expert_embeddings 에서 읽기만 한다(재학습 없음, 새 누수 없음).
여기서 유일하게 새로 학습되는 것은 이 파일의 GateM0 뿐이다.

Run: python 실험11_MoE/exp_gate_m0.py --targets os,pfs --seeds 42,43,44,45,46
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sclc import cohort, expert_embeddings as ee

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "moe_gate_m0")
PCA_K = 8
EPOCHS = 400
LR = 3e-2
WEIGHT_DECAY = 1e-2
BASELINE = {  # outputs/late_fusion_B/results.json 에 기록된 기존 late fusion 기준선
    "os": {"tabular_only": 0.7076, "image_only": 0.6570, "late_coxph": 0.7221},
    "pfs": {"tabular_only": 0.6678, "image_only": 0.6154, "late_coxph": 0.6531},
}
# MODEL_SUMMARY.md §2-1/§3-2 RadBERT 열 (fix_brain_meta=True). image 축은 판독지
# 인코더와 무관해 TF-IDF 와 동일값.
BASELINE_RADBERT = {
    "os": {"tabular_only": 0.7153, "image_only": 0.6570, "late_coxph": 0.7224},
    "pfs": {"tabular_only": 0.6456, "image_only": 0.6154, "late_coxph": 0.6470},
}
DELTA_THRESHOLD = 0.016  # 5-fold paired 비교의 관례적 최소 유의차 (RESULTS_TABLE_final.md)


def baseline_for(target: str, report_encoder: str = "tfidf") -> dict:
    return (BASELINE if report_encoder == "tfidf" else BASELINE_RADBERT)[target]


def cox_partial_log_likelihood(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    """표준 Cox partial log-likelihood (Breslow 근사). MCL 논문 공개코드
    (survival.py::cox_partial_log_likelihood)와 동일한 정의를 그대로 채택."""
    order = torch.argsort(time, descending=True)
    risk = risk[order]
    event = event[order].float()
    log_risk_set = torch.logcumsumexp(risk, dim=0)
    observed = event > 0
    if observed.sum() == 0:
        return risk.new_tensor(0.0)
    return -(risk[observed] - log_risk_set[observed]).mean()


class GateM0(nn.Module):
    def __init__(self, gate_input_dim: int, k: int = 2, constant_gate: bool = False):
        super().__init__()
        self.constant_gate = constant_gate
        if constant_gate:
            self.logits = nn.Parameter(torch.zeros(k))
        else:
            self.gate = nn.Linear(gate_input_dim, k)
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, gate_features: torch.Tensor, risks: torch.Tensor):
        if self.constant_gate:
            logits = self.logits.unsqueeze(0).expand(risks.size(0), -1)
        else:
            logits = self.gate(gate_features)
        g = torch.softmax(logits, dim=-1)
        combined = (g * risks).sum(dim=-1)
        return self.alpha * combined, g


def build_fold_tensors(fold_data: dict):
    """train 통계로만 표준화/PCA 를 fit 하고 train+test 에 적용."""
    train, test = fold_data["train"], fold_data["test"]

    risk_train = np.stack([train["image_risk"], train["tabular_risk"]], axis=1)
    risk_test = np.stack([test["image_risk"], test["tabular_risk"]], axis=1)
    risk_scaler = StandardScaler().fit(risk_train)
    risk_train_std = risk_scaler.transform(risk_train)
    risk_test_std = risk_scaler.transform(risk_test)

    gate_train_parts, gate_test_parts = [], []
    for key in ("image_emb", "clinical_emb", "report_emb"):
        k = min(PCA_K, train[key].shape[1], train[key].shape[0] - 1)
        pca = PCA(n_components=k, random_state=0).fit(train[key])
        gate_train_parts.append(pca.transform(train[key]))
        gate_test_parts.append(pca.transform(test[key]))
    gate_train = np.concatenate(gate_train_parts, axis=1)
    gate_test = np.concatenate(gate_test_parts, axis=1)
    gate_scaler = StandardScaler().fit(gate_train)
    gate_train = gate_scaler.transform(gate_train)
    gate_test = gate_scaler.transform(gate_test)

    return {
        "risk_train": risk_train_std, "risk_test": risk_test_std,
        "gate_train": gate_train, "gate_test": gate_test,
        "duration_train": train["duration"], "event_train": train["event"],
        "duration_test": test["duration"], "event_test": test["event"],
        "raw_risk_train": risk_train, "raw_risk_test": risk_test,  # G3/G4 용 원본 스케일
    }


def train_one(tensors: dict, constant_gate: bool, seed: int, device):
    torch.manual_seed(seed)
    gate_dim = tensors["gate_train"].shape[1]
    model = GateM0(gate_dim, k=2, constant_gate=constant_gate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    risk_tr = torch.from_numpy(tensors["risk_train"]).float().to(device)
    gate_tr = torch.from_numpy(tensors["gate_train"]).float().to(device)
    dur_tr = torch.from_numpy(tensors["duration_train"]).float().to(device)
    evt_tr = torch.from_numpy(tensors["event_train"]).float().to(device)

    for _ in range(EPOCHS):
        optimizer.zero_grad()
        pred, _ = model(gate_tr, risk_tr)  # full-batch (n=171) -- 실험2 의 bs16 함정 회피
        loss = cox_partial_log_likelihood(pred, dur_tr, evt_tr)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        risk_te = torch.from_numpy(tensors["risk_test"]).float().to(device)
        gate_te = torch.from_numpy(tensors["gate_test"]).float().to(device)
        pred_te, g_te = model(gate_te, risk_te)
        pred_tr, g_tr = model(gate_tr, risk_tr)

    ci_test = concordance_index(tensors["duration_test"], -pred_te.cpu().numpy(), tensors["event_test"])
    ci_train = concordance_index(tensors["duration_train"], -pred_tr.cpu().numpy(), tensors["event_train"])
    return {
        "ci_test": float(ci_test), "ci_train": float(ci_train),
        "gate_test": g_te.cpu().numpy(), "gate_train": g_tr.cpu().numpy(),
        "alpha": float(model.alpha.item()),
        "gate_weight_norm": float(model.gate.weight.norm().item()) if not constant_gate else 0.0,
    }


def coxph_reproduction(tensors: dict) -> float:
    """G7(a) -- 같은 risk 배열로 고전 CoxPH 2변수 결합을 재현 (기존 late_simplecnn 대조용)."""
    import pandas as pd
    df = pd.DataFrame(tensors["raw_risk_train"], columns=["image", "tabular"])
    df["duration"] = tensors["duration_train"]
    df["event"] = tensors["event_train"]
    cph = CoxPHFitter(penalizer=0.0).fit(df, "duration", "event")
    test_df = pd.DataFrame(tensors["raw_risk_test"], columns=["image", "tabular"])
    risk_te = cph.predict_partial_hazard(test_df).to_numpy()
    return float(concordance_index(tensors["duration_test"], -risk_te, tensors["event_test"])), {
        "image": float(cph.params_["image"]), "tabular": float(cph.params_["tabular"]),
    }


def run_target(target: str, seeds: list[int], max_folds: int | None = None, report_encoder: str = "tfidf"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 70}\n[gate_m0/{target}] 임베딩 로딩 (image=캐시 읽기, tabular=체크포인트 forward, "
          f"report_encoder={report_encoder})\n{'=' * 70}")
    data = ee.load_gate_inputs(target, max_folds=max_folds, report_encoder=report_encoder)
    cohort_df = cohort.load_trimodal_cohort()
    clinical_frame = cohort_df.drop_duplicates("research_id").set_index("research_id")

    fold_results = []
    for fold, fold_data in data.items():
        tensors = build_fold_tensors(fold_data)
        coxph_ci, coxph_coef = coxph_reproduction(tensors)

        seed_runs_learned = [train_one(tensors, constant_gate=False, seed=s, device=device) for s in seeds]
        seed_runs_const = [train_one(tensors, constant_gate=True, seed=s, device=device) for s in seeds]

        ci_learned = [r["ci_test"] for r in seed_runs_learned]
        ci_const = [r["ci_test"] for r in seed_runs_const]

        # G1: 학습된 gate 의 환자별 분산(첫 seed 대표값 + seed 평균)
        gate_img_std_per_seed = [float(r["gate_test"][:, 0].std()) for r in seed_runs_learned]
        gate_img_mean_per_seed = [float(r["gate_test"][:, 0].mean()) for r in seed_runs_learned]

        # G4: expert 별 in-sample(train) vs OOF(test) C-index
        img_train_ci = concordance_index(tensors["duration_train"], -tensors["raw_risk_train"][:, 0], tensors["event_train"])
        img_test_ci = concordance_index(tensors["duration_test"], -tensors["raw_risk_test"][:, 0], tensors["event_test"])
        tab_train_ci = concordance_index(tensors["duration_train"], -tensors["raw_risk_train"][:, 1], tensors["event_train"])
        tab_test_ci = concordance_index(tensors["duration_test"], -tensors["raw_risk_test"][:, 1], tensors["event_test"])

        # G6: 첫 seed 학습게이트의 image 가중치가 임상변수/라벨과 상관있는지
        test_rid = fold_data["test"]["rid"]
        sub = clinical_frame.loc[test_rid]
        g_img = seed_runs_learned[0]["gate_test"][:, 0]
        g6 = {}
        for col in ("age", "ecog_ps", f"{target}_days", f"{target}_event"):
            if col in sub.columns:
                vals = sub[col].to_numpy(dtype=float)
                if np.std(vals) > 0:
                    g6[col] = float(np.corrcoef(g_img, vals)[0, 1])

        fold_results.append({
            "fold": fold,
            "n_train": len(tensors["duration_train"]), "n_test": len(tensors["duration_test"]),
            "coxph_reproduction_ci": coxph_ci, "coxph_coef": coxph_coef,
            "learned_gate_ci": {"mean": float(np.mean(ci_learned)), "std": float(np.std(ci_learned)), "per_seed": ci_learned},
            "constant_gate_ci": {"mean": float(np.mean(ci_const)), "std": float(np.std(ci_const)), "per_seed": ci_const},
            "g1_gate_image_weight_std": {"mean": float(np.mean(gate_img_std_per_seed)), "per_seed": gate_img_std_per_seed},
            "g1_gate_image_weight_mean": {"mean": float(np.mean(gate_img_mean_per_seed)), "per_seed": gate_img_mean_per_seed},
            "g4_in_sample_vs_oof": {
                "image": {"train": img_train_ci, "test": img_test_ci, "gap": img_train_ci - img_test_ci},
                "tabular": {"train": tab_train_ci, "test": tab_test_ci, "gap": tab_train_ci - tab_test_ci},
            },
            "g6_gate_correlation": g6,
        })
        print(f"[gate_m0/{target}] fold {fold}: coxph_repro={coxph_ci:.4f}  "
              f"constant_gate={np.mean(ci_const):.4f}±{np.std(ci_const):.4f}  "
              f"learned_gate={np.mean(ci_learned):.4f}±{np.std(ci_learned):.4f}  "
              f"g_image_std={np.mean(gate_img_std_per_seed):.4f}")

    summary = {
        "target": target, "seeds": seeds, "report_encoder": report_encoder,
        "baseline": baseline_for(target, report_encoder),
        "coxph_reproduction_mean": float(np.mean([f["coxph_reproduction_ci"] for f in fold_results])),
        "constant_gate_mean": float(np.mean([f["constant_gate_ci"]["mean"] for f in fold_results])),
        "learned_gate_mean": float(np.mean([f["learned_gate_ci"]["mean"] for f in fold_results])),
        "folds": fold_results,
    }
    delta_learned_vs_const = summary["learned_gate_mean"] - summary["constant_gate_mean"]
    delta_learned_vs_baseline = summary["learned_gate_mean"] - baseline_for(target, report_encoder)["late_coxph"]
    summary["delta_learned_vs_constant_gate"] = delta_learned_vs_const
    summary["delta_learned_vs_existing_late_fusion"] = delta_learned_vs_baseline
    summary["g2_verdict"] = (
        "학습게이트가 상수게이트 대비 유의미(>0.016)"
        if delta_learned_vs_const > DELTA_THRESHOLD else
        "학습게이트 ≈ 상수게이트 (gate가 patient-adaptive 신호를 못 찾음, 장식화 의심)"
    )

    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="os,pfs")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--max_folds", type=int, default=None)
    ap.add_argument("--report_encoder", choices=("tfidf", "radbert"), default="tfidf",
                    help="tfidf(기본, 기존 결과 재현) 또는 radbert -- outputs/late_fusion_B_radbert 체크포인트")
    ap.add_argument("--out_dir", default=None,
                    help=f"기본: tfidf -> {OUT_DIR}, radbert -> {OUT_DIR}_radbert")
    args = ap.parse_args()

    out_dir = args.out_dir or (OUT_DIR if args.report_encoder == "tfidf" else f"{OUT_DIR}_radbert")
    if args.max_folds is not None and out_dir == OUT_DIR:
        raise SystemExit("부분 실행(--max_folds)이 기록된 5-fold 결과를 덮어쓴다 -- --out_dir 를 따로 주라")
    os.makedirs(out_dir, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    all_results = {}
    for target in [t.strip() for t in args.targets.split(",") if t.strip()]:
        all_results[target] = run_target(target, seeds, max_folds=args.max_folds,
                                         report_encoder=args.report_encoder)

    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 70}\n요약\n{'=' * 70}")
    for target, r in all_results.items():
        base = r["baseline"]
        print(f"\n[{target.upper()}] 기존 late fusion(CoxPH) = {base['late_coxph']:.4f} "
              f"(tabular단독 {base['tabular_only']:.4f} / image단독 {base['image_only']:.4f})")
        print(f"  이번 실험 CoxPH 재현       = {r['coxph_reproduction_mean']:.4f}")
        print(f"  상수게이트(neural)         = {r['constant_gate_mean']:.4f}")
        print(f"  학습게이트(neural, M0)     = {r['learned_gate_mean']:.4f}")
        print(f"  Δ(학습게이트-상수게이트)    = {r['delta_learned_vs_constant_gate']:+.4f}  -> {r['g2_verdict']}")
        print(f"  Δ(학습게이트-기존late fusion) = {r['delta_learned_vs_existing_late_fusion']:+.4f}")

    print(f"\n-> {os.path.join(out_dir, 'results.json')}")


if __name__ == "__main__":
    main()
