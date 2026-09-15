# -*- coding: utf-8 -*-
"""M0 게이트의 하이퍼파라미터를 outer-train 안 inner-CV 로 안전하게 탐색.

exp_gate_m0.py 의 결과(학습게이트가 상수게이트를 못 이김)가 진단(G1: 게이트가
환자마다 변하지만 성능엔 도움 안 됨)상 **과적합**을 가리키므로, "레이어를 늘리는"
방향이 아니라 "입력 차원을 줄이고 정칙화를 세게 주는" 방향으로 후보를 만든다.

실험8/exp_image_cph.py 의 select_by_inner_cv 패턴을 그대로 따른다:
  outer train(171명)을 inner 5-fold 로 쪼개 후보별 평균 C-index 를 낸 뒤,
  outer test 는 최종 후보가 정해진 뒤 **한 번만** 본다. 후보 선택 과정에서
  outer test 의 어떤 값도 계산하거나 들여다보지 않는다.

candidates (gate_input, weight_decay) 축소:
  gate_input: "risk_only"(2차원, 원래 risk 표준화값 그대로) / "pca2"(전문가당
              PCA-2, 6차원) / "pca4"(전문가당 PCA-4, 12차원) -- 기존 pca8(24차원)
              은 과적합 용의선상이라 후보에서 제외하고 더 낮은 차원만 본다.
  weight_decay: 1e-1 / 3e-2 / 1e-2 (기존 1e-2 보다 센 정칙화 위주)
  epochs 는 기존과 동일(400)로 고정 -- 과적합 원인은 "얼마나 오래" 가 아니라
  "얼마나 많은 입력을 보는가" 라는 진단(G1)에 따라 이 축은 안 건드린다.

Run: python 실험11_MoE/exp_gate_m0_tuned.py --targets os,pfs
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import itertools

import numpy as np
import torch
from lifelines.utils import concordance_index
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from sclc import cohort, expert_embeddings as ee
from exp_gate_m0 import (  # 같은 실험 폴더의 형제 모듈 -- 관례상 허용
    GateM0, cox_partial_log_likelihood, coxph_reproduction, baseline_for, DELTA_THRESHOLD, EPOCHS, LR,
)

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "moe_gate_m0_tuned")
INNER_FOLDS = 5
SELECTION_SEED = 42  # 후보 선택 단계는 1개 seed 로 빠르게 (최종 평가만 5-seed)
FINAL_SEEDS = [42, 43, 44, 45, 46]

GATE_INPUT_OPTIONS = ("risk_only", "pca2", "pca4")
WEIGHT_DECAY_OPTIONS = (1e-1, 3e-2, 1e-2)
CANDIDATES = list(itertools.product(GATE_INPUT_OPTIONS, WEIGHT_DECAY_OPTIONS))


def make_gate_features(gate_input: str, image_emb, clinical_emb, report_emb, fit: bool, scalers=None):
    """gate_input 옵션에 따라 게이트 입력 벡터를 만든다. fit=True 면 PCA/scaler를
    새로 만들어 반환하고(반드시 train 데이터에만), fit=False 면 주어진 scalers 로
    transform 만 한다(test 데이터용)."""
    if gate_input == "risk_only":
        return None, {}  # GateM0 forward 에서 risk 자체를 gate_features 로 재사용
    per_expert_k = {"pca2": 2, "pca4": 4}[gate_input]
    parts = []
    new_scalers = {} if fit else None
    for name, emb in (("image", image_emb), ("clinical", clinical_emb), ("report", report_emb)):
        k = min(per_expert_k, emb.shape[1], emb.shape[0] - 1)
        if fit:
            pca = PCA(n_components=k, random_state=0).fit(emb)
            new_scalers[name] = pca
        else:
            pca = scalers[name]
        parts.append(pca.transform(emb))
    feats = np.concatenate(parts, axis=1)
    return feats, new_scalers


def build_tensors_for_candidate(fold_data: dict, gate_input: str):
    train, test = fold_data["train"], fold_data["test"]

    risk_train_raw = np.stack([train["image_risk"], train["tabular_risk"]], axis=1)
    risk_test_raw = np.stack([test["image_risk"], test["tabular_risk"]], axis=1)
    risk_scaler = StandardScaler().fit(risk_train_raw)
    risk_train = risk_scaler.transform(risk_train_raw)
    risk_test = risk_scaler.transform(risk_test_raw)

    if gate_input == "risk_only":
        gate_train, gate_test = risk_train, risk_test
    else:
        gate_train_raw, scalers = make_gate_features(
            gate_input, train["image_emb"], train["clinical_emb"], train["report_emb"], fit=True)
        gate_test_raw, _ = make_gate_features(
            gate_input, test["image_emb"], test["clinical_emb"], test["report_emb"], fit=False, scalers=scalers)
        gate_scaler = StandardScaler().fit(gate_train_raw)
        gate_train, gate_test = gate_scaler.transform(gate_train_raw), gate_scaler.transform(gate_test_raw)

    return {
        "risk_train": risk_train, "risk_test": risk_test,
        "gate_train": gate_train, "gate_test": gate_test,
        "duration_train": train["duration"], "event_train": train["event"],
        "duration_test": test["duration"], "event_test": test["event"],
    }


def train_and_eval(tensors: dict, weight_decay: float, seed: int, device) -> float:
    torch.manual_seed(seed)
    model = GateM0(tensors["gate_train"].shape[1], k=2, constant_gate=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)

    risk_tr = torch.from_numpy(tensors["risk_train"]).float().to(device)
    gate_tr = torch.from_numpy(tensors["gate_train"]).float().to(device)
    dur_tr = torch.from_numpy(tensors["duration_train"]).float().to(device)
    evt_tr = torch.from_numpy(tensors["event_train"]).float().to(device)

    for _ in range(EPOCHS):
        optimizer.zero_grad()
        pred, _ = model(gate_tr, risk_tr)
        loss = cox_partial_log_likelihood(pred, dur_tr, evt_tr)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        risk_te = torch.from_numpy(tensors["risk_test"]).float().to(device)
        gate_te = torch.from_numpy(tensors["gate_test"]).float().to(device)
        pred_te, _ = model(gate_te, risk_te)
    return float(concordance_index(tensors["duration_test"], -pred_te.cpu().numpy(), tensors["event_test"]))


def inner_cv_select(fold_data: dict, device) -> tuple[str, float]:
    """outer train(171명)을 inner 5-fold 로 쪼개 후보 조합의 평균 C-index 를 낸다.
    outer test 는 이 함수 안 어디에서도 참조하지 않는다."""
    train = fold_data["train"]
    n = len(train["duration"])
    kf = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SELECTION_SEED)

    scores = {cand: [] for cand in CANDIDATES}
    for tr_idx, va_idx in kf.split(np.arange(n)):
        inner_train = {k: v[tr_idx] for k, v in train.items() if k != "rid"}
        inner_val = {k: v[va_idx] for k, v in train.items() if k != "rid"}
        inner_fold_data = {"train": inner_train, "test": inner_val}
        for gate_input, wd in CANDIDATES:
            tensors = build_tensors_for_candidate(inner_fold_data, gate_input)
            ci = train_and_eval(tensors, wd, seed=SELECTION_SEED, device=device)
            scores[(gate_input, wd)].append(ci)

    mean_scores = {cand: float(np.mean(v)) for cand, v in scores.items()}
    best = max(mean_scores, key=mean_scores.get)
    return best, mean_scores


def run_target(target: str, max_folds: int | None = None, report_encoder: str = "tfidf"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = ee.load_gate_inputs(target, max_folds=max_folds, report_encoder=report_encoder)

    fold_results = []
    for fold, fold_data in data.items():
        print(f"\n[gate_m0_tuned/{target}] fold {fold}: inner-CV 로 후보 {len(CANDIDATES)}개 탐색 중...")
        best_cand, inner_scores = inner_cv_select(fold_data, device)
        print(f"  선택됨: gate_input={best_cand[0]} weight_decay={best_cand[1]:g} "
              f"(inner-CV C-index={inner_scores[best_cand]:.4f})")

        tensors = build_tensors_for_candidate(fold_data, best_cand[0])
        coxph_ci, _ = coxph_reproduction({
            "raw_risk_train": np.stack([fold_data["train"]["image_risk"], fold_data["train"]["tabular_risk"]], axis=1),
            "raw_risk_test": np.stack([fold_data["test"]["image_risk"], fold_data["test"]["tabular_risk"]], axis=1),
            "duration_train": tensors["duration_train"], "event_train": tensors["event_train"],
            "duration_test": tensors["duration_test"], "event_test": tensors["event_test"],
        })
        test_cis = [train_and_eval(tensors, best_cand[1], seed=s, device=device) for s in FINAL_SEEDS]

        fold_results.append({
            "fold": fold, "selected_candidate": {"gate_input": best_cand[0], "weight_decay": best_cand[1]},
            "inner_cv_scores": {f"{c[0]}_wd{c[1]:g}": v for c, v in inner_scores.items()},
            "coxph_reproduction_ci": coxph_ci,
            "tuned_gate_ci": {"mean": float(np.mean(test_cis)), "std": float(np.std(test_cis)), "per_seed": test_cis},
        })
        print(f"  outer test: coxph_repro={coxph_ci:.4f}  tuned_gate={np.mean(test_cis):.4f}±{np.std(test_cis):.4f}")

    summary = {
        "target": target, "report_encoder": report_encoder,
        "baseline": baseline_for(target, report_encoder),
        "coxph_reproduction_mean": float(np.mean([f["coxph_reproduction_ci"] for f in fold_results])),
        "tuned_gate_mean": float(np.mean([f["tuned_gate_ci"]["mean"] for f in fold_results])),
        "folds": fold_results,
    }
    summary["delta_vs_coxph_reproduction"] = summary["tuned_gate_mean"] - summary["coxph_reproduction_mean"]
    summary["delta_vs_existing_late_fusion"] = summary["tuned_gate_mean"] - baseline_for(target, report_encoder)["late_coxph"]
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="os,pfs")
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
    all_results = {}
    for target in [t.strip() for t in args.targets.split(",") if t.strip()]:
        all_results[target] = run_target(target, max_folds=args.max_folds, report_encoder=args.report_encoder)

    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 70}\n요약 (inner-CV 로 하이퍼파라미터 튜닝, outer-test 는 최종 1회만 확인)\n{'=' * 70}")
    for target, r in all_results.items():
        print(f"\n[{target.upper()}] 기존 late fusion = {r['baseline']['late_coxph']:.4f}")
        print(f"  이번 CoxPH 재현        = {r['coxph_reproduction_mean']:.4f}")
        print(f"  튜닝된 게이트          = {r['tuned_gate_mean']:.4f}")
        print(f"  Δ(튜닝게이트-CoxPH재현) = {r['delta_vs_coxph_reproduction']:+.4f}")
        selected = [f["selected_candidate"] for f in r["folds"]]
        print(f"  fold별 선택된 설정: {selected}")

    print(f"\n-> {os.path.join(out_dir, 'results.json')}")


if __name__ == "__main__":
    main()
