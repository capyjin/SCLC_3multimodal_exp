# -*- coding: utf-8 -*-
"""Prior-guided Gating -- gate_input 을 학습 임베딩 PCA 대신 임상 prior(stage, age)로.

동기: 실험9(병기별 stage-aware 융합)가 이미 "CoxPH 2변수 결합의 image/tabular
가중치가 병기(LS/ES)에 따라 다른가"를 테스트해 NO_EVIDENCE 판정을 받았지만,
OS 5/5, PFS 4/5 fold 에서 ES>LS 방향으로 이미지 가중치가 일관되게 높았다 --
LS 표본(n=72)이 근본적으로 검정력 부족이라는 결론이었다. exp_gate_m0/tuned
도 "입력이 풍부할수록(PCA 임베딩) 171명 규모에서 과적합"이라는 같은 패턴을
보였다. 이 실험은 그 두 결론을 결합한다: 저차원·비-노이즈(clinical) 입력이면
같은 gate 구조가 실제로 patient-adaptive 신호를 찾아낼 수 있는지 검증한다.

Opus 설계안(2026-08-31, exp_gate_m0_clinical 사양)을 따른다:
  gate_input(주 실험) = [stage(±0.5, 고정 인코딩), age_z(train fit)] -- 2차원.
  stage 를 z-score 하지 않는 이유: fold 별 LS 비율이 27~36%로 흔들려서
  중심화하면 계수의 "0" 기준점 자체가 fold 마다 달라진다(실험9 build_design
  docstring 과 동일 근거). ±0.5 고정 인코딩이면 weight_decay 로 계수가
  0으로 수축할 때 정확히 상수게이트(=기존 late fusion) 해로 수렴한다.
  tumor size 는 이 코호트에 컬럼 자체가 없어(2D MIP, 세그멘테이션 마스크
  없음) 제외한다.

GateM0/cox_partial_log_likelihood/coxph_reproduction/train_one/BASELINE/
DELTA_THRESHOLD/EPOCHS/LR/WEIGHT_DECAY 는 exp_gate_m0.py 에서, inner-CV
튜닝된 2차 실험용 train_and_eval 은 exp_gate_m0_tuned.py 에서 그대로
가져다 쓴다(같은 폴더 형제 모듈 -- 관례상 허용, core/ 승격 기준인
"2개 이상 실험 폴더가 공유" 는 아직 아님).

Run: python 실험11_MoE/exp_gate_m0_clinical.py --targets os,pfs
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
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from sclc import cohort, expert_embeddings as ee
from exp_gate_m0 import (  # 같은 실험 폴더의 형제 모듈 -- 관례상 허용
    train_one, coxph_reproduction, baseline_for, DELTA_THRESHOLD,
)
from exp_gate_m0_tuned import train_and_eval  # weight_decay 파라미터화된 트레이너(2차 실험용)

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "moe_gate_prior")
INNER_FOLDS = 5
SELECTION_SEED = 42
FINAL_SEEDS = [42, 43, 44, 45, 46]

STAGE_LS, STAGE_ES = 1, 2  # 실험9 exp_stage_aware_fusion.py 와 동일 상수

PRIMARY_GATE_INPUT = "prior_only"  # 사전등록: 탐색 없이 이 조합 하나만 본다
GATE_INPUT_OPTIONS = ("prior_only", "risk_prior")  # 2차(inner-CV 탐색) 후보
WEIGHT_DECAY_OPTIONS = (1e-1, 3e-2, 1e-2)
CANDIDATES = list(itertools.product(GATE_INPUT_OPTIONS, WEIGHT_DECAY_OPTIONS))


def attach_clinical_prior(data: dict) -> dict:
    """각 fold 의 train/test 에 prior_raw(=[stage_sc, age_raw]) 를 rid 정렬 그대로 붙인다.
    stage/age 는 baseline 공변량이라 시점 누수는 없지만, rid 정렬이 깨지면 다른
    환자의 라벨이 섞이는 조용한 오류가 되므로 순서를 명시적으로 검증한다."""
    cohort_df = cohort.load_trimodal_cohort()
    cf = cohort_df.drop_duplicates("research_id").set_index("research_id")
    assert cf.index.is_unique, "research_id 가 유일 인덱스가 아님"

    for fold, fold_data in data.items():
        for split_name in ("train", "test"):
            split = fold_data[split_name]
            rid = split["rid"]
            missing = set(int(r) for r in rid) - set(cf.index)
            if missing:
                raise ValueError(f"fold {fold}/{split_name}: clinical frame 에 없는 research_id {sorted(missing)}")
            sub = cf.loc[rid]
            if not np.array_equal(sub.index.to_numpy(), np.asarray(rid)):
                raise AssertionError(f"fold {fold}/{split_name}: clinical_frame 행 순서가 rid 정렬과 어긋남")

            stage = sub["stage"].to_numpy(dtype=int)
            if not set(np.unique(stage)) <= {STAGE_LS, STAGE_ES}:
                raise AssertionError(f"fold {fold}/{split_name}: 예상 밖 stage 값 {sorted(set(stage))}")
            age = sub["age_at_diagnosis"].to_numpy(dtype=float)
            if np.isnan(age).any():
                raise AssertionError(f"fold {fold}/{split_name}: age_at_diagnosis 결측 존재")

            sc_stage = np.where(stage == STAGE_ES, 0.5, -0.5)
            split["prior_raw"] = np.stack([sc_stage, age], axis=1)
    return data


def build_tensors_for_candidate(fold_data: dict, gate_input: str) -> dict:
    train, test = fold_data["train"], fold_data["test"]

    risk_train_raw = np.stack([train["image_risk"], train["tabular_risk"]], axis=1)
    risk_test_raw = np.stack([test["image_risk"], test["tabular_risk"]], axis=1)
    risk_scaler = StandardScaler().fit(risk_train_raw)
    risk_train = risk_scaler.transform(risk_train_raw)
    risk_test = risk_scaler.transform(risk_test_raw)

    # stage(sc_stage) 는 고정 ±0.5 인코딩 그대로 두고, age 만 train 통계로 z-score 한다.
    # exp_gate_m0_tuned 처럼 concat 뒤 통째로 StandardScaler 를 걸면 stage 인코딩이
    # fold 마다 재조정돼 "수축 시 상수게이트로 정확히 수렴" 성질이 깨진다.
    age_scaler = StandardScaler().fit(train["prior_raw"][:, 1:2])
    prior_train = np.column_stack([
        train["prior_raw"][:, 0], age_scaler.transform(train["prior_raw"][:, 1:2]).ravel(),
    ])
    prior_test = np.column_stack([
        test["prior_raw"][:, 0], age_scaler.transform(test["prior_raw"][:, 1:2]).ravel(),
    ])

    if gate_input == "prior_only":
        gate_train, gate_test = prior_train, prior_test
    elif gate_input == "risk_prior":
        gate_train = np.hstack([risk_train, prior_train])
        gate_test = np.hstack([risk_test, prior_test])
    else:
        raise ValueError(f"알 수 없는 gate_input: {gate_input}")

    return {
        "risk_train": risk_train, "risk_test": risk_test,
        "gate_train": gate_train, "gate_test": gate_test,
        "duration_train": train["duration"], "event_train": train["event"],
        "duration_test": test["duration"], "event_test": test["event"],
        "raw_risk_train": risk_train_raw, "raw_risk_test": risk_test_raw,
        "prior_train": prior_train, "prior_test": prior_test,
    }


def inner_cv_select(fold_data: dict, device) -> tuple[tuple, dict]:
    """outer train(171명) 을 inner 5-fold 로 쪼개 후보별 평균 C-index 를 낸다.
    outer test 는 이 함수 안 어디에서도 참조하지 않는다(exp_gate_m0_tuned.py 와 동일 패턴)."""
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
    print(f"\n{'=' * 70}\n[gate_prior/{target}] 임베딩 로딩 + 임상 prior(stage,age) 부착 "
          f"(report_encoder={report_encoder})\n{'=' * 70}")
    data = ee.load_gate_inputs(target, max_folds=max_folds, report_encoder=report_encoder)
    data = attach_clinical_prior(data)

    fold_results = []
    for fold, fold_data in data.items():
        # ---- 주 실험: prior_only, 고정 wd=1e-2(=exp_gate_m0.WEIGHT_DECAY, train_one 그대로 재사용) ----
        tensors = build_tensors_for_candidate(fold_data, PRIMARY_GATE_INPUT)
        coxph_ci, coxph_coef = coxph_reproduction(tensors)
        learned_runs = [train_one(tensors, constant_gate=False, seed=s, device=device) for s in FINAL_SEEDS]
        const_runs = [train_one(tensors, constant_gate=True, seed=s, device=device) for s in FINAL_SEEDS]
        ci_learned = [r["ci_test"] for r in learned_runs]
        ci_const = [r["ci_test"] for r in const_runs]

        # ---- 진단: 게이트가 실험9 의 방향성(ES 에서 이미지 가중치 ↑)을 재현하는가 ----
        g_test0 = learned_runs[0]["gate_test"]  # [n,2] softmax, col0 = image
        prior_test = tensors["prior_test"]  # [n,2] = [sc_stage, age_z]
        n_test = g_test0.shape[0]
        A = np.column_stack([prior_test, np.ones(n_test)])
        y = np.log(np.clip(g_test0[:, 0], 1e-12, None) / np.clip(g_test0[:, 1], 1e-12, None))
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        fit_err = float(np.abs(A @ coef - y).max())
        if fit_err > 1e-4:
            raise AssertionError(f"fold {fold}: gate logit 선형성 self-check 실패 (max err {fit_err})")

        stage_raw_test = fold_data["test"]["prior_raw"][:, 0]  # ±0.5, sc>0 == ES
        age_raw_test = fold_data["test"]["prior_raw"][:, 1]
        is_es = stage_raw_test > 0
        g_img = g_test0[:, 0]
        n_ls, n_es = int((~is_es).sum()), int(is_es.sum())
        g_img_es_mean = float(g_img[is_es].mean()) if n_es else float("nan")
        g_img_ls_mean = float(g_img[~is_es].mean()) if n_ls else float("nan")
        rho_age, _ = spearmanr(g_img, age_raw_test)

        # ---- 2차 실험: {prior_only, risk_prior} x weight_decay, inner-CV 로 안전하게 선택 ----
        best_cand, inner_scores = inner_cv_select(fold_data, device)
        tuned_tensors = build_tensors_for_candidate(fold_data, best_cand[0])
        tuned_cis = [train_and_eval(tuned_tensors, best_cand[1], seed=s, device=device) for s in FINAL_SEEDS]

        fold_results.append({
            "fold": fold,
            "coxph_reproduction_ci": coxph_ci, "coxph_coef": coxph_coef,
            "learned_gate_ci": {"mean": float(np.mean(ci_learned)), "std": float(np.std(ci_learned)), "per_seed": ci_learned},
            "constant_gate_ci": {"mean": float(np.mean(ci_const)), "std": float(np.std(ci_const)), "per_seed": ci_const},
            "gate_logit_coef": {"stage": float(coef[0]), "age_z": float(coef[1]), "intercept": float(coef[2])},
            "g_image_by_stage": {
                "es_mean": g_img_es_mean, "ls_mean": g_img_ls_mean,
                "diff_es_minus_ls": g_img_es_mean - g_img_ls_mean, "n_es": n_es, "n_ls": n_ls,
            },
            "g_image_spearman_age": float(rho_age),
            "g_image_std": float(g_img.std()),
            "secondary": {
                "selected_candidate": {"gate_input": best_cand[0], "weight_decay": best_cand[1]},
                "inner_cv_ci": inner_scores[best_cand],
                "tuned_gate_ci": {"mean": float(np.mean(tuned_cis)), "std": float(np.std(tuned_cis)), "per_seed": tuned_cis},
            },
        })
        print(f"[gate_prior/{target}] fold {fold}: coxph_repro={coxph_ci:.4f}  "
              f"constant_gate={np.mean(ci_const):.4f}  prior_gate={np.mean(ci_learned):.4f}  "
              f"coef(stage={coef[0]:+.3f}, age={coef[1]:+.3f})  "
              f"g_img(ES={g_img_es_mean:.3f} n={n_es} / LS={g_img_ls_mean:.3f} n={n_ls})  "
              f"secondary_tuned={np.mean(tuned_cis):.4f}({best_cand[0]}/wd={best_cand[1]:g})")

    summary = {
        "target": target, "report_encoder": report_encoder,
        "baseline": baseline_for(target, report_encoder),
        "coxph_reproduction_mean": float(np.mean([f["coxph_reproduction_ci"] for f in fold_results])),
        "constant_gate_mean": float(np.mean([f["constant_gate_ci"]["mean"] for f in fold_results])),
        "prior_gate_mean": float(np.mean([f["learned_gate_ci"]["mean"] for f in fold_results])),
        "secondary_tuned_gate_mean": float(np.mean([f["secondary"]["tuned_gate_ci"]["mean"] for f in fold_results])),
        "folds": fold_results,
    }
    summary["delta_vs_coxph_reproduction"] = summary["prior_gate_mean"] - summary["coxph_reproduction_mean"]
    summary["delta_vs_constant_gate"] = summary["prior_gate_mean"] - summary["constant_gate_mean"]
    summary["delta_vs_existing_late_fusion"] = summary["prior_gate_mean"] - baseline_for(target, report_encoder)["late_coxph"]
    summary["performance_verdict"] = (
        "PASS" if (summary["delta_vs_coxph_reproduction"] > DELTA_THRESHOLD
                   and summary["delta_vs_constant_gate"] > DELTA_THRESHOLD)
        else "FAIL"
    )

    n_folds = len(fold_results)
    n_stage_pos = sum(1 for f in fold_results if f["gate_logit_coef"]["stage"] > 0)
    n_age_pos = sum(1 for f in fold_results if f["gate_logit_coef"]["age_z"] > 0)
    mean_es_minus_ls = float(np.mean([f["g_image_by_stage"]["diff_es_minus_ls"] for f in fold_results]))
    summary["sign_consistency"] = {
        "stage_coef_positive_folds": f"{n_stage_pos}/{n_folds}",
        "age_coef_positive_folds": f"{n_age_pos}/{n_folds}",
        "mean_within_fold_g_image_es_minus_ls": mean_es_minus_ls,
    }
    summary["diagnostic_verdict"] = (
        "실험9 방향성(ES>LS) 확인" if (n_stage_pos >= 4 and mean_es_minus_ls > 0)
        else "실험9 방향성 미확인/반박"
    )
    return summary


def negative_control(fold_data: dict, n_draws: int, device, rng: np.random.Generator) -> list[float]:
    """prior_raw 를 train/test 안에서 각각 독립적으로 섞어 재실행 -- Δ 의 널 분포.
    성능 검정이 PASS 로 나온 target 에 한해서만(--nc_draws>0) 돌린다."""
    null_deltas = []
    for _ in range(n_draws):
        shuffled = {
            "train": dict(fold_data["train"]), "test": dict(fold_data["test"]),
        }
        shuffled["train"]["prior_raw"] = rng.permutation(fold_data["train"]["prior_raw"])
        shuffled["test"]["prior_raw"] = rng.permutation(fold_data["test"]["prior_raw"])
        tensors = build_tensors_for_candidate(shuffled, PRIMARY_GATE_INPUT)
        coxph_ci, _ = coxph_reproduction(tensors)
        run = train_one(tensors, constant_gate=False, seed=SELECTION_SEED, device=device)
        null_deltas.append(run["ci_test"] - coxph_ci)
    return null_deltas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="os,pfs")
    ap.add_argument("--max_folds", type=int, default=None)
    ap.add_argument("--nc_draws", type=int, default=0, help="PASS 판정 target 에 한해 실행할 negative-control 순열 수")
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

    if args.nc_draws > 0:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rng = np.random.default_rng(SELECTION_SEED)
        for target, summary in all_results.items():
            if summary["performance_verdict"] != "PASS":
                continue
            data = attach_clinical_prior(
                ee.load_gate_inputs(target, max_folds=args.max_folds, report_encoder=args.report_encoder))
            null_deltas = []
            for fold_data in data.values():
                null_deltas.extend(negative_control(fold_data, args.nc_draws, device, rng))
            obs = summary["delta_vs_coxph_reproduction"]
            summary["negative_control"] = {
                "n_draws": args.nc_draws,
                "p_value": float(np.mean(np.array(null_deltas) >= obs)),
            }

    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 70}\n요약 (Prior-guided Gating: stage + age)\n{'=' * 70}")
    for target, r in all_results.items():
        print(f"\n[{target.upper()}] 기존 late fusion = {r['baseline']['late_coxph']:.4f}")
        print(f"  CoxPH 재현         = {r['coxph_reproduction_mean']:.4f}")
        print(f"  상수게이트         = {r['constant_gate_mean']:.4f}")
        print(f"  prior 게이트(주실험) = {r['prior_gate_mean']:.4f}")
        print(f"  Δ(prior-CoxPH재현)  = {r['delta_vs_coxph_reproduction']:+.4f}")
        print(f"  Δ(prior-상수게이트) = {r['delta_vs_constant_gate']:+.4f}  -> {r['performance_verdict']}")
        print(f"  2차(inner-CV 튜닝)  = {r['secondary_tuned_gate_mean']:.4f}")
        sc = r["sign_consistency"]
        print(f"  stage계수>0: {sc['stage_coef_positive_folds']}  age계수>0: {sc['age_coef_positive_folds']}  "
              f"g_image(ES-LS) 평균={sc['mean_within_fold_g_image_es_minus_ls']:+.4f}  -> {r['diagnostic_verdict']}")
        if "negative_control" in r:
            print(f"  negative control p={r['negative_control']['p_value']:.4f} (n_draws={r['negative_control']['n_draws']})")

    print(f"\n-> {os.path.join(out_dir, 'results.json')}")


if __name__ == "__main__":
    main()
