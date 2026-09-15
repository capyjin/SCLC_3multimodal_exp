# -*- coding: utf-8 -*-
"""서브그룹(병기 LS/ES 또는 고위험 복합군) C-index — late fusion vs MoE 게이트.

[연구 질문]
  RESULTS_ALL.md §3은 "전체 평균"에서 MoE 게이트가 late fusion을 못 이긴다는
  결론을 냈다. 하지만 전체 평균이 비슷하거나 MoE가 근소하게 못 미치더라도,
  **특정 서브그룹에서는 게이트가 더 나을 수 있다** — 이질적인 하위집단의
  평균은 비슷해도 내부에서는 방향이 반대로 상쇄될 수 있기 때문이다. 이
  스크립트는 그 가능성을 직접 잰다.

  두 가지 서브그룹 정의를 지원한다(--subgroup):
    stage      : LS(제한기) vs ES(확장기) — 실험9/실험11 exp_gate_m0_clinical의
                 연장선. 병기 자체는 clinical 변수 13개 중 하나라 예후 프록시일
                 뿐, "고위험군"과 동의어는 아니다.
    high_risk  : LDH 상위25% / ECOG≥2 / (간전이 or 뇌전이) 중 **2개 이상 동시
                 만족**을 고위험군으로 정의(238명 코호트 기준 n=50, os_event
                 49/50=98%). 사용자가 지정한 "예후가 매우 나쁜" 그룹에 가장
                 가까운 조작적 정의 — AND(3개 전부)는 n=6로 분석 불가, OR(1개
                 이상)는 n=147(62%)로 ES와 거의 겹쳐 변별력이 없어 기각했다.

[방법론 — 실험9의 교훈을 그대로 적용]
  - pooled-OOF 금지: fold 내부·그룹 내부에서만 concordant/comparable pair를
    세고, fold를 가로질러 pair 개수를 합산한다(sclc.evaluation.concordant_pair_
    counts). 이렇게 안 하면 fold별 위험점수 스케일 drift가 신호로 섞인다.
  - late fusion 기준선은 **이 실행에서 재구성한 CoxPH 2변수 결합**이다(저장된
    historical 값과 비교하면 착시 — RESULTS_ALL.md §5).
  - Δ(게이트-late fusion)의 유의성은 실험9 paired_bootstrap_delta와 동일한
    발상의 환자 단위 Poisson(1) 가중 부트스트랩(B=2000)으로 낸다.
  - **검증**: 이 파일이 로컬로 재구성한 환자별 예측이 각 fold·seed에서
    exp_gate_m0.coxph_reproduction()/train_one() (M0/Prior-gate) 또는
    exp_gate_m0_tuned.train_and_eval() (M0-tuned)의 반환 CI와 소수점 6자리까지
    일치하는지 매번 assert한다. 이미 결과가 문서화된 완결 실험 파일들은
    건드리지 않고 이 파일 안에 예측-추출용 복제본을 둔다.
  - 게이트 예측은 5-seed(42~46) 평균을 부트스트랩의 입력으로 쓴다. M0-tuned는
    fold마다 inner-CV(exp_gate_m0_tuned.inner_cv_select, outer-test 미참조)로
    고른 (gate_input, weight_decay) 후보를 그대로 재사용한다.

Run: python 실험11_MoE/exp_gate_stage_subgroup.py --targets os,pfs --gates m0_tuned \
       --subgroup high_risk
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
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

from sclc import cohort, expert_embeddings as ee
from sclc.evaluation import concordant_pair_counts

from exp_gate_m0 import (  # 같은 폴더 형제 모듈 — exp_gate_m0_clinical.py와 동일 관례
    GateM0, cox_partial_log_likelihood, build_fold_tensors,
    coxph_reproduction, train_one, baseline_for, EPOCHS, LR, WEIGHT_DECAY,
)
from exp_gate_m0_clinical import (
    attach_clinical_prior, build_tensors_for_candidate as build_tensors_prior,
)
from exp_gate_m0_tuned import (
    inner_cv_select, train_and_eval, build_tensors_for_candidate as build_tensors_tuned,
)

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "moe_gate_stage_subgroup")
STAGE_LS, STAGE_ES = 1, 2  # 실험9/실험11 exp_gate_m0_clinical.py 와 동일 상수
SEEDS = [42, 43, 44, 45, 46]  # exp_gate_m0.py 기본값과 동일
ASSERT_TOL = 1e-6
BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 42


# ---------------------------------------------------------------------------
# 서브그룹 정의
# ---------------------------------------------------------------------------

def group_flags(subgroup: str, clinical_frame: pd.DataFrame) -> tuple[pd.Series, str, str]:
    """research_id -> bool(그룹A 소속) Series와 (그룹A 이름, 그룹B 이름)을 반환.
    전체 238명은 항상 A/B 둘 중 하나에 정확히 속한다(누락 없음)."""
    if subgroup == "stage":
        stage = clinical_frame["stage"]
        bad = set(stage.unique()) - {STAGE_LS, STAGE_ES}
        if bad:
            raise AssertionError(f"예상 밖 stage 값 {sorted(bad)}")
        return stage == STAGE_LS, "LS", "ES"
    if subgroup == "high_risk":
        ldh_thr = clinical_frame["ldh"].quantile(0.75)
        ldh_high = clinical_frame["ldh"] >= ldh_thr
        ecog_high = clinical_frame["ecog"] >= 2
        meta = (clinical_frame["liver_meta"] == 1) | (clinical_frame["brain_meta"] == 1)
        score = ldh_high.astype(int) + ecog_high.astype(int) + meta.astype(int)
        flag = score >= 2
        print(f"[high_risk 정의] LDH>={ldh_thr:.0f}(상위25%) / ECOG>=2 / 간전이·뇌전이 중 "
              f"2개 이상 만족 -> n={int(flag.sum())}/{len(flag)}")
        return flag, "HR", "REST"
    raise ValueError(f"알 수 없는 subgroup: {subgroup}")


# ---------------------------------------------------------------------------
# 캐노니컬 함수 복제본(환자별 예측 추출용) — 기존 실험 파일은 수정하지 않는다
# ---------------------------------------------------------------------------

def coxph_reproduction_with_predictions(tensors: dict):
    train_df = pd.DataFrame(tensors["raw_risk_train"], columns=["image", "tabular"])
    train_df["duration"] = tensors["duration_train"]
    train_df["event"] = tensors["event_train"]
    cph = CoxPHFitter(penalizer=0.0).fit(train_df, "duration", "event")
    test_df = pd.DataFrame(tensors["raw_risk_test"], columns=["image", "tabular"])
    risk_te = cph.predict_partial_hazard(test_df).to_numpy().ravel()
    ci = float(concordance_index(tensors["duration_test"], -risk_te, tensors["event_test"]))
    return ci, risk_te


def _run_gate_m0(tensors: dict, weight_decay: float, seed: int, device):
    """GateM0 학습 루프 복제(exp_gate_m0.train_one / exp_gate_m0_tuned.train_and_eval
    과 동일 코드, weight_decay만 인자로 받아 두 캐노니컬 함수를 모두 재현 가능)."""
    torch.manual_seed(seed)
    gate_dim = tensors["gate_train"].shape[1]
    model = GateM0(gate_dim, k=2, constant_gate=False).to(device)
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

    pred_te_np = pred_te.cpu().numpy()
    ci_test = float(concordance_index(tensors["duration_test"], -pred_te_np, tensors["event_test"]))
    return ci_test, pred_te_np


def train_one_with_predictions(tensors: dict, seed: int, device):
    return _run_gate_m0(tensors, WEIGHT_DECAY, seed, device)


def train_and_eval_with_predictions(tensors: dict, weight_decay: float, seed: int, device):
    return _run_gate_m0(tensors, weight_decay, seed, device)


# ---------------------------------------------------------------------------
# 공용 집계
# ---------------------------------------------------------------------------

def accumulate(buckets: dict, key: str, dur, evt, risk):
    if len(dur) < 2:
        return
    c, n = concordant_pair_counts(dur, evt, risk)
    acc = buckets.setdefault(key, [0.0, 0.0])
    acc[0] += c
    acc[1] += n


def finalize(buckets: dict) -> dict:
    return {k: {"cindex": (v[0] / v[1]) if v[1] else float("nan"), "pairs": int(v[1])}
            for k, v in buckets.items()}


def bootstrap_delta_by_group(fold_records: list[dict], group_names: tuple[str, str],
                              *, B: int = BOOTSTRAP_B, seed: int = BOOTSTRAP_SEED) -> dict:
    """환자 단위 Poisson(1) 가중 부트스트랩으로 Δ(risk_b − risk_a)의 CI를 낸다.

    실험9 exp_stage_aware_fusion.py::paired_bootstrap_delta와 동일 발상(두 모델을
    같은 리샘플에서 동시에 계산 — 주변 CI를 빼는 것보다 좁고 정확)을, 이 스크립트의
    자료구조에 맞게 새로 구현했다. group_records의 'is_a'는 bool(그룹A 소속).
    """
    rng = np.random.default_rng(seed)
    pre = []
    for rec in fold_records:
        dur, evt, is_a = rec["dur"], rec["evt"], rec["is_a"]
        ra, rb = rec["risk_a"], rec["risk_b"]  # a=late fusion, b=게이트
        n = len(dur)
        idx = []
        for i, j in itertools.combinations(range(n), 2):
            if dur[i] < dur[j] and evt[i] == 1:
                idx.append((i, j))
            elif dur[j] < dur[i] and evt[j] == 1:
                idx.append((j, i))
        pre.append((n, is_a, ra, rb, idx))

    def stat(weights, want_a: bool):
        ca = cb = n = 0.0
        for (npat, is_a, ra, rb, idx), w in zip(pre, weights):
            for s, l in idx:
                if is_a[s] != want_a or is_a[l] != want_a:
                    continue
                ww = w[s] * w[l]
                if ww == 0:
                    continue
                n += ww
                ca += ww * (1.0 if ra[s] > ra[l] else 0.5 if ra[s] == ra[l] else 0.0)
                cb += ww * (1.0 if rb[s] > rb[l] else 0.5 if rb[s] == rb[l] else 0.0)
        return (cb / n - ca / n) if n else np.nan

    out = {}
    for name, want_a in ((group_names[0], True), (group_names[1], False)):
        deltas = []
        for _ in range(B):
            weights = [rng.poisson(1.0, size=npat).astype(float) for (npat, *_ ) in pre]
            v = stat(weights, want_a)
            if not np.isnan(v):
                deltas.append(v)
        deltas = np.array(deltas)
        out[name] = {
            "delta_mean": float(deltas.mean()), "sd": float(deltas.std()),
            "ci95": [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))],
            "P_gt_0": float((deltas > 0).mean()), "B": int(len(deltas)),
        }
    return out


# ---------------------------------------------------------------------------
# 메인 루프
# ---------------------------------------------------------------------------

def run_target(target: str, gates: list[str], subgroup: str, report_encoder: str = "tfidf"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 70}\n[{target}] 임베딩 로딩 (subgroup={subgroup})\n{'=' * 70}")
    data_plain = ee.load_gate_inputs(target, report_encoder=report_encoder)
    cohort_df = cohort.load_trimodal_cohort()
    clinical_frame = cohort_df.drop_duplicates("research_id").set_index("research_id")
    is_group_a_flag, name_a, name_b = group_flags(subgroup, clinical_frame)

    # ---- late fusion(CoxPH) 기준선: fold마다 재구성 + 캐노니컬 함수와 대조검증 ----
    late_buckets = {name_a: [0.0, 0.0], name_b: [0.0, 0.0]}
    late_overall = []
    late_by_fold = {}
    for fold, fold_data in data_plain.items():
        tensors = build_fold_tensors(fold_data)
        canon_ci, _ = coxph_reproduction(tensors)
        local_ci, risk_te = coxph_reproduction_with_predictions(tensors)
        assert abs(canon_ci - local_ci) < ASSERT_TOL, (
            f"late fusion fold {fold}: 로컬 재구성({local_ci:.6f})이 "
            f"exp_gate_m0.coxph_reproduction({canon_ci:.6f})과 불일치")
        late_overall.append(local_ci)

        rid = fold_data["test"]["rid"]
        is_a = is_group_a_flag.loc[rid].to_numpy(dtype=bool)
        dur, evt = tensors["duration_test"], tensors["event_test"]
        for key, mask in ((name_a, is_a), (name_b, ~is_a)):
            accumulate(late_buckets, key, dur[mask], evt[mask], risk_te[mask])
        late_by_fold[fold] = {"rid": rid, "is_a": is_a, "dur": dur, "evt": evt, "risk": risk_te}

    late_within = finalize(late_buckets)
    print(f"[late fusion] 전체 fold평균 CI(검증됨) = {np.mean(late_overall):.4f} | "
          f"{name_a}={late_within[name_a]['cindex']:.4f}(pairs={late_within[name_a]['pairs']}) "
          f"{name_b}={late_within[name_b]['cindex']:.4f}(pairs={late_within[name_b]['pairs']})")

    results = {"target": target, "report_encoder": report_encoder, "subgroup": subgroup,
               "group_sizes": {name_a: int(is_group_a_flag.sum()), name_b: int((~is_group_a_flag).sum())},
               "baseline_late_coxph_saved": baseline_for(target, report_encoder)["late_coxph"],
               "late_fusion_reproduced_overall_mean": float(np.mean(late_overall)),
               "late_fusion_within_group": late_within}

    # ---- 게이트 모델(M0 / Prior-gate / M0-tuned) ----
    for gate_name in gates:
        per_seed_buckets = {s: {name_a: [0.0, 0.0], name_b: [0.0, 0.0]} for s in SEEDS}
        per_seed_overall = {s: [] for s in SEEDS}
        pred_by_fold_seed = {}
        selected_candidates = {}

        if gate_name == "prior":
            gate_data = attach_clinical_prior(ee.load_gate_inputs(target, report_encoder=report_encoder))
        else:
            gate_data = data_plain

        for fold, fold_data in gate_data.items():
            rid = fold_data["test"]["rid"]
            assert np.array_equal(rid, late_by_fold[fold]["rid"]), (
                f"{gate_name} fold {fold}: rid 정렬이 late fusion과 다름")
            is_a = late_by_fold[fold]["is_a"]

            if gate_name == "m0":
                tensors = build_fold_tensors(fold_data)
                wd_for_fold = WEIGHT_DECAY
            elif gate_name == "prior":
                tensors = build_tensors_prior(fold_data, "prior_only")
                wd_for_fold = WEIGHT_DECAY
            elif gate_name == "m0_tuned":
                best_cand, inner_scores = inner_cv_select(fold_data, device)
                tensors = build_tensors_tuned(fold_data, best_cand[0])
                wd_for_fold = best_cand[1]
                selected_candidates[fold] = {"gate_input": best_cand[0], "weight_decay": best_cand[1]}
                print(f"  [m0_tuned] fold {fold}: 선택된 후보 gate_input={best_cand[0]} "
                      f"wd={best_cand[1]:g} (inner-CV={inner_scores[best_cand]:.4f})")
            else:
                raise ValueError(f"알 수 없는 gate: {gate_name}")

            dur, evt = tensors["duration_test"], tensors["event_test"]
            pred_by_fold_seed[fold] = {}
            for seed in SEEDS:
                if gate_name == "m0_tuned":
                    canon_ci = train_and_eval(tensors, wd_for_fold, seed=seed, device=device)
                else:
                    canon_ci = train_one(tensors, constant_gate=False, seed=seed, device=device)["ci_test"]
                local_ci, pred_te = train_and_eval_with_predictions(tensors, wd_for_fold, seed=seed, device=device)
                assert abs(canon_ci - local_ci) < ASSERT_TOL, (
                    f"{gate_name} fold {fold} seed {seed}: 로컬 재구성({local_ci:.6f})이 "
                    f"캐노니컬 함수({canon_ci:.6f})과 불일치")
                per_seed_overall[seed].append(local_ci)
                pred_by_fold_seed[fold][seed] = pred_te
                for key, mask in ((name_a, is_a), (name_b, ~is_a)):
                    accumulate(per_seed_buckets[seed], key, dur[mask], evt[mask], pred_te[mask])

        within_per_seed = {s: finalize(b) for s, b in per_seed_buckets.items()}
        a_vals = [within_per_seed[s][name_a]["cindex"] for s in SEEDS]
        b_vals = [within_per_seed[s][name_b]["cindex"] for s in SEEDS]
        overall_vals = [float(np.mean(per_seed_overall[s])) for s in SEEDS]
        gate_summary = {
            "overall_mean": float(np.mean(overall_vals)), "overall_std": float(np.std(overall_vals)),
            f"{name_a}_mean": float(np.mean(a_vals)), f"{name_a}_std": float(np.std(a_vals)),
            f"{name_b}_mean": float(np.mean(b_vals)), f"{name_b}_std": float(np.std(b_vals)),
            f"{name_a}_pairs": within_per_seed[SEEDS[0]][name_a]["pairs"],
            f"{name_b}_pairs": within_per_seed[SEEDS[0]][name_b]["pairs"],
            "per_seed": {name_a: a_vals, name_b: b_vals, "overall": overall_vals},
        }
        if selected_candidates:
            gate_summary["selected_candidates"] = selected_candidates
        gate_summary[f"delta_{name_a}_vs_late_fusion"] = gate_summary[f"{name_a}_mean"] - late_within[name_a]["cindex"]
        gate_summary[f"delta_{name_b}_vs_late_fusion"] = gate_summary[f"{name_b}_mean"] - late_within[name_b]["cindex"]

        fold_records = []
        for fold in late_by_fold:
            mean_pred = np.mean([pred_by_fold_seed[fold][s] for s in SEEDS], axis=0)
            rec = late_by_fold[fold]
            fold_records.append({"dur": rec["dur"], "evt": rec["evt"], "is_a": rec["is_a"],
                                  "risk_a": rec["risk"], "risk_b": mean_pred})
        boot = bootstrap_delta_by_group(fold_records, (name_a, name_b))
        gate_summary["bootstrap"] = boot

        results[f"gate_{gate_name}"] = gate_summary
        print(f"[{gate_name}] 전체={gate_summary['overall_mean']:.4f}±{gate_summary['overall_std']:.4f} | "
              f"{name_a}={gate_summary[f'{name_a}_mean']:.4f}±{gate_summary[f'{name_a}_std']:.4f} "
              f"(Δvs late={gate_summary[f'delta_{name_a}_vs_late_fusion']:+.4f}, "
              f"boot95%CI={[round(x,4) for x in boot[name_a]['ci95']]}, P(Δ>0)={boot[name_a]['P_gt_0']:.2f}, "
              f"pairs={gate_summary[f'{name_a}_pairs']}) | "
              f"{name_b}={gate_summary[f'{name_b}_mean']:.4f}±{gate_summary[f'{name_b}_std']:.4f} "
              f"(Δvs late={gate_summary[f'delta_{name_b}_vs_late_fusion']:+.4f}, "
              f"boot95%CI={[round(x,4) for x in boot[name_b]['ci95']]}, P(Δ>0)={boot[name_b]['P_gt_0']:.2f}, "
              f"pairs={gate_summary[f'{name_b}_pairs']})")

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="os,pfs")
    ap.add_argument("--gates", default="m0,prior")
    ap.add_argument("--subgroup", choices=("stage", "high_risk"), default="stage")
    ap.add_argument("--report_encoder", choices=("tfidf", "radbert"), default="tfidf")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    gates = [g.strip() for g in args.gates.split(",") if g.strip()]
    all_results = {}
    for target in [t.strip() for t in args.targets.split(",") if t.strip()]:
        all_results[target] = run_target(target, gates, args.subgroup, report_encoder=args.report_encoder)

    out_path = os.path.join(OUT_DIR, f"results_{args.subgroup}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, ensure_ascii=False, indent=2)
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
