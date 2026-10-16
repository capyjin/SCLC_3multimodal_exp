# -*- coding: utf-8 -*-
"""⑤ GMU 위험점수의 선형 부분 분석. 학습 환자에서 r_D ≈ a·z_tab + b·z_img + c 를 회귀로 적합하고,
시험 환자에서 선형 부분만 남긴 점수의 C-index 를 잰다 (재학습 없음).

Run:  python experiments/실험16_GMU_nestedCV/analyze_gmu_linear_part.py"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hashlib
import json

import numpy as np
import pandas as pd
import torch

from nested_fusion.config import ARMS, CANDIDATES, MODALITIES, RUN_TAG, SPLIT_DIR, guard
from nested_fusion.dataset import feats, load_labels, load_manifest, load_outer_split
from nested_fusion.model import build
from nested_fusion.processing import CheckpointRebuilder, standardize, standardizer
from sclc import cohort, features, paths
from sclc import foldsafe_clinical as fsc
from sclc.encoders import build_encoder
from sclc.evaluation import cindex

OUT = paths.outputs("gmu_nested", "linear_projection_train_only")
TOL_SCORE = 1e-4    # 재구성 점수와 저장값의 허용 오차


def file_sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fit_linear(z_train: np.ndarray, r_train: np.ndarray) -> np.ndarray:
    """OLS 계수 [a, b, c] (학습 환자 점수만 사용, 레이블 미사용)."""
    X = np.column_stack([z_train, np.ones(len(z_train))])
    coef, *_ = np.linalg.lstsq(X, r_train, rcond=None)
    return coef


def apply_linear(coef: np.ndarray, z: np.ndarray) -> np.ndarray:
    return z @ coef[:2] + coef[2]


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))


def run_outer(rb: CheckpointRebuilder, run_dir: str, target: str, k: int, ctx: dict) -> tuple[dict, dict, pd.DataFrame]:
    """outer k: 체크포인트 재구성 → GMU 위험점수 → 선형 적합 → C-index."""
    refit_csv = os.path.join(SPLIT_DIR, f"outer{k}_refit.csv")
    t_ids, o_ids = map(list, load_outer_split(refit_csv))
    om = next(o for o in ctx["manifest"]["outer"] if o["k"] == k)
    v = {"T_O_disjoint": not (set(t_ids) & set(o_ids)), "T_plus_O_is_238": len(set(t_ids) | set(o_ids)) == 238,
         "O_matches_manifest": sorted(o_ids) == om["outer_test_ids"]}
    guard(v["T_O_disjoint"] and v["T_plus_O_is_238"] and v["O_matches_manifest"], f"outer {k}: 분할 이상 {v}")

    ck = os.path.join(run_dir, "ckpt", f"outer{k}")
    tab_store = rb.tabular(refit_csv, os.path.join(ck, "tabular_refit"), None)[k]
    img_store = rb.image(refit_csv, os.path.join(ck, "image_refit"), None)[k]
    RT, RI = tab_store["splits"], img_store["splits"]
    for name, st in (("tabular", tab_store), ("image", img_store)):
        guard(sorted(st["ids"]["train"]) == sorted(t_ids) and sorted(st["ids"]["test"]) == sorted(o_ids),
              f"outer {k} {name}: 재구성 분할이 T_k/O_k 와 다르다")
        guard(list(st["ids"]["train"]) == t_ids and list(st["ids"]["test"]) == o_ids,
              f"outer {k} {name}: 재구성 환자 순서가 refit plan 과 다르다")
    man = ctx["enc_manifest"][str(k)]
    v["tab_state_hash_matches_manifest"] = tab_store["hash"] == man["tabular"]["refit"][str(k)]["state_sha256"]
    v["img_state_hash_matches_manifest"] = img_store["hash"] == man["image"]["refit"][str(k)]["state_sha256"]
    guard(v["tab_state_hash_matches_manifest"] and v["img_state_hash_matches_manifest"],
          f"outer {k}: 인코더 가중치가 본 실행 기록과 다르다")

    dev = rb.dev
    cfg_name = ctx["selection"][str(k)]["D_gmu"]["selected"]
    d_path = os.path.join(run_dir, "fusion_state", f"outer{k}", "D_gmu.pt")
    model = build("D_gmu", CANDIDATES[cfg_name]).to(dev)
    model.load_state_dict(torch.load(d_path, map_location=dev), strict=True)
    model.eval()
    t_feats, o_feats = feats(RT["train"], RI["train"], dev), feats(RT["test"], RI["test"], dev)
    with torch.no_grad():
        rD_t = model(t_feats).squeeze(1).double().cpu().numpy()
        rD_o = model(o_feats).squeeze(1).double().cpu().numpy()
    saved_stats = ctx["gate_analysis"][str(k)]["outer_selected"]["D_gmu"]
    now_stats = {"T_k": model.feature_stats(t_feats), "O_k": model.feature_stats(o_feats)}
    gate_diff = max(abs(now_stats[s]["gates"][m][q] - saved_stats[s]["gates"][m][q])
                    for s in ("T_k", "O_k") for m in MODALITIES for q in ("mean", "sd", "q05", "q50", "q95"))
    v["T_O_gate_stats_vs_saved_maxdiff"] = gate_diff
    guard(gate_diff < 1e-3, f"outer {k}: D gate 분포(T_k/O_k)가 저장값과 다르다 ({gate_diff})")

    so = ctx["saved_oof"].loc[o_ids]
    raw = {"tabular": (RT["train"]["risk"].double().numpy(), RT["test"]["risk"].double().numpy()),
           "image": (RI["train"]["risk"].double().numpy(), RI["test"]["risk"].double().numpy())}
    v["O_tab_vs_saved"] = float(np.abs(raw["tabular"][1] - so.risk_tabular_only).max())
    v["O_img_vs_saved"] = float(np.abs(raw["image"][1] - so.risk_image_only).max())
    v["O_D_vs_saved"] = float(np.abs(rD_o - so.risk_D_gmu).max())
    guard(max(v["O_tab_vs_saved"], v["O_img_vs_saved"], v["O_D_vs_saved"]) < TOL_SCORE,
          f"outer {k}: 재구성 O_k 점수가 저장값과 다르다 {v}")

    stats = {arm: standardizer(raw[arm][0]) for arm in ARMS}
    z_t = np.column_stack([standardize(raw[arm][0], stats[arm]) for arm in ARMS])
    z_o = np.column_stack([standardize(raw[arm][1], stats[arm]) for arm in ARMS])
    fit_ids = list(t_ids)
    coef = fit_linear(z_t, rD_t)
    v["fit_ids_equal_T"] = sorted(fit_ids) == sorted(t_ids) and len(fit_ids) == len(z_t) == len(rD_t)
    v["fit_ids_disjoint_O"] = not (set(fit_ids) & set(o_ids))
    guard(v["fit_ids_equal_T"] and v["fit_ids_disjoint_O"], f"outer {k}: 회귀 적합 행 이상")
    lin_t, lin_o = apply_linear(coef, z_t), apply_linear(coef, z_o)
    res_t, res_o = rD_t - lin_t, rD_o - lin_o

    lab = ctx["labels"]
    o_lab, t_lab = lab.loc[o_ids], lab.loc[t_ids]
    ci_o = lambda r: cindex(o_lab[f"{target}_days"].to_numpy(), r, o_lab[f"{target}_event"].to_numpy())
    ci_t = lambda r: cindex(t_lab[f"{target}_days"].to_numpy(), r, t_lab[f"{target}_event"].to_numpy())
    saved_fm = ctx["fold_metrics"]
    saved_ci = lambda m: float(saved_fm[(saved_fm.outer == k) & (saved_fm.model == m)].cindex.iloc[0])
    rec = {"outer": k, "n_train": len(t_ids), "n_test": len(o_ids),
           "events_train": int(t_lab[f"{target}_event"].sum()), "events_test": int(o_lab[f"{target}_event"].sum()),
           "D_selected": cfg_name,
           "A": ci_o(so.risk_A_primary.to_numpy()), "D": ci_o(rD_o), "linear": ci_o(lin_o),
           "residual": ci_o(res_o), "tabular_only": ci_o(raw["tabular"][1]), "image_only": ci_o(raw["image"][1]),
           "a_tab": float(coef[0]), "b_img": float(coef[1]), "c": float(coef[2]),
           "image_share_abs": float(abs(coef[1]) / (abs(coef[0]) + abs(coef[1]))),
           "A_coef_tab": ctx["saved_A"][str(k)]["coef_z"]["z_tabular"],
           "A_coef_img": ctx["saved_A"][str(k)]["coef_z"]["z_image"],
           "R2_train": r2(rD_t, lin_t), "R2_test": r2(rD_o, lin_o),
           "corr_train": float(np.corrcoef(rD_t, lin_t)[0, 1]), "corr_test": float(np.corrcoef(rD_o, lin_o)[0, 1]),
           "sd_rD_train": float(rD_t.std(ddof=1)), "sd_rD_test": float(rD_o.std(ddof=1)),
           "sd_resid_train": float(res_t.std(ddof=1)), "sd_resid_test": float(res_o.std(ddof=1)),
           "train_insample_D": ci_t(rD_t), "train_insample_linear": ci_t(lin_t), "train_insample_residual": ci_t(res_t)}
    rec["A_cindex_vs_saved"] = abs(rec["A"] - saved_ci("A_primary"))
    rec["D_cindex_vs_saved"] = abs(rec["D"] - saved_ci("D_gmu"))
    v["A_cindex_vs_saved"], v["D_cindex_vs_saved"] = rec["A_cindex_vs_saved"], rec["D_cindex_vs_saved"]
    guard(max(rec["A_cindex_vs_saved"], rec["D_cindex_vs_saved"]) < 1e-9, f"outer {k}: A/D C-index 재현 실패 {v}")
    v["checkpoints"] = {"D_gmu": file_sha(d_path),
                        "tabular_refit": file_sha(os.path.join(ck, "tabular_refit", f"fold{k}_early_fusion_{target}.pt")),
                        "image_refit": file_sha(os.path.join(ck, "image_refit", f"fold{k}_image_simplecnn_{target}.pt")),
                        "same_object_for_T_and_O": True}
    v["standardizer_T"] = {arm: list(stats[arm]) for arm in ARMS}

    rows = pd.concat([pd.DataFrame({"research_id": ids, "outer": k, "split": split,
                                    "z_tabular": z[:, 0], "z_image": z[:, 1], "risk_D": r,
                                    "linear_part": lp, "residual": rs})
                      for split, ids, z, r, lp, rs in (("train", t_ids, z_t, rD_t, lin_t, res_t),
                                                       ("test", o_ids, z_o, rD_o, lin_o, res_o))],
                     ignore_index=True)
    print(f"[linear] {target} outer {k}: A {rec['A']:.4f} | D {rec['D']:.4f} | linear {rec['linear']:.4f} | "
          f"resid {rec['residual']:.4f} | a {coef[0]:.3f} b {coef[1]:.3f} | R2 tr {rec['R2_train']:.3f} "
          f"te {rec['R2_test']:.3f}")
    return rec, v, rows


def run_target(target: str, ctx_base: dict):
    run_dir = paths.outputs("gmu_nested", f"{RUN_TAG}_{target}")
    load = lambda f: json.load(open(os.path.join(run_dir, f), encoding="utf-8"))
    ctx = {**ctx_base,
           "saved_oof": pd.read_csv(os.path.join(run_dir, "oof_predictions.csv")).set_index("research_id"),
           "fold_metrics": pd.read_csv(os.path.join(run_dir, "fold_metrics.csv")),
           "saved_A": load("late_fusion_A.json"), "selection": load("selection.json"),
           "enc_manifest": load("encoder_manifest.json"), "gate_analysis": load("gate_analysis.json"),
           "labels": load_labels(target)}
    rb = CheckpointRebuilder(target, ctx_base)
    recs, verif, rows = [], {}, []
    for k in range(1, 6):
        rec, v, r = run_outer(rb, run_dir, target, k, ctx)
        recs.append(rec)
        verif[k] = v
        rows.append(r)
    scores = pd.concat(rows, ignore_index=True)
    guard(scores[scores.split == "test"].research_id.nunique() == 238, f"{target}: O_k 합이 238명이 아니다")
    return pd.DataFrame(recs), verif, scores


def summarize(df: pd.DataFrame) -> dict:
    s = {}
    for col in ("A", "D", "linear", "residual", "tabular_only", "image_only", "R2_train", "R2_test",
                "corr_train", "corr_test", "a_tab", "b_img", "image_share_abs",
                "train_insample_D", "train_insample_linear", "train_insample_residual"):
        s[col] = {"mean": float(df[col].mean()), "sd": float(df[col].std(ddof=1))}
    for ref in ("A", "D"):
        d = df["linear"] - df[ref]
        s[f"linear_minus_{ref}"] = {"mean": float(d.mean()), "sd": float(d.std(ddof=1)),
                                    "by_fold": [round(float(x), 4) for x in d], "n_positive": int((d > 0).sum())}
    s["R2_train_minus_test"] = {"mean": float((df.R2_train - df.R2_test).mean()),
                                "by_fold": [round(float(x), 4) for x in df.R2_train - df.R2_test]}
    return s


def main():
    if os.path.exists(os.path.join(OUT, "results.json")):
        raise SystemExit(f"{OUT} 에 이미 결과가 있다 — 덮어쓰지 않는다.")
    os.makedirs(OUT, exist_ok=True)
    corpus, _ = features.load_text_corpus(cohort.DEFAULT_MERGED_CSV)
    ctx = {"raw": fsc.load_raw_values(), "corpus": corpus,
           "text_fn": build_encoder("radbert").evaluator_kwargs(corpus)["text_encoder_fn"],
           "manifest": load_manifest()}
    out = {"status": "사후 탐색적 실험 — 본 결과(A/B/C/D) 확인 뒤 2026-10-15 추가. 사전 지정 비교 아님",
           "design": {"model": "r_D = a*z_tab + b*z_img + c + residual (OLS)",
                      "fit_rows": "outer train T_k 만 (생존 레이블 미사용)",
                      "z": "outer 재학습 인코더 위험점수를 T_k mean/sd(ddof=1)로 표준화 (A 와 같은 규칙)",
                      "test": "O_k 에 고정 계수 적용 -> 선형 부분, 잔차 = r_D - 선형 부분",
                      "reused": "run_v2 refit 인코더 체크포인트 + fusion_state D_gmu.pt (학습 없음), A 는 저장된 risk_A_primary",
                      "note": "T_k 점수는 인코더·D 의 학습 환자 위 in-sample 값. 기존 0.7291/0.6765 (O_k 적합 진단)은 재현 대상 아님"},
           "targets": {}}
    for target in ("os", "pfs"):
        df, verif, scores = run_target(target, ctx)
        df.to_csv(os.path.join(OUT, f"fold_metrics_{target}.csv"), index=False)
        scores.to_csv(os.path.join(OUT, f"scores_{target}.csv"), index=False)
        out["targets"][target] = {"summary": summarize(df), "folds": df.to_dict(orient="records"),
                                  "verification": verif}
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
    print(f"[analyze_gmu_linear_part] wrote {OUT}")


if __name__ == "__main__":
    main()
