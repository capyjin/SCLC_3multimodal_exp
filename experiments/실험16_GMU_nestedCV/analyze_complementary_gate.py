# -*- coding: utf-8 -*-
"""⑧ Complementary gate 비교 (사후 탐색, 재실행 기준).

지도교수 질문(2026-11-03): 일반적인 gating r = g·z_img + (1−g)·z_tab 도 실험해 봤는가.
context(임상+판독지)·image 인코더의 위험점수는 재실행 체크포인트(outputs/gmu_nested/arms_rerun_{os,pfs})로
재구성만 하고(재학습 없음), 결합 단계만 바꿔 같은 입력·같은 fold 에서 비교한다.

    Late fusion      r = b_tab·z_tab + b_img·z_img                       nested_late_fusion (CoxPH)
    Hybrid fusion    r = Σ_m b_m·2σ(c_mᵀz)·z_m                           GatedLateFusion
    Complementary    r = g·z_img + (1−g)·z_tab,  g = σ(cᵀz)              ComplementaryGateFusion
    (참고) 단순 평균 r = 0.5·z_img + 0.5·z_tab                            Complementary 의 c = 0

두 gate 결합기는 같은 손실(Efron + λ‖c‖²)·λ 후보·inner 4-fold 선택 규칙(동점이면 큰 λ)을 쓴다.
수치는 재실행 체크포인트 기준이라 논문 Table 3–5(run_v2)와 다르다 — Figure 2·Table 7(km_nested)과 같은 점수다.

재구성 검증 (outer 마다. 구조가 어긋나면 저장하지 않고 멈춘다)
  정렬    inner Fit/S/V · refit 학습/시험 환자가 학습 당시 manifest 와 같다, 두 arm 의 환자 순서가 같다,
          inner 분할이 T_k 를 정확히 나눈다, V 를 모으면 T_k, T_k ∩ O_k = ∅, 저장 점수의 레이블이 코호트와 같다
  가중치  불러온 모델의 state SHA-256 = manifest
  전처리  영상 정규화 값, fold-safe 결측 대치 기록(학습 환자 중앙값·대치 수·환자 수) = manifest
  표준화  inner OOF z = 자기 inner 모델 Fit 환자 점수의 평균·SD, 시험 z = refit 모델 T_k 점수의 평균·SD,
          세 결합기가 같은 Z 를 받는다
  재현    재구성 시험 점수 vs 저장 점수(SD 대비 차이·순위 뒤바뀜), arm C-index vs 저장 값,
          Late·Hybrid 의 C-index·λ·점수 vs km_nested
  재실행 체크포인트는 GPU(합성곱 TF32 기본값)에서 점수를 저장했고 여기서는 CPU(FP32)로 재구성하므로
  영상 점수에 SD 의 0.1% 수준 차이가 난다 (outer 1 OS: 최대 8.1e-4, 순위 뒤바뀜 0, C-index 같음).

Run:  python experiments/실험16_GMU_nestedCV/analyze_complementary_gate.py
      경로 점검만: ... --targets os --outers 1 --out_dir <새 폴더>"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from nested_fusion.config import ARMS, SPLIT_DIR, guard
from nested_fusion.dataset import load_labels, load_outer_split
from nested_fusion.model import F64, ComplementaryGateFusion, GatedLateFusion, nested_late_fusion
from nested_fusion.processing import CheckpointRebuilder, arm_records, pool_inner_oof, standardize, standardizer
from sclc import cohort, features, paths
from sclc import foldsafe_clinical as fsc
from sclc.encoders import build_encoder
from sclc.evaluation import cindex

OUT = paths.outputs("gmu_nested", "complementary_gate")
KM_DIR = paths.outputs("gmu_nested", "km_nested")
ARM_DIRS = {"tabular": "clin_report", "image": "image_simplecnn"}     # 결과 키 → 재실행 폴더 이름
BASIS = ("재실행 기준: outputs/gmu_nested/arms_rerun_{os,pfs} 체크포인트로 재구성한 위험점수. 논문 Table 3–5 의 "
         "run_v2 수치와 다르며 Figure 2·Table 7(km_nested)과 같은 점수다.")
REL_TOL = 1e-2           # 재구성 시험 점수 vs 저장 점수: 최대 차이 / 저장 점수 SD (CPU FP32 vs GPU TF32 차이 허용)
METHODS = (("late", "Late fusion"), ("hybrid", "Hybrid fusion"),
           ("comp", "Complementary gate"), ("avg", "Equal average (c = 0)"))
DELTAS = (("comp", "late"), ("comp", "hybrid"), ("hybrid", "late"))


def rerun_dir(target: str) -> str:
    return paths.outputs("gmu_nested", f"arms_rerun_{target}")


def _json(*parts):
    with open(os.path.join(*parts), encoding="utf-8") as fh:
        return json.load(fh)


def _ints(xs) -> list[int]:
    return [int(x) for x in xs]


# ── 재구성 ──
def rebuild_outer(rb: CheckpointRebuilder, target: str, k: int):
    """outer k 의 inner·refit 체크포인트로 두 arm 의 위험점수를 다시 계산한다 (재학습 없음)."""
    inner_csv = os.path.join(SPLIT_DIR, f"outer{k}_inner.csv")
    refit_csv = os.path.join(SPLIT_DIR, f"outer{k}_refit.csv")
    t_ids, o_ids = load_outer_split(refit_csv)
    ck = lambda arm, stage: os.path.join(rerun_dir(target), "ckpt", ARM_DIRS[arm], f"outer{k}", stage)
    audit = {"inner": [], "refit": []}
    stores = {"tabular": {"inner": rb.tabular(inner_csv, ck("tabular", "inner"), o_ids, audit=audit["inner"]),
                          "refit": rb.tabular(refit_csv, ck("tabular", "refit"), None, audit=audit["refit"])},
              "image": {"inner": rb.image(inner_csv, ck("image", "inner"), o_ids),
                        "refit": rb.image(refit_csv, ck("image", "refit"), None)}}
    inner, outer = arm_records(stores["tabular"]["inner"], stores["image"]["inner"],
                               stores["tabular"]["refit"][k]["splits"], stores["image"]["refit"][k]["splits"], o_ids)
    return t_ids, o_ids, stores, audit, inner, outer


# ── 재구성 검증 ──
def _same_norm(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return all(np.allclose(a[key], b[key], rtol=0, atol=1e-8) for key in ("mean", "std"))


def _audit_key(entry) -> str:
    return json.dumps(entry, sort_keys=True)


def verify_rebuild(target, k, t_ids, o_ids, stores, audit, inner, outer, labels) -> dict:
    """재구성 점수가 학습 당시와 같은 환자·가중치·전처리·표준화 기준인지 확인한다."""
    t_set, o_set = set(_ints(t_ids)), set(_ints(o_ids))
    guard(not (t_set & o_set) and len(t_set) + len(o_set) == len(labels), f"outer {k}: T_k/O_k 분할 이상")
    rec = {"n_train": len(t_set), "n_test": len(o_set)}

    # (1) 학습 당시 manifest 와 환자 분할·가중치·영상 정규화 대조
    stage_keys = {"inner": {"train": "train_ids", "val": "stop_ids", "test": "eval_ids"},
                  "refit": {"train": "train_ids", "test": "eval_ids"}}
    for arm, name in ARM_DIRS.items():
        d = os.path.join(rerun_dir(target), name, f"outer{k}")
        man = _json(d, "encoder_manifest.json")
        arm_rec = {"ids_same_order": True, "n_models": 0}
        for stage, key_map in stage_keys.items():
            store = stores[arm][stage]
            entries = [man[stage][key] for key in sorted(man[stage], key=int)]
            guard(len(entries) == len(store), f"{name} outer {k} {stage}: 모델 수 {len(store)} != manifest {len(entries)}")
            for fold, m in zip(sorted(store), entries):
                ids = store[fold]["ids"]
                for split, mk in key_map.items():
                    guard(set(_ints(ids[split])) == set(_ints(m[mk])),
                          f"{name} outer {k} {stage} fold {fold}: {split} 환자 집합이 manifest 와 다르다")
                    arm_rec["ids_same_order"] &= _ints(ids[split]) == _ints(m[mk])
                guard(store[fold]["hash"] == m["state_sha256"], f"{name} outer {k} {stage} fold {fold}: 가중치 해시 불일치")
                guard(_same_norm(store[fold]["image_norm"], m.get("image_norm")),
                      f"{name} outer {k} {stage} fold {fold}: 영상 정규화 값 불일치")
                arm_rec["n_models"] += 1
        arm_rec["weights_sha256_match"] = True
        arm_rec["image_norm_match"] = True

        # (2) fold-safe 결측 대치 기록 (tabular arm 만 해당)
        if arm == "tabular":
            pm = _json(d, "preprocessing_manifest.json")
            for stage in ("inner", "refit"):
                ref = pm[f"foldsafe_audit_{stage}"]
                guard(sorted(map(_audit_key, audit[stage])) == sorted(map(_audit_key, ref)),
                      f"{name} outer {k} {stage}: 결측 대치 기록(학습 환자 중앙값 등)이 manifest 와 다르다")
                arm_rec[f"imputation_{stage}_match"] = True
                arm_rec[f"imputation_{stage}_same_order"] = [_audit_key(a) for a in audit[stage]] == list(map(_audit_key, ref))
        rec[arm] = arm_rec

    # (3) 두 arm 의 환자 순서 일치, inner 분할이 T_k 를 정확히 나누는지
    for j in sorted(stores["tabular"]["inner"]):
        tab_ids, img_ids = stores["tabular"]["inner"][j]["ids"], stores["image"]["inner"][j]["ids"]
        for split in ("train", "val", "test"):
            guard(_ints(tab_ids[split]) == _ints(img_ids[split]), f"outer {k} inner {j}: 두 arm 의 {split} 순서가 다르다")
        parts = [set(_ints(tab_ids[s])) for s in ("train", "val", "test")]
        guard(set().union(*parts) == t_set and sum(map(len, parts)) == len(t_set),
              f"outer {k} inner {j}: Fit/S/V 가 T_k 를 정확히 나누지 않는다")
    for split in ("train", "test"):
        guard(_ints(stores["tabular"]["refit"][k]["ids"][split]) == _ints(stores["image"]["refit"][k]["ids"][split]),
              f"outer {k} refit: 두 arm 의 {split} 순서가 다르다")
    guard(_ints(stores["tabular"]["refit"][k]["ids"]["test"]) == _ints(o_ids), f"outer {k}: refit 시험 환자 순서 != O_k")
    guard(set(_ints(stores["tabular"]["refit"][k]["ids"]["train"])) == t_set, f"outer {k}: refit 학습 환자 != T_k")
    oof = pool_inner_oof(inner, t_ids)          # V 를 모으면 T_k 와 정확히 같아야 한다 (어긋나면 ValueError)
    rec["oof_covers_T_k_once"] = True

    # (4) 표준화 기준: inner OOF z 는 자기 inner 모델의 Fit 환자, 시험 z 는 refit 모델의 T_k 환자 점수로
    std = {}
    for arm in ARMS:
        for r in inner:
            j = r["j"]
            a = r["arms"][arm]
            n_fit = len(stores[arm]["inner"][j]["ids"]["train"])
            guard(len(a["fit_risk"]) == n_fit and len(a["val_risk"]) == len(r["val_ids"]),
                  f"outer {k} inner {j} {arm}: 표준화에 쓴 점수 수가 Fit/V 환자 수와 다르다")
            mu, sd = standardizer(a["fit_risk"])
            std[f"{arm}_inner{j}"] = {"basis": "inner Fit", "n": n_fit, "mean": mu, "sd": sd}
        o = outer["arms"][arm]
        guard(len(o["train_risk"]) == len(t_set) and len(o["test_risk"]) == len(o_set),
              f"outer {k} {arm}: 시험 z 의 표준화 기준이 T_k 가 아니다")
        mu, sd = standardizer(o["train_risk"])
        std[f"{arm}_outer"] = {"basis": "refit T_k", "n": len(t_set), "mean": mu, "sd": sd}
    rec["standardization"] = std

    # (5) 저장된 시험 점수·레이블·C-index 와 대조
    o_lab = labels.loc[_ints(o_ids)]
    tte, ete = o_lab[f"{target}_days"].to_numpy(float), o_lab[f"{target}_event"].to_numpy(float)
    for arm, name in ARM_DIRS.items():
        d = os.path.join(rerun_dir(target), name, f"outer{k}")
        saved = pd.read_csv(os.path.join(d, "oof_predictions.csv")).set_index("research_id")
        guard(set(saved.index) == o_set, f"{name} outer {k}: 저장 점수의 환자 != O_k")
        s = saved.loc[_ints(o_ids)]
        guard(np.array_equal(s["duration"].to_numpy(float), tte) and np.array_equal(s["event"].to_numpy(float), ete),
              f"{name} outer {k}: 저장 점수의 레이블이 코호트와 다르다")
        risk = np.asarray(outer["arms"][arm]["test_risk"], dtype=float)
        ref = s[f"risk_{name}"].to_numpy(float)
        diff = float(np.abs(risk - ref).max())
        rel = diff / float(ref.std())
        guard(rel < REL_TOL, f"{name} outer {k}: 재구성 시험 점수가 저장 점수와 SD 의 {rel:.1%} 다르다")
        flips = int((np.sign(risk[:, None] - risk[None, :]) * np.sign(ref[:, None] - ref[None, :]) < 0).sum() // 2)
        rec[arm].update({"saved_order_equal": _ints(saved.index) == _ints(o_ids), "test_risk_max_abs_diff": diff,
                         "test_risk_max_diff_over_sd": rel, "test_risk_order_flips": flips,
                         "cindex_rebuilt": cindex(tte, risk, ete),
                         "cindex_saved": float(_json(d, "fold_metrics.json")["cindex"])})
    return rec, oof


# ── 결합기 비교 ──
def fit_combiners(target, k, inner, outer, oof, t_ids, o_ids, labels) -> tuple[dict, pd.DataFrame]:
    """같은 inner OOF z(학습)·시험 z(평가)로 Late·Hybrid·Complementary·단순 평균을 적합·평가한다."""
    a = nested_late_fusion(inner, outer, t_ids, labels.loc[t_ids], target)
    guard(a["oof"].equals(oof), f"outer {k}: Late fusion 이 받은 OOF 가 다르다")
    t_set, o_set = set(_ints(t_ids)), set(_ints(o_ids))
    guard(set(oof.index) == t_set and not (set(oof.index) & o_set), f"outer {k}: 결합기 학습 환자 != T_k")

    lab = labels.loc[oof.index]
    Ztr = torch.tensor(oof[[f"z_{arm}" for arm in ARMS]].to_numpy(float), dtype=F64)
    ttr = torch.tensor(lab[f"{target}_days"].to_numpy(float), dtype=F64)
    etr = torch.tensor(lab[f"{target}_event"].to_numpy(float), dtype=F64)
    folds = oof["inner_fold"].to_numpy()
    zte = {arm: standardize(outer["arms"][arm]["test_risk"], standardizer(outer["arms"][arm]["train_risk"]))
           for arm in ARMS}
    Zte = torch.tensor(np.stack([zte[arm] for arm in ARMS], axis=1), dtype=F64)
    lin = sum(a["coef_z"][f"z_{arm}"] * zte[arm] for arm in ARMS)
    guard(np.allclose(lin, a["pred"]["primary"], rtol=0, atol=1e-12), f"outer {k}: Late fusion 시험 z 가 다르다")

    o_lab = labels.loc[_ints(o_ids)]
    tte, ete = o_lab[f"{target}_days"].to_numpy(float), o_lab[f"{target}_event"].to_numpy(float)

    lam_h, sc_h = GatedLateFusion.select_lambda(Ztr, ttr, etr, folds)
    hyb = GatedLateFusion(lam_h).fit(Ztr, ttr, etr)
    r_h, g_h = hyb.predict(Zte)
    lam_c, sc_c = ComplementaryGateFusion.select_lambda(Ztr, ttr, etr, folds)
    comp = ComplementaryGateFusion(lam_c).fit(Ztr, ttr, etr)
    r_c, w_c = comp.predict(Zte)
    r_avg = 0.5 * (Zte[:, 0] + Zte[:, 1]).numpy()

    risks = {"late": np.asarray(a["pred"]["primary"], float), "hybrid": r_h.numpy(), "comp": r_c.numpy(), "avg": r_avg}
    g_img = w_c[:, 1].numpy()
    gs_h = GatedLateFusion.gate_summary(g_h)
    rec = {"outer": k, "n_test": len(o_set), **{m: cindex(tte, risks[m], ete) for m, _ in METHODS},
           "late_coef": a["coef_z"],
           "hybrid_lambda": lam_h, "hybrid_inner_scores": sc_h, "hybrid_beta": hyb.beta.tolist(), "hybrid_C": hyb.C.tolist(),
           "hybrid_gate": gs_h, "hybrid_gate_off": bool(max(max(abs(gs_h[m]["max"] - 1), abs(gs_h[m]["min"] - 1))
                                                             for m in ARMS) < 0.01),
           "comp_lambda": lam_c, "comp_inner_scores": sc_c, "comp_c": comp.C.tolist(),
           "comp_g_img": {"mean": float(g_img.mean()), "sd": float(g_img.std(ddof=1)),
                          "min": float(g_img.min()), "max": float(g_img.max())},
           "comp_near_equal_average": bool(np.abs(g_img - 0.5).max() < 0.01)}
    scores = pd.DataFrame({"research_id": _ints(o_ids), "outer": k, "days": tte, "event": ete.astype(int),
                           "z_tabular": zte["tabular"], "z_image": zte["image"],
                           **{f"risk_{m}": risks[m] for m, _ in METHODS},
                           "hybrid_gate_tabular": g_h[:, 0].numpy(), "hybrid_gate_image": g_h[:, 1].numpy(),
                           "comp_g_image": g_img})
    return rec, scores


def compare_km_nested(target, k, rec, scores) -> dict:
    """Late·Hybrid 를 같은 재실행 점수로 만든 km_nested(Figure 2·Table 7)와 대조한다."""
    km = next(f for f in _json(KM_DIR, "folds.json")[target]["folds"] if f["outer"] == k)
    km_scores = pd.read_csv(os.path.join(KM_DIR, f"scores_{target}.csv")).set_index("research_id")
    s = km_scores.loc[scores["research_id"]]
    guard((s["fold"] == k).all(), f"outer {k}: km_nested 의 fold 배정이 다르다")
    return {"late_cindex_diff": abs(rec["late"] - km["cindex_A"]),
            "hybrid_cindex_diff": abs(rec["hybrid"] - km["cindex_E1"]),
            "hybrid_lambda_equal": float(rec["hybrid_lambda"]) == float(km["lambda"]),
            "hybrid_risk_max_abs_diff": float(np.abs(scores["risk_hybrid"].to_numpy() - s["risk"].to_numpy()).max())}


# ── 요약 ──
def summarize(rows: list[dict]) -> dict:
    df = pd.DataFrame(rows)
    out = {m: {"mean": float(df[m].mean()), "sd": float(df[m].std(ddof=1)) if len(df) > 1 else None,
               "folds": df[m].tolist()} for m, _ in METHODS}
    for x, y in DELTAS:
        d = df[x] - df[y]
        out[f"delta_{x}_minus_{y}"] = {"mean": float(d.mean()), "n_positive": int((d > 0).sum()),
                                       "n_zero": int((d == 0).sum()), "folds": d.tolist()}
    return out


def write_summary(path, results):
    lines = ["# ⑧ Complementary gate 비교 — 재실행 기준 (사후 탐색)", "",
             f"> {BASIS}", "> 지도교수 질문(2026-11-03)에 따라 추가한 사후 비교다. 사전 지정 비교가 아니며 검정은 하지 않았다.", "",
             "결합식: Complementary r = g·z_img + (1−g)·z_tab, g = σ(cᵀz) (bias·결합계수 없음). "
             "두 gate 결합기 모두 Efron + λ‖c‖², λ ∈ {0.01, 0.1, 1, 10, 100} 를 inner 4-fold 로 선택(동점이면 큰 λ).", "",
             "## outer test C-index (5 fold 평균 ± 표본 SD)", "",
             "| | " + " | ".join(lbl for _, lbl in METHODS) + " |", "|---|" + "---|" * len(METHODS)]
    for t, r in results.items():
        s = r["summary"]
        lines.append(f"| {t.upper()} | " + " | ".join(
            f"{s[m]['mean']:.3f}" + (f" ± {s[m]['sd']:.3f}" if s[m]["sd"] is not None else "") for m, _ in METHODS) + " |")
    lines += ["", "## fold 별 차이 (평균, 양수 fold 수 / 전체)", "", "| | Complementary − Late | Complementary − Hybrid | Hybrid − Late |",
              "|---|---|---|---|"]
    for t, r in results.items():
        s, n = r["summary"], len(r["folds"])
        lines.append(f"| {t.upper()} | " + " | ".join(
            f"{s[f'delta_{x}_minus_{y}']['mean']:+.4f} ({s[f'delta_{x}_minus_{y}']['n_positive']}/{n})" for x, y in DELTAS) + " |")
    lines += ["", "## fold 별 값", "",
              "| target | outer | Late | Hybrid | Complementary | 평균(c=0) | λ Hybrid | λ Comp | c (tab, img) | g_img 범위 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for t, r in results.items():
        for f in r["folds"]:
            g = f["comp_g_img"]
            lines.append(f"| {t.upper()} | {f['outer']} | {f['late']:.4f} | {f['hybrid']:.4f} | {f['comp']:.4f} | {f['avg']:.4f} | "
                         f"{f['hybrid_lambda']} | {f['comp_lambda']} | {f['comp_c'][0]:+.3f}, {f['comp_c'][1]:+.3f} | "
                         f"{g['min']:.2f}–{g['max']:.2f} |")
    lines += ["", "## 재구성 검증 (outer 마다 전부 통과해야 저장된다)", "",
              "| target | outer | 분할·순서 | 가중치 해시 | 전처리 | 시험 점수 최대 차이/SD (ctx / img) | 순위 뒤바뀜 (ctx / img) | "
              "arm C-index 차이 (ctx / img) | Late·Hybrid C-index 차이 vs km_nested | Hybrid λ 일치 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for t, r in results.items():
        for k, v in r["verification"].items():
            km = v["km_nested"]
            lines.append(f"| {t.upper()} | {k} | 일치 | {v['tabular']['n_models'] + v['image']['n_models']}개 일치 | 일치 | "
                         f"{v['tabular']['test_risk_max_diff_over_sd']:.1e} / {v['image']['test_risk_max_diff_over_sd']:.1e} | "
                         f"{v['tabular']['test_risk_order_flips']} / {v['image']['test_risk_order_flips']} | "
                         f"{abs(v['tabular']['cindex_rebuilt'] - v['tabular']['cindex_saved']):.1e} / "
                         f"{abs(v['image']['cindex_rebuilt'] - v['image']['cindex_saved']):.1e} | "
                         f"{km['late_cindex_diff']:.1e} / {km['hybrid_cindex_diff']:.1e} | {km['hybrid_lambda_equal']} |")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", default="os,pfs")
    ap.add_argument("--outers", default="1,2,3,4,5")
    ap.add_argument("--out_dir", default=OUT)
    args = ap.parse_args()
    targets, outers = args.targets.split(","), [int(x) for x in args.outers.split(",")]
    if os.path.exists(os.path.join(args.out_dir, "results.json")):
        raise SystemExit(f"{args.out_dir} 에 이미 결과가 있다 — 덮어쓰지 않는다.")
    os.makedirs(args.out_dir, exist_ok=True)

    corpus, _ = features.load_text_corpus(cohort.DEFAULT_MERGED_CSV)
    ctx = {"raw": fsc.load_raw_values(), "corpus": corpus,
           "text_fn": build_encoder("radbert").evaluator_kwargs(corpus)["text_encoder_fn"]}
    results = {}
    for target in targets:
        labels = load_labels(target)
        rb = CheckpointRebuilder(target, ctx)
        rows, verif, scores = [], {}, []
        for k in outers:
            t0 = time.time()
            t_ids, o_ids, stores, audit, inner, outer = rebuild_outer(rb, target, k)
            v, oof = verify_rebuild(target, k, t_ids, o_ids, stores, audit, inner, outer, labels)
            rec, sc = fit_combiners(target, k, inner, outer, oof, t_ids, o_ids, labels)
            v["km_nested"] = compare_km_nested(target, k, rec, sc)
            rows.append(rec)
            verif[k] = v
            scores.append(sc)
            print(f"[⑧ {target} outer {k}] Late {rec['late']:.4f} | Hybrid {rec['hybrid']:.4f} (λ {rec['hybrid_lambda']}) | "
                  f"Complementary {rec['comp']:.4f} (λ {rec['comp_lambda']}, g_img {rec['comp_g_img']['min']:.2f}–"
                  f"{rec['comp_g_img']['max']:.2f}) | 평균 {rec['avg']:.4f} | 검증 통과, km_nested 차이 "
                  f"{v['km_nested']['late_cindex_diff']:.1e}/{v['km_nested']['hybrid_cindex_diff']:.1e} | {time.time() - t0:.0f}s",
                  flush=True)
        results[target] = {"folds": rows, "summary": summarize(rows), "verification": verif}
        pd.concat(scores).to_csv(os.path.join(args.out_dir, f"scores_{target}.csv"), index=False)
        pd.DataFrame([{key: val for key, val in r.items() if not isinstance(val, (dict, list))} for r in rows]).to_csv(
            os.path.join(args.out_dir, f"fold_metrics_{target}.csv"), index=False)

    out = {"basis": BASIS,
           "status": "사후 탐색 — 지도교수 질문(2026-11-03)으로 추가. 사전 지정 비교 아님, 검정 없음",
           "design": {"complementary": "r = g*z_img + (1-g)*z_tab, g = sigmoid(c . [z_tab, z_img]), bias·결합계수 없음",
                      "hybrid": "r = sum_m b_m * 2*sigmoid(c_m . z) * z_m (GatedLateFusion)",
                      "late": "CoxPH(Efron) on inner OOF z (nested_late_fusion primary)",
                      "loss": "Efron (사건당) + lambda*||c||^2", "lambdas": list(GatedLateFusion.LAMBDAS),
                      "selection": "inner 4-fold V C-index 평균 최대, 동점이면 큰 lambda (Hybrid 와 같음)",
                      "inputs": "inner OOF z (inner Fit 기준 표준화) / 시험 z (refit T_k 기준 표준화), 세 결합기 공통"},
           "targets": results}
    with open(os.path.join(args.out_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
    write_summary(os.path.join(args.out_dir, "SUMMARY.md"), results)
    print(f"[analyze_complementary_gate] wrote {args.out_dir}")


if __name__ == "__main__":
    main()
