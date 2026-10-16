# -*- coding: utf-8 -*-
"""[실험16-6] run_nested_cv.py --smoke 산출물의 구조 점검 — A/B/C/D 학습·평가 경로가 맞게 연결됐는가.

성능을 보지 않는다: C-index 는 [0, 1] 의 유한값인지만 확인한다.

Run:  python experiments/실험16_GMU_nestedCV/diagnostics/check_smoke_outputs.py --target os
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))                 # diagnostics/
EXP_DIR = os.path.dirname(HERE)                                   # 실험16 폴더 (nested_fusion 패키지)
PROJECT_ROOT = os.path.dirname(os.path.dirname(EXP_DIR))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, HERE)

import argparse
import json
import subprocess

import numpy as np
import pandas as pd

from checklist import CHECKS, check, expected_saved

from nested_fusion.config import CANDIDATES
from nested_fusion.dataset import load_manifest
from nested_fusion.model import FUSION_MODELS
from nested_fusion.train import effective_candidates, refit_epochs, select_candidate
from sclc import paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=("os", "pfs"))
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    out = args.out_dir or paths.outputs("gmu_nested", f"pathcheck_{args.target}")
    man = load_manifest()
    om = next(o for o in man["outer"] if o["k"] == 1)
    o_ids = set(om["outer_test_ids"])
    load = lambda f: json.load(open(os.path.join(out, f), encoding="utf-8"))

    cfg = load("experiment_config.json")
    check("config: smoke flag, target, environment recorded",
          cfg["smoke"] and cfg["target"] == args.target and cfg["environment"]["torch"] is not None)

    inner = pd.read_csv(os.path.join(out, "inner_results.csv"))
    n_expect = 4 * sum(len(effective_candidates(m)) for m in FUSION_MODELS)
    check(f"inner: 4 folds x effective candidates (B5+C5+D6) = {n_expect} rows",
          len(inner) == n_expect and all(set(inner[inner.model == m].candidate) == set(effective_candidates(m))
                                         for m in FUSION_MODELS), len(inner))
    check("inner: V C-index finite in [0,1]", inner.cindex_V.between(0, 1).all())
    check("inner: best_epoch within [min_epoch, max_epochs] (smoke 1..3)",
          inner.best_epoch.between(1, 3).all(), sorted(inner.best_epoch.unique().tolist()))

    enc = load("encoder_manifest.json")["1"]
    for kind in ("tabular", "image"):
        ok = True
        for j, f in enumerate(om["inner"], start=1):
            e = enc[kind]["inner"][str(j)]
            ok &= (e["train_ids"] == f["ids"]["fit"] and e["stop_ids"] == f["ids"]["stop"]
                   and e["eval_ids"] == f["ids"]["val"] and not (set(e["train_ids"]) & o_ids))
        check(f"encoder/{kind}: inner Fit/S/V == locked split, O_1 never trained on", ok)
        r = enc[kind]["refit"]["1"]
        check(f"encoder/{kind}: refit on T_1 (190) without stop split, evaluated on O_1 (48)",
              len(r["train_ids"]) == 190 and r["stop_ids"] == [] and set(r["eval_ids"]) == o_ids
              and not (set(r["train_ids"]) & o_ids))
        hashes = {enc[kind]["inner"][str(j)]["state_sha256"] for j in range(1, 5)} | {r["state_sha256"]}
        check(f"encoder/{kind}: 5 distinct models (4 inner + refit)", len(hashes) == 5)
    dims = {**enc["tabular"]["inner"]["1"]["feature_dims"], **enc["image"]["inner"]["1"]["feature_dims"]}
    check("encoder: feature dims 128/16/512", dims == {"clinical": 128, "report": 16, "image": 512}, dims)

    sel = load("selection.json")["1"]
    check("selection: every model has a selected candidate and integer refit epochs",
          all(sel[m]["selected"] in CANDIDATES and isinstance(sel[m]["refit_epochs"], int)
              for m in FUSION_MODELS), {m: (sel[m]["selected"], sel[m]["refit_epochs"]) for m in FUSION_MODELS})
    recompute = all(select_candidate(sel[m]["inner_mean_cindex"]) == sel[m]["selected"]
                    and refit_epochs(sel[m]["inner_best_epochs"]) == sel[m]["refit_epochs"]
                    for m in FUSION_MODELS)
    check("selection: N4/N5 rules reproduce the recorded choice", recompute)
    means_ok = all(abs(inner[(inner.model == m) & (inner.candidate == c)].cindex_V.mean() - v) < 1e-6
                   for m in FUSION_MODELS for c, v in sel[m]["inner_mean_cindex"].items())
    check("selection: inner means computed from inner V only (match inner_results)", means_ok)

    pre = load("preprocessing_manifest.json")["1"]
    ai, ar = pre["foldsafe_audit_inner"], pre["foldsafe_audit_refit"]
    train_sizes = [a["n"]["train"] for a in ai]
    check("prep: fold-safe audits — inner (4 x [train fit, O_k extra]) and refit (T_k)",
          len(ai) == 8 and all(a["n"]["test"] in (47, 48) for a in ai)
          and len(ar) == 1 and ar[0]["n"] == {"train": 190, "val": 0, "test": 48}, train_sizes)
    eh = pd.read_csv(os.path.join(out, "encoder_history.csv"))
    check("encoder: S Efron NLL recorded every epoch and used for saving (monitor=loss)",
          eh[eh.phase == "inner"].val_loss.notna().all()
          and all(expected_saved(g.val_loss, 1e-4) == g.saved.tolist()
                  for _, g in eh[eh.phase == "inner"].groupby(["encoder", "fold"])))
    same_med = all(ai[2 * i]["train_medians"] == ai[2 * i + 1]["train_medians"] for i in range(4))
    check("prep: O_k extra features use the same Fit-only medians as training", same_med)

    fmx = pd.read_csv(os.path.join(out, "fold_metrics.csv"))
    expect = {"B_concat", "C_fixed_gate", "D_gmu", "A_primary", "A_raw", "A_ensemble",
              "tabular_only", "image_only"}
    check("metrics: all comparison models evaluated on O_1 (n=48), C-index finite in [0,1]",
          set(fmx.model) == expect and (fmx.n == 48).all() and fmx.cindex.between(0, 1).all(),
          sorted(fmx.model))

    oof = pd.read_csv(os.path.join(out, "oof_predictions.csv"))
    check("oof: 48 rows = O_1, one per patient", len(oof) == 48 and set(oof.research_id) == o_ids
          and not oof.research_id.duplicated().any())
    risk_cols = [c for c in oof if c.startswith("risk_")]
    check("oof: risk columns finite", np.isfinite(oof[risk_cols].to_numpy()).all(), risk_cols)
    gc = [c for c in oof if c.startswith("gate_C_fixed_gate")]
    gd = [c for c in oof if c.startswith("gate_D_gmu")]
    check("oof: gates in (0,1); C patient-common; D varies by patient",
          len(gc) == 3 and len(gd) == 3 and ((oof[gc + gd] > 0) & (oof[gc + gd] < 1)).all().all()
          and (oof[gc].nunique() == 1).all() and (oof[gd].nunique() > 1).all())

    a = load("late_fusion_A.json")["1"]
    d = a["diagnostics"]
    finite = lambda x: all(np.isfinite(v) for v in x.values())
    check("A: coefficients and diagnostics finite (slopes for 4 inner folds)",
          finite(a["coef_z"]) and finite(a["coef_raw"]) and finite(d["sd_ratio_outer_over_inner"])
          and finite(d["effective_image_weight"]) and sorted(d["calibration_slopes"]) == ["1", "2", "3", "4"],
          {"sd_ratio": {k: round(v, 3) for k, v in d["sd_ratio_outer_over_inner"].items()},
           "eff_img_w": {k: round(v, 3) for k, v in d["effective_image_weight"].items()}})

    ga = load("gate_analysis.json")["1"]
    check("N6: pre/post projection norms and gate distributions recorded",
          {"pre_projection_norm", "post_projection_norm"} <= set(ga["outer_selected"]["D_gmu"]["O_k"])
          and "gates" in ga["outer_selected"]["D_gmu"]["O_k"] and len(ga["inner_G1"]) == 12)

    before = os.path.getmtime(os.path.join(out, "outer1", "fold_metrics.csv"))
    rerun = subprocess.run([sys.executable, os.path.join(EXP_DIR, "run_nested_cv.py"),
                            "--target", args.target, "--smoke", "--out_dir", out], capture_output=True, text=True)
    check("guard: rerun into the same folder skips the finished outer (no overwrite)",
          rerun.returncode == 0 and "건너뛴다" in rerun.stdout
          and os.path.getmtime(os.path.join(out, "outer1", "fold_metrics.csv")) == before,
          rerun.stderr[-300:] if rerun.returncode else None)

    n_fail = sum(not c["pass"] for c in CHECKS)
    with open(os.path.join(out, "pathcheck_report.json"), "w", encoding="utf-8") as fh:
        json.dump({"status": "PATH CHECK ONLY — numbers are not results", "n_checks": len(CHECKS),
                   "n_fail": n_fail, "checks": CHECKS}, fh, ensure_ascii=False, indent=1, default=str)
    print(f"\n=== check_smoke_outputs {args.target}: {len(CHECKS) - n_fail}/{len(CHECKS)} PASS")


if __name__ == "__main__":
    main()
