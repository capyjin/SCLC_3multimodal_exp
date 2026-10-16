# -*- coding: utf-8 -*-
"""[실험16-9] 본 학습 산출물 점검 — 끝난 outer 마다 규칙·분할·누출·유한값이 맞는지 확인한다.

성능은 보지 않는다 (C-index 는 [0,1] 유한값인지만). run_nested_cv.py 의 가드와 별개로,
저장된 파일만 읽어 같은 조건을 독립적으로 다시 확인한다.

Run:  python experiments/실험16_GMU_nestedCV/diagnostics/check_run_outputs.py --target os
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
import glob
import json

import numpy as np
import pandas as pd

from checklist import CHECKS, check, expected_saved

from nested_fusion.config import CANDIDATES, ENCODERS, RUN_TAG
from nested_fusion.dataset import load_manifest
from nested_fusion.model import FUSION_MODELS
from nested_fusion.train import EarlyStopper, effective_candidates, refit_epochs, rule_for, select_candidate
from sclc import paths

def check_outer(out, k, man):
    d = os.path.join(out, f"outer{k}")
    load = lambda f: json.load(open(os.path.join(d, f), encoding="utf-8"))
    om = next(o for o in man["outer"] if o["k"] == k)
    o_ids = set(om["outer_test_ids"])
    tag = f"outer {k}"

    inner = pd.read_csv(os.path.join(d, "inner_results.csv"))
    n_expect = 4 * sum(len(effective_candidates(m)) for m in FUSION_MODELS)
    check(f"{tag}: inner rows {n_expect}, V C-index finite", len(inner) == n_expect and inner.cindex_V.between(0, 1).all())
    ok_rule, bad = True, []
    for _, r in inner.iterrows():
        rule = rule_for(CANDIDATES[r.candidate])
        stop_ok = r.epochs_run == rule["max_epochs"] or r.epochs_run == r.best_epoch + rule["patience"]
        if not (rule["min_epoch"] <= r.best_epoch <= r.epochs_run <= rule["max_epochs"] and stop_ok
                and r.rule_max_epochs == rule["max_epochs"]):
            ok_rule = False
            bad.append((r.inner, r.model, r.candidate, r.best_epoch, r.epochs_run))
    check(f"{tag}: GMU stop rule per lr (best>=30, stop = best+patience or max)", ok_rule, bad[:5] or None)
    hit_max = inner[inner.epochs_run == inner.rule_max_epochs]
    check(f"{tag}: GMU runs that hit max epochs (기록용)", True,
          hit_max[["inner", "model", "candidate", "best_epoch"]].values.tolist() or "none")
    fin = np.isfinite(inner[["train_nll_at_best", "stop_nll_at_best"]].to_numpy()).all()
    check(f"{tag}: GMU train/stop NLL at best finite", fin)

    hist = pd.read_csv(os.path.join(d, "gmu_inner_history.csv.gz"))
    finite_h = np.isfinite(hist[["train_loss", "stop_loss", "train_loss_eval"]].to_numpy()).all()
    check(f"{tag}: GMU histories finite (no NaN)", finite_h, len(hist))
    replay_ok = True
    for (j, m, c), g in hist.groupby(["inner", "model", "candidate"]):
        rule = rule_for(CANDIDATES[c])
        st = EarlyStopper(rule["patience"], rule["min_epoch"], rule["min_delta"])
        for e, v in zip(g.epoch, g.stop_loss):
            st.update(int(e), float(v))
        rec = inner[(inner.inner == j) & (inner.model == m) & (inner.candidate == c)].iloc[0]
        replay_ok &= st.best_epoch == rec.best_epoch and len(g) == rec.epochs_run
    check(f"{tag}: best epochs reproduced by replaying EarlyStopper on saved histories", replay_ok)

    sel = load("selection.json")
    re_ok = all(select_candidate(sel[m]["inner_mean_cindex"]) == sel[m]["selected"]
                and refit_epochs(sel[m]["inner_best_epochs"]) == sel[m]["refit_epochs"]
                and set(sel[m]["inner_mean_cindex"]) == set(effective_candidates(m)) for m in FUSION_MODELS)
    check(f"{tag}: selection (N4/N5) reproducible from inner results", re_ok,
          {m: (sel[m]["selected"], sel[m]["refit_epochs"]) for m in FUSION_MODELS})
    means_ok = all(abs(inner[(inner.model == m) & (inner.candidate == c)].cindex_V.mean() - v) < 1e-6
                   for m in FUSION_MODELS for c, v in sel[m]["inner_mean_cindex"].items())
    check(f"{tag}: inner means come from inner V only", means_ok)
    rh = pd.read_csv(os.path.join(d, "gmu_refit_history.csv"))
    check(f"{tag}: GMU refit epochs = selected refit_epochs, losses finite",
          all(len(rh[rh.model == m]) == sel[m]["refit_epochs"] for m in FUSION_MODELS)
          and np.isfinite(rh.train_loss).all())

    enc = load("encoder_manifest.json")
    eh = pd.read_csv(os.path.join(d, "encoder_history.csv"))
    for kind in ENCODERS:
        ok = True
        for j, f in enumerate(om["inner"], start=1):
            e = enc[kind]["inner"][str(j)]
            ok &= (e["train_ids"] == f["ids"]["fit"] and e["stop_ids"] == f["ids"]["stop"]
                   and e["eval_ids"] == f["ids"]["val"] and not (set(e["train_ids"] + e["stop_ids"]) & o_ids))
        r = enc[kind]["refit"][str(k)]
        ok &= (len(r["train_ids"]) + len(r["eval_ids"]) == 238 and r["stop_ids"] == []
               and set(r["eval_ids"]) == o_ids and not (set(r["train_ids"]) & o_ids))
        check(f"{tag} {kind}: inner/refit patients == locked split, O_k never trained", ok)
        g_in = eh[(eh.encoder == kind) & (eh.phase == "inner")]
        mono = all(expected_saved(g.val_loss, ENCODERS[kind]["min_delta"], ENCODERS[kind]["min_epoch"])
                   == g.saved.tolist()
                   for _, g in g_in.groupby("fold"))
        stops = {int(f): (int(g.epoch[g.saved].iloc[-1]), len(g)) for f, g in g_in.groupby("fold")}
        stop_ok = all((n == ENCODERS[kind]["max_epochs"] or n == b + ENCODERS[kind]["patience"])
                      and b >= ENCODERS[kind]["min_epoch"] for b, n in stops.values())
        check(f"{tag} {kind}: encoder saves on S Efron NLL (min_delta 1e-4, min epoch "
              f"{ENCODERS[kind]['min_epoch']}), stop = best+10 or max", mono and stop_ok,
              {"inner (best, run)": stops, "refit epochs": len(eh[(eh.encoder == kind) & (eh.phase == "refit")])})
        refit_ep = refit_epochs([b for b, _ in stops.values()])
        check(f"{tag} {kind}: refit epochs = median of inner best", refit_ep == sel["encoders_refit_epochs"][kind]
              == len(eh[(eh.encoder == kind) & (eh.phase == "refit")]), refit_ep)
        check(f"{tag} {kind}: encoder losses finite",
              np.isfinite(g_in[["train_loss", "val_loss", "train_loss_eval"]].to_numpy()).all())

    pre = load("preprocessing_manifest.json")
    ai = pre["foldsafe_audit_inner"]
    check(f"{tag}: fold-safe imputation fitted on Fit only; O_k extra uses the same medians",
          len(ai) == 8 and all(ai[2 * j]["n"]["train"] == len(om["inner"][j]["ids"]["fit"])
                               and ai[2 * j + 1]["train_medians"] == ai[2 * j]["train_medians"] for j in range(4))
          and pre["foldsafe_audit_refit"][0]["n"]["train"] == 238 - len(o_ids))

    fmx = pd.read_csv(os.path.join(d, "fold_metrics.csv"))
    oof = pd.read_csv(os.path.join(d, "oof_predictions.csv"))
    check(f"{tag}: 8 models evaluated on O_k, C-index finite", len(fmx) == 8 and fmx.cindex.between(0, 1).all()
          and (fmx.n == len(o_ids)).all())
    check(f"{tag}: OOF rows = O_k exactly, risks finite", set(oof.research_id) == o_ids and len(oof) == len(o_ids)
          and np.isfinite(oof[[c for c in oof if c.startswith("risk_")]].to_numpy()).all())
    gd = [c for c in oof if c.startswith("gate_D_gmu")]
    gc = [c for c in oof if c.startswith("gate_C_fixed_gate")]
    check(f"{tag}: gates in (0,1), C patient-common", ((oof[gd + gc] > 0) & (oof[gd + gc] < 1)).all().all()
          and (oof[gc].nunique() == 1).all())
    a = load("late_fusion_A.json")
    check(f"{tag}: A coefficients/diagnostics finite",
          all(np.isfinite(v) for v in list(a["coef_z"].values()) + list(a["coef_raw"].values())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=("os", "pfs"))
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    out = args.out_dir or paths.outputs("gmu_nested", f"{RUN_TAG}_{args.target}")
    man = load_manifest()
    done = sorted(int(os.path.basename(p)[5:]) for p in glob.glob(os.path.join(out, "outer[0-9]")))
    for k in done:
        check_outer(out, k, man)
    n_fail = sum(not c["pass"] for c in CHECKS)
    with open(os.path.join(out, "check_run_report.json"), "w", encoding="utf-8") as fh:
        json.dump({"outers": done, "n_checks": len(CHECKS), "n_fail": n_fail, "checks": CHECKS},
                  fh, ensure_ascii=False, indent=1, default=str)
    print(f"\n=== check_run_outputs {args.target} outers {done}: {len(CHECKS) - n_fail}/{len(CHECKS)} PASS")


if __name__ == "__main__":
    main()
