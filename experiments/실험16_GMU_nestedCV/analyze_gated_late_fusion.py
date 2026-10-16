# -*- coding: utf-8 -*-
"""④ Gated Late Fusion 사후 분석. 저장된 체크포인트로 위험점수를 재구성하고,
Late Fusion 과 같은 입력에 GatedLateFusion 을 학습해 outer 시험 환자에서 비교한다.

Run:  python experiments/실험16_GMU_nestedCV/analyze_gated_late_fusion.py"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json

import numpy as np
import pandas as pd
import torch

from nested_fusion.config import ARMS, RUN_TAG, SPLIT_DIR
from nested_fusion.dataset import load_labels, load_outer_split
from nested_fusion.model import F64, GatedLateFusion, nested_late_fusion
from nested_fusion.processing import CheckpointRebuilder, arm_records, standardize, standardizer
from sclc import cohort, features, paths
from sclc import foldsafe_clinical as fsc
from sclc.encoders import build_encoder
from sclc.evaluation import cindex

OUT = paths.outputs("gmu_nested", "gated_nested")


def rebuild_outer(rb: CheckpointRebuilder, run_dir: str, k: int):
    """outer k 의 inner·outer 위험점수를 체크포인트로 재구성한다."""
    inner_csv = os.path.join(SPLIT_DIR, f"outer{k}_inner.csv")
    refit_csv = os.path.join(SPLIT_DIR, f"outer{k}_refit.csv")
    t_ids, o_ids = load_outer_split(refit_csv)
    ck = os.path.join(run_dir, "ckpt", f"outer{k}")
    tab_s = rb.tabular(inner_csv, os.path.join(ck, "tabular_inner"), o_ids)
    img_s = rb.image(inner_csv, os.path.join(ck, "image_inner"), o_ids)
    rt = rb.tabular(refit_csv, os.path.join(ck, "tabular_refit"), None)[k]["splits"]
    ri = rb.image(refit_csv, os.path.join(ck, "image_refit"), None)[k]["splits"]
    inner, outer = arm_records(tab_s, img_s, rt, ri, o_ids)
    return inner, outer, t_ids, o_ids


def run_target(target, ctx):
    """outer 5개에서 Late Fusion 재현을 확인한 뒤 GatedLateFusion 을 학습·평가한다."""
    run_dir = paths.outputs("gmu_nested", f"{RUN_TAG}_{target}")
    saved_oof = pd.read_csv(os.path.join(run_dir, "oof_predictions.csv")).set_index("research_id")
    saved_A = json.load(open(os.path.join(run_dir, "late_fusion_A.json"), encoding="utf-8"))
    labels = load_labels(target)
    rb = CheckpointRebuilder(target, ctx)
    folds_out, verif, score_rows = [], {}, []
    for k in range(1, 6):
        inner, outer, t_ids, o_ids = rebuild_outer(rb, run_dir, k)
        so = saved_oof.loc[o_ids]
        a = nested_late_fusion(inner, outer, t_ids, labels.loc[t_ids], target)
        v = {"refit_tab_vs_saved": float(np.abs(outer["arms"]["tabular"]["test_risk"] - so.risk_tabular_only).max()),
             "refit_img_vs_saved": float(np.abs(outer["arms"]["image"]["test_risk"] - so.risk_image_only).max()),
             "A_coef_vs_saved": float(max(abs(a["coef_z"][c] - saved_A[str(k)]["coef_z"][c]) for c in a["coef_z"])),
             "A_pred_vs_saved": float(np.abs(a["pred"]["primary"] - so.risk_A_primary).max())}
        oof = a["oof"]
        Ztr = torch.tensor(oof[["z_tabular", "z_image"]].to_numpy(float), dtype=F64)
        lab = labels.loc[oof.index]
        ttr = torch.tensor(lab[f"{target}_days"].to_numpy(float), dtype=F64)
        etr = torch.tensor(lab[f"{target}_event"].to_numpy(float), dtype=F64)
        folds = oof["inner_fold"].to_numpy()
        Zte = torch.tensor(np.stack([standardize(outer["arms"][arm]["test_risk"],
                                                     standardizer(outer["arms"][arm]["train_risk"]))
                                     for arm in ARMS], axis=1), dtype=F64)
        o_lab = labels.loc[o_ids]
        tte, ete = o_lab[f"{target}_days"].to_numpy(float), o_lab[f"{target}_event"].to_numpy(float)
        bA = torch.tensor([a["coef_z"]["z_tabular"], a["coef_z"]["z_image"]], dtype=F64)
        r0, g0 = GatedLateFusion.forward(bA, torch.zeros(2, 2, dtype=F64), Zte)
        v["E_c0_bA_vs_A_pred"] = float(np.abs(r0.numpy() - a["pred"]["primary"]).max())
        v["E_c0_gate_is_1"] = bool(torch.all(g0 == 1.0))
        b_own = GatedLateFusion(0.0, gated=False).fit(Ztr, ttr, etr).beta
        v["Efron_beta_vs_A_coef"] = float((b_own - bA).abs().max())
        verif[k] = v

        rec = {"outer": k, "n_train": len(t_ids), "n_test": len(o_ids),
               "A_primary": cindex(tte, a["pred"]["primary"], ete)}
        name = "E1_learn_beta"
        lam, sc = GatedLateFusion.select_lambda(Ztr, ttr, etr, folds)
        model = GatedLateFusion(lam).fit(Ztr, ttr, etr)
        beta, C = model.beta, model.C
        r, g = model.predict(Zte)
        gs = GatedLateFusion.gate_summary(g)
        rec.update({f"{name}": cindex(tte, r.numpy(), ete), f"{name}_lambda": lam,
                    f"{name}_inner_scores": sc, f"{name}_beta": beta.tolist(), f"{name}_C": C.tolist(),
                    f"{name}_gate": gs,
                    f"{name}_gate_off": bool(max(abs(gs[m]["max"] - 1) for m in ARMS) < 0.01
                                             and max(abs(gs[m]["min"] - 1) for m in ARMS) < 0.01)})
        score_rows += [{"outer": k, "model": name, "research_id": i, "risk": float(x),
                        "gate_tabular": float(gg[0]), "gate_image": float(gg[1])}
                       for i, x, gg in zip(o_ids, r.numpy(), g.numpy())]
        folds_out.append(rec)
        print(f"[GatedLateFusion] {target} outer {k}: Late Fusion {rec['A_primary']:.4f} | Gated {rec['E1_learn_beta']:.4f} "
              f"(lam {rec['E1_learn_beta_lambda']}) | verify {v}")
    return folds_out, verif, pd.DataFrame(score_rows)


def main():
    if os.path.exists(os.path.join(OUT, "results.json")):
        raise SystemExit(f"{OUT} 에 이미 결과가 있다 — 덮어쓰지 않는다.")
    os.makedirs(OUT, exist_ok=True)
    corpus, _ = features.load_text_corpus(cohort.DEFAULT_MERGED_CSV)
    ctx = {"raw": fsc.load_raw_values(), "corpus": corpus,
           "text_fn": build_encoder("radbert").evaluator_kwargs(corpus)["text_encoder_fn"]}
    out = {"status": "탐색적 실험 — 본 결과(A/B/C/D) 확인 뒤 2026-10-15 추가. 사전 지정 비교 아님",
           "design": {"gate": "g = 2*sigmoid(c . [z_tab, z_img])", "loss": "Efron (사건당) + lambda*||c||^2",
                      "lambdas": list(GatedLateFusion.LAMBDAS), "selection": "확정 inner 4-fold V C-index 평균, 동점이면 큰 lambda",
                      "E1": "b, c 동시 학습",
                      "inputs": "A 와 같은 inner OOF z / outer 재학습 인코더 z (체크포인트 재구성)"},
           "targets": {}}
    for target in ("os", "pfs"):
        folds, verif, scores = run_target(target, ctx)
        scores.to_csv(os.path.join(OUT, f"oof_scores_{target}.csv"), index=False)
        df = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in folds])
        df.to_csv(os.path.join(OUT, f"fold_metrics_{target}.csv"), index=False)
        out["targets"][target] = {"folds": folds, "verification": verif}
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
    print(f"[analyze_gated_late_fusion] wrote {OUT}")


if __name__ == "__main__":
    main()
