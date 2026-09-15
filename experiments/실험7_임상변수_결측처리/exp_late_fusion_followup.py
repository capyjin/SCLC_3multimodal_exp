# -*- coding: utf-8 -*-
"""Step 3 follow-up -- does the improved clinical+report model still help once
it is combined with the image branch in the project's adopted late-fusion
pipeline (MODEL_SUMMARY.md 3-2, OS 0.7143 / PFS 0.6621)?

A gain measured on the tabular arm alone does not automatically survive
fusion: late fusion refits a per-fold CoxPH on the two OOF risk scores, and if
the image score already carries whatever the improved clinical features added,
the combined C-index can stay flat. This script checks that directly.

No retraining happens here. Both inputs already exist:
  - tabular risk scores : the OOF predictions saved by
    ``clinical/exp_missing_handling.py`` for every (variant, seed) --
    clin_report config, bs32/ep60, frozen split.
  - image risk scores   : ``outputs/late_fusion_B/oof_{target}.json`` ["image"],
    the SimpleCNN arm (bs16/ep30, seed 42) that the adopted late-fusion result
    was built from. Reusing it is the same practice as
    ``fusion/exp_late_fusion_3modal_rerun.py``; the image arm never reads
    clinical columns, so none of this experiment's changes can affect it.

The combination step itself is ``late_fusion_tab_image.combine_two``, imported
unmodified, so the fusion rule is identical to the recorded pipeline: per fold,
fit CoxPH on the train-fold patients' two risk scores, evaluate on the test fold.

Run:  python clinical/exp_late_fusion_followup.py --variants original,C
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "fusion"))

from core import cohort
from core.fusion_stack import combine_two

RESULTS_JSON = os.path.join(PROJECT_ROOT, "outputs", "EXP_20260805_clinical_missing_handling", "results.json")
IMAGE_OOF_DIR = os.path.join(PROJECT_ROOT, "outputs", "late_fusion_B")
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "EXP_20260805_clinical_missing_handling")


def load_image_oof(target: str) -> dict:
    path = os.path.join(IMAGE_OOF_DIR, f"oof_{target}.json")
    with open(path) as fh:
        payload = json.load(fh)
    return {int(k): float(v) for k, v in payload["image"].items()}


def load_tabular_oof(results_path: str, variant: str, target: str, seed: int) -> dict:
    with open(results_path) as fh:
        payload = json.load(fh)
    for run in payload["runs"]:
        if (run["config"] == "clin_report" and run["variant"] == variant
                and run["target"] == target and run["seed"] == seed):
            return {int(r["research_id"]): float(r["risk_score"]) for r in run["oof_predictions"]}
    raise KeyError(f"no clin_report/{variant}/{target}/seed{seed} run found in {results_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=RESULTS_JSON)
    ap.add_argument("--variants", default="original,A,B,C")
    ap.add_argument("--targets", default="os,pfs")
    ap.add_argument("--seeds", default="42,142,242")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "late_fusion_followup.json"))
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",")]
    targets = [t.strip() for t in args.targets.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    cohort_df = cohort.load_trimodal_cohort()
    results = []

    for target in targets:
        image_risk = load_image_oof(target)
        for variant in variants:
            for seed in seeds:
                tab_risk = load_tabular_oof(args.results, variant, target, seed)
                missing = set(image_risk) ^ set(tab_risk)
                assert not missing, f"tabular/image OOF cover different patients: {sorted(missing)[:10]}"

                combo = combine_two(cohort_df, target, tab_risk, image_risk)
                tab_only_folds = None
                results.append({
                    "target": target, "variant": variant, "seed": seed,
                    "fused_mean": combo["mean"], "fused_std": combo["std"],
                    "fused_folds": [round(c, 6) for c in combo["fold_cindex"]],
                    "mean_coef": combo["mean_coef"],
                })
                print(f"[latefusion] {target}/{variant}/seed{seed}: fused={combo['mean']:.4f} "
                      f"+/- {combo['std']:.4f}  coef(tab,img)="
                      f"({combo['mean_coef']['risk_tabular']:.3f},{combo['mean_coef']['risk_image']:.3f})")

    with open(args.out, "w") as fh:
        json.dump({"runs": results}, fh, indent=2)

    print(f"\n{'='*80}\nLATE FUSION SUMMARY (tabular=clin_report + image=SimpleCNN)\n{'='*80}")
    print(f"{'target':<7}{'variant':<10}{'fused mean':>12}{'sd across seeds':>18}   per-seed")
    for target in targets:
        for variant in variants:
            sel = [r for r in results if r["target"] == target and r["variant"] == variant]
            if not sel:
                continue
            means = [r["fused_mean"] for r in sel]
            print(f"{target:<7}{variant:<10}{np.mean(means):>12.4f}{np.std(means):>18.4f}   "
                  f"{[round(m,4) for m in means]}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
