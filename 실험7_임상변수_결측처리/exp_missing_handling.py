# -*- coding: utf-8 -*-
"""Step 3 -- does correct (fold-safe) handling of the five clinical indicators
improve OS/PFS prediction over the current whole-cohort-median-imputed inputs?

Model ladder (each step adds exactly one thing to the previous one):

  original : the pipeline as it stands today -- LDH/WBC/FVC/FEV1/DLCO come
             from the merged CSV where they were already imputed with the
             **whole-cohort** median. Reference point, reproduces the recorded
             numbers bit-exactly (clin_only OS 0.6388, clin_report OS 0.7057).
  A        : same 21 features, but the five indicators are rebuilt from the raw
             excel values and imputed with the **train-fold** median (scaler
             also train-fold only). Same dimensionality as `original`, so the
             A-vs-original delta isolates *imputation correctness alone*.
  B        : A + five `{indicator}_missing` indicators (dim 26).
  C        : B + pre-specified LDH/FEV1 representations (dim 29):
             log2(LDH) + LDH>=400 and FEV1>=80 cutoff flags.

Held fixed across every run (protocol sections 2 and 5.4): the 238-patient
tri-modal cohort, the frozen 5-fold split file, model architecture, Cox loss,
optimizer, lr, weight decay, epochs=60, batch_size=32, checkpoint rule, and
the C-index implementation. Only the clinical feature block changes.

Repeated seeds (42/142/242) change **weight init and batch shuffling only** --
fold membership comes from the frozen split CSV and is identical across every
seed, so all comparisons stay paired at the patient level.

Run:
  python clinical/exp_missing_handling.py --smoke        # 1 fold, 2 epochs -- pipeline check
  python clinical/exp_missing_handling.py                # full: 4 variants x 2 configs x 2 targets x 3 seeds
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import json
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import pandas as pd

import fold_safe_features as fsf
import raw_clinical_values as rcv
from core.model import MODALITY_CONFIGS, make_model_factory
from core.train import TrimodalEvaluator

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "EXP_20260805_clinical_missing_handling")

VARIANTS = ("original", "A", "B", "C")
MODEL_CONFIGS = ("clin_only", "clin_report")
DEFAULT_SEEDS = (42, 142, 242)


def run_one(variant: str, config: str, target: str, seed: int, epochs: int, batch_size: int,
            max_folds=None, raw_frame=None) -> dict:
    """One 5-fold evaluation. Returns fold C-indices + OOF predictions + the
    per-fold leakage audit (empty for `original`, which has no extra block)."""
    flags = MODALITY_CONFIGS[config]
    audit: list = []
    kwargs = {}

    if variant != "original":
        # Both hooks must be passed together: reduced_clinical_columns removes the
        # five pre-imputed columns from the clinical block, make_extra_numeric_fn
        # re-adds the same quantities fold-safely. fold_safe_features asserts the
        # pairing on first call, so passing only one of them fails loudly.
        from core import cohort as cohort_mod
        cdf = cohort_mod.load_trimodal_cohort().drop_duplicates("research_id").set_index("research_id")
        red_std, red_cat = fsf.reduced_clinical_columns(cdf)
        kwargs["clinical_columns_fn"] = fsf.reduced_clinical_columns
        kwargs["extra_numeric_fn"] = fsf.make_extra_numeric_fn(
            variant, raw_frame=raw_frame, audit=audit,
            standardize_cols=red_std, categorical_cols=red_cat,
        )

    tag = f"{config}_{variant}_{target}_seed{seed}"
    ev = TrimodalEvaluator(
        target=target, epochs=epochs, batch_size=batch_size, seed=seed,
        save_dir=os.path.join(OUT_DIR, "checkpoints", tag),
        model_factory=make_model_factory(flags),
        max_folds=max_folds,
        **kwargs,
    ).run()

    return {
        "variant": variant, "config": config, "target": target, "seed": seed,
        "epochs": epochs, "batch_size": batch_size,
        "folds": [round(float(c), 6) for c in ev.c_indices],
        "mean": float(np.mean(ev.c_indices)), "std": float(np.std(ev.c_indices)),
        "fold_records": ev.fold_records,
        "oof_predictions": ev.oof_predictions,
        "leakage_audit": audit,
    }


def _verify_oof(result: dict, cohort_frame: pd.DataFrame) -> None:
    """Guards the paired-bootstrap input: every cohort patient must appear
    exactly once in the pooled OOF predictions, and each row's duration/event
    must match that patient's label in the cohort frame.

    This is not paranoia for its own sake -- TrimodalEvaluator builds OOF rows
    by ``zip(ids['test'], durations, ...)``, which silently misaligns if image
    loading ever drops a patient. Currently no image is missing (all 238
    present), so the zip is safe; this check makes that a verified fact per
    run instead of a standing assumption.
    """
    oof = pd.DataFrame(result["oof_predictions"])
    target = result["target"]
    dur_col, evt_col = f"{target}_days", f"{target}_event"

    counts = oof["research_id"].value_counts()
    assert (counts == 1).all(), f"{(counts != 1).sum()} patient(s) appear more than once in OOF"
    expected_ids = set(cohort_frame.index.astype(int))
    got_ids = set(oof["research_id"].astype(int))
    assert got_ids == expected_ids, (
        f"OOF patient set != cohort: missing {sorted(expected_ids - got_ids)[:10]}, "
        f"unexpected {sorted(got_ids - expected_ids)[:10]}"
    )

    ref = cohort_frame.loc[oof["research_id"].astype(int)]
    assert np.allclose(oof["duration"].to_numpy(float), ref[dur_col].to_numpy(float)), \
        "OOF duration does not match the cohort label -- risk scores are misaligned to patients"
    assert np.allclose(oof["event"].to_numpy(float), ref[evt_col].to_numpy(float)), \
        "OOF event does not match the cohort label -- risk scores are misaligned to patients"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--configs", default=",".join(MODEL_CONFIGS))
    ap.add_argument("--targets", default="os,pfs")
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--smoke", action="store_true",
                    help="1 fold x 2 epochs, seed 42 only -- pipeline sanity check, NOT a result")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",")]
    configs = [c.strip() for c in args.configs.split(",")]
    targets = [t.strip() for t in args.targets.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    epochs, batch_size, max_folds = args.epochs, args.batch_size, None
    if args.smoke:
        epochs, max_folds, seeds = 2, 1, [42]

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = args.out or os.path.join(
        OUT_DIR, "results_smoke.json" if args.smoke else "results.json")

    # Load the raw indicator frame once and share it across every run -- it is
    # read-only reference data, so re-reading the excel 48x would only cost time.
    raw_frame = rcv.load_cohort_indicators()
    from core import cohort as cohort_mod
    cohort_frame = (cohort_mod.load_trimodal_cohort()
                    .drop_duplicates("research_id").set_index("research_id"))

    runs, started = [], time.time()
    total = len(variants) * len(configs) * len(targets) * len(seeds)
    for config in configs:
        for target in targets:
            for variant in variants:
                for seed in seeds:
                    idx = len(runs) + 1
                    print(f"\n{'#'*90}\n### [{idx}/{total}] config={config} variant={variant} "
                          f"target={target} seed={seed} bs={batch_size} ep={epochs}\n{'#'*90}")
                    res = run_one(variant, config, target, seed, epochs, batch_size,
                                  max_folds=max_folds, raw_frame=raw_frame)
                    if max_folds is None:
                        _verify_oof(res, cohort_frame)
                    runs.append(res)
                    print(f"[RESULT] {config}/{variant}/{target}/seed{seed}: "
                          f"{res['mean']:.4f} +/- {res['std']:.4f}  folds={res['folds']}")
                    with open(out_path, "w") as fh:
                        json.dump({"runs": runs, "epochs": epochs, "batch_size": batch_size,
                                   "smoke": args.smoke}, fh, indent=2)

    print(f"\n{'='*90}\nSUMMARY (mean C-index over folds, averaged across seeds)\n{'='*90}")
    print(f"{'config':<13}{'target':<7}{'variant':<10}{'mean':>9}{'std(fold)':>11}   per-seed means")
    for config in configs:
        for target in targets:
            for variant in variants:
                sel = [r for r in runs if r["config"] == config and r["target"] == target
                       and r["variant"] == variant]
                if not sel:
                    continue
                means = [r["mean"] for r in sel]
                print(f"{config:<13}{target:<7}{variant:<10}{np.mean(means):>9.4f}"
                      f"{np.mean([r['std'] for r in sel]):>11.4f}   {[round(m,4) for m in means]}")

    print(f"\nelapsed {(time.time()-started)/60:.1f} min -- wrote {out_path}")


if __name__ == "__main__":
    main()
