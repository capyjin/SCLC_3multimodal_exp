# -*- coding: utf-8 -*-
"""Step 3 -- fold-safe replacement for the five whole-cohort-median-imputed
clinical indicators (LDH / WBC / FVC%pred / FEV1%pred / DLCOadj%pred).

The problem being fixed
-----------------------
``DATA/merged_tabular_with_reports.csv`` ships these five columns already
imputed with the **whole-cohort** median (verified in Step 1: 0 NaN in the CSV
but 3--52 genuinely missing values per indicator in the source excel). Feeding
those columns to a per-fold model is a double defect:

  1. leakage -- the imputed constant was computed from all 238 patients, so
     every train fold silently carries val/test-fold information;
  2. lost signal -- "this patient never had a PFT" is erased and replaced by a
     value indistinguishable from a real median-valued measurement.

What this module does
---------------------
Rebuilds those five inputs from the raw excel values (``raw_clinical_values``)
with **every cross-patient statistic fit on the train fold only** -- median for
imputation, StandardScaler for scaling -- and hands them to
``features.build_fold_multimodal_tabular`` through its ``extra_numeric_fn``
hook, exactly like ``report/suv_features.py`` does for SUV.

Critically, the five *pre-imputed* columns must simultaneously be **removed**
from ``ClinicalEncoder``'s standardize list (via ``reduced_clinical_columns``,
passed to ``TrimodalEvaluator(clinical_columns_fn=...)``), or the model would
see each indicator twice -- once leaked/global-median-imputed and once
fold-safe. ``assert_no_preimputed_leakage`` enforces that pairing and is
called automatically by ``make_extra_numeric_fn``'s returned closure on its
first invocation, so a caller who forgets ``clinical_columns_fn`` gets a hard
failure instead of a silently double-fed tensor.

Model variants (per the experiment brief)
-----------------------------------------
  A  "Corrected Baseline"  : 5 fold-safe continuous columns.
                             Total clinical dim 16 + 5 = 21 == baseline's 21,
                             so A isolates *imputation correctness* alone with
                             the feature count held fixed.
  B  "Missing-aware"       : A + 5 ``{indicator}_missing`` indicators (dim 26).
  C  "Clinical-enhanced"   : B + pre-specified LDH/FEV1 representations
                             (dim 29): log2(LDH) -- Step 2 showed the raw LDH
                             distribution is right-skewed (mean/median=1.50)
                             and the log2 fit is markedly better (OS p
                             2.5e-10 -> 4.9e-13) -- plus the clinical cutoff
                             flags for LDH (400) and FEV1 (80), the two
                             indicators pre-specified by the clinician.

Cutoff / missing encoding (fixed in advance, never data-driven)
---------------------------------------------------------------
  observed & value >= cutoff : cutoff=1, missing=0
  observed & value <  cutoff : cutoff=0, missing=0
  missing                    : cutoff=0, missing=1

The two columns are always emitted together so that "cutoff=0" is never
ambiguous: the interpretation reference group is **observed & below cutoff**
(cutoff=0 AND missing=0). A missing patient is *not* a member of the low/normal
group -- reading a cutoff coefficient without conditioning on its missing
partner would do exactly that.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


import numpy as np
from sklearn.preprocessing import StandardScaler

import raw_clinical_values as rcv

# raw excel column  ->  the pre-imputed column of the same quantity in the merged CSV
#
# SCOPE WARNING: this covers the five indicators the clinician pre-specified for
# this experiment, NOT every whole-cohort-median-imputed column. Of the 8
# continuous clinical columns, 7 carry global-median imputation (238-cohort raw
# missing counts: ldh 39, dlco 52, pack_years 33, fvc 27, fev1 26, wbc 3, hb 3;
# only age_at_diagnosis is genuinely complete). `pack_years` (13.9% missing --
# comparable to LDH) and `hb` are therefore STILL globally imputed in every
# variant built here. Comparisons stay internally valid because those two are
# held identical across original/A/B/C, but no variant produced by this module
# is a fully leakage-free pipeline. Before adding pack_years here, first resolve
# how the source column ('흡연력 ', with a trailing space) encodes non-smokers
# vs. unrecorded -- a 0 may mean either, and cross-checking `smoking_status` is
# required to avoid imputing over real zeros.
RAW_TO_PREIMPUTED = {
    "ldh_raw": "ldh",
    "wbc_raw": "wbc",
    "fvc_raw": "fvc_pre_percent_ref",
    "fev1_raw": "fev1_pre_percent_ref",
    "dlco_raw": "dlcoadj_pre_percent_ref",
}
PREIMPUTED_COLUMNS = tuple(RAW_TO_PREIMPUTED.values())
RAW_COLUMNS = tuple(RAW_TO_PREIMPUTED)

# Pre-specified core indicators (clinician-designated); included regardless of p-value.
CUTOFF_FLAG_COLUMNS = ("ldh_raw", "fev1_raw")

VARIANTS = ("A", "B", "C")
VARIANT_DESCRIPTION = {
    "A": "Corrected Baseline -- fold-safe median imputation only (same dim as baseline)",
    "B": "Missing-aware -- A + per-indicator missing indicators",
    "C": "Clinical-enhanced -- B + log2(LDH) + LDH/FEV1 clinical cutoff flags",
}


def block_columns(variant: str) -> tuple[list[str], list[str]]:
    """Returns ``(continuous_cols, binary_cols)`` emitted for a variant, in
    output order. Continuous columns are median-imputed + standardized on the
    train fold; binary columns are passed through as raw 0/1 (matching how
    ``ClinicalEncoder`` leaves its categorical block unscaled)."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {list(VARIANTS)}")

    continuous = list(RAW_COLUMNS)
    binary: list[str] = []
    if variant in ("B", "C"):
        binary += [f"{c}_missing" for c in RAW_COLUMNS]
    if variant == "C":
        continuous = continuous + ["ldh_log2"]
        binary += [f"{c}_cutoff" for c in CUTOFF_FLAG_COLUMNS]
    return continuous, binary


def block_width(variant: str) -> int:
    cont, binary = block_columns(variant)
    return len(cont) + len(binary)


def reduced_clinical_columns(df):
    """Drop-in replacement for ``features.resolve_clinical_columns`` that
    **removes the five pre-imputed columns** from the standardize list. Pass
    as ``TrimodalEvaluator(clinical_columns_fn=...)``. The categorical list is
    untouched (none of the five are categorical)."""
    from core import features

    standardize, categorical = features.resolve_clinical_columns(df)
    kept = [c for c in standardize if c not in PREIMPUTED_COLUMNS]
    dropped = [c for c in standardize if c in PREIMPUTED_COLUMNS]
    if len(dropped) != len(PREIMPUTED_COLUMNS):
        raise ValueError(
            f"expected to drop all {len(PREIMPUTED_COLUMNS)} pre-imputed columns "
            f"{list(PREIMPUTED_COLUMNS)} from the standardize list, but only dropped {dropped}. "
            "The merged CSV schema changed -- update RAW_TO_PREIMPUTED."
        )
    return kept, categorical


def assert_no_preimputed_leakage(standardize_cols, categorical_cols, variant: str) -> None:
    """Hard guarantees requested for this experiment:

      (1) none of the five whole-cohort-median-imputed columns survive in the
          clinical block (no residual imputed values), and
      (2) no name collision between the clinical block and this extra block
          (no duplicated input of the same quantity).
    """
    clinical_cols = list(standardize_cols) + list(categorical_cols)

    residual = sorted(set(clinical_cols) & set(PREIMPUTED_COLUMNS))
    assert not residual, (
        f"pre-imputed column(s) {residual} are still in the clinical block while the fold-safe "
        f"extra block also supplies them -- the model would see the same quantity twice, once "
        f"with whole-cohort-median leakage. Pass clinical_columns_fn=reduced_clinical_columns "
        f"to TrimodalEvaluator."
    )

    cont, binary = block_columns(variant)
    extra_cols = cont + binary
    collision = sorted(set(clinical_cols) & set(extra_cols))
    assert not collision, f"duplicate column name(s) between clinical block and extra block: {collision}"
    assert len(set(extra_cols)) == len(extra_cols), f"duplicate column inside extra block: {extra_cols}"


def make_extra_numeric_fn(variant: str, raw_frame=None, audit: list | None = None,
                          standardize_cols=None, categorical_cols=None):
    """Builds the ``extra_numeric_fn`` closure for ``variant``.

    ``raw_frame`` defaults to ``raw_clinical_values.load_cohort_indicators()``
    (raw observed values + cutoff/missing flags, indexed by research_id).

    Fold discipline, identical to ``report/suv_features.make_extra_numeric_fn``:
    the imputation median and the StandardScaler are fit on **train fold rows
    only**; val/test rows are transform-only. Binary flags need neither (they
    are derived per-patient from that patient's own observed value, so they
    carry no cross-patient statistic and cannot leak).

    Every fold appends a record to ``audit`` (train-fold medians, scaler
    mean/scale, how many rows were imputed per split, n per split) so the
    leakage discipline is inspectable in the saved results rather than assumed.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {list(VARIANTS)}")
    if raw_frame is None:
        raw_frame = rcv.load_cohort_indicators()

    cont_cols, binary_cols = block_columns(variant)
    frame = raw_frame.copy()
    if "ldh_log2" in cont_cols:
        # log2 on the *observed* value; missing stays NaN and is imputed below
        # with the train-fold median of log2(LDH). (Because log2 is monotone,
        # median(log2(x)) == log2(median(x)) -- imputing in log space and
        # log-transforming an imputed median give the same number; doing it in
        # log space keeps the imputation consistent with the column's own scale.)
        frame["ldh_log2"] = np.log2(frame["ldh_raw"])

    checked = {"done": False}

    def fn(train_ids, val_ids, test_ids):
        if not checked["done"]:
            if standardize_cols is not None:
                assert_no_preimputed_leakage(standardize_cols, categorical_cols or [], variant)
            checked["done"] = True

        ids = {"train": [int(i) for i in train_ids],
               "val": [int(i) for i in val_ids],
               "test": [int(i) for i in test_ids]}

        # ── (1) continuous: median-impute + standardize, both fit on TRAIN only ──
        raw = {name: frame.reindex(id_list)[cont_cols].to_numpy(dtype="float64")
               for name, id_list in ids.items()}

        medians = np.nanmedian(raw["train"], axis=0)
        if np.isnan(medians).any():
            bad = [c for c, m in zip(cont_cols, medians) if np.isnan(m)]
            raise ValueError(f"train fold has no observed value at all for column(s) {bad}")

        filled, n_imputed = {}, {}
        for name, arr in raw.items():
            mask = np.isnan(arr)
            filled[name] = np.where(mask, medians, arr)
            n_imputed[name] = {c: int(mask[:, j].sum()) for j, c in enumerate(cont_cols)}

        scaler = StandardScaler().fit(filled["train"])
        cont_out = {name: (scaler.transform(arr).astype("float32") if len(arr)
                           else np.empty((0, len(cont_cols)), dtype="float32"))
                    for name, arr in filled.items()}

        # ── (2) binary flags: per-patient, no cross-patient statistic, no scaling ──
        if binary_cols:
            bin_out = {name: frame.reindex(id_list)[binary_cols].to_numpy(dtype="float32")
                       for name, id_list in ids.items()}
            for name, arr in bin_out.items():
                if arr.size and not np.isin(arr, (0.0, 1.0)).all():
                    raise ValueError(f"non-binary value in flag columns for split {name}")
            out = {name: np.concatenate([cont_out[name], bin_out[name]], axis=1).astype("float32")
                   if len(ids[name]) else np.empty((0, block_width(variant)), dtype="float32")
                   for name in ids}
        else:
            out = cont_out

        for name in ids:
            assert out[name].shape == (len(ids[name]), block_width(variant)), (
                f"extra block width mismatch for split {name}: got {out[name].shape}, "
                f"expected {(len(ids[name]), block_width(variant))}"
            )

        if audit is not None:
            audit.append({
                "variant": variant,
                "n_train": len(ids["train"]), "n_val": len(ids["val"]), "n_test": len(ids["test"]),
                "continuous_cols": cont_cols, "binary_cols": binary_cols,
                "train_fold_medians": {c: round(float(m), 4) for c, m in zip(cont_cols, medians)},
                "train_fold_scaler_mean": {c: round(float(m), 4) for c, m in zip(cont_cols, scaler.mean_)},
                "train_fold_scaler_scale": {c: round(float(s), 4) for c, s in zip(cont_cols, scaler.scale_)},
                "n_imputed": n_imputed,
                "n_observed_train": {c: int((~np.isnan(raw["train"][:, j])).sum())
                                     for j, c in enumerate(cont_cols)},
            })
        return out

    return fn


def verify_fold_medians_are_train_only(audit: list, split_df, raw_frame=None) -> dict:
    """Independent re-derivation of the leakage check: recomputes each fold's
    imputation median directly from the raw values restricted to that fold's
    train ids, and confirms it equals what the closure actually used. Also
    confirms those medians differ from the whole-cohort median that the old
    pipeline baked into the CSV (when they happen to coincide numerically the
    check is reported, not failed -- coincidence is possible and harmless).
    """
    if raw_frame is None:
        raw_frame = rcv.load_cohort_indicators()
    frame = raw_frame.copy()
    frame["ldh_log2"] = np.log2(frame["ldh_raw"])

    folds = sorted(int(f) for f in split_df["fold"].unique())
    results, mismatches = [], []
    for record, fold in zip(audit, folds):
        train_ids = split_df.loc[(split_df["fold"] == fold) & (split_df["split"] == "train"),
                                 "research_id"].astype(int).tolist()
        for col, used in record["train_fold_medians"].items():
            expected = float(np.nanmedian(frame.reindex(train_ids)[col].to_numpy(dtype="float64")))
            global_median = float(np.nanmedian(frame[col].to_numpy(dtype="float64")))
            ok = np.isclose(used, expected, rtol=0, atol=1e-4)
            if not ok:
                mismatches.append({"fold": fold, "col": col, "used": used, "recomputed_train_only": expected})
            results.append({
                "fold": fold, "col": col, "used_median": used, "recomputed_train_only": round(expected, 4),
                "whole_cohort_median": round(global_median, 4),
                "matches_train_only": bool(ok),
                "differs_from_whole_cohort": bool(not np.isclose(expected, global_median, rtol=0, atol=1e-4)),
            })

    assert not mismatches, (
        "imputation median used at training time does not match a train-fold-only recomputation "
        f"-- leakage or an id-alignment bug: {mismatches}"
    )
    return {"per_fold_column": results,
            "n_checked": len(results),
            "n_differ_from_whole_cohort": sum(r["differs_from_whole_cohort"] for r in results)}


if __name__ == "__main__":
    import pandas as pd

    from core import cohort, features

    split = pd.read_csv(rcv.DEFAULT_SPLIT_CSV, dtype={"research_id": str})
    split["research_id"] = split["research_id"].astype(int)
    raw = rcv.load_cohort_indicators()

    cdf = cohort.load_trimodal_cohort().drop_duplicates("research_id").set_index("research_id")
    base_std, base_cat = features.resolve_clinical_columns(cdf)
    red_std, red_cat = reduced_clinical_columns(cdf)
    print(f"baseline clinical dim : {len(base_std) + len(base_cat)} ({len(base_std)} cont + {len(base_cat)} cat)")
    print(f"reduced  clinical dim : {len(red_std) + len(red_cat)} ({len(red_std)} cont + {len(red_cat)} cat)")
    print(f"dropped pre-imputed   : {sorted(set(base_std) - set(red_std))}")

    for variant in VARIANTS:
        cont, binary = block_columns(variant)
        audit: list = []
        fn = make_extra_numeric_fn(variant, raw_frame=raw, audit=audit,
                                   standardize_cols=red_std, categorical_cols=red_cat)
        for fold in sorted(split["fold"].unique()):
            f = split[split["fold"] == fold]
            get = lambda s: f.loc[f["split"] == s, "research_id"].astype(int).tolist()
            out = fn(get("train"), get("val"), get("test"))
        total_dim = len(red_std) + len(red_cat) + block_width(variant)
        check = verify_fold_medians_are_train_only(audit, split, raw_frame=raw)
        print(f"\n[{variant}] {VARIANT_DESCRIPTION[variant]}")
        print(f"     extra block width={block_width(variant)}  -> total clinical dim={total_dim}")
        print(f"     continuous={cont}")
        print(f"     binary={binary}")
        print(f"     median leakage check: {check['n_checked']} fold-column pairs verified train-only, "
              f"{check['n_differ_from_whole_cohort']} differ from the whole-cohort median")
