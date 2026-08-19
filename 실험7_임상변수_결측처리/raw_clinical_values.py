# -*- coding: utf-8 -*-
"""Recovers true observed values + missingness for LDH/WBC/FVC%/FEV1%/DLCO%/gender
from the original ``DATA/20260619 SCLC PET 전달용.xlsx`` (``Whole`` sheet),
re-linked to patients by ``research_id`` (연구번호).

Why this module exists: ``DATA/merged_tabular_with_reports.csv`` (and its
upstream ``DATA/Clinical/fillna_tabular_data_260619.csv``) already have these
five continuous indicators imputed with the **whole-cohort** median -- there
are zero NaNs in the merged CSV for these columns, but repeated-value spikes
(e.g. ldh=539.5 appears 51/321x) show the imputation. That pre-imputation
both hides true missingness and leaks val/test-fold information into any
per-fold statistic computed downstream, so subgroup analysis (data_audit.py,
km_cox_analysis.py) and fold-safe model features (later step 3) must both
start from the raw excel values, not the merged CSV.

Neither the original excel nor the merged/fillna CSVs are modified anywhere
in this module -- everything here is read-only against those files.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
DEFAULT_SOURCE_XLSX = os.path.join(PROJECT_ROOT, "DATA", "20260619 SCLC PET 전달용.xlsx")
DEFAULT_MERGED_CSV = os.path.join(PROJECT_ROOT, "DATA", "merged_tabular_with_reports.csv")
DEFAULT_SPLIT_CSV = os.path.join(PROJECT_ROOT, "splits", "trimodal_common_5fold_seed42_v1.csv")

# Raw excel column -> canonical short name used throughout this experiment.
RAW_COLUMN_MAP = {
    "연구번호": "research_id",
    "성별": "gender_raw",
    "WBC": "wbc_raw",
    "LDH": "ldh_raw",
    "FVC\nPre%ref": "fvc_raw",
    "FEV1\nPre%ref": "fev1_raw",
    "DLCOAdj\nPre%ref": "dlco_raw",
}

# Clinical cutoffs specified by the requesting clinician (SCLC_EXPERIMENT
# protocol section 1 of this experiment's brief). Direction is NOT normalized
# to "high risk = 1" here -- e.g. FVC/FEV1/DLCO >=80% is the *normal* lung
# function group, opposite polarity from LDH/WBC >= cutoff (abnormal/high).
# Cox/KM output must label groups by their literal ">=cutoff" / "<cutoff"
# meaning, not by an assumed risk direction.
CUTOFFS = {"ldh_raw": 400.0, "wbc_raw": 10000.0, "fvc_raw": 80.0, "fev1_raw": 80.0, "dlco_raw": 80.0}

CONTINUOUS_INDICATORS = ("ldh_raw", "wbc_raw", "fvc_raw", "fev1_raw", "dlco_raw")


def load_raw_indicators(source_xlsx: str = DEFAULT_SOURCE_XLSX) -> pd.DataFrame:
    """Reads the ``Whole`` sheet and returns one row per patient (research_id
    index) with the 6 raw columns above, NaN preserved exactly as recorded.

    Raises if ``research_id`` has duplicates after dropping the trailing
    blank row (row 324 in the source sheet has an empty 연구번호 and is not a
    patient) -- silently keeping a dup would let ``.reindex()`` downstream
    return multiple rows per id without anyone noticing.
    """
    src = pd.read_excel(source_xlsx, sheet_name="Whole")
    missing_cols = set(RAW_COLUMN_MAP) - set(src.columns)
    if missing_cols:
        raise ValueError(f"expected columns not found in {source_xlsx!r} 'Whole' sheet: {sorted(missing_cols)}")

    df = src[list(RAW_COLUMN_MAP)].rename(columns=RAW_COLUMN_MAP)
    df["research_id"] = pd.to_numeric(df["research_id"], errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=["research_id"])
    n_dropped = n_before - len(df)
    if n_dropped > 1:
        raise ValueError(
            f"expected at most 1 blank trailing row with no research_id, found {n_dropped} -- investigate"
        )
    df["research_id"] = df["research_id"].astype(int)

    if df["research_id"].duplicated().any():
        dups = df.loc[df["research_id"].duplicated(keep=False), "research_id"].tolist()
        raise ValueError(f"duplicate research_id in raw excel: {sorted(set(dups))}")

    for col in list(CONTINUOUS_INDICATORS) + ["gender_raw"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.set_index("research_id")


def attach_cutoff_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Adds ``{col}_cutoff`` / ``{col}_missing`` for each continuous indicator,
    per the pre-agreed encoding (both columns must always be read together):

      observed, value >= cutoff : cutoff=1, missing=0
      observed, value <  cutoff : cutoff=0, missing=0
      missing                   : cutoff=0, missing=1   (never assigned to
                                   the >=cutoff OR the <cutoff group -- the
                                   reference/low group for interpretation is
                                   always "observed & below cutoff", i.e.
                                   cutoff=0 & missing=0; missing=1 rows must
                                   not be read as "low/normal")

    Both output columns are int8 (0/1), no NaN, so they drop straight into a
    model tensor or a pandas groupby without special-casing.
    """
    out = df.copy()
    for col, cutoff in CUTOFFS.items():
        observed = out[col].notna()
        cutoff_flag = np.where(observed & (out[col] >= cutoff), 1, 0)
        missing_flag = np.where(observed, 0, 1)
        out[f"{col}_cutoff"] = cutoff_flag.astype("int8")
        out[f"{col}_missing"] = missing_flag.astype("int8")
    return out


def _validate_gender_matches_merged(raw: pd.DataFrame, merged_csv: str) -> None:
    """Sanity check: gender has 0 missing in both sources, so raw vs merged
    values must match exactly for every id both files share. A mismatch would
    mean the id linkage (연구번호 -> research_id) itself is wrong -- this is
    the cheapest possible check of that linkage, so it's run unconditionally."""
    merged = pd.read_csv(merged_csv, encoding="utf-8-sig", low_memory=False)
    merged = merged.set_index("research_id")[["gender"]]
    joined = merged.join(raw[["gender_raw"]], how="inner")
    if joined["gender_raw"].isna().any():
        bad = joined[joined["gender_raw"].isna()].index.tolist()
        raise ValueError(f"gender_raw missing for research_id present in merged CSV: {bad}")
    mismatched = joined[joined["gender"] != joined["gender_raw"]]
    if len(mismatched):
        raise ValueError(
            f"gender mismatch between raw excel and merged CSV for {len(mismatched)} patient(s) "
            f"-- research_id<->연구번호 linkage is likely wrong: {mismatched.index.tolist()[:20]}"
        )


def load_cohort_indicators(
    source_xlsx: str = DEFAULT_SOURCE_XLSX,
    merged_csv: str = DEFAULT_MERGED_CSV,
    split_csv: str = DEFAULT_SPLIT_CSV,
    cohort_ids=None,
) -> pd.DataFrame:
    """Returns the analysis-ready frame for the 238-patient tri-modal common
    cohort (or an explicit ``cohort_ids`` subset): raw observed values +
    cutoff/missing flags for the 5 continuous indicators + gender, joined
    with OS/PFS duration and event from the merged CSV.

    Runs (and raises on failure) the following linkage checks so a broken
    join fails loudly instead of silently producing a wrong subgroup:
      - every cohort research_id exists in the raw excel (no unmatched ids)
      - no duplicate research_id in the raw excel
      - gender (0% missing in both sources) matches exactly raw vs merged,
        as an independent check that the id join itself is correct
    """
    raw = load_raw_indicators(source_xlsx)
    _validate_gender_matches_merged(raw, merged_csv)

    merged = pd.read_csv(merged_csv, encoding="utf-8-sig", low_memory=False).set_index("research_id")

    if cohort_ids is None:
        split = pd.read_csv(split_csv, dtype={"research_id": str})
        cohort_ids = split["research_id"].astype(int).unique().tolist()
    cohort_ids = sorted(int(i) for i in cohort_ids)

    missing_from_raw = sorted(set(cohort_ids) - set(raw.index))
    if missing_from_raw:
        raise ValueError(f"{len(missing_from_raw)} cohort research_id(s) not found in raw excel: {missing_from_raw}")
    missing_from_merged = sorted(set(cohort_ids) - set(merged.index))
    if missing_from_merged:
        raise ValueError(f"{len(missing_from_merged)} cohort research_id(s) not found in merged CSV: {missing_from_merged}")

    sub_raw = raw.reindex(cohort_ids)
    sub_raw = attach_cutoff_flags(sub_raw)
    sub_labels = merged.reindex(cohort_ids)[["os_days", "os_event", "pfs_days", "pfs_event"]]

    out = sub_raw.join(sub_labels)
    out.index.name = "research_id"
    return out


if __name__ == "__main__":
    frame = load_cohort_indicators()
    print(f"cohort n={len(frame)}")
    for col in CONTINUOUS_INDICATORS:
        n_missing = int(frame[col].isna().sum())
        print(f"  {col:10s} missing={n_missing:3d} ({n_missing/len(frame)*100:.1f}%)")
    print("gender_raw value_counts:")
    print(frame["gender_raw"].value_counts().to_string())
