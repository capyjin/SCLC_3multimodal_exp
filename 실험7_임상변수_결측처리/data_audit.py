# -*- coding: utf-8 -*-
"""Step 1 -- data audit for LDH/WBC/FVC%/FEV1%/DLCO%/gender on the 238-patient
tri-modal common cohort, using the *raw* (excel-recovered) observed values
from ``raw_clinical_values.py`` -- never the whole-cohort-median-imputed
merged CSV columns.

Per the experiment brief: subgroups are built only from patients with a real
observed value; missing patients are excluded from the main subgroup table
and reported as a separate count (never assigned to a cutoff group by their
imputed value). DLCO (highest missingness, 21.8%) additionally gets an
observed-vs-missing descriptive comparison -- reported as a sensitivity
check, not as a biological risk group.

Run:  python clinical/data_audit.py
Writes: outputs/EXP_20260805_clinical_data_audit/{data_manifest.csv,
        indicator_audit.csv, distribution_table.csv, cutoff_group_table.csv,
        missing_sensitivity_dlco.csv}
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


import numpy as np
import pandas as pd

import raw_clinical_values as rcv

OUT_DIR = os.path.join(rcv.PROJECT_ROOT, "outputs", "EXP_20260805_clinical_data_audit")

LABELS = {
    "ldh_raw": "LDH", "wbc_raw": "WBC", "fvc_raw": "FVC%pred",
    "fev1_raw": "FEV1%pred", "dlco_raw": "DLCOadj%pred", "gender_raw": "Gender",
}


# ── ID / linkage audit ───────────────────────────────────────────────────────
def id_audit(frame: pd.DataFrame, split_csv: str = rcv.DEFAULT_SPLIT_CSV,
             merged_csv: str = rcv.DEFAULT_MERGED_CSV, source_xlsx: str = rcv.DEFAULT_SOURCE_XLSX) -> dict:
    """Independently recomputes the linkage checks (raw_clinical_values.py
    already raises hard on any of these during load_cohort_indicators, so
    reaching this function means they passed -- this just makes the passing
    checks visible in the audit report instead of silent)."""
    raw = rcv.load_raw_indicators(source_xlsx)
    merged = pd.read_csv(merged_csv, encoding="utf-8-sig", low_memory=False)
    split = pd.read_csv(split_csv, dtype={"research_id": str})
    split["research_id"] = split["research_id"].astype(int)
    cohort_ids = set(split["research_id"].unique())

    n_dup_raw = int(raw.index.duplicated().sum())
    n_dup_merged = int(merged["research_id"].duplicated().sum())
    # split.csv has 5 rows per patient by design (one per fold), so a patient
    # duplicate only matters *within* a single fold (same id assigned to two
    # splits, or listed twice, inside one fold) -- checking across the whole
    # file would flag all 238*4=952 legitimate repeat-across-folds rows.
    n_dup_split = int(split.groupby("fold")["research_id"].apply(lambda s: s.duplicated().sum()).sum())
    n_folds_per_patient = split.groupby("research_id")["fold"].nunique()
    n_patients_missing_a_fold = int((n_folds_per_patient != split["fold"].nunique()).sum())
    missing_from_raw = sorted(cohort_ids - set(raw.index))
    missing_from_merged = sorted(cohort_ids - set(merged["research_id"]))

    label_cols = ["os_days", "os_event", "pfs_days", "pfs_event"]
    label_frame = frame[label_cols]
    n_incomplete_labels = int(label_frame.isna().any(axis=1).sum())

    return {
        "cohort_n": len(cohort_ids),
        "duplicate_research_id_raw_excel": n_dup_raw,
        "duplicate_research_id_merged_csv": n_dup_merged,
        "duplicate_research_id_within_fold_split_csv": n_dup_split,
        "patients_not_present_in_all_5_folds": n_patients_missing_a_fold,
        "cohort_ids_missing_from_raw_excel": len(missing_from_raw),
        "cohort_ids_missing_from_merged_csv": len(missing_from_merged),
        "gender_raw_vs_merged_mismatch": 0,  # load_cohort_indicators() would have raised otherwise
        "cohort_patients_with_incomplete_os_pfs_labels": n_incomplete_labels,
        "all_checks_passed": bool(
            n_dup_raw == 0 and n_dup_merged == 0 and n_dup_split == 0 and n_patients_missing_a_fold == 0
            and not missing_from_raw and not missing_from_merged and n_incomplete_labels == 0
        ),
    }


# ── per-indicator n / missing / event counts ────────────────────────────────
def build_indicator_audit(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n_total = len(frame)
    for col in rcv.CONTINUOUS_INDICATORS:
        observed = frame[col].notna()
        missing = ~observed
        rows.append({
            "indicator": LABELS[col], "n_total": n_total,
            "n_observed": int(observed.sum()), "n_missing": int(missing.sum()),
            "missing_pct": round(100 * missing.sum() / n_total, 1),
            "os_event_observed": int(frame.loc[observed, "os_event"].sum()),
            "os_event_missing": int(frame.loc[missing, "os_event"].sum()),
            "pfs_event_observed": int(frame.loc[observed, "pfs_event"].sum()),
            "pfs_event_missing": int(frame.loc[missing, "pfs_event"].sum()),
        })
    # gender: complete (0 missing), included for symmetry in the same table
    rows.append({
        "indicator": LABELS["gender_raw"], "n_total": n_total,
        "n_observed": int(frame["gender_raw"].notna().sum()), "n_missing": int(frame["gender_raw"].isna().sum()),
        "missing_pct": round(100 * frame["gender_raw"].isna().sum() / n_total, 1),
        "os_event_observed": int(frame.loc[frame["gender_raw"].notna(), "os_event"].sum()),
        "os_event_missing": 0,
        "pfs_event_observed": int(frame.loc[frame["gender_raw"].notna(), "pfs_event"].sum()),
        "pfs_event_missing": 0,
    })
    return pd.DataFrame(rows)


# ── observed-value distribution ─────────────────────────────────────────────
def build_distribution_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in rcv.CONTINUOUS_INDICATORS:
        vals = frame[col].dropna().to_numpy(dtype=float)
        rows.append({
            "indicator": LABELS[col], "n_observed": len(vals),
            "min": round(float(vals.min()), 2), "p25": round(float(np.percentile(vals, 25)), 2),
            "median": round(float(np.median(vals)), 2), "p75": round(float(np.percentile(vals, 75)), 2),
            "max": round(float(vals.max()), 2),
            "mean": round(float(vals.mean()), 2), "sd": round(float(vals.std(ddof=1)), 2),
            "skew_mean_over_median": round(float(vals.mean() / np.median(vals)), 3),
        })
    return pd.DataFrame(rows)


# ── cutoff-group sizes + events (observed patients only) ───────────────────
def build_cutoff_group_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, cutoff in rcv.CUTOFFS.items():
        observed_df = frame[frame[col].notna()]
        for group_name, group_df in (
            (f"<{cutoff:g}", observed_df[observed_df[col] < cutoff]),
            (f">={cutoff:g}", observed_df[observed_df[col] >= cutoff]),
        ):
            rows.append({
                "indicator": LABELS[col], "group": group_name, "n": len(group_df),
                "os_event": int(group_df["os_event"].sum()), "os_censored": int((group_df["os_event"] == 0).sum()),
                "pfs_event": int(group_df["pfs_event"].sum()), "pfs_censored": int((group_df["pfs_event"] == 0).sum()),
            })
        rows.append({
            "indicator": LABELS[col], "group": "missing (excluded from subgroup)",
            "n": int(frame[col].isna().sum()),
            "os_event": int(frame.loc[frame[col].isna(), "os_event"].sum()),
            "os_censored": int((frame.loc[frame[col].isna(), "os_event"] == 0).sum()),
            "pfs_event": int(frame.loc[frame[col].isna(), "pfs_event"].sum()),
            "pfs_censored": int((frame.loc[frame[col].isna(), "pfs_event"] == 0).sum()),
        })
    # gender: no cutoff, but same shape (group by raw code) for the report table
    for code, group_df in frame.groupby("gender_raw"):
        rows.append({
            "indicator": LABELS["gender_raw"], "group": f"code={int(code)}", "n": len(group_df),
            "os_event": int(group_df["os_event"].sum()), "os_censored": int((group_df["os_event"] == 0).sum()),
            "pfs_event": int(group_df["pfs_event"].sum()), "pfs_censored": int((group_df["pfs_event"] == 0).sum()),
        })
    return pd.DataFrame(rows)


# ── DLCO (highest missingness) observed-vs-missing sensitivity comparison ──
def build_missing_sensitivity(frame: pd.DataFrame, col: str = "dlco_raw") -> pd.DataFrame:
    rows = []
    for name, sub in (("observed", frame[frame[col].notna()]), ("missing", frame[frame[col].isna()])):
        rows.append({
            "group": name, "n": len(sub),
            "os_event": int(sub["os_event"].sum()), "os_event_rate": round(sub["os_event"].mean(), 3),
            "os_median_days_all": round(float(sub["os_days"].median()), 1),
            "pfs_event": int(sub["pfs_event"].sum()), "pfs_event_rate": round(sub["pfs_event"].mean(), 3),
            "pfs_median_days_all": round(float(sub["pfs_days"].median()), 1),
        })
    return pd.DataFrame(rows)


def run() -> dict:
    os.makedirs(OUT_DIR, exist_ok=True)
    frame = rcv.load_cohort_indicators()

    audit = id_audit(frame)
    indicator_tbl = build_indicator_audit(frame)
    dist_tbl = build_distribution_table(frame)
    cutoff_tbl = build_cutoff_group_table(frame)
    dlco_sens = build_missing_sensitivity(frame, "dlco_raw")

    frame.to_csv(os.path.join(OUT_DIR, "data_manifest.csv"))
    indicator_tbl.to_csv(os.path.join(OUT_DIR, "indicator_audit.csv"), index=False)
    dist_tbl.to_csv(os.path.join(OUT_DIR, "distribution_table.csv"), index=False)
    cutoff_tbl.to_csv(os.path.join(OUT_DIR, "cutoff_group_table.csv"), index=False)
    dlco_sens.to_csv(os.path.join(OUT_DIR, "missing_sensitivity_dlco.csv"), index=False)
    pd.Series(audit).to_csv(os.path.join(OUT_DIR, "id_audit.csv"), header=["value"])

    print("=== ID / linkage audit ===")
    for k, v in audit.items():
        print(f"  {k}: {v}")
    print("\n=== indicator n/missing/events ===")
    print(indicator_tbl.to_string(index=False))
    print("\n=== observed-value distribution ===")
    print(dist_tbl.to_string(index=False))
    print("\n=== cutoff group sizes + events ===")
    print(cutoff_tbl.to_string(index=False))
    print("\n=== DLCO observed-vs-missing sensitivity ===")
    print(dlco_sens.to_string(index=False))
    print(f"\nwrote outputs to {OUT_DIR}")

    return {
        "frame": frame, "id_audit": audit, "indicator_audit": indicator_tbl,
        "distribution": dist_tbl, "cutoff_groups": cutoff_tbl, "dlco_sensitivity": dlco_sens,
    }


if __name__ == "__main__":
    run()
