# -*- coding: utf-8 -*-
"""Step 2 -- KM/Cox association analysis for LDH/WBC/FVC%/FEV1%/DLCO%/gender
vs. OS and PFS, on the 238-patient tri-modal common cohort, using the raw
(excel-recovered) observed values from ``raw_clinical_values.py``.

Design (per experiment brief):
  - PRIMARY: continuous univariable Cox on observed patients only. Reported
    as both HR-per-SD (comparable across indicators) and HR-per-clinical-unit
    (directly interpretable). LDH additionally gets a log2(LDH) fit (its
    natural unit becomes "per doubling of LDH") because the raw distribution
    is strongly right-skewed (see distribution_table.csv: mean/median=1.5).
  - SUPPLEMENTARY: clinical-cutoff binary Cox + log-rank + KM curve, on the
    same observed-only subset (cutoff group membership is never assigned to
    missing patients -- see raw_clinical_values.attach_cutoff_flags).
  - EXPLORATORY: missing-indicator univariable Cox (missing=1 vs observed=0),
    fit on ALL 238 patients per indicator. This tests whether *not having the
    test done* associates with survival -- it does not test whether the
    imputed value is informative, and a significant HR here must not be read
    as "the missing group is biologically higher/lower risk"; it may simply
    reflect who was well enough to complete a PFT or blood draw.
  - Every fit also runs lifelines' rank-transform proportional-hazards test
    (Schoenfeld-residual based) as the required PH diagnostic.
  - gender is a native binary covariate (code 1 vs 2, no missingness) -- one
    Cox/KM/log-rank, not split into primary/supplementary.
  - LDH and FEV1 are pre-specified core indicators (per protocol) and are
    reported here regardless of p-value; p-values throughout this module are
    exploratory association/direction signals, not a variable-selection gate.

Run:  python clinical/km_cox_analysis.py
Writes tables + KM plots under outputs/EXP_20260805_clinical_data_audit/km_cox/
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test, proportional_hazard_test

import raw_clinical_values as rcv
from core import plotstyle as ps
from core.plotstyle import BLUE, GRID, ORANGE
from data_audit import LABELS, OUT_DIR

PLOT_DIR = os.path.join(OUT_DIR, "km_cox", "plots")
TABLE_DIR = os.path.join(OUT_DIR, "km_cox")

TARGETS = {"os": ("os_days", "os_event"), "pfs": ("pfs_days", "pfs_event")}
CLINICAL_UNIT = {"ldh_raw": 100.0, "wbc_raw": 1000.0, "fvc_raw": 10.0, "fev1_raw": 10.0, "dlco_raw": 10.0}
CLINICAL_UNIT_LABEL = {
    "ldh_raw": "per 100 U/L", "wbc_raw": "per 1000/uL",
    "fvc_raw": "per 10%pred", "fev1_raw": "per 10%pred", "dlco_raw": "per 10%pred",
}
CORE_PRESPECIFIED = {"ldh_raw", "fev1_raw"}  # included in modeling candidates regardless of p-value

ps.apply(font_size=10.5)   # 프로젝트 공통 팔레트/폰트 (core/plotstyle.py)


# ── core Cox fit helper ──────────────────────────────────────────────────────
def _cox_fit(sub: pd.DataFrame, duration_col: str, event_col: str, x_col: str) -> dict:
    """Fits a single-covariate CoxPHFitter and returns coef/se/HR/CI/p plus
    the rank-transform proportional-hazards diagnostic p-value. ``x_col`` is
    used as-is (caller decides raw/standardized/binary/log2 scaling) -- the
    Wald p-value and PH-test p-value are invariant to any positive linear
    rescaling of x, so callers may rescale coef/se afterward without refitting.
    """
    d = sub[[duration_col, event_col, x_col]].dropna()
    n, events = len(d), int(d[event_col].sum())
    if events < 5 or d[x_col].nunique() < 2:
        return {"n": n, "events": events, "coef": np.nan, "se": np.nan, "p": np.nan,
                "concordance": np.nan, "ph_p": np.nan, "note": "too few events or no variance"}

    cph = CoxPHFitter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(d, duration_col=duration_col, event_col=event_col, formula=f"{x_col}")
    row = cph.summary.loc[x_col]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ph = proportional_hazard_test(cph, d, time_transform="rank")
        ph_p = float(ph.summary["p"].iloc[0])
    except Exception as exc:  # pragma: no cover -- diagnostic only, must not abort the analysis
        ph_p = np.nan
        print(f"  [warn] PH test failed for x={x_col} dur={duration_col}: {exc}")

    return {
        "n": n, "events": events,
        "coef": float(row["coef"]), "se": float(row["se(coef)"]), "p": float(row["p"]),
        "concordance": float(cph.concordance_index_), "ph_p": ph_p,
    }


def _hr_ci(coef: float, se: float, scale: float = 1.0) -> tuple[float, float, float]:
    """HR / 95% CI for a `scale`-unit change in x, given coef/se fit on x itself."""
    c, s = coef * scale, se * scale
    return float(np.exp(c)), float(np.exp(c - 1.96 * s)), float(np.exp(c + 1.96 * s))


# ── PRIMARY: continuous univariable Cox (standardized), + clinical-unit HR ──
def continuous_cox_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, (dur, evt) in TARGETS.items():
        for col in rcv.CONTINUOUS_INDICATORS:
            sub = frame[frame[col].notna()].copy()
            mean, sd = sub[col].mean(), sub[col].std(ddof=1)
            sub["_z"] = (sub[col] - mean) / sd
            fit = _cox_fit(sub, dur, evt, "_z")
            row = {"target": target, "indicator": LABELS[col], "column": col,
                   "n": fit["n"], "events": fit["events"], "pre_specified_core": col in CORE_PRESPECIFIED,
                   "p": fit["p"], "ph_assumption_p": fit["ph_p"], "concordance": fit["concordance"]}
            if not np.isnan(fit["coef"]):
                hr_sd, lo_sd, hi_sd = _hr_ci(fit["coef"], fit["se"], scale=1.0)
                unit = CLINICAL_UNIT[col] / sd
                hr_u, lo_u, hi_u = _hr_ci(fit["coef"], fit["se"], scale=unit)
                row.update({
                    "hr_per_sd": round(hr_sd, 3), "ci95_per_sd": f"[{lo_sd:.3f}, {hi_sd:.3f}]",
                    "hr_clinical_unit": round(hr_u, 3), "ci95_clinical_unit": f"[{lo_u:.3f}, {hi_u:.3f}]",
                    "clinical_unit_label": CLINICAL_UNIT_LABEL[col],
                })
            rows.append(row)
    return pd.DataFrame(rows)


# ── LDH log2 transform (own natural unit: per doubling) ─────────────────────
def ldh_log2_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, (dur, evt) in TARGETS.items():
        sub = frame[frame["ldh_raw"].notna()].copy()
        sub["_log2ldh"] = np.log2(sub["ldh_raw"])
        fit = _cox_fit(sub, dur, evt, "_log2ldh")
        row = {"target": target, "indicator": "LDH (log2)", "n": fit["n"], "events": fit["events"],
               "p": fit["p"], "ph_assumption_p": fit["ph_p"], "concordance": fit["concordance"]}
        if not np.isnan(fit["coef"]):
            hr, lo, hi = _hr_ci(fit["coef"], fit["se"], scale=1.0)
            row.update({"hr_per_doubling": round(hr, 3), "ci95_per_doubling": f"[{lo:.3f}, {hi:.3f}]"})
        rows.append(row)
    return pd.DataFrame(rows)


# ── SUPPLEMENTARY: cutoff-based binary Cox + log-rank + KM ──────────────────
def cutoff_cox_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, (dur, evt) in TARGETS.items():
        for col, cutoff in rcv.CUTOFFS.items():
            sub = frame[frame[col].notna()].copy()
            flag_col = f"{col}_cutoff"
            fit = _cox_fit(sub, dur, evt, flag_col)

            lo_grp = sub[sub[flag_col] == 0]
            hi_grp = sub[sub[flag_col] == 1]
            lr = logrank_test(lo_grp[dur], hi_grp[dur], lo_grp[evt], hi_grp[evt])

            row = {
                "target": target, "indicator": LABELS[col], "cutoff": cutoff,
                "n_low": len(lo_grp), "events_low": int(lo_grp[evt].sum()),
                "n_high": len(hi_grp), "events_high": int(hi_grp[evt].sum()),
                "logrank_p": float(lr.p_value), "ph_assumption_p": fit["ph_p"],
            }
            if not np.isnan(fit["coef"]):
                hr, lo, hi = _hr_ci(fit["coef"], fit["se"], scale=1.0)
                row.update({"hr_high_vs_low": round(hr, 3), "ci95": f"[{lo:.3f}, {hi:.3f}]", "cox_p": fit["p"]})
            rows.append(row)
            _km_plot_two_groups(
                {f"<{cutoff:g} (n={len(lo_grp)})": (lo_grp[dur], lo_grp[evt]),
                 f">={cutoff:g} (n={len(hi_grp)})": (hi_grp[dur], hi_grp[evt])},
                title=f"{LABELS[col]} cutoff={cutoff:g} -- {target.upper()}\nlog-rank p={lr.p_value:.4f}",
                out_path=os.path.join(PLOT_DIR, f"km_cutoff_{col}_{target}.png"),
            )
    return pd.DataFrame(rows)


# ── gender (native binary, no missingness) ──────────────────────────────────
def gender_cox_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, (dur, evt) in TARGETS.items():
        sub = frame.copy()
        sub["_g"] = sub["gender_raw"] - 1.0  # code=1 (n=208, majority) is the reference (0)
        fit = _cox_fit(sub, dur, evt, "_g")
        g1 = sub[sub["gender_raw"] == 1]
        g2 = sub[sub["gender_raw"] == 2]
        lr = logrank_test(g1[dur], g2[dur], g1[evt], g2[evt])
        row = {"target": target, "indicator": "Gender (code2 vs code1)",
               "n_code1": len(g1), "events_code1": int(g1[evt].sum()),
               "n_code2": len(g2), "events_code2": int(g2[evt].sum()),
               "logrank_p": float(lr.p_value), "ph_assumption_p": fit["ph_p"]}
        if not np.isnan(fit["coef"]):
            hr, lo, hi = _hr_ci(fit["coef"], fit["se"], scale=1.0)
            row.update({"hr_code2_vs_code1": round(hr, 3), "ci95": f"[{lo:.3f}, {hi:.3f}]", "cox_p": fit["p"]})
        rows.append(row)
        _km_plot_two_groups(
            {f"code=1 (n={len(g1)})": (g1[dur], g1[evt]), f"code=2 (n={len(g2)})": (g2[dur], g2[evt])},
            title=f"Gender -- {target.upper()}\nlog-rank p={lr.p_value:.4f}",
            out_path=os.path.join(PLOT_DIR, f"km_gender_{target}.png"),
        )
    return pd.DataFrame(rows)


# ── EXPLORATORY: missing-indicator vs OS/PFS (all 238 patients) ────────────
def missing_indicator_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target, (dur, evt) in TARGETS.items():
        for col in rcv.CONTINUOUS_INDICATORS:
            sub = frame.copy()
            missing_col = f"{col}_missing"
            fit = _cox_fit(sub, dur, evt, missing_col)
            obs_grp = sub[sub[missing_col] == 0]
            miss_grp = sub[sub[missing_col] == 1]
            lr = logrank_test(obs_grp[dur], miss_grp[dur], obs_grp[evt], miss_grp[evt])
            row = {"target": target, "indicator": LABELS[col],
                   "n_observed": len(obs_grp), "events_observed": int(obs_grp[evt].sum()),
                   "n_missing": len(miss_grp), "events_missing": int(miss_grp[evt].sum()),
                   "logrank_p": float(lr.p_value)}
            if not np.isnan(fit["coef"]):
                hr, lo, hi = _hr_ci(fit["coef"], fit["se"], scale=1.0)
                row.update({"hr_missing_vs_observed": round(hr, 3), "ci95": f"[{lo:.3f}, {hi:.3f}]", "cox_p": fit["p"]})
            rows.append(row)
            if col == "dlco_raw":  # highest missingness -> dedicated sensitivity KM plot
                _km_plot_two_groups(
                    {f"observed (n={len(obs_grp)})": (obs_grp[dur], obs_grp[evt]),
                     f"missing (n={len(miss_grp)})": (miss_grp[dur], miss_grp[evt])},
                    title=f"{LABELS[col]} missing vs observed -- {target.upper()} (sensitivity check, "
                          f"NOT a risk-group claim)\nlog-rank p={lr.p_value:.4f}",
                    out_path=os.path.join(PLOT_DIR, f"km_missingness_{col}_{target}.png"),
                )
    return pd.DataFrame(rows)


# ── plotting ─────────────────────────────────────────────────────────────────
def _km_plot_two_groups(groups: dict, title: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    for (label, (dur, evt)), color in zip(groups.items(), (BLUE, ORANGE)):
        kmf = KaplanMeierFitter()
        kmf.fit(dur, evt, label=label)
        kmf.plot_survival_function(ax=ax, color=color, ci_show=True, ci_alpha=0.12)
    ax.set_xlabel("Days")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.02)
    ax.set_title(title, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _ldh_distribution_plot(frame: pd.DataFrame) -> None:
    vals = frame["ldh_raw"].dropna().to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].hist(vals, bins=30, color=BLUE, edgecolor="white")
    axes[0].axvline(400, color=ORANGE, linestyle="--", label="cutoff=400")
    axes[0].set_title(f"LDH raw (n={len(vals)}, mean={vals.mean():.0f}, median={np.median(vals):.0f})", fontsize=9.5, loc="left")
    axes[0].legend(frameon=False, fontsize=8.5)
    axes[1].hist(np.log2(vals), bins=30, color=BLUE, edgecolor="white")
    axes[1].axvline(np.log2(400), color=ORANGE, linestyle="--", label="log2(400)")
    axes[1].set_title("LDH log2-transformed", fontsize=9.5, loc="left")
    axes[1].legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "ldh_distribution.png"), dpi=150)
    plt.close(fig)


def run() -> dict:
    os.makedirs(PLOT_DIR, exist_ok=True)
    frame = rcv.load_cohort_indicators()

    print("=== PRIMARY: continuous univariable Cox (observed only) ===")
    cont = continuous_cox_table(frame)
    print(cont.to_string(index=False))
    cont.to_csv(os.path.join(TABLE_DIR, "cox_continuous_primary.csv"), index=False)

    print("\n=== LDH log2-transformed Cox ===")
    ldh_log2 = ldh_log2_table(frame)
    print(ldh_log2.to_string(index=False))
    ldh_log2.to_csv(os.path.join(TABLE_DIR, "cox_log2_ldh.csv"), index=False)
    _ldh_distribution_plot(frame)

    print("\n=== SUPPLEMENTARY: clinical-cutoff binary Cox + log-rank ===")
    cutoff = cutoff_cox_table(frame)
    print(cutoff.to_string(index=False))
    cutoff.to_csv(os.path.join(TABLE_DIR, "cox_cutoff_binary.csv"), index=False)

    print("\n=== Gender Cox + log-rank ===")
    gender = gender_cox_table(frame)
    print(gender.to_string(index=False))
    gender.to_csv(os.path.join(TABLE_DIR, "cox_gender.csv"), index=False)

    print("\n=== EXPLORATORY: missing-indicator vs OS/PFS (all 238) ===")
    missing = missing_indicator_table(frame)
    print(missing.to_string(index=False))
    missing.to_csv(os.path.join(TABLE_DIR, "cox_missing_indicator.csv"), index=False)

    print(f"\nwrote tables to {TABLE_DIR}, plots to {PLOT_DIR}")
    return {"continuous": cont, "ldh_log2": ldh_log2, "cutoff": cutoff, "gender": gender, "missing": missing}


if __name__ == "__main__":
    run()
