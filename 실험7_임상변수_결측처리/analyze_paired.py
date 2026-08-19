# -*- coding: utf-8 -*-
"""Step 3 analysis -- paired comparison of the model ladder with bootstrap CIs.

Why paired, and paired on what
------------------------------
Every variant is trained on the **same frozen folds** (splits/trimodal_common_
5fold_seed42_v1.csv) with the same seeds, so each patient gets exactly one
out-of-fold (OOF) risk score per (variant, seed). That lets us compare
variants on identical patients rather than comparing two independent
5-fold averages -- an unpaired comparison would throw away the fact that the
same 238 people are being scored twice and would give needlessly wide
intervals.

Two aggregations are reported because they answer different questions and
protocol section 9 requires them to be distinguished:
  - fold-mean C-index : average of the 5 per-fold C-indices (what
                        MODEL_SUMMARY.md records; comparable to prior numbers)
  - pooled-OOF C-index: one C-index over all 238 pooled OOF scores

Uncertainty -- two bootstrap CIs, not one
------------------------------------------
Early runs of this analysis showed delta_fold_mean and delta_pooled_oof
disagreeing in sign on some cells. The reason: pooled-OOF concordance compares
patients across *different* folds, i.e. against risk scores from a different
fold's independently-trained model. A Cox partial-likelihood risk score is
only meaningful up to an arbitrary monotone transform *within* the model that
produced it, so a cross-fold pair's "concordance" partly reflects how two
unrelated models' arbitrary scales happened to align, not predictive skill.
The fold-mean metric never has this problem because every comparison inside a
fold's C-index uses one model's scores. Two bootstraps are reported so both
metrics carry an interval instead of only the (leakier) pooled one:
  - `paired_bootstrap`            -- patient-level resample of all 238 pooled
    OOF rows (seed-averaged risk ranks). Kept for continuity/diagnosis, but
    its cross-fold comparisons carry the scale-mixing risk above.
  - `paired_bootstrap_fold_level` -- resamples patients *within* every
    individual (seed, fold) test set separately, using that run's own raw
    risk scores; the delta is the mean over all (seed x fold) cells, which is
    exactly what `delta_fold_mean` already reports as a point estimate. This
    is the primary interval -- it never lets two differently-scaled models'
    scores enter the same comparison.
  - seed spread (3 seeds, init/shuffle only) is reported separately so that
    "the model got luckier" and "the features are better" stay distinguishable.

A delta whose 95% CI contains 0 is reported as not distinguishable from zero.
One seed's point estimate going up is never, on its own, treated as improvement.

Run:  python clinical/analyze_paired.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import itertools
import json

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "EXP_20260805_clinical_missing_handling")

# Comparisons that carry a specific question, evaluated per (config, target).
# Each step of the ladder isolates exactly one change from its predecessor.
LADDER = [
    ("original", "A", "fold-safe imputation alone (same feature count)"),
    ("A", "B", "+ missing indicators"),
    ("B", "C", "+ log2(LDH) and LDH/FEV1 cutoff flags"),
    ("original", "B", "cumulative: fold-safe imputation + missing indicators"),
    ("original", "C", "cumulative: full clinical-enhanced model"),
]


def _cindex(dur, evt, risk) -> float:
    """Harrell C-index with the project's sign convention (train.py: higher
    risk score = higher risk = shorter survival, so scores are negated)."""
    return float(concordance_index(dur, -risk, evt))


def load_runs(path: str) -> pd.DataFrame:
    with open(path) as fh:
        payload = json.load(fh)
    rows = []
    for run in payload["runs"]:
        oof = pd.DataFrame(run["oof_predictions"]).set_index("research_id").sort_index()
        rows.append({
            "config": run["config"], "target": run["target"], "variant": run["variant"],
            "seed": run["seed"], "fold_cindices": run["folds"],
            "fold_mean": float(np.mean(run["folds"])),
            "oof": oof,
        })
    return pd.DataFrame(rows)


def _seed_averaged_ranks(runs_for_variant: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapses the repeated seeds into one risk vector per patient.

    Risk scores from different seeds live on different arbitrary scales (a Cox
    partial-likelihood risk score is only defined up to monotone comparison),
    so averaging the raw scores across seeds would let whichever seed happened
    to produce the largest spread dominate. Averaging the **within-seed ranks**
    is scale-free and leaves the C-index -- itself a rank statistic --
    well-defined. Returns (ids, duration, event, mean_rank) aligned by id.
    """
    frames = list(runs_for_variant["oof"])
    ids = frames[0].index.to_numpy()
    for f in frames[1:]:
        assert np.array_equal(f.index.to_numpy(), ids), "seed runs cover different patients"
    dur = frames[0]["duration"].to_numpy(float)
    evt = frames[0]["event"].to_numpy(float)
    ranks = np.vstack([pd.Series(f["risk_score"].to_numpy(float)).rank().to_numpy() for f in frames])
    return dur, evt, ranks.mean(axis=0)


def paired_bootstrap(dur, evt, risk_a, risk_b, n_boot: int = 2000, seed: int = 12345) -> dict:
    """Patient-level paired bootstrap of ``C(b) - C(a)``.

    The same resampled patient index is used for both variants, so the
    per-resample difference cancels the shared patient-sampling noise. Draws
    that contain fewer than 2 events (no comparable pair) are skipped rather
    than counted as 0 difference -- counting them would shrink the interval
    toward zero artificially.
    """
    rng = np.random.default_rng(seed)
    n = len(dur)
    deltas, skipped = [], 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if evt[idx].sum() < 2:
            skipped += 1
            continue
        try:
            deltas.append(_cindex(dur[idx], evt[idx], risk_b[idx]) - _cindex(dur[idx], evt[idx], risk_a[idx]))
        except ZeroDivisionError:  # no admissible pair in this resample
            skipped += 1
    deltas = np.asarray(deltas)
    return {
        "delta_point": _cindex(dur, evt, risk_b) - _cindex(dur, evt, risk_a),
        "ci95_low": float(np.percentile(deltas, 2.5)),
        "ci95_high": float(np.percentile(deltas, 97.5)),
        "p_delta_gt_0": float((deltas > 0).mean()),
        "n_boot_used": int(len(deltas)), "n_boot_skipped": int(skipped),
    }


def _fold_cells(ref_runs: pd.DataFrame, new_runs: pd.DataFrame) -> list[dict]:
    """One dict per (seed, fold): that cell's patients' duration/event and
    each variant's *own* raw risk scores (no cross-seed/cross-fold mixing)."""
    cells = []
    for _, ref_r in ref_runs.iterrows():
        match = new_runs[new_runs["seed"] == ref_r["seed"]]
        assert len(match) == 1, f"expected exactly one seed={ref_r['seed']} run in new_runs, got {len(match)}"
        new_r = match.iloc[0]
        ref_oof, new_oof = ref_r["oof"], new_r["oof"]
        for f in sorted(ref_oof["fold"].unique()):
            ref_sub = ref_oof[ref_oof["fold"] == f]
            new_sub = new_oof.loc[ref_sub.index]  # same patients, same order (index = research_id)
            assert np.allclose(ref_sub["duration"].to_numpy(float), new_sub["duration"].to_numpy(float)), \
                f"seed={ref_r['seed']} fold={f}: duration mismatch between reference and comparison variant"
            assert np.allclose(ref_sub["event"].to_numpy(float), new_sub["event"].to_numpy(float)), \
                f"seed={ref_r['seed']} fold={f}: event mismatch between reference and comparison variant"
            cells.append({
                "dur": ref_sub["duration"].to_numpy(float), "evt": ref_sub["event"].to_numpy(float),
                "risk_ref": ref_sub["risk_score"].to_numpy(float), "risk_new": new_sub["risk_score"].to_numpy(float),
            })
    return cells


def paired_bootstrap_fold_level(ref_runs: pd.DataFrame, new_runs: pd.DataFrame,
                                n_boot: int = 2000, seed: int = 12345) -> dict:
    """Fold-and-seed-stratified paired bootstrap for the delta of **fold-mean**
    C-index -- the project's primary metric -- across all (seed x fold) cells.

    Each draw resamples patients *within* every individual (seed, fold) test
    set independently, exactly mirroring how the reported fold C-index is
    computed, and never compares two patients scored by different
    fold-specific models. ``delta_point`` here is computed directly from the
    cells and should closely match the ``delta_fold_mean`` point estimate
    computed elsewhere from the stored per-fold C-indices -- both come from
    the same underlying numbers, just recombined; a large discrepancy would
    indicate a bug in one of the two code paths.
    """
    rng = np.random.default_rng(seed)
    cells = _fold_cells(ref_runs, new_runs)

    point_deltas = [_cindex(c["dur"], c["evt"], c["risk_new"]) - _cindex(c["dur"], c["evt"], c["risk_ref"])
                    for c in cells]
    point = float(np.mean(point_deltas))

    boot_deltas, skipped = [], 0
    for _ in range(n_boot):
        draw, ok = [], True
        for c in cells:
            idx = rng.integers(0, len(c["dur"]), size=len(c["dur"]))
            if c["evt"][idx].sum() < 2:
                ok = False
                break
            try:
                draw.append(_cindex(c["dur"][idx], c["evt"][idx], c["risk_new"][idx])
                           - _cindex(c["dur"][idx], c["evt"][idx], c["risk_ref"][idx]))
            except ZeroDivisionError:
                ok = False
                break
        if not ok:
            skipped += 1
            continue
        boot_deltas.append(np.mean(draw))

    boot_deltas = np.asarray(boot_deltas)
    return {
        "delta_point": point,
        "ci95_low": float(np.percentile(boot_deltas, 2.5)),
        "ci95_high": float(np.percentile(boot_deltas, 97.5)),
        "p_delta_gt_0": float((boot_deltas > 0).mean()),
        "n_boot_used": int(len(boot_deltas)), "n_boot_skipped": int(skipped),
        "n_cells": len(cells),
    }


def analyze(df: pd.DataFrame, n_boot: int = 2000) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows, delta_rows = [], []

    for (config, target), sub in df.groupby(["config", "target"]):
        variants = [v for v in ("original", "A", "B", "C") if v in set(sub["variant"])]
        cache = {}
        for variant in variants:
            runs = sub[sub["variant"] == variant].sort_values("seed")
            dur, evt, mean_rank = _seed_averaged_ranks(runs)
            cache[variant] = (dur, evt, mean_rank)

            per_seed_pooled = [_cindex(r["oof"]["duration"].to_numpy(float),
                                       r["oof"]["event"].to_numpy(float),
                                       r["oof"]["risk_score"].to_numpy(float))
                               for _, r in runs.iterrows()]
            summary_rows.append({
                "config": config, "target": target, "variant": variant,
                "n_seeds": len(runs),
                "fold_mean_avg": round(float(runs["fold_mean"].mean()), 4),
                "fold_mean_sd_across_seeds": round(float(runs["fold_mean"].std(ddof=0)), 4),
                "pooled_oof_avg": round(float(np.mean(per_seed_pooled)), 4),
                "pooled_oof_sd_across_seeds": round(float(np.std(per_seed_pooled)), 4),
                "pooled_oof_seed_averaged_rank": round(_cindex(dur, evt, mean_rank), 4),
                "per_seed_fold_means": [round(v, 4) for v in runs["fold_mean"]],
            })

        for ref, new, question in LADDER:
            if ref not in cache or new not in cache:
                continue
            dur, evt, risk_ref = cache[ref]
            _, _, risk_new = cache[new]
            pooled_boot = paired_bootstrap(dur, evt, risk_ref, risk_new, n_boot=n_boot)

            ref_runs = sub[sub["variant"] == ref].sort_values("seed")
            new_runs = sub[sub["variant"] == new].sort_values("seed")
            fold_boot = paired_bootstrap_fold_level(ref_runs, new_runs, n_boot=n_boot)

            per_fold_delta = (np.array([r for rr in new_runs["fold_cindices"] for r in rr])
                              - np.array([r for rr in ref_runs["fold_cindices"] for r in rr]))
            # fold_boot's own point estimate is recomputed straight from the raw OOF
            # risk scores (independent code path from the stored fold_cindices) --
            # this assert is the leakage/bug check the docstring promises.
            assert abs(fold_boot["delta_point"] - float(per_fold_delta.mean())) < 1e-6, (
                f"{config}/{target} {new}-{ref}: fold-level bootstrap point estimate "
                f"{fold_boot['delta_point']:.6f} != stored per-fold delta {per_fold_delta.mean():.6f}"
            )

            delta_rows.append({
                "config": config, "target": target, "comparison": f"{new} - {ref}", "question": question,
                # PRIMARY: fold-mean delta with a fold-stratified bootstrap CI -- every
                # concordance comparison inside this CI stays within one fold's own model.
                "delta_fold_mean": round(float(new_runs["fold_mean"].mean() - ref_runs["fold_mean"].mean()), 4),
                "ci95_fold_level": f"[{fold_boot['ci95_low']:+.4f}, {fold_boot['ci95_high']:+.4f}]",
                "ci_excludes_zero_fold_level": bool(fold_boot["ci95_low"] > 0 or fold_boot["ci95_high"] < 0),
                "p_boot_fold_level_gt_0": round(fold_boot["p_delta_gt_0"], 3),
                "delta_per_fold_sd": round(float(per_fold_delta.std(ddof=0)), 4),
                "n_fold_pairs_improved": int((per_fold_delta > 0).sum()),
                "n_fold_pairs": int(len(per_fold_delta)),
                # SECONDARY / diagnostic: pooled-OOF delta -- kept for continuity, but its
                # bootstrap mixes concordance comparisons across different folds' models
                # (see module docstring), so a disagreement with the fold-level CI above
                # is itself informative, not a bug to reconcile.
                "delta_pooled_oof": round(pooled_boot["delta_point"], 4),
                "ci95_pooled_oof": f"[{pooled_boot['ci95_low']:+.4f}, {pooled_boot['ci95_high']:+.4f}]",
                "ci_excludes_zero_pooled_oof": bool(pooled_boot["ci95_low"] > 0 or pooled_boot["ci95_high"] < 0),
            })

    return pd.DataFrame(summary_rows), pd.DataFrame(delta_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(OUT_DIR, "results.json"))
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()

    df = load_runs(args.results)
    summary, deltas = analyze(df, n_boot=args.n_boot)

    summary.to_csv(os.path.join(OUT_DIR, "summary_by_variant.csv"), index=False)
    deltas.to_csv(os.path.join(OUT_DIR, "paired_deltas.csv"), index=False)

    pd.set_option("display.width", 200)
    print("=== C-index by variant (averaged over seeds) ===")
    print(summary.to_string(index=False))
    print("\n=== Paired deltas with patient-level bootstrap 95% CI ===")
    print(deltas.to_string(index=False))
    print(f"\nwrote summary_by_variant.csv / paired_deltas.csv to {OUT_DIR}")


if __name__ == "__main__":
    main()
