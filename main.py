# -*- coding: utf-8 -*-
"""Entrypoint for the two tri-modal fusion experiments.

    python main.py --experiment early_fusion --mode smoke_test
    python main.py --experiment early_fusion --mode batch_smoke
    python main.py --experiment early_fusion --mode train --target all

    python main.py --experiment late_fusion  --mode smoke_test
    python main.py --experiment late_fusion  --mode batch_smoke
    python main.py --experiment late_fusion  --mode train --target all

- smoke_test: no torch required. Validates paths, cohort/manifest/split
  agreement, and that the report text corpus loads. Seconds.
- batch_smoke: 1 fold, 2 epochs -- confirms the real training pipeline runs
  end-to-end (dataset -> model -> loss -> backward -> optimizer step ->
  C-index) without committing to a full run. For late_fusion, the final
  weighted-sum combination step is skipped (matches clinical+report/main.py's
  precedent: with max_folds=1 the unimodal OOF risk scores don't cover the
  full cohort, so CoxPHFitter can't be fit on them yet).
- train: full 5-fold x configured-epochs run for OS and/or PFS.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
TARGETS = ("os", "pfs")


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(cfg: dict, key: str) -> str:
    return str((ROOT / cfg["paths"][key]).resolve())


def cmd_smoke_test(cfg: dict) -> None:
    """No-torch check: paths exist, cohort/manifest/split agree, corpus loads."""
    import cohort

    checks = []
    merged_csv = resolve(cfg, "merged_csv")
    image_dir = resolve(cfg, "image_dir")
    split_csv = resolve(cfg, "split_csv")

    checks.append(("merged_csv exists", os.path.exists(merged_csv)))
    checks.append(("image_dir exists", os.path.isdir(image_dir)))
    checks.append(("split_csv exists", os.path.exists(split_csv)))

    manifest = cohort.build_manifest(merged_csv, image_dir, split_csv)
    n_trimodal = int((manifest["cohort_name"] == "trimodal_common").sum())
    checks.append((f"tri-modal cohort n == 238 (actual {n_trimodal})", n_trimodal == 238))

    cohort_df = cohort.load_trimodal_cohort(merged_csv, split_csv)
    n_cohort = cohort_df["research_id"].nunique()
    checks.append((f"loaded cohort n == 238 (actual {n_cohort})", n_cohort == 238))

    for fold in sorted(cohort_df["fold"].unique()):
        fold_df = cohort_df[cohort_df["fold"] == fold]
        tr = set(fold_df.loc[fold_df["split"] == "train", "research_id"])
        va = set(fold_df.loc[fold_df["split"] == "val", "research_id"])
        te = set(fold_df.loc[fold_df["split"] == "test", "research_id"])
        overlap = (tr & va) | (tr & te) | (va & te)
        checks.append((f"fold {fold} train/val/test overlap == 0", len(overlap) == 0))
        checks.append((f"fold {fold} cohort union == {n_cohort}", len(tr | va | te) == n_cohort))

    import features
    corpus, _ = features.load_text_corpus(merged_csv)
    checks.append(("text corpus loads (has_report=1 patients)", len(corpus) > 0))

    ok = all(c[1] for c in checks)
    print()
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print(f"\nsmoke_test: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


def _experiment_id(experiment: str, mode: str) -> str:
    import datetime as dt
    date = dt.datetime.now().strftime("%Y%m%d")
    return f"EXP_{date}_{experiment}_{mode}"


def _write_artifacts(cfg, experiment_id, contract, fold_records_by_target, oof_by_target, extra_history_by_target=None):
    import reporting

    output_root = resolve(cfg, "output_dir")
    dirs = reporting.prepare_experiment_dir(output_root, experiment_id)
    reporting.write_resolved_config(dirs, contract)
    reporting.write_data_manifest(dirs)
    reporting.write_splits(dirs, resolve(cfg, "split_csv"))
    reporting.write_environment(dirs)

    summaries = {}
    for target, fold_records in fold_records_by_target.items():
        summaries[target] = {}
        by_modality: dict[str, list[dict]] = {}
        for row in fold_records:
            by_modality.setdefault(row["modality"], []).append(row)
        for modality, rows in by_modality.items():
            summaries[target][modality] = reporting.summarize(rows)

    all_fold_records = [row for rows in fold_records_by_target.values() for row in rows]
    all_oof = [row for rows in oof_by_target.values() for row in rows]
    reporting.write_metrics(dirs, all_fold_records, all_oof, summaries)
    reporting.write_experiment_report(dirs, contract, summaries)
    print(f"\nArtifacts written to: {dirs['root']}")
    return dirs, summaries


def cmd_early_fusion(cfg: dict, mode: str, target_arg: str) -> None:
    import torch
    from train import TrimodalEvaluator

    targets = TARGETS if target_arg == "all" else (target_arg,)
    max_folds = 1 if mode == "batch_smoke" else None
    epochs = 2 if mode == "batch_smoke" else int(cfg["training"]["epochs"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    experiment_id = _experiment_id("early_fusion", mode)
    fold_records_by_target, oof_by_target = {}, {}
    for target in targets:
        print(f"\n{'=' * 60}\n  early_fusion [{mode}] target={target}\n{'=' * 60}")
        ev = TrimodalEvaluator(
            target=target, merged_csv=resolve(cfg, "merged_csv"), image_dir=resolve(cfg, "image_dir"),
            split_csv=resolve(cfg, "split_csv"), resize=int(cfg["model"]["resize"]),
            gray_scale=bool(cfg["model"]["gray_scale"]),
            tfidf_max_features=int(cfg["model"]["tfidf_max_features"]),
            tfidf_ngram_range=tuple(cfg["model"]["tfidf_ngram_range"]),
            image_proj_dim=int(cfg["model"]["image_proj_dim"]), image_dropout=float(cfg["model"]["image_dropout"]),
            clinical_hidden=tuple(cfg["model"]["clinical_hidden"]), clinical_dropout=float(cfg["model"]["clinical_dropout"]),
            report_hidden=tuple(cfg["model"]["report_hidden"]), report_dropout=float(cfg["model"]["report_dropout"]),
            fusion_dropout=float(cfg["model"]["fusion_dropout"]),
            lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]),
            epochs=epochs, batch_size=int(cfg["training"]["batch_size"]),
            save_dir=str(ROOT / "outputs" / experiment_id / "checkpoints" / target),
            max_folds=max_folds, device=device, num_workers=int(cfg["training"]["num_workers"]),
            seed=int(cfg["reproducibility"]["seed"]),
        ).run()
        fold_records_by_target[target] = ev.fold_records
        oof_by_target[target] = ev.oof_predictions
        print(f"\n[early_fusion/{target}] mean C-index: {sum(ev.c_indices) / len(ev.c_indices):.4f}")

    contract = {
        "experiment_id": experiment_id,
        "experiment_name": "Tri-modal early (concat) fusion: image + clinical + report",
        "hypothesis": "Concatenating image, clinical, and report embeddings before a single Cox head "
                       "outperforms any single modality and (per earlyfusion.md) is expected to also "
                       "outperform late/weighted-sum fusion.",
        "primary_comparison": "early_fusion (this run) vs late_fusion (see EXP_*_late_fusion_*)",
        "baseline_result_source": "clinical+image (image+clinical, 257 cohort) and clinical+report "
                                    "(clinical+report, 238 cohort) unimodal/2-modal runs -- historical "
                                    "reference only, not a direct comparison (different cohorts; see "
                                    "SCLC_EXPERIMENT_PROTOCOL.md 4.4/13).",
        "intentional_change": "Add report (TF-IDF) modality on top of the validated image+clinical concat "
                               "architecture; fusion point is feature-level concat before the Cox head.",
        "fixed_conditions": "Image backbone (SimpleCNN, 512D), image projection (128D), clinical branch "
                             "(128x4, BN+Dropout0.5), report branch (TFIDF400->32->16, BN+Dropout0.3), "
                             "Adam lr=1e-4 wd=1e-4, batch=16, epochs=30, seed=42, 5-fold.",
        "unavoidable_changes": "Cohort is 238 (tri-modal common) vs 257 (image+clinical baseline) or "
                                "238 (report_common) individually -- same n as report_common by construction.",
        "modalities": ["image", "clinical", "report"],
        "cohort_rule": cfg["experiment"]["cohort_rule"],
        "manifest_path": "data_manifest.csv (this experiment folder)",
        "actual_n": 238,
        "target": list(targets),
        "split_file": "splits/trimodal_common_5fold_seed42_v1.csv",
        "split_method": "identical to clinical+report/report_common_5fold_seed42_v1.csv (StratifiedKFold on joint OS/PFS event)",
        "n_folds": 5,
        "seed": int(cfg["reproducibility"]["seed"]),
        "image_preprocessing": "grayscale, resize 512, per-train-fold mean/std normalize, random h-flip (train only)",
        "clinical_preprocessing": "21 features (8 standardized continuous + 13 categorical passthrough), StandardScaler fit on train fold only",
        "text_preprocessing": "char n-gram(2,4) TF-IDF, max_features=400, vocabulary fit on train fold only; dates/hospital names masked",
        "model": "TrimodalConcatDeepSurv (model.py)",
        "pretrained_or_scratch": "scratch (SimpleCNN has no pretrained weights)",
        "freeze_policy": "none -- all branches trained end-to-end",
        "loss": "Cox negative partial log-likelihood (train.cox_ph_loss)",
        "optimizer": "Adam",
        "learning_rate": float(cfg["training"]["learning_rate"]),
        "batch_size": int(cfg["training"]["batch_size"]),
        "epochs": epochs,
        "early_stopping": "none (best-val-C-index checkpoint selection, matching clinical+image/train.py)",
        "checkpoint_rule": "save state_dict whenever val C-index improves; reload best before test evaluation",
        "primary_metric": "C-index",
        "secondary_metrics": ["train/val C-index gap", "fold std", "OOF pooled C-index"],
        "output_dir": str(ROOT / "outputs"),
        "mode": mode,
    }
    _write_artifacts(cfg, contract["experiment_id"], contract, fold_records_by_target, oof_by_target)


def cmd_late_fusion(cfg: dict, mode: str, target_arg: str) -> None:
    import torch
    import late_fusion_3modal as lf
    import cohort

    targets = TARGETS if target_arg == "all" else (target_arg,)
    max_folds = 1 if mode == "batch_smoke" else None
    epochs = 2 if mode == "batch_smoke" else int(cfg["training"]["epochs"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cohort_df = cohort.load_trimodal_cohort(resolve(cfg, "merged_csv"), resolve(cfg, "split_csv"))

    experiment_id = _experiment_id("late_fusion", mode)
    fold_records_by_target, oof_by_target = {}, {}
    for target in targets:
        print(f"\n{'=' * 60}\n  late_fusion [{mode}] target={target}\n{'=' * 60}")
        image_res = lf.run_image_only(
            target, resolve(cfg, "merged_csv"), resolve(cfg, "image_dir"), resolve(cfg, "split_csv"),
            epochs=epochs, batch_size=int(cfg["training"]["batch_size"]), resize=int(cfg["model"]["resize"]),
            gray_scale=bool(cfg["model"]["gray_scale"]), lr=float(cfg["training"]["learning_rate"]),
            weight_decay=float(cfg["training"]["weight_decay"]),
            save_dir=str(ROOT / "outputs" / experiment_id / "checkpoints" / "image_only" / target),
            max_folds=max_folds, seed=int(cfg["reproducibility"]["seed"]),
            num_workers=int(cfg["training"]["num_workers"]), device=device,
        )
        clinical_res = lf.run_clinical_only(
            cohort_df, target, num_nodes=tuple(cfg["model"]["clinical_hidden"]), dropout=float(cfg["model"]["clinical_dropout"]),
            lr=float(cfg["training"]["learning_rate"]), epochs=epochs, batch_size=int(cfg["training"]["batch_size"]),
            max_folds=max_folds, seed=int(cfg["reproducibility"]["seed"]),
        )
        report_res = lf.run_report_only(
            cohort_df, target, resolve(cfg, "merged_csv"), num_nodes=tuple(cfg["model"]["report_hidden"]),
            dropout=float(cfg["model"]["report_dropout"]), tfidf_max_features=int(cfg["model"]["tfidf_max_features"]),
            tfidf_ngram_range=tuple(cfg["model"]["tfidf_ngram_range"]), lr=float(cfg["training"]["learning_rate"]),
            epochs=epochs, batch_size=int(cfg["training"]["batch_size"]), max_folds=max_folds,
            seed=int(cfg["reproducibility"]["seed"]),
        )

        fold_records = list(image_res["fold_records"]) + list(clinical_res["fold_records"]) + list(report_res["fold_records"])
        oof = list(image_res["oof_predictions"]) + list(clinical_res["oof_predictions"]) + list(report_res["oof_predictions"])

        if max_folds is None:
            combined = lf.combine_weighted_sum(
                cohort_df, target,
                lf._oof_lookup(image_res["oof_predictions"]),
                lf._oof_lookup(clinical_res["oof_predictions"]),
                lf._oof_lookup(report_res["oof_predictions"]),
                max_folds=max_folds,
            )
            fold_records += combined["fold_records"]
            oof += combined["oof_predictions"]
        else:
            print("  [skip] late_fusion weighted-sum combine: max_folds is set (batch_smoke) so the "
                  "unimodal OOF predictions don't cover the full cohort yet -- combine only runs in --mode train.")

        fold_records_by_target[target] = fold_records
        oof_by_target[target] = oof

    contract = {
        "experiment_id": experiment_id,
        "experiment_name": "Tri-modal late (out-level, weighted-sum) fusion: image + clinical + report",
        "hypothesis": "A learned linear combination of independently-trained unimodal OOF risk scores "
                       "(image-only, clinical-only, report-only) is a weaker fusion strategy than "
                       "early/concat fusion, consistent with clinical+report/earlyfusion.md's 2-modal finding.",
        "primary_comparison": "late_fusion (this run) vs early_fusion (see EXP_*_early_fusion_*)",
        "baseline_result_source": "clinical+report/CODE/train_early_fusion_clinical_report.py's 2-modal "
                                    "late-fusion result (OS 0.5907, PFS 0.5496) -- historical reference only.",
        "intentional_change": "Fusion point: output-level weighted sum (lifelines CoxPHFitter on 3 OOF risk "
                               "scores) instead of feature-level concat.",
        "fixed_conditions": "Same cohort/split/seed as early_fusion; same per-modality encoder shapes "
                             "(image SimpleCNN 512D, clinical 128x4 BN+Dropout0.5, report TFIDF400->32->16 "
                             "BN+Dropout0.3) trained standalone per modality; same Cox loss/optimizer/epochs "
                             "for each unimodal arm.",
        "unavoidable_changes": "Each unimodal arm has its own risk-score scale before combination; the "
                                "combiner is fit fold-wise on OOF risk (not on raw features).",
        "modalities": ["image", "clinical", "report"],
        "cohort_rule": cfg["experiment"]["cohort_rule"],
        "manifest_path": "data_manifest.csv (this experiment folder)",
        "actual_n": 238,
        "target": list(targets),
        "split_file": "splits/trimodal_common_5fold_seed42_v1.csv",
        "split_method": "identical to clinical+report/report_common_5fold_seed42_v1.csv (StratifiedKFold on joint OS/PFS event)",
        "n_folds": 5,
        "seed": int(cfg["reproducibility"]["seed"]),
        "image_preprocessing": "grayscale, resize 512, per-train-fold mean/std normalize, random h-flip (train only)",
        "clinical_preprocessing": "21 features (8 standardized continuous + 13 categorical passthrough), StandardScaler fit on train fold only",
        "text_preprocessing": "char n-gram(2,4) TF-IDF, max_features=400, vocabulary fit on train fold only; dates/hospital names masked",
        "model": "ImageOnlyDeepSurv + generate_net(clinical) + generate_net(report) + lifelines CoxPHFitter combiner",
        "pretrained_or_scratch": "scratch",
        "freeze_policy": "none; each unimodal arm trained independently, combiner fit on frozen OOF risk scores",
        "loss": "Cox negative partial log-likelihood per unimodal arm; CoxPHFitter partial likelihood for the combiner",
        "optimizer": "Adam (unimodal arms); Newton-Raphson (lifelines CoxPHFitter combiner)",
        "learning_rate": float(cfg["training"]["learning_rate"]),
        "batch_size": int(cfg["training"]["batch_size"]),
        "epochs": epochs,
        "early_stopping": "none (image-only: best-val-C-index checkpoint; clinical/report-only: pycox CoxPH default)",
        "checkpoint_rule": "image-only: save state_dict whenever val C-index improves. clinical/report-only: in-memory pycox model.",
        "primary_metric": "C-index",
        "secondary_metrics": ["per-modality unimodal C-index", "learned combination coefficients"],
        "output_dir": str(ROOT / "outputs"),
        "mode": mode,
    }
    _write_artifacts(cfg, contract["experiment_id"], contract, fold_records_by_target, oof_by_target)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Tri-modal (image+clinical+report) fusion experiments")
    ap.add_argument("--experiment", choices=("early_fusion", "late_fusion"), required=True)
    ap.add_argument("--mode", choices=("smoke_test", "batch_smoke", "train"), default="smoke_test")
    ap.add_argument("--target", choices=("os", "pfs", "all"), default="all")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    if args.mode == "smoke_test":
        cmd_smoke_test(cfg)
        return 0

    if args.experiment == "early_fusion":
        cmd_early_fusion(cfg, args.mode, args.target)
    else:
        cmd_late_fusion(cfg, args.mode, args.target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
