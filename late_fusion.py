# -*- coding: utf-8 -*-
"""Late (out-level) fusion: train 3 independent unimodal models on the same
238-patient / 5-fold tri-modal cohort, then learn a linear combination
(weighted sum) of their fold-wise out-of-fold risk scores.

- image-only:    train.ImageOnlyEvaluator (custom Cox-loss training loop over
                  DataLoader, since image loading can't go through pycox).
- clinical-only: pycox CoxPH + torchtuples MLPVanilla, num_nodes=[128]*4,
                  dropout=0.5 -- same 128x4 capacity validated in
                  clinical+report/earlyfusion.md's clinical-only ablation.
- report-only:   pycox CoxPH + torchtuples MLPVanilla, num_nodes=[32,16],
                  dropout=0.3 -- matches the report branch used in early fusion.

The 3 fold-wise OOF risk scores are combined with lifelines' CoxPHFitter
(3 covariates -> 3 learned coefficients = the "weighted sum"), extending
clinical+report/main.py's `_run_late_fusion` (2 covariates) to 3 modalities.
This is the same late-fusion pattern earlyfusion.md compared early fusion
against for the 2-modal case.
"""
import pandas as pd
import torch.optim as optim
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

import features
from model import generate_net, get_cox_ph_model
from train import ImageOnlyEvaluator, _seed_everything


def _labels_by_id(cohort_df: pd.DataFrame, target: str) -> pd.DataFrame:
    return cohort_df.drop_duplicates("research_id").set_index("research_id")[[f"{target}_days", f"{target}_event"]]


def run_clinical_only(cohort_df: pd.DataFrame, target: str, num_nodes=(128, 128, 128, 128), dropout=0.5,
                       lr=1e-4, epochs=30, batch_size=16, max_folds=None, seed=42) -> dict:
    """Same TabularSplitEvaluator-style K-fold Cox training as
    clinical+report/CODE/common_eval.py, restricted to the tri-modal cohort."""
    from train import _fold_plan

    clinical_frame = cohort_df.drop_duplicates("research_id").set_index("research_id")
    standardize_cols, categorical_cols = features.resolve_clinical_columns(clinical_frame)
    labels = _labels_by_id(cohort_df, target)

    fold_records, oof = [], []
    for fold, ids in _fold_plan(cohort_df, max_folds=max_folds):
        _seed_everything(seed + fold)
        enc = features.ClinicalEncoder(clinical_frame, standardize_cols, categorical_cols)
        x_train = enc.fit_transform(ids["train"])
        x_val = enc.transform(ids["val"])
        x_test = enc.transform(ids["test"])

        y_train = (labels.loc[ids["train"], f"{target}_days"].to_numpy("float32"), labels.loc[ids["train"], f"{target}_event"].to_numpy("float32"))
        y_val = (labels.loc[ids["val"], f"{target}_days"].to_numpy("float32"), labels.loc[ids["val"], f"{target}_event"].to_numpy("float32"))
        y_test_dur = labels.loc[ids["test"], f"{target}_days"].to_numpy("float32")
        y_test_evt = labels.loc[ids["test"], f"{target}_event"].to_numpy("float32")

        net = generate_net(in_features=x_train.shape[1], num_nodes=list(num_nodes), dropout=dropout)
        model = get_cox_ph_model(net, optim.Adam, lr)
        model.fit(x_train, y_train, batch_size, epochs, verbose=False, val_data=(x_val, y_val))

        risk_test = model.predict(x_test).flatten()
        ci = concordance_index(y_test_dur, -risk_test, y_test_evt)
        fold_records.append({"target": target, "modality": "clinical_only", "fold": fold, "c_index": float(ci), "n": len(ids["test"])})
        oof.extend({"research_id": rid, "target": target, "modality": "clinical_only", "fold": fold,
                     "duration": float(d), "event": float(e), "risk_score": float(r)}
                    for rid, d, e, r in zip(ids["test"], y_test_dur, y_test_evt, risk_test))
        print(f"[late_fusion/clinical_only/{target}] fold {fold}: C-index={ci:.4f}")

    return {"fold_records": fold_records, "oof_predictions": oof}


def run_report_only(cohort_df: pd.DataFrame, target: str, merged_csv: str, num_nodes=(32, 16), dropout=0.3,
                     tfidf_max_features=400, tfidf_ngram_range=(2, 4),
                     lr=1e-4, epochs=30, batch_size=16, max_folds=None, seed=42) -> dict:
    """Same shape as clinical+report/CODE/train_early_fusion_clinical_report.py's
    TF-IDF branch, trained standalone via pycox CoxPH."""
    from train import _fold_plan

    corpus, _ = features.load_text_corpus(merged_csv)
    labels = _labels_by_id(cohort_df, target)

    fold_records, oof = [], []
    for fold, ids in _fold_plan(cohort_df, max_folds=max_folds):
        _seed_everything(seed + fold)
        enc = features.TfidfEncoder(max_features=tfidf_max_features, ngram_range=tfidf_ngram_range)
        x_train = enc.fit_transform([corpus.get(rid, "") for rid in ids["train"]])
        x_val = enc.transform([corpus.get(rid, "") for rid in ids["val"]])
        x_test = enc.transform([corpus.get(rid, "") for rid in ids["test"]])

        y_train = (labels.loc[ids["train"], f"{target}_days"].to_numpy("float32"), labels.loc[ids["train"], f"{target}_event"].to_numpy("float32"))
        y_val = (labels.loc[ids["val"], f"{target}_days"].to_numpy("float32"), labels.loc[ids["val"], f"{target}_event"].to_numpy("float32"))
        y_test_dur = labels.loc[ids["test"], f"{target}_days"].to_numpy("float32")
        y_test_evt = labels.loc[ids["test"], f"{target}_event"].to_numpy("float32")

        net = generate_net(in_features=x_train.shape[1], num_nodes=list(num_nodes), dropout=dropout)
        model = get_cox_ph_model(net, optim.Adam, lr)
        model.fit(x_train, y_train, batch_size, epochs, verbose=False, val_data=(x_val, y_val))

        risk_test = model.predict(x_test).flatten()
        ci = concordance_index(y_test_dur, -risk_test, y_test_evt)
        fold_records.append({"target": target, "modality": "report_only", "fold": fold, "c_index": float(ci), "n": len(ids["test"])})
        oof.extend({"research_id": rid, "target": target, "modality": "report_only", "fold": fold,
                     "duration": float(d), "event": float(e), "risk_score": float(r)}
                    for rid, d, e, r in zip(ids["test"], y_test_dur, y_test_evt, risk_test))
        print(f"[late_fusion/report_only/{target}] fold {fold}: C-index={ci:.4f}")

    return {"fold_records": fold_records, "oof_predictions": oof}


def run_image_only(target: str, merged_csv, image_dir, split_csv, epochs=30, batch_size=16,
                    resize=512, gray_scale=True, lr=1e-4, weight_decay=1e-4,
                    save_dir="checkpoints/late_fusion_image_only", max_folds=None, seed=42,
                    num_workers=4, device=None) -> dict:
    ev = ImageOnlyEvaluator(
        target=target, merged_csv=merged_csv, image_dir=image_dir, split_csv=split_csv,
        resize=resize, gray_scale=gray_scale, lr=lr, weight_decay=weight_decay,
        epochs=epochs, batch_size=batch_size, save_dir=save_dir, max_folds=max_folds,
        device=device, num_workers=num_workers, seed=seed,
    ).run()
    return {"fold_records": ev.fold_records, "oof_predictions": ev.oof_predictions, "training_history": ev.training_history}


def _oof_lookup(oof_predictions: list[dict]) -> dict[int, float]:
    return {row["research_id"]: row["risk_score"] for row in oof_predictions}


def combine_weighted_sum(cohort_df: pd.DataFrame, target: str, image_risk: dict, clinical_risk: dict,
                          report_risk: dict, max_folds: int | None = None) -> dict:
    """Learns a 3-covariate linear combination (lifelines CoxPHFitter) of the
    three unimodal OOF risk scores per fold -- the "weighted sum" out-level
    fusion, extending clinical+report/main.py's 2-covariate `_run_late_fusion`."""
    from train import _fold_plan

    labels = _labels_by_id(cohort_df, target)

    def _frame(ids):
        return pd.DataFrame({
            "risk_image": [image_risk[i] for i in ids],
            "risk_clinical": [clinical_risk[i] for i in ids],
            "risk_report": [report_risk[i] for i in ids],
            "duration": labels.loc[ids, f"{target}_days"].to_numpy(),
            "event": labels.loc[ids, f"{target}_event"].to_numpy(),
        })

    fold_records, oof = [], []
    for fold, ids in _fold_plan(cohort_df, max_folds=max_folds):
        train_df = _frame(ids["train"])
        cph = CoxPHFitter()
        cph.fit(train_df, duration_col="duration", event_col="event")

        def _risk(frame):
            return cph.predict_partial_hazard(frame[["risk_image", "risk_clinical", "risk_report"]]).to_numpy()

        test_df = _frame(ids["test"])
        test_risk = _risk(test_df)
        ci = concordance_index(test_df["duration"], -test_risk, test_df["event"])
        coefs = {name: float(cph.params_[name]) for name in ("risk_image", "risk_clinical", "risk_report")}
        fold_records.append({"target": target, "modality": "late_fusion_weighted_sum", "fold": fold,
                              "c_index": float(ci), "n": len(ids["test"]), "coefficients": coefs})
        oof.extend({"research_id": rid, "target": target, "modality": "late_fusion_weighted_sum", "fold": fold,
                     "duration": float(d), "event": int(e), "risk_score": float(r)}
                    for rid, d, e, r in zip(ids["test"], test_df["duration"], test_df["event"], test_risk))
        print(f"[late_fusion/combined/{target}] fold {fold}: C-index={ci:.4f} "
              f"coef(image,clinical,report)=({coefs['risk_image']:.3f},{coefs['risk_clinical']:.3f},{coefs['risk_report']:.3f})")

    return {"fold_records": fold_records, "oof_predictions": oof}
