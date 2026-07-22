# -*- coding: utf-8 -*-
"""Clinical(tabular) and report(text) feature encoders.

Column lists and ``ClinicalEncoder`` are copied verbatim from
``clinical+report/SCLC_report_unimodal_test-main/CODE/clinical_features.py``
(itself copied from ``clinical+image/SCLC_simple_CNN-main/dataset.py``) — both
source projects used the identical 21-column clinical schema, so there is
nothing to reconcile here.

``TfidfEncoder``/``load_text_corpus``/``mask_text`` are copied verbatim from
``clinical+report/SCLC_report_unimodal_test-main/CODE/text_dataset.py``.

All encoders must be ``fit()`` on the training fold only — validation/test
folds only ever call ``transform()`` — to avoid leakage (SCLC_EXPERIMENT_PROTOCOL.md 8.2/8.3).
"""
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# --- clinical column schema (identical in both source projects) -----------------
DEFAULT_STANDARDIZE_COLUMNS = [
    "birth_year", "age_at_diagnosis", "pack_years", "hb", "hct", "wbc", "plt",
    "neutrophil_percent", "lymphocyte_percent", "mpv", "ldh",
    "fvc_pre_meas", "fvc_pre_percent_ref", "fev1_pre_meas", "fev1_pre_percent_ref",
    "fev1_fvc_pre_meas", "dlcoadj_pre_percent_ref", "dlco_va_pre_percent_ref",
    "tumor_x", "horizontal_dist", "asymmetry_area", "mean_intensity",
    "asymmetry_index", "dose", "rt",
]
DEFAULT_CATEGORICAL_COLUMNS = [
    "gender", "ecog", "mmrc", "smoking_status", "dm", "htn", "tb", "lung_disease",
    "stage", "liver_meta", "brain_meta", "brain_metastasis_timing", "tumor_location",
    "horizontal_distance_category", "asymmetry_value", "atezolizumab", "rt_type",
]


def resolve_clinical_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Only columns actually present in the merged CSV are used (21 of them:
    8 continuous + 13 categorical for ``merged_tabular_with_reports.csv``)."""
    standardize = [c for c in DEFAULT_STANDARDIZE_COLUMNS if c in df.columns]
    categorical = [c for c in DEFAULT_CATEGORICAL_COLUMNS if c in df.columns]
    return standardize, categorical


class ClinicalEncoder:
    """Fit on the train fold's research_ids, transform any patient list."""

    def __init__(self, df: pd.DataFrame, standardize_columns: list[str], categorical_columns: list[str]):
        self.df = df
        self.standardize_columns = standardize_columns
        self.categorical_columns = categorical_columns
        self.scaler = StandardScaler()
        self._fitted = False

    @property
    def dim(self) -> int:
        return len(self.standardize_columns) + len(self.categorical_columns)

    def fit(self, train_ids):
        self.scaler.fit(self.df.loc[train_ids, self.standardize_columns])
        self._fitted = True
        return self

    def transform(self, ids) -> np.ndarray:
        assert self._fitted, "fit() must be called on the train fold first"
        std = self.scaler.transform(self.df.loc[ids, self.standardize_columns]).astype("float32")
        cat = self.df.loc[ids, self.categorical_columns].to_numpy(dtype="float32")
        return np.concatenate([std, cat], axis=1)

    def fit_transform(self, train_ids) -> np.ndarray:
        self.fit(train_ids)
        return self.transform(train_ids)


# --- report text: masking + TF-IDF ------------------------------------------------
DATE_PATTERNS = [
    re.compile(r"\d{4}[-.\s/]\d{1,2}[-.\s/]\d{1,2}"),
    re.compile(r"\d{4}년\s*\d{1,2}월\s*\d{1,2}일"),
    re.compile(r"\b20\d{6}\b"),
]
HOSPITAL_PATTERN = re.compile(
    r"[가-힣A-Za-z0-9]{1,20}(?:대학교병원|대학병원|의료원|병원|메디컬센터)"
)


def mask_text(text):
    if not text:
        return text
    masked = text
    for pat in DATE_PATTERNS:
        masked = pat.sub("[DATE]", masked)
    masked = HOSPITAL_PATTERN.sub("[HOSPITAL]", masked)
    return masked


def load_text_corpus(merged_csv: str) -> tuple[dict[int, str], dict]:
    """{research_id: masked_text} for every has_report=1 patient.

    Conclusion first, then finding (report_recommendation excluded — mostly
    boilerplate). See source ``text_dataset.py`` docstring for rationale.
    """
    import csv

    out = {}
    stats = {"n": 0, "total_chars": 0}
    required = {"research_id", "has_report", "report_finding", "report_conclusion"}
    with open(merged_csv, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"merged CSV missing columns: {sorted(missing)}")
        for row in reader:
            if int(row["has_report"] or 0) != 1:
                continue
            rid = int(row["research_id"])
            finding = row["report_finding"] or ""
            conclusion = row["report_conclusion"] or ""
            combined = f"{conclusion}\n{finding}".strip()
            masked = mask_text(combined)
            out[rid] = masked
            stats["n"] += 1
            stats["total_chars"] += len(masked)
    return out, stats


class TfidfEncoder:
    """Char n-gram TF-IDF -> dense array. Fit vocabulary on train-fold texts only."""

    def __init__(self, max_features=400, ngram_range=(2, 4), min_df=1, max_df=1.0):
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=tuple(ngram_range), max_features=max_features,
            min_df=min_df, max_df=max_df,
        )
        self._fitted = False
        self.max_features = int(max_features)

    def fit(self, train_texts):
        self.vectorizer.fit(train_texts)
        self._fitted = True
        return self

    def transform(self, texts) -> np.ndarray:
        assert self._fitted, "call fit() on train-fold texts first"
        dense = self.vectorizer.transform(texts).toarray().astype("float32")
        if dense.shape[1] < self.max_features:
            # Zero-pad so every fold's tabular tensor has the same fixed width
            # even when the train-fold vocabulary comes in under max_features
            # (small folds can have fewer distinct n-grams than the cap).
            pad = np.zeros((dense.shape[0], self.max_features - dense.shape[1]), dtype="float32")
            dense = np.concatenate([dense, pad], axis=1)
        return dense

    def fit_transform(self, train_texts) -> np.ndarray:
        self.fit(train_texts)
        return self.transform(train_texts)


def build_fold_multimodal_tabular(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
    corpus: dict[int, str], clinical_standardize_cols, clinical_categorical_cols,
    tfidf_max_features: int = 400, tfidf_ngram_range=(2, 4),
) -> tuple[dict[str, np.ndarray], int, int]:
    """Builds one combined [clinical | tfidf] feature matrix per split, fit on
    the train fold only. Layout: columns [0:clinical_dim) = clinical,
    [clinical_dim:clinical_dim+tfidf_dim) = TF-IDF. The fusion model splits
    the combined tensor back into two branches in ``forward()`` — this lets
    the existing image+tabular ``PetSurvivalDataset`` plumbing carry both
    modalities through a single ``tabular`` slot unmodified.

    Returns ``({'train': X, 'val': X, 'test': X}, clinical_dim, tfidf_dim)``.
    """
    # ClinicalEncoder.transform() looks rows up via self.df.loc[ids]; it must
    # be able to see all three splits' research_ids even though it only
    # fits (StandardScaler) on the train fold, or val/test transform() would
    # KeyError on ids that aren't in train_df.
    full_df = pd.concat([train_df, val_df, test_df])
    clin_enc = ClinicalEncoder(full_df, clinical_standardize_cols, clinical_categorical_cols)
    clin_train = clin_enc.fit_transform(train_df.index)
    clin_val = clin_enc.transform(val_df.index) if len(val_df) else np.empty((0, clin_enc.dim), dtype="float32")
    clin_test = clin_enc.transform(test_df.index) if len(test_df) else np.empty((0, clin_enc.dim), dtype="float32")

    tfidf_enc = TfidfEncoder(max_features=tfidf_max_features, ngram_range=tfidf_ngram_range)
    texts_train = [corpus.get(int(rid), "") for rid in train_df.index]
    tfidf_train = tfidf_enc.fit_transform(texts_train)
    texts_val = [corpus.get(int(rid), "") for rid in val_df.index]
    tfidf_val = tfidf_enc.transform(texts_val) if len(val_df) else np.empty((0, tfidf_enc.max_features), dtype="float32")
    texts_test = [corpus.get(int(rid), "") for rid in test_df.index]
    tfidf_test = tfidf_enc.transform(texts_test) if len(test_df) else np.empty((0, tfidf_enc.max_features), dtype="float32")

    combined = {
        "train": np.concatenate([clin_train, tfidf_train], axis=1).astype("float32"),
        "val": np.concatenate([clin_val, tfidf_val], axis=1).astype("float32"),
        "test": np.concatenate([clin_test, tfidf_test], axis=1).astype("float32"),
    }
    return combined, clin_enc.dim, tfidf_enc.max_features
