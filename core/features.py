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


TEXT_SOURCES = ("concl_find", "concl_only", "find_only")


def load_text_corpus(merged_csv: str, source: str = "concl_find") -> tuple[dict[int, str], dict]:
    """{research_id: masked_text} for every has_report=1 patient.

    Conclusion first, then finding (report_recommendation excluded — mostly
    boilerplate). See source ``text_dataset.py`` docstring for rationale.

    ``source`` 는 판독지의 **어느 부분**을 텍스트로 쓸지 고르는 스위치다
    ("리포트의 어디에 신호가 있나?" 실험용, exp_text_source.py 참고).
      - "concl_find" : conclusion + finding (기존 동작, **기본값**)
      - "concl_only" : conclusion 만
      - "find_only"  : finding 만
    기본값이 기존 동작과 글자 단위로 동일해야 기존 결과가 그대로 재현된다
    (concl_find 분기는 예전 코드의 f-string/strip/mask 순서를 그대로 유지).
    마스킹(날짜·병원명 삭제)은 어느 source든 항상 적용한다.
    """
    import csv

    if source not in TEXT_SOURCES:
        raise ValueError(f"unknown text source {source!r}; expected one of {list(TEXT_SOURCES)}")

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
            if source == "concl_find":
                raw = f"{conclusion}\n{finding}"
            elif source == "concl_only":
                raw = conclusion
            else:  # "find_only"
                raw = finding
            combined = raw.strip()
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
    extra_numeric_fn=None, text_encoder_fn=None,
) -> tuple[dict[str, np.ndarray], int, int]:
    """Builds one combined [clinical | extra | tfidf] feature matrix per split,
    fit on the train fold only. Layout: columns [0:clinical_dim) = clinical
    (+ optional extra numeric block appended right after it),
    [clinical_dim:clinical_dim+tfidf_dim) = TF-IDF. The fusion model splits
    the combined tensor back into two branches in ``forward()`` — this lets
    the existing image+tabular ``PetSurvivalDataset`` plumbing carry both
    modalities through a single ``tabular`` slot unmodified.

    ``extra_numeric_fn`` (optional, default None = byte-identical to the
    previous behaviour) is a callable ``(train_ids, val_ids, test_ids) ->
    {'train': X, 'val': X, 'test': X}`` supplying extra numeric columns
    (e.g. SUVmax parsed out of the report text, see ``suv_features.py``).
    It is called with the **train ids first** and, exactly like
    ``ClinicalEncoder``/``TfidfEncoder`` above, must fit every cross-patient
    statistic it needs (imputation medians, scaling) on the train fold only.
    The block is placed in the clinical half of the tensor, so the returned
    first dim is ``clinical_dim + n_extra`` and the extra columns are consumed
    by the clinical branch — no model change needed.

    ``text_encoder_fn`` (optional, default None = byte-identical to the
    previous behaviour) 는 판독지 텍스트 블록을 **TF-IDF 대신** 만들어 주는
    콜러블이다 (``(train_ids, val_ids, test_ids) -> {'train': X, 'val': X,
    'test': X}``). frozen BERT 임베딩을 쓰는 실험용 훅 (``bert_features.py``,
    ``exp_bert_text.py`` 참고). None 이면 기존 TF-IDF 경로가 그대로 돈다.
    ``extra_numeric_fn`` 과 동일한 규율이 적용된다 — 환자를 가로질러 계산되는
    통계(SVD 기저, StandardScaler)는 전부 **train fold 환자만**으로 fit 해야 한다.
    이 훅이 주어지면 TF-IDF 는 아예 계산되지 않으므로 ``tfidf_max_features``/
    ``tfidf_ngram_range`` 는 무시되고, 반환되는 세 번째 값(report 브랜치 차원)은
    이 블록의 폭이 된다. 첫 번째 반환값(clinical_dim)은 영향받지 않는다.

    Returns ``({'train': X, 'val': X, 'test': X}, clinical_dim(+extra), text_dim)``
    (``text_dim`` 은 기본 경로에서는 ``tfidf_dim`` 과 같다).
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

    extra_dim = 0
    extra = {"train": None, "val": None, "test": None}
    if extra_numeric_fn is not None:
        extra = extra_numeric_fn(train_df.index, val_df.index, test_df.index)
        extra_dim = int(extra["train"].shape[1])

    if text_encoder_fn is None:
        tfidf_enc = TfidfEncoder(max_features=tfidf_max_features, ngram_range=tfidf_ngram_range)
        texts_train = [corpus.get(int(rid), "") for rid in train_df.index]
        tfidf_train = tfidf_enc.fit_transform(texts_train)
        texts_val = [corpus.get(int(rid), "") for rid in val_df.index]
        tfidf_val = tfidf_enc.transform(texts_val) if len(val_df) else np.empty((0, tfidf_enc.max_features), dtype="float32")
        texts_test = [corpus.get(int(rid), "") for rid in test_df.index]
        tfidf_test = tfidf_enc.transform(texts_test) if len(test_df) else np.empty((0, tfidf_enc.max_features), dtype="float32")
        text_blocks = {"train": tfidf_train, "val": tfidf_val, "test": tfidf_test}
        text_dim = tfidf_enc.max_features
    else:
        text_blocks = text_encoder_fn(train_df.index, val_df.index, test_df.index)
        text_blocks = {k: np.asarray(v, dtype="float32") for k, v in text_blocks.items()}
        text_dim = int(text_blocks["train"].shape[1])

    def _stack(split, clin, text):
        blocks = [clin] if extra[split] is None else [clin, extra[split]]
        return np.concatenate(blocks + [text], axis=1).astype("float32")

    combined = {
        "train": _stack("train", clin_train, text_blocks["train"]),
        "val": _stack("val", clin_val, text_blocks["val"]),
        "test": _stack("test", clin_test, text_blocks["test"]),
    }
    return combined, clin_enc.dim + extra_dim, text_dim
