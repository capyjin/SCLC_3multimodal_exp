# -*- coding: utf-8 -*-
"""채택 모델의 연속형 임상변수 7개를 **훈련 fold 중앙값**으로 다시 대치하는 블록.

[무엇을 고치나]
  ``data/merged_tabular_with_reports.csv`` 의 연속형 임상변수 중 7개는 원본
  엑셀에 결측이 있는데, CSV 에는 이미 **전체 코호트 중앙값**으로 채워져 있다
  (238명 코호트 기준 결측: 흡연량 14 · Hb 3 · WBC 3 · LDH 39 · FVC 27 ·
  FEV1 26 · DLCO 52). 그 중앙값에는 test fold 환자도 들어가 있으므로 fold
  바깥 정보가 학습에 섞인다(미세한 누수).

  실험7 의 ``fold_safe_features`` 는 이 중 5개(LDH/WBC/FVC/FEV1/DLCO)만 다뤘다.
  Hb 와 흡연량도 같은 방식으로 채워져 있었으므로 여기서는 7개를 전부 다룬다.

[흡연량(pack-years)의 특수 규칙]
  원본 엑셀에서 흡연량이 비어 있는 33명 중 19명은 비흡연자(흡연유무=3)다.
  CSV 는 이들을 0 으로 뒀는데, 이건 대치가 아니라 **정의상 0** 이므로 관측값으로
  취급한다. 나머지 14명만 결측으로 보고 대치한다.

[fold 규율] 실험7 과 같다 — 중앙값과 StandardScaler 를 **train fold 환자만으로**
  적합하고 val/test 는 transform 만 한다. fold 마다 감사 기록(중앙값·대치 건수)을
  ``audit`` 에 남긴다.

[열 개수 고정] ``reduced_clinical_columns`` 가 CSV 의 대치된 7열을 임상 블록에서
  빼고, ``make_extra_numeric_fn`` 이 같은 7개를 fold-safe 로 다시 붙인다.
  입력 폭은 21 로 그대로다 (1 연속 + 13 범주 + 7 fold-safe 연속).
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from sclc import cohort, features

#: 원본 엑셀 열 -> CSV 의 (전체 중앙값으로 대치된) 열
RAW_TO_CSV = {
    "흡연력 ": "pack_years",
    "Hb": "hb",
    "WBC": "wbc",
    "LDH": "ldh",
    "FVC\nPre%ref": "fvc_pre_percent_ref",
    "FEV1\nPre%ref": "fev1_pre_percent_ref",
    "DLCOAdj\nPre%ref": "dlcoadj_pre_percent_ref",
}
PREIMPUTED_COLUMNS = tuple(RAW_TO_CSV.values())
SMOKING_COL = "흡연유무\n1. Ex\n2. current \n3. Non"
NEVER_SMOKER = 3


def load_raw_values(source_xlsx: str = cohort.DEFAULT_SOURCE_XLSX,
                    merged_csv: str = cohort.DEFAULT_MERGED_CSV) -> pd.DataFrame:
    """research_id 인덱스, CSV 열 이름으로 된 원본 관측값 (결측은 NaN 그대로).

    관측된 칸이 CSV 값과 한 칸이라도 다르면 죽는다 — 연결(research_id)이나
    열 대응이 틀렸다는 뜻이기 때문이다.
    """
    src = pd.read_excel(source_xlsx)
    src["research_id"] = pd.to_numeric(src["연구번호"], errors="coerce")
    src = src.dropna(subset=["research_id"])
    src["research_id"] = src["research_id"].astype(int)
    if src["research_id"].duplicated().any():
        raise ValueError("duplicate research_id in source excel")
    src = src.set_index("research_id")

    raw = pd.DataFrame(index=src.index)
    for xl_col, csv_col in RAW_TO_CSV.items():
        raw[csv_col] = pd.to_numeric(src[xl_col], errors="coerce")
    never = pd.to_numeric(src[SMOKING_COL], errors="coerce").eq(NEVER_SMOKER)
    raw.loc[never & raw["pack_years"].isna(), "pack_years"] = 0.0

    merged = cohort.load_merged_clinical(merged_csv).set_index("research_id")
    common = raw.index.intersection(merged.index)
    for col in PREIMPUTED_COLUMNS:
        obs = raw.loc[common, col].notna()
        a = raw.loc[common, col][obs].to_numpy(float)
        b = merged.loc[common, col][obs].to_numpy(float)
        if not np.allclose(a, b):
            n_bad = int((~np.isclose(a, b)).sum())
            raise ValueError(f"{col}: {n_bad} observed values differ between excel and merged CSV")
    return raw


def reduced_clinical_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """``features.resolve_clinical_columns`` 에서 대치된 7열만 뺀 목록."""
    standardize, categorical = features.resolve_clinical_columns(df)
    dropped = [c for c in standardize if c in PREIMPUTED_COLUMNS]
    if len(dropped) != len(PREIMPUTED_COLUMNS):
        raise ValueError(f"expected to drop {list(PREIMPUTED_COLUMNS)}, dropped only {dropped}")
    return [c for c in standardize if c not in PREIMPUTED_COLUMNS], categorical


def make_extra_numeric_fn(raw: pd.DataFrame, audit: list | None = None):
    """``TrimodalEvaluator(extra_numeric_fn=...)`` 용 fold-safe 블록 생성기."""
    audit = [] if audit is None else audit
    cols = list(PREIMPUTED_COLUMNS)

    def fn(train_ids, val_ids, test_ids):
        ids = {"train": [int(i) for i in train_ids],
               "val": [int(i) for i in val_ids],
               "test": [int(i) for i in test_ids]}
        arrs = {k: raw.reindex(v)[cols].to_numpy(float) for k, v in ids.items()}
        medians = np.nanmedian(arrs["train"], axis=0)
        filled = {k: np.where(np.isnan(a), medians, a) for k, a in arrs.items()}
        scaler = StandardScaler().fit(filled["train"])
        out = {k: (scaler.transform(a).astype("float32") if len(a)
                   else np.empty((0, len(cols)), dtype="float32")) for k, a in filled.items()}
        audit.append({
            "train_medians": {c: float(m) for c, m in zip(cols, medians)},
            "n_imputed": {k: {c: int(np.isnan(a[:, j]).sum()) for j, c in enumerate(cols)}
                          for k, a in arrs.items()},
            "n": {k: len(v) for k, v in ids.items()},
        })
        return out

    return fn
