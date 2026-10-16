# -*- coding: utf-8 -*-
"""[실험16-0] GMU nested CV 학습 전 데이터 정합성 감사 — 학습 없음, 읽기 전용.

project.md 2.3 과 2026-10-14 점검 요청(D6·D12 후속)을 재현 가능한 스크립트로 남긴다.

  1. Clinical 21개 입력에 실제 시행 치료 변수(rt_type, atezolizumab)가 들어 있는가
  2. lung_disease 를 원본 엑셀 범주로 재코딩하면 몇 명의 값이 바뀌는가
  3. 판독지 원문에서 PET 검사일·검사 목적·선행 치료(S/P ...) 기록을 뽑아
     진단일·PFS/OS 사건일과 대조 (촬영 전에 사건 또는 치료가 있었던 환자)
  4. 진단일 기준 vs PET 검사일 기준 생존시간을 쓸 때 Harrell C-index 의 비교 가능 쌍이
     얼마나 달라지는가 (endpoint 를 바꾸지 않는다 — 비교만 한다)
  5. 지정 환자(297, 234, 127, 99, 224) 상세

**레이블·코호트를 고치지 않는다.** 결과는 환자 단위 파일이므로 gitignore 대상인
outputs/gmu_nested/data_audit/ 에만 쓴다 (저장소는 공개 — 환자 정보 커밋 금지).

Run:  python experiments/실험16_GMU_nestedCV/diagnostics/check_data.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))                 # diagnostics/
EXP_DIR = os.path.dirname(HERE)                                   # 실험16 폴더 (nested_fusion 패키지)
PROJECT_ROOT = os.path.dirname(os.path.dirname(EXP_DIR))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, HERE)

import json
import re

import numpy as np
import pandas as pd

from sclc import cohort, features, paths

OUT_DIR = paths.outputs("gmu_nested", "data_audit")
REPORT_DIRS = (paths.data("REPORTS", "extracted", "docx_text"),
               paths.data("REPORTS", "extracted", "jpg_ocr_verified"))
TREATMENT_COLUMNS = ("rt_type", "atezolizumab")
FOCUS_IDS = (297, 234, 127, 99, 224)

#: 선행 치료 문구 분류. 다른 암의 과거력("for prostate cancer" 등)은 따로 거른다.
SYSTEMIC = re.compile(r"(?i)chemo|chemoradiation|CCRT|radiation therapy to (?!brain)|immunotherapy|atezolizumab")
BRAIN_LOCAL = re.compile(r"(?i)gamma knife|craniotomy|craniectomy|radiation therapy to brain")
SURGERY = re.compile(r"(?i)resection|lobectomy|wedge")
OTHER_CANCER = re.compile(r"(?i)for (?!small cell|lung|brain)[a-z ,()/-]*?(cancer|carcinoma|lymphoma|tumor)")


def _raw_excel() -> pd.DataFrame:
    x = pd.read_excel(cohort.DEFAULT_SOURCE_XLSX)
    x["research_id"] = pd.to_numeric(x["연구번호"], errors="coerce")
    x = x.dropna(subset=["research_id"])
    x["research_id"] = x["research_id"].astype(int)
    if x["research_id"].duplicated().any():
        raise ValueError("duplicate research_id in source excel")
    return x.set_index("research_id")


def _report_text(rid: int) -> tuple[str | None, str | None]:
    for d in REPORT_DIRS:
        path = os.path.join(d, f"{rid}.txt")
        if os.path.exists(path):
            with open(path, encoding="utf-8-sig") as fh:
                return fh.read(), os.path.basename(d)
    return None, None


def _exam_date(text: str) -> tuple[pd.Timestamp | None, str]:
    """판독지 원문의 날짜와 그 출처 문구. 외부 PET 은 '촬영일'이 판독일과 다르다."""
    m = re.search(r"taken on (\d{4})\.(\d{2})\.(\d{2})", text)
    if m:
        return pd.Timestamp(f"{m[1]}-{m[2]}-{m[3]}"), "outside_pet_taken_on"
    m = re.search(r"일\s*자\s*:\s*(\d{4}-\d{1,2}-\d{1,2})", text)
    if m:
        return pd.Timestamp(m[1]), "docx_일자"
    m = re.search(r"검사일\s*:\s*(\d{4}-\d{2}-\d{2})", text)
    if m:
        return pd.Timestamp(m[1]), "jpg_검사일"
    m = re.search(r"\b(20\d{6})[:\s]", text)
    if m:
        return pd.Timestamp(m[1]), "jpg_header_yyyymmdd"
    return None, "not_found"


def _purpose(text: str) -> str | None:
    m = re.search(r"검사\s*목적\s*:\s*([^-\[\n]{0,40})", text)
    return m[1].strip() if m else None


def _prior_treatment(text: str, dx: pd.Timestamp) -> dict:
    """'S/P ...' 절을 나눠 SCLC 관련(진단 60일 전 이후 날짜) 선행 치료를 분류한다."""
    info = re.search(r"임상\s*정보\s*:(.*?)-\s*검사\s*목적", text, re.S)
    clause_src = re.sub(r"\s+", " ", info[1]) if info else ""
    out = {"systemic": [], "brain_local": [], "surgery": [], "other_history": []}
    prev = None
    for clause in re.split(r"(?=S/P)", clause_src):
        if not clause.startswith("S/P"):
            continue
        # "S/P Pre/Post-operative ..." 는 바로 앞 수술 절(다른 암일 수 있음)의 보조치료다
        if prev == "other_history" and re.match(r"(?i)S/P\s*(pre|post)[- ]?operative", clause):
            out["other_history"].append(clause.strip()[:120])
            continue
        dates = [pd.Timestamp(f"{y}-{m}-{d}") for y, m, d in
                 re.findall(r"(\d{4})-(\d{1,2})-(\d{1,2})", clause)]
        recent = any(d >= dx - pd.Timedelta(days=60) for d in dates)
        # "(1983" 처럼 연도만 적힌 과거력: 전부 진단 전년도 이전이면 오래된 병력
        years = [int(y) for y in re.findall(r"\b((?:19|20)\d{2})\b", clause)]
        old_years_only = not dates and years and max(years) < dx.year - 1
        short = clause.strip()[:120]
        if OTHER_CANCER.search(clause) or (dates and not recent) or old_years_only:
            prev = "other_history"
        elif BRAIN_LOCAL.search(clause):
            prev = "brain_local"
        elif SYSTEMIC.search(clause):
            prev = "systemic"
        elif SURGERY.search(clause):
            prev = "surgery"
        else:
            prev = "other_history"
        out[prev].append(short)
    return out


def lung_recode_table(raw: pd.DataFrame, merged: pd.DataFrame, ids) -> pd.DataFrame:
    new = cohort.recode_lung_disease_from_excel(raw.loc[ids])
    t = pd.DataFrame({"raw": raw.loc[ids, cohort.LUNG_DISEASE_COL].astype(str),
                      "current": merged.loc[ids, "lung_disease"], "recoded": new})
    t["changed"] = t["current"] != t["recoded"]
    return t


def comparable_pairs(time: np.ndarray, event: np.ndarray) -> np.ndarray:
    """Harrell 비교 가능 쌍 행렬 [i, j] = i 가 j 보다 먼저 사건 (i 사건, t_i < t_j)."""
    return (time[:, None] < time[None, :]) & (event[:, None] == 1)


def anchor_comparison(df: pd.DataFrame, target: str) -> dict:
    t_dx = df[f"{target}_days"].to_numpy(float)
    e = df[f"{target}_event"].to_numpy(int)
    lag = df["exam_lag_days"].to_numpy(float)
    ok = ~np.isnan(lag)
    t_pet = t_dx - np.where(ok, lag, 0.0)
    a, b = comparable_pairs(t_dx[ok], e[ok]), comparable_pairs(t_pet[ok], e[ok])
    return {
        "n_with_exam_date": int(ok.sum()),
        "n_time_from_pet_le0": int((t_pet[ok] <= 0).sum()),
        "ids_time_from_pet_le0": [int(i) for i in df.index[ok][t_pet[ok] <= 0]],
        "comparable_pairs_dx": int(a.sum()),
        "comparable_pairs_pet": int(b.sum()),
        "pairs_changed": int((a != b).sum()),
        "pairs_changed_frac_of_dx": round(float((a != b).sum() / a.sum()), 5),
    }


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    cdf = cohort.load_trimodal_cohort()
    clin = cdf.drop_duplicates("research_id").set_index("research_id")
    ids = sorted(int(i) for i in clin.index)
    raw = _raw_excel()
    merged = cohort.load_merged_clinical().set_index("research_id")
    summary: dict = {"n_cohort": len(ids)}

    # 1. 치료 변수
    std_cols, cat_cols = features.resolve_clinical_columns(clin)
    summary["clinical_columns"] = {"continuous": std_cols, "categorical": cat_cols,
                                   "n": len(std_cols) + len(cat_cols)}
    summary["treatment_columns_in_input"] = [c for c in TREATMENT_COLUMNS if c in std_cols + cat_cols]

    # 2. lung_disease 재코딩
    lung = lung_recode_table(raw, merged, ids)
    lung.to_csv(os.path.join(OUT_DIR, "lung_disease_recode.csv"))
    summary["lung_disease"] = {
        "raw_counts": lung["raw"].value_counts().to_dict(),
        "current_counts": {int(k): int(v) for k, v in lung["current"].value_counts().items()},
        "recoded_counts": {int(k): int(v) for k, v in lung["recoded"].value_counts().items()},
        "n_changed": int(lung["changed"].sum()),
        "changed_by_transition": {f"{c}->{r}": int(n) for (c, r), n in
                                  lung[lung.changed].groupby(["current", "recoded"]).size().items()},
    }

    # 3. 판독지 원문 대조
    rows = []
    for rid in ids:
        text, src = _report_text(rid)
        dx = pd.Timestamp(raw.loc[rid, "진단일"])
        exam, how = _exam_date(text) if text else (None, "no_report_file")
        pt = _prior_treatment(text, dx) if text else {}
        pfs_date_col = pd.to_datetime(raw.loc[rid, "PFS Date"], errors="coerce")
        row = {
            "research_id": rid, "report_source_dir": src, "dx_date": dx.date(),
            "exam_date": exam.date() if exam is not None else None, "exam_date_source": how,
            "csv_report_date": clin.loc[rid, "report_date"],
            "exam_lag_days": (exam - dx).days if exam is not None else np.nan,
            "purpose": _purpose(text) if text else None,
            "pfs_days": int(clin.loc[rid, "pfs_days"]), "pfs_event": int(clin.loc[rid, "pfs_event"]),
            "os_days": int(clin.loc[rid, "os_days"]), "os_event": int(clin.loc[rid, "os_event"]),
            "excel_pfs_date": pfs_date_col.date() if pd.notna(pfs_date_col) else None,
            "excel_pfs_date_minus_dx": (pfs_date_col - dx).days if pd.notna(pfs_date_col) else np.nan,
            "prior_systemic_or_thoracic_rt": " | ".join(pt.get("systemic", [])),
            "prior_brain_local": " | ".join(pt.get("brain_local", [])),
            "prior_surgery": " | ".join(pt.get("surgery", [])),
        }
        rows.append(row)
    rep = pd.DataFrame(rows).set_index("research_id")
    lag = rep["exam_lag_days"]
    rep["exam_on_or_after_pfs_event"] = (rep.pfs_event == 1) & (lag >= rep.pfs_days)
    rep["exam_on_or_after_death"] = (rep.os_event == 1) & (lag >= rep.os_days)
    rep["exam_after_excel_pfs_date"] = lag > rep["excel_pfs_date_minus_dx"]
    rep["excel_pfs_date_mismatch"] = (rep["excel_pfs_date_minus_dx"].notna()
                                      & (rep["excel_pfs_date_minus_dx"] != rep.pfs_days))
    rep["csv_report_date_differs"] = (pd.to_datetime(rep.csv_report_date, errors="coerce")
                                      != pd.to_datetime(rep.exam_date, errors="coerce"))
    rep.to_csv(os.path.join(OUT_DIR, "report_timing.csv"))

    def _ids(mask) -> list[int]:
        return [int(i) for i in rep.index[mask.fillna(False)]]

    summary["exam_lag_days"] = {
        "describe": {k: round(float(v), 1) for k, v in lag.describe().items()},
        "n_missing": int(lag.isna().sum()), "ids_missing": _ids(lag.isna()),
        "n_lt0": int((lag < 0).sum()), "n_0_14": int(lag.between(0, 14).sum()),
        "n_15_30": int(lag.between(15, 30).sum()), "n_31_60": int(lag.between(31, 60).sum()),
        "n_gt60": int((lag > 60).sum()), "ids_gt30": _ids(lag > 30),
    }
    summary["purpose_counts"] = rep["purpose"].fillna("NA").value_counts().to_dict()
    summary["ids_purpose_not_initial"] = _ids(rep.purpose.notna() & ~rep.purpose.isin(
        ["Initial staging", "진단과 병기 설정"]))
    summary["ids_prior_systemic_or_thoracic_rt"] = _ids(rep.prior_systemic_or_thoracic_rt != "")
    summary["ids_prior_brain_local_only"] = _ids((rep.prior_brain_local != "")
                                                 & (rep.prior_systemic_or_thoracic_rt == ""))
    summary["ids_prior_surgery_only"] = _ids((rep.prior_surgery != "") & (rep.prior_brain_local == "")
                                             & (rep.prior_systemic_or_thoracic_rt == ""))
    summary["ids_exam_on_or_after_pfs_event"] = _ids(rep.exam_on_or_after_pfs_event)
    summary["ids_exam_on_or_after_death"] = _ids(rep.exam_on_or_after_death)
    summary["ids_exam_after_excel_pfs_date"] = _ids(rep.exam_after_excel_pfs_date)
    summary["ids_excel_pfs_date_mismatch"] = _ids(rep.excel_pfs_date_mismatch)
    summary["ids_csv_report_date_differs_from_text"] = _ids(rep.csv_report_date_differs)

    # 4. 기준일 비교
    summary["anchor_comparison"] = {t: anchor_comparison(rep, t) for t in ("os", "pfs")}

    # 5. 지정 환자
    summary["focus"] = {int(i): {k: (str(v) if not isinstance(v, (int, float, bool, np.integer,
                                                                  np.floating, np.bool_)) else
                                     (v.item() if hasattr(v, "item") else v))
                                 for k, v in rep.loc[i].items()} for i in FOCUS_IDS}

    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, default=str)
    print(json.dumps({k: v for k, v in summary.items() if k != "focus"},
                     ensure_ascii=False, indent=1, default=str))
    print(f"\n[audit] wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
