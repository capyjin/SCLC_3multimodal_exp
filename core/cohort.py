# -*- coding: utf-8 -*-
"""Tri-modal cohort + manifest, per DATA/SCLC_EXPERIMENT_PROTOCOL.md section 6.

The authoritative patient set and fold assignment come from
``splits/trimodal_common_5fold_seed42_v1.csv`` — a byte-identical copy of
``clinical+report``'s ``report_common_5fold_seed42_v1.csv``. That file's 238
patients already equal the exact intersection of the image+clinical baseline
cohort (257) and the report_common cohort (238): every report_common patient
also has an image, so report_common *is* the tri-modal common cohort. No new
split is generated (protocol section 7: "모델마다 새로운 split을 생성하지 않는다").

``build_manifest`` reproduces the report_usable exclusion rule from
``clinical+report/CODE/build_manifest.py`` (EVIDENCE_CONFIRMED_LEAK_IDS) for
manifest documentation purposes only — actual cohort membership for training
always comes from the split file, never recomputed here.
"""
import os

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MERGED_CSV = os.path.join(PROJECT_ROOT, "DATA", "merged_tabular_with_reports.csv")
DEFAULT_IMAGE_DIR = os.path.join(PROJECT_ROOT, "DATA", "CUT IMAGE")
DEFAULT_SPLIT_CSV = os.path.join(PROJECT_ROOT, "splits", "trimodal_common_5fold_seed42_v1.csv")

# Copied from clinical+report/CODE/build_manifest.py: report_date vs. PFS
# progression date leakage confirmed for these 8 patients (see
# leakage_check_report.md in the source repo). Documentation-only here.
EVIDENCE_CONFIRMED_LEAK_IDS = {5, 78, 101, 131, 167, 171, 188, 306}

MANIFEST_COLUMNS = [
    "research_id", "has_clinical", "has_image", "has_report",
    "clinical_usable", "image_usable", "report_usable",
    "os_duration", "os_event", "pfs_duration", "pfs_event",
    "exclusion_reason", "cohort_name", "fold", "split",
]


def load_merged_clinical(merged_csv: str = DEFAULT_MERGED_CSV) -> pd.DataFrame:
    df = pd.read_csv(merged_csv, encoding="utf-8-sig", low_memory=False)
    if df["research_id"].duplicated().any():
        raise ValueError("merged CSV contains duplicate research_id values")
    return df


def load_split(split_csv: str = DEFAULT_SPLIT_CSV) -> pd.DataFrame:
    split = pd.read_csv(split_csv, dtype={"research_id": str})
    split["research_id"] = split["research_id"].astype(int)
    required = {"research_id", "split", "fold"}
    missing = required - set(split.columns)
    if missing:
        raise ValueError(f"Split manifest missing columns: {sorted(missing)}")
    return split


def build_manifest(
    merged_csv: str = DEFAULT_MERGED_CSV,
    image_dir: str = DEFAULT_IMAGE_DIR,
    split_csv: str = DEFAULT_SPLIT_CSV,
) -> pd.DataFrame:
    """Builds the full-cohort manifest (321 clinical patients), documenting
    who is/isn't in the 238-patient tri-modal common cohort and why."""
    clinical = load_merged_clinical(merged_csv)
    image_ids = {
        int(os.path.splitext(name)[0])
        for name in os.listdir(image_dir)
        if name.lower().endswith(".png") and os.path.splitext(name)[0].isdigit()
    }
    split = load_split(split_csv)
    split_by_id = {int(row.research_id): (int(row.fold), row.split) for row in split.itertuples()}

    rows = []
    for _, r in clinical.iterrows():
        rid = int(r["research_id"])
        has_image = int(rid in image_ids)
        has_report = int(r["has_report"]) if pd.notna(r["has_report"]) else 0

        os_dur, os_evt = r.get("os_days"), r.get("os_event")
        pfs_dur, pfs_evt = r.get("pfs_days"), r.get("pfs_event")
        label_complete = all(pd.notna(v) for v in (os_dur, os_evt, pfs_dur, pfs_evt))

        clinical_usable = int(label_complete)
        image_usable = has_image
        report_usable = int(has_report and rid not in EVIDENCE_CONFIRMED_LEAK_IDS)

        reasons = []
        if not label_complete:
            reasons.append("incomplete_survival_label")
        if not has_image:
            reasons.append("no_image")
        if not has_report:
            reasons.append("no_report")
        elif rid in EVIDENCE_CONFIRMED_LEAK_IDS:
            reasons.append("report_leakage_confirmed")

        in_split = rid in split_by_id
        is_trimodal_eligible = bool(clinical_usable and image_usable and report_usable)
        fold, split_name = split_by_id.get(rid, (None, None))

        # The split file is authoritative for actual cohort membership (protocol
        # section 7: never regenerate a split from current data). cohort_name
        # reflects that. A patient who is *newly* eligible under the current
        # merged CSV but isn't in the frozen split (data pipeline moved on
        # after the split was created -- e.g. an OCR/report fix) is flagged
        # separately rather than silently dropped or forced into training.
        if in_split:
            cohort_name = "trimodal_common"
        elif is_trimodal_eligible:
            cohort_name = "trimodal_eligible_not_in_split"
            reasons.append("eligible_under_current_data_but_absent_from_frozen_split")
        else:
            cohort_name = "excluded"

        rows.append({
            "research_id": rid,
            "has_clinical": 1,
            "has_image": has_image,
            "has_report": has_report,
            "clinical_usable": clinical_usable,
            "image_usable": image_usable,
            "report_usable": report_usable,
            "os_duration": os_dur, "os_event": os_evt,
            "pfs_duration": pfs_dur, "pfs_event": pfs_evt,
            "exclusion_reason": ";".join(reasons) if reasons else "",
            "cohort_name": cohort_name,
            "fold": fold,
            "split": split_name,
        })
        if in_split and not is_trimodal_eligible:
            print(
                f"[cohort] WARNING research_id {rid}: in the frozen split but no longer "
                "passes the current eligibility recompute (clinical_usable="
                f"{clinical_usable}, image_usable={image_usable}, report_usable={report_usable}). "
                "Training will still use it (split is authoritative) -- investigate if unexpected."
            )

    drifted = [row["research_id"] for row in rows if row["cohort_name"] == "trimodal_eligible_not_in_split"]
    if drifted:
        print(
            f"[cohort] NOTE: {len(drifted)} patient(s) are tri-modal-eligible under the current "
            f"merged CSV but are not in the frozen split (likely added/fixed after the split was "
            f"created): {drifted}. They are excluded from training to keep the split reproducible; "
            "a new split version would be needed to include them (protocol section 7)."
        )

    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


DEFAULT_SOURCE_XLSX = os.path.join(PROJECT_ROOT, "DATA", "20260619 SCLC PET 전달용.xlsx")
_BRAIN_TIMING_COL = "Brain metastasis 1. Initial\n2. META"


def apply_brain_meta_fix(df: pd.DataFrame, source_xlsx: str = DEFAULT_SOURCE_XLSX) -> pd.DataFrame:
    """`brain_meta` 를 **기저(baseline) 시점 뇌전이만** 1 로 다시 코딩한다.

    [왜 필요한가 — 누수와 불멸시간 편향]
      merged CSV 의 `brain_meta` 는 원본 엑셀의 `Brain metastasis 1.Initial / 2.META`
      두 경우를 구분 없이 합쳐 놓은 값이다. 238명 코호트에서 brain_meta=1 인 84명 중
        - 48명: 진단 시점(기저)에 이미 뇌전이가 있었음   (1 = Initial)
        - 34명: 진단 **이후에** 새로 생긴 뇌전이          (2 = META, 전체의 40.5%)
        -  2명: 시점 불명 (원본 결측)
      "진단 이후에 생긴 전이"는 그때까지 **생존해 있어야** 관측되므로, 이를 기저
      공변량으로 쓰면 미래 정보가 과거 시점 변수로 새어 들어온다(immortal time bias).
      실제로 이시성 전이군의 OS 중앙값은 **564일**로, 뇌전이가 아예 없는 군(352일)보다
      길다 — 전이가 예후를 좋게 만든 것이 아니라 오래 산 사람에게서 발견된 것이다.

      그 결과 현재 변수는 방향이 반대인 두 집단이 섞여 예후 정보가 상쇄돼 있다:
        수정 전 brain_meta        OS HR 1.21 (p=0.203)   PFS HR 1.31 (p=0.055)
        수정 후 기저 전이만        OS HR 1.82 (p=0.0005)  PFS HR 1.85 (p=0.0002)

      시점이 불명인 2명은 기저임을 확인할 수 없으므로 **0 으로 둔다**(누수를 되살리는
      쪽보다 정보를 조금 잃는 쪽을 택한다).

    원본 엑셀을 읽을 수 없으면 예외를 던진다 — 조용히 수정 없이 넘어가면
    "고쳤다고 생각했는데 안 고쳐진" 상태가 되기 때문이다.
    """
    if "brain_meta" not in df.columns:
        raise ValueError("brain_meta column not found; cannot apply the fix")
    src = pd.read_excel(source_xlsx)
    if _BRAIN_TIMING_COL not in src.columns:
        raise ValueError(f"timing column {_BRAIN_TIMING_COL!r} not found in {source_xlsx}")
    src = src.assign(rid=pd.to_numeric(src["연구번호"], errors="coerce")).dropna(subset=["rid"])
    timing = pd.to_numeric(
        src.set_index(src["rid"].astype(int))[_BRAIN_TIMING_COL], errors="coerce")
    timing = timing[~timing.index.duplicated(keep="first")]

    out = df.copy()
    is_baseline = out["research_id"].astype(int).map(timing).eq(1)
    out["brain_meta"] = (out["brain_meta"].eq(1) & is_baseline).astype(int)
    return out


def load_trimodal_cohort(
    merged_csv: str = DEFAULT_MERGED_CSV, split_csv: str = DEFAULT_SPLIT_CSV,
    fix_brain_meta: bool = True,
) -> pd.DataFrame:
    """Returns the merged clinical+report frame restricted to the 238-patient
    tri-modal common cohort, indexed by research_id, with split/fold columns
    attached. This is the frame every experiment (early/late fusion, and any
    unimodal arm) should read cohort membership from.

    ``fix_brain_meta`` 는 **기본값 True** — `brain_meta` 를 기저 시점 전이만 1 로
    다시 코딩한다(누수 수정, apply_brain_meta_fix 의 docstring 참고).
    False 로 주면 수정 전(legacy) 동작이 되며, 2026-08-02 이전에 기록된 결과
    (예: late fusion OS 0.7221)를 재현할 때 쓴다.
    """
    clinical = load_merged_clinical(merged_csv)
    split = load_split(split_csv)
    df = clinical.merge(split, on="research_id", how="inner", validate="one_to_many")
    if df["research_id"].nunique() != split["research_id"].nunique():
        raise ValueError("Some split research_ids are missing from the merged clinical CSV")
    if fix_brain_meta:
        df = apply_brain_meta_fix(df)
    return df


if __name__ == "__main__":
    manifest = build_manifest()
    n_trimodal = int((manifest["cohort_name"] == "trimodal_common").sum())
    print(f"Full clinical cohort: {len(manifest)}")
    print(f"Tri-modal common cohort (image+clinical+report): {n_trimodal}")
    print("\nExclusion reasons (non-trimodal patients):")
    excluded = manifest[manifest["cohort_name"] != "trimodal_common"]
    print(excluded["exclusion_reason"].value_counts().to_string())
