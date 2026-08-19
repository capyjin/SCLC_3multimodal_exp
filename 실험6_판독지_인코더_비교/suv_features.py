# -*- coding: utf-8 -*-
"""판독지(report_finding) 텍스트에서 SUVmax 수치를 정규식으로 뽑아 숫자 feature로 만든다.

[근거 — 판독지 포맷]
  PET/CT 판독지 finding 안에는 범례 문장이 들어 있다:
      "괄호안 숫자는 maxSUV임"
  그 뒤로 병변 서술이 이어지고, 괄호 안에 SUV 값이 붙는다. 예)
      "... 에 hypermetabolic mass (7.2)"
  => 괄호 안이 **숫자 하나뿐**인 경우만 SUV 값으로 본다.

[실제 데이터에서 확인한 사실 (248 has_report 환자 기준, 이 파일 하단 self-check로 재확인 가능)]
  - 범례 문장(괄호…숫자)은 237/248(95.6%) 판독지에 있고, **전부 finding 쪽**에 있다
    (conclusion 에는 0건).
  - 괄호 안이 순수 숫자인 그룹은 finding 에서만 374건 나오며, **그 374건 전부가
    범례가 있는 판독지 안에 있다.** 즉 "범례 있는 문서로 한정"은 값을 하나도
    잃지 않으면서 '괄호숫자=SUV'라는 근거가 텍스트로 명시된 범위만 쓰게 해준다.
  - 괄호 안에 숫자가 섞여 있지만 SUV 가 아닌 것들(날짜 2019-05-03, 슬라이스 번호
    IM 47, 늑골 레벨, 범위 ~5 등 139건)은 정규식이 "괄호 안 전체가 숫자"를
    요구하므로 자동으로 걸러진다.
  - 값 분포: n=374, min 1.1 / median 9.05 / mean 9.52 / p95 18.0 / max 37.9
    → 100% 가 임상적으로 타당한 SUVmax 범위(0.5–50) 안에 있다.
  - 괄호숫자 바로 앞 단어는 mass(251) / lesion(63) / metabolism(18) / nodule(4) …
    전부 병변 어휘 → 퍼센트·개수·병기 숫자가 섞여 들어온 흔적이 없다.

[파생 feature]
  suv_max   = 문서 내 최대값   (병변 중 가장 높은 SUVmax)
  suv_count = 문서 내 값 개수  (수치가 기재된 병변 수 ≈ 병변 부담)
  suv_mean  = 문서 내 평균값

[결측 처리]  코호트 238명 중 25명(10.5%)은 값이 0개다(범례는 있으나 hypermetabolic
  병변 수치 기재가 없거나, 범례 자체가 없는 판독지).
  - suv_count : 0  (자연스러운 값이고, 동시에 "SUV 기재 없음" 지시자 역할을 겸한다.
                    그래서 별도 availability indicator 열을 추가하지 않는다 —
                    step2/step3 에서는 suv_count 가 그 역할을 그대로 한다.)
  - suv_max / suv_mean : NaN → **train fold 환자만으로 계산한 median** 으로 대치.
    (전체 238명 median 을 쓰면 val/test 정보가 train 으로 새어 들어간다.)

[누수 방지]  정규식 파싱 자체는 환자별 결정론적 연산이라 fold와 무관하다(전역 1회 OK).
  fold 마다 달라져야 하는 건 **환자들을 가로질러 계산되는 통계**뿐이다:
  결측 대치용 median 과 StandardScaler(mean/std). 둘 다 train fold id 로만 fit 하고
  val/test 는 transform 만 한다 (features.ClinicalEncoder 와 동일한 규율).
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import csv
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from core import cohort

# 범례 문장. "괄호안 숫자는 maxSUV임" / "괄호 안의 숫자는 max SUV 임" 등 띄어쓰기·조사
# 변형을 흡수한다. (원안 r'괄호\s?안?\s?숫자는' 은 236건, 이 정규식은 237건을 잡는다.
# 늘어난 1건도 괄호숫자를 가진 정상 판독지였다.)
LEGEND_RE = re.compile(r"괄호\s*안?\s*의?\s*숫자")
# 괄호 안이 '숫자 하나'뿐일 때만 매치. (2019-05-03), (IM 47), (SUVmax 5, IM 22) 등은
# 괄호 바로 뒤가 숫자로 시작해도 닫는 괄호 전까지 숫자가 아니므로 매치되지 않는다.
VALUE_RE = re.compile(r"\(([0-9]+\.?[0-9]*)\)")

FEATURE_COLUMNS = ("suv_max", "suv_count", "suv_mean")
# 3단계 실험에서 쓰는 feature 조합 (누적)
STEPS = {
    "none":                 (),
    "suv_max":              ("suv_max",),
    "suv_max+count":        ("suv_max", "suv_count"),
    "suv_max+count+mean":   ("suv_max", "suv_count", "suv_mean"),
}
# 임상적으로 말이 되는 SUVmax 범위. 벗어나면 정규식이 SUV 아닌 숫자를 물었다는 뜻이라
# 조용히 통과시키지 않고 경고한다.
PLAUSIBLE_RANGE = (0.5, 50.0)


def extract_suv_table(merged_csv: str = cohort.DEFAULT_MERGED_CSV,
                      require_legend: bool = True) -> pd.DataFrame:
    """{research_id -> suv_max/suv_count/suv_mean/suv_available} 표를 만든다.

    fold 와 무관한 순수 파싱이므로 전역 1회 호출해도 누수가 없다.
    값이 없는 환자는 suv_max/suv_mean = NaN, suv_count = 0, suv_available = False.
    """
    records, raw_values, n_legend, n_reports = [], [], 0, 0
    with open(merged_csv, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"research_id", "has_report", "report_finding"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"merged CSV missing columns: {sorted(missing)}")
        for row in reader:
            if int(row["has_report"] or 0) != 1:
                continue
            n_reports += 1
            finding = row["report_finding"] or ""
            has_legend = bool(LEGEND_RE.search(finding))
            n_legend += int(has_legend)
            # 범례가 있는 문서 안에서만 '괄호숫자 = SUV' 라는 근거가 성립한다.
            vals = [float(v) for v in VALUE_RE.findall(finding)] if (has_legend or not require_legend) else []
            raw_values.extend(vals)
            records.append({
                "research_id": int(row["research_id"]),
                "suv_max": float(np.max(vals)) if vals else np.nan,
                "suv_count": float(len(vals)),
                "suv_mean": float(np.mean(vals)) if vals else np.nan,
                "suv_available": bool(vals),
                "has_legend": has_legend,
            })

    table = pd.DataFrame.from_records(records).set_index("research_id")
    table.attrs["n_reports"] = n_reports
    table.attrs["n_legend"] = n_legend
    table.attrs["n_values"] = len(raw_values)
    if raw_values:
        arr = np.array(raw_values)
        lo, hi = PLAUSIBLE_RANGE
        n_out = int(((arr < lo) | (arr > hi)).sum())
        table.attrs["value_stats"] = {
            "n": len(arr), "min": float(arr.min()), "median": float(np.median(arr)),
            "mean": float(arr.mean()), "p95": float(np.percentile(arr, 95)), "max": float(arr.max()),
            "n_outside_plausible_range": n_out,
        }
        if n_out:
            print(f"[suv] WARNING: {n_out}/{len(arr)} extracted values fall outside the plausible "
                  f"SUVmax range {PLAUSIBLE_RANGE} -- the regex may be matching non-SUV numbers.")
    return table


def make_extra_numeric_fn(table: pd.DataFrame, cols, audit: list | None = None):
    """features.build_fold_multimodal_tabular 에 넘길 fold-safe 블록 생성기를 만든다.

    반환되는 함수는 (train_ids, val_ids, test_ids) 를 받아
    {"train": X, "val": X, "test": X} (float32, 열 개수 = len(cols)) 를 돌려준다.
    **median(결측 대치)과 StandardScaler 는 train_ids 행으로만 fit** 하고,
    val/test 는 transform 만 한다. 사용한 median/표본수는 audit 리스트에 기록해
    로그에서 "train 몇 명으로 계산한 median 인가"를 눈으로 감사할 수 있게 한다.
    """
    cols = list(cols)

    def fn(train_ids, val_ids, test_ids):
        if not cols:
            return {k: np.empty((len(ids), 0), dtype="float32")
                    for k, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids))}

        train_ids = [int(i) for i in train_ids]
        raw = {name: table.reindex([int(i) for i in ids])[cols].to_numpy(dtype="float64")
               for name, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids))}

        # ── (1) 결측 대치: median 은 train fold 환자 중 값이 있는 사람만으로 계산 ──
        train_raw = raw["train"]
        medians = np.nanmedian(train_raw, axis=0)
        if np.isnan(medians).any():  # train fold 전원이 결측인 열은 없어야 정상
            raise ValueError(f"all-NaN column in train fold for cols={cols}")
        filled = {}
        n_imputed = {}
        for name, arr in raw.items():
            mask = np.isnan(arr)
            arr = np.where(mask, medians, arr)
            filled[name] = arr
            n_imputed[name] = mask.any(axis=1).sum().item() if arr.size else 0

        # ── (2) 표준화: StandardScaler 도 train fold 로만 fit ──
        scaler = StandardScaler().fit(filled["train"])
        out = {name: scaler.transform(arr).astype("float32") if len(arr)
               else np.empty((0, len(cols)), dtype="float32")
               for name, arr in filled.items()}

        if audit is not None:
            n_avail_train = int((~np.isnan(train_raw[:, 0])).sum()) if train_raw.size else 0
            audit.append({
                "cols": cols,
                "n_train": len(train_ids), "n_val": len(val_ids), "n_test": len(test_ids),
                "n_train_with_suv": n_avail_train,
                "train_fold_medians": {c: round(float(m), 4) for c, m in zip(cols, medians)},
                "train_fold_scaler_mean": {c: round(float(m), 4) for c, m in zip(cols, scaler.mean_)},
                "train_fold_scaler_scale": {c: round(float(s), 4) for c, s in zip(cols, scaler.scale_)},
                "n_imputed": n_imputed,
            })
        return out

    return fn


def cohort_summary(table: pd.DataFrame, merged_csv: str = cohort.DEFAULT_MERGED_CSV,
                   split_csv: str = cohort.DEFAULT_SPLIT_CSV) -> dict:
    """실제 학습에 들어가는 238명 코호트 기준 요약 (전체 248명 corpus 기준과 구분)."""
    cohort_df = cohort.load_trimodal_cohort(merged_csv, split_csv)
    rids = sorted({int(r) for r in cohort_df["research_id"]})
    sub = table.reindex(rids)
    mx = sub["suv_max"].dropna().to_numpy()
    return {
        "n_cohort": len(rids),
        "n_available": int(sub["suv_available"].fillna(False).sum()),
        "n_missing": int((~sub["suv_available"].fillna(False)).sum()),
        "suv_max": {"min": float(mx.min()), "median": float(np.median(mx)),
                    "mean": float(mx.mean()), "max": float(mx.max()),
                    "n_unique": int(len(np.unique(mx)))},
        "suv_count_hist": {int(k): int(v) for k, v in
                           sub["suv_count"].fillna(0).value_counts().sort_index().items()},
    }


if __name__ == "__main__":
    # self-check: 문서화된 숫자가 실제로 재현되는지 확인한다.
    t = extract_suv_table()
    print(f"reports={t.attrs['n_reports']}  legend={t.attrs['n_legend']} "
          f"({t.attrs['n_legend'] / t.attrs['n_reports'] * 100:.1f}%)  "
          f"docs_with_value={int(t['suv_available'].sum())} "
          f"({t['suv_available'].mean() * 100:.1f}%)  values={t.attrs['n_values']}")
    print("value stats:", t.attrs["value_stats"])
    # 범례 없는 문서에서도 뽑아보면 값이 늘어나는지 (=범례 스코프가 값을 버리는지) 확인
    t_all = extract_suv_table(require_legend=False)
    print(f"without legend scoping: docs_with_value={int(t_all['suv_available'].sum())} "
          f"values={t_all.attrs['n_values']}  (동일해야 정상: 괄호숫자는 전부 범례 문서 안에 있음)")
    print("cohort(238):", cohort_summary(t))
