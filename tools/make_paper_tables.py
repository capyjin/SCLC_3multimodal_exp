# -*- coding: utf-8 -*-
"""논문 Results 3.2–3.6 의 표(Table 2–6)와 KM 그림을 Overleaf 용 LaTeX 로 만든다.

재학습은 하지 않는다. 이미 저장된 결과 JSON 과 채택 모델의 OOF 위험점수만 읽는다.

    표                       절    원본
    tab2_unimodal            3.2   outputs/fusion_grouping (임상) · reportonly_* (판독지) ·
                                   late_fusion_B · image_radimagenet (영상)
    tab3_multimodal          3.3   outputs/fusion_grouping (2모달 concat) · radbert_full (채택)
    tab4_fusion              3.4   outputs/fusion_methods (실험12) · fusion_grouping (실험13)
    tab5_risk_groups         3.5   outputs/late_fusion_B_radbert_tests/oof_{os,pfs}.json
    tab6_subgroups           3.6   같은 OOF + 코호트 임상변수
    overleaf_table1          3.1   기저 특성 longtable (원자료 연속형 관측값 + 모델 입력 범주형)
    overleaf_tables          —     위 표 5개 + KM 그림 2개를 기존 main.tex 에 붙여 넣는 float 묶음
    overleaf_preamble        —     그 묶음이 요구하는 main.tex 프리앰블 줄
    fig_km_overall           3.5   같은 OOF
    fig_km_stage             3.6   같은 OOF

규약 (README ★절 · 논문 결정 2026-09-30):
- C-index 는 test fold 별 값의 평균 ± SD (np.std, ddof=0 — README 수치와 같은 정의).
- P 는 채택 모델 대비 fold 쌍대 t 검정 (n = 5 fold, 참고용).
- 위험군 컷오프 = fold 마다 CoxPH 결합기를 적합한 train 환자 위험점수의 중앙값.
  test 환자를 그 값으로 나눈 뒤 5 fold 를 모아 KM · log-rank · Cox HR 을 낸다.
- 서브그룹 C-index 는 pooled-OOF 를 쓰지 않는다. (fold, 서브그룹) 안에서만 쌍을 세어
  합산하고, 95% CI 는 환자 단위 Poisson 부트스트랩(B = 2000)이다 — 실험9·11 과 같은 방법.

Run:  python tools/make_paper_tables.py
      tools/build_paper_figures.sh tab2_unimodal tab3_multimodal tab4_fusion \\
          tab5_risk_groups tab6_subgroups fig_km_overall fig_km_stage
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import median_survival_times
from scipy import stats

from sclc import paths
from sclc.cohort import load_trimodal_cohort
from sclc.evaluation import concordant_pair_counts
from sclc.late_fusion import labels_by_id
from sclc.late_fusion_tests import iter_fold_stack, risk_frame
from sclc.train import fold_plan

PAPER_DIR = os.path.join(PROJECT_ROOT, "report_assets", "paper")
FINAL_OOF_DIR = paths.outputs("late_fusion_B_radbert_tests")
DAYS_PER_MONTH = 30.4375
N_BOOT = 2000
TARGETS = ("os", "pfs")


# ---------------------------------------------------------------------------
# 결과 JSON 읽기
# ---------------------------------------------------------------------------
def _load(*parts):
    with open(paths.outputs(*parts), encoding="utf-8") as fh:
        return json.load(fh)


def _arm(block):
    """결과 JSON 의 arm 블록 -> {mean, sd, folds}. 저장된 mean/std 를 우선 쓴다
    (folds 는 소수 4자리로 반올림돼 있어 다시 평균하면 셋째 자리가 달라질 수 있다)."""
    folds = [float(f) for f in block["folds"]]
    out = {"mean": float(block.get("mean", np.mean(folds))),
           "sd": float(block.get("std", np.std(folds))), "folds": folds}
    if "constant_gate_folds" in block:
        out["reference_folds"] = [float(f) for f in block["constant_gate_folds"]]
    if "fold_records" in block and "gate_max_rel_change" in block["fold_records"][0]:
        out["records"] = [{"gate_max_rel_change": r["gate_max_rel_change"]} for r in block["fold_records"]]
    return out


def collect_cv_results() -> dict:
    """표 2–4 의 fold 별 C-index 를 {행이름: {target: arm}} 로 모은다."""
    rows = {}

    def put(name, target, block):
        rows.setdefault(name, {})[target] = _arm(block)

    for t in TARGETS:
        full = _load("radbert_full", f"results_{t}.json")["results"]
        grouping = _load("fusion_grouping", f"results_{t}.json")
        methods = _load("fusion_methods", "results.json")[t]
        lfb = _load("late_fusion_B", "results.json")[t]
        radimagenet = _load("image_radimagenet", "results.json")[t]

        put("proposed", t, full["late_tab_radbert+img"])
        put("clin_report", t, full["tab_radbert"])
        put("img_simplecnn", t, full["image_simplecnn"])
        put("img_resnet18", t, lfb["image_resnet18_only"])
        put("img_radimagenet", t, radimagenet["pretrained_cindex"])
        put("clin", t, grouping["axes"]["clin_only"])
        put("clin_img", t, grouping["axes"]["clin_image"])
        put("report_img", t, grouping["axes"]["report_image"])
        put("g2", t, grouping["combinations"]["G2_clin+img|report"])
        put("g3", t, grouping["combinations"]["G3_report+img|clin"])
        put("early", t, methods["early_concat"])
        put("attention", t, _load("attention_fusion", f"results_{t}.json"))   # 실험12 exp_attention_fusion.py
        # 실험12 exp_gate_oof.py --gate_inputs risk — 게이트 입력은 두 위험점수뿐(임상변수 재투입 없음, 지도교수 지적)
        put("gate", t, _load("gate_oof", f"results_{t}_riskonly.json"))
        put("late3", t, methods["late_3way"])
        for enc, arm in (("tfidf", "tfidf"), ("mbert", "bert_raw"),
                         ("kmbert", "bert_raw"), ("radbert", "bert_ko2en")):
            put(f"report_{enc}", t, _load(f"reportonly_{enc}", f"results_{t}.json")["arms"][arm])

    for name, by_t in rows.items():
        for t, arm in by_t.items():
            # 게이트는 같은 실행에서 잰 상수 게이트(= 채택 모델) fold 값과 비교한다 — 둘 다 반올림 전 값이라 정밀도가 같다
            ref = arm.get("reference_folds", rows["proposed"][t]["folds"])
            arm["p_vs_proposed"] = (None if name == "proposed"
                                    else float(stats.ttest_rel(arm["folds"], ref).pvalue))
    return rows


# ---------------------------------------------------------------------------
# 채택 모델의 test 위험점수 + 위험군
# ---------------------------------------------------------------------------
def final_model_test_scores(cohort: pd.DataFrame, target: str) -> pd.DataFrame:
    """fold 마다 결합기를 다시 적합해 test 위험점수와 train 중앙값 기준 위험군을 낸다."""
    with open(os.path.join(FINAL_OOF_DIR, f"oof_{target}.json"), encoding="utf-8") as fh:
        payload = json.load(fh)
    risks = {f"risk_{k}": {int(i): float(v) for i, v in payload[k].items()}
             for k in ("tabular", "image")}
    labels = labels_by_id(cohort, target)
    rows = []
    for fold, ids, cph, test_df, names in iter_fold_stack(risks, labels, fold_plan(cohort), target):
        train_df = risk_frame(risks, ids["train"], labels, target, names)
        cutoff = float(np.median(cph.predict_partial_hazard(train_df[names])))
        test_risk = cph.predict_partial_hazard(test_df[names]).to_numpy()
        for rid, d, e, r in zip(ids["test"], test_df["duration"], test_df["event"], test_risk):
            rows.append({"research_id": int(rid), "fold": int(fold), "months": d / DAYS_PER_MONTH,
                         "event": int(e), "risk": float(r), "high": int(r > cutoff)})
    return pd.DataFrame(rows)


def km_summary(df: pd.DataFrame) -> dict:
    """저위험/고위험 두 군의 KM 요약 + log-rank + Cox HR(고위험 vs 저위험)."""
    out = {}
    for name, g in (("low", df[df.high == 0]), ("high", df[df.high == 1])):
        kmf = KaplanMeierFitter().fit(g["months"], g["event"])
        ci = median_survival_times(kmf.confidence_interval_).to_numpy().ravel()
        out[name] = {"n": int(len(g)), "events": int(g["event"].sum()),
                     "median": float(kmf.median_survival_time_),
                     "median_ci": [float(ci[0]), float(ci[1])],
                     "rate_12": float(kmf.predict(12)), "rate_24": float(kmf.predict(24))}
    lo, hi = df[df.high == 0], df[df.high == 1]
    out["logrank_p"] = float(logrank_test(hi["months"], lo["months"], hi["event"], lo["event"]).p_value)
    out["hr"] = cox_hr(df)
    return out


def cox_hr(df: pd.DataFrame, extra=None) -> dict:
    cols = ["months", "event", "high"] + list(extra or [])
    s = CoxPHFitter().fit(df[cols], "months", "event").summary.loc["high"]
    return {"hr": float(s["exp(coef)"]), "lo": float(s["exp(coef) lower 95%"]),
            "hi": float(s["exp(coef) upper 95%"]), "p": float(s["p"])}


def interaction_p(df: pd.DataFrame, member: pd.Series) -> float:
    """위험군 HR 이 서브그룹에 따라 다른가 — Cox 상호작용항의 Wald P."""
    d = df.assign(sub=member.astype(int).to_numpy())
    d["high_x_sub"] = d["high"] * d["sub"]
    s = CoxPHFitter().fit(d[["months", "event", "high", "sub", "high_x_sub"]], "months", "event").summary
    return float(s.loc["high_x_sub", "p"])


# ---------------------------------------------------------------------------
# 다변량 Cox (Supplementary Table S1, 3.5 절)
#   위험군(이분형)을 넣는다 — 연속 위험점수는 fold 마다 척도가 달라 5 fold 를 모으면 pooled-OOF 함정이 된다.
#   이분형 위험군은 fold 별 train 중앙값으로 잘라 fold 마다 상대적으로 정의돼 있다.
#   공변량 = 결측 없는 SCLC 표준 예후인자. LDH 는 16% 결측(전체 중앙값 대치)이라 넣지 않았다.
# ---------------------------------------------------------------------------
MV_COVARIATES = [  # (열 이름, 표 라벨)
    ("high", "High-risk group (vs. low risk)"),
    ("es", "Extensive stage (vs. limited)"),
    ("ecog2", r"ECOG performance status $\geq$ 2 (vs. 0--1)"),
    ("age10", "Age, per 10 years"),
    ("female", "Female sex (vs. male)"),
    ("liver_meta", "Liver metastasis"),
    ("brain_meta", "Brain metastasis at diagnosis"),
]


def multivariable_cox(cohort: pd.DataFrame, scores: dict) -> dict:
    clin = cohort.drop_duplicates("research_id").set_index("research_id")
    out = {}
    for t in TARGETS:
        d = scores[t].set_index("research_id").join(
            clin[["stage", "ecog", "age_at_diagnosis", "gender", "liver_meta", "brain_meta"]])
        d = d.assign(es=(d.stage == 2).astype(int), ecog2=(d.ecog >= 2).astype(int),
                     age10=d.age_at_diagnosis / 10, female=(d.gender == 2).astype(int))
        cols = [c for c, _ in MV_COVARIATES]

        def summ(fit_cols):
            sm = CoxPHFitter().fit(d[["months", "event", *fit_cols]], "months", "event").summary
            return {c: {"hr": float(sm.loc[c, "exp(coef)"]), "lo": float(sm.loc[c, "exp(coef) lower 95%"]),
                        "hi": float(sm.loc[c, "exp(coef) upper 95%"]), "p": float(sm.loc[c, "p"])}
                    for c in fit_cols}
        out[t] = {"uni": {c: summ([c])[c] for c in cols}, "multi": summ(cols), "n": int(len(d)),
                  "events": int(d.event.sum())}
    return out


# ---------------------------------------------------------------------------
# 서브그룹 C-index (fold 안에서만 쌍을 센다) + Poisson 부트스트랩
# ---------------------------------------------------------------------------
def _pair_matrices(t, e, r):
    """comp[i,j]=1: i 가 먼저 사건(비교가능 쌍), conc[i,j]: 그 쌍의 일치 점수(동점 0.5)."""
    comp = (t[:, None] < t[None, :]) & (e[:, None] == 1)
    conc = np.where(r[:, None] > r[None, :], 1.0, np.where(r[:, None] == r[None, :], 0.5, 0.0))
    return comp.astype(float), conc * comp


def subgroup_cindex(df: pd.DataFrame, mask: np.ndarray, rng) -> dict:
    blocks = []
    for _fold, g in df[mask].groupby("fold"):
        comp, conc = _pair_matrices(g["months"].to_numpy(), g["event"].to_numpy(), g["risk"].to_numpy())
        blocks.append((comp, conc))
    point = sum(c.sum() for _, c in blocks) / sum(m.sum() for m, _ in blocks)

    num, den = np.zeros(N_BOOT), np.zeros(N_BOOT)
    for comp, conc in blocks:
        w = rng.poisson(1.0, size=(N_BOOT, comp.shape[0])).astype(float)
        num += np.einsum("bi,ij,bj->b", w, conc, w)
        den += np.einsum("bi,ij,bj->b", w, comp, w)
    boot = num[den > 0] / den[den > 0]
    return {"c": float(point), "lo": float(np.percentile(boot, 2.5)),
            "hi": float(np.percentile(boot, 97.5))}


SUBGROUPS = [
    # (변수 제목, [(수준 이름, 조건 함수)]) — 코호트 원자료 코드 기준
    ("Age", [(r"$<$ 70 years", lambda c: c.age_at_diagnosis < 70),
             (r"$\geq$ 70 years", lambda c: c.age_at_diagnosis >= 70)]),
    # 성별 코드 1 = 남성, 2 = 여성 (원자료에 범례가 없어 2026-10-11 저자에게 확인받음)
    ("Sex", [("Male", lambda c: c.gender == 1), ("Female", lambda c: c.gender == 2)]),
    ("ECOG performance status", [("0--1", lambda c: c.ecog <= 1),
                                 (r"$\geq$ 2", lambda c: c.ecog >= 2)]),
    ("Stage", [("Limited", lambda c: c.stage == 1), ("Extensive", lambda c: c.stage == 2)]),
    ("Liver metastasis", [("Absent", lambda c: c.liver_meta == 0), ("Present", lambda c: c.liver_meta == 1)]),
    ("Brain metastasis", [("Absent", lambda c: c.brain_meta == 0), ("Present", lambda c: c.brain_meta == 1)]),
    ("Atezolizumab", [("Not used", lambda c: c.atezolizumab == 0), ("Used", lambda c: c.atezolizumab == 1)]),
]


def subgroup_analysis(cohort: pd.DataFrame, scores: dict) -> list:
    clin = cohort.drop_duplicates("research_id").set_index("research_id")
    rng = np.random.default_rng(42)
    out = []
    all_rows = {}
    for t in TARGETS:
        df = scores[t]
        mask = np.ones(len(df), dtype=bool)
        all_rows[t] = {"c": subgroup_cindex(df, mask, rng), "hr": cox_hr(df)}
    out.append({"variable": "All patients", "levels": [{"name": None, "n": len(scores["os"]), **all_rows}]})
    for var, levels in SUBGROUPS:
        entry = {"variable": var, "levels": [], "p_int": {}}
        for name, cond in levels:
            lvl = {"name": name}
            for t in TARGETS:
                df = scores[t]
                mask = cond(clin.loc[df["research_id"]]).to_numpy()
                lvl["n"] = int(mask.sum())
                lvl[t] = {"c": subgroup_cindex(df, mask, rng), "hr": cox_hr(df[mask])}
            entry["levels"].append(lvl)
        for t in TARGETS:
            df = scores[t]
            second = levels[1][1](clin.loc[df["research_id"]])
            entry["p_int"][t] = interaction_p(df, second)
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# Table 1 — 기저 특성 (사용자 원고의 longtable 형식 그대로, 238명 분석 코호트)
# ---------------------------------------------------------------------------
SOURCE_XLSX = paths.data("20260619 SCLC PET 전달용.xlsx")

# 연속형은 대치 전 관측값으로 요약한다. merged CSV 는 결측을 전체 중앙값으로 채워 둔 값이라
# (실험7 raw_clinical_values.py docstring) 중앙값·IQR 이 그 값 쪽으로 쏠린다.
T1_CONTINUOUS = [  # (표 라벨, 원자료 열, 결측 주석 이름)
    ("Age at diagnosis, years", "진단시 나이", "age"),
    ("Smoking, pack-years\\textsuperscript{a}", "흡연력", "pack-years"),   # 과거·현재 흡연자만 (비흡연자는 원자료가 비어 있다)
    ("Hemoglobin, g/dL", "Hb", "hemoglobin"),
    (r"White blood cell count, /\textmu L", "WBC", "white blood cell count"),
    ("LDH, U/L", "LDH", "LDH"),
    (r"FVC, \% predicted", "FVC Pre%ref", "FVC"),
    (r"FEV1, \% predicted", "FEV1 Pre%ref", "FEV1"),
    (r"DLCO (adjusted), \% predicted", "DLCOAdj Pre%ref", "DLCO"),
]

# 범주형은 모델 입력(코호트 로더, brain_meta 누수 수정 적용)의 코드를 쓴다 — 원자료와 1:1 대조 확인(2026-10-11).
# 만성 폐질환만 예외: 모델 입력은 천식·ILD/IPF 를 결측(-1)과 같은 코드로 묶어 두었으므로 원자료 범주를 쓴다.
T1_CATEGORICAL = [
    ("Sex", "gender", [(1, "Male"), (2, "Female")]),
    ("ECOG performance status", "ecog", [(0, "0"), (1, "1"), (2, "2"), (3, "3")]),
    ("mMRC dyspnea scale", "mmrc", [(k, str(k)) for k in range(5)]),
    ("Smoking status", "smoking_status", [(3, "Never"), (1, "Former"), (2, "Current"), (0, "Unknown")]),
    ("Diabetes mellitus", "dm", [(0, "No"), (1, "Yes")]),
    ("Hypertension", "htn", [(0, "No"), (1, "Yes")]),
    ("History of tuberculosis", "tb", [(0, "No"), (1, "Yes"), (-1, "Unknown")]),
    ("Chronic lung disease", "lung_raw", [("0", "None"), ("1", "COPD"), ("2", "Asthma"), ("3", "ILD/IPF"),
                                          ("1, 3", "COPD and ILD/IPF"), ("nan", "Unknown")]),
    ("Liver metastasis", "liver_meta", [(0, "No"), (1, "Yes")]),
    ("Brain metastasis at diagnosis", "brain_meta", [(0, "No"), (1, "Yes")]),
    ("Atezolizumab use", "atezolizumab", [(0, "No"), (1, "Yes")]),
    ("Radiotherapy type", "rt_type", [(0, "None"), (1, "CCRT"), (2, "Palliative RT"), (3, "Curative RT"),
                                      (-1, "Not recorded")]),
]


def baseline_frame(cohort: pd.DataFrame) -> pd.DataFrame:
    """238명 × (모델 입력 범주형 + 원자료 연속형·만성 폐질환)."""
    clin = cohort.drop_duplicates("research_id").set_index("research_id")
    raw = pd.read_excel(SOURCE_XLSX, sheet_name="Whole")
    raw.columns = [str(c).replace("\n", " ").strip() for c in raw.columns]
    raw["research_id"] = pd.to_numeric(raw["연구번호"], errors="coerce")
    raw = raw.dropna(subset=["research_id"]).set_index("research_id")
    raw.index = raw.index.astype(int)
    raw = raw.loc[clin.index]
    out = clin.copy()
    for _label, col, _ in T1_CONTINUOUS:
        out[col] = pd.to_numeric(raw[col], errors="coerce")
    out["lung_raw"] = raw["1 COPD 2 Asthma  3 ILDIPF"].astype(str).str.replace(".0", "", regex=False)
    return out


def table1(cohort: pd.DataFrame) -> str:
    df = baseline_frame(cohort)
    groups = [df, df[df.stage == 1], df[df.stage == 2]]
    ns = [len(g) for g in groups]

    def med_iqr(x):
        x = x.dropna()
        q1, q2, q3 = np.percentile(x, [25, 50, 75])
        return f"{q2:.1f} [{q1:.1f}--{q3:.1f}]"

    def subset(g, col):
        return g[g.smoking_status.isin([1, 2])] if col == "흡연력" else g

    rows = []
    for label, col, _ in T1_CONTINUOUS:
        rows.append(f"{label} & " + " & ".join(med_iqr(subset(g, col)[col]) for g in groups) + r" \\")
    for label, col, levels in T1_CATEGORICAL:
        rows.append(f"{label} & & & \\\\")
        covered = 0
        for code, name in levels:
            counts = [int((g[col] == code).sum()) for g in groups]
            covered += counts[0]
            rows.append(rf"\quad {name} & " + " & ".join(
                f"{k} ({100 * k / n:.0f})" for k, n in zip(counts, ns)) + r" \\")
        assert covered == ns[0], (col, covered)          # 범주 빠짐 방지

    missing = [(name, int(subset(df, col)[col].isna().sum())) for _l, col, name in T1_CONTINUOUS
               if subset(df, col)[col].isna().any()]
    n_smokers = int(df.smoking_status.isin([1, 2]).sum())
    head = (f"Characteristic & Overall (N={ns[0]}) & Limited stage (N={ns[1]}) & "
            f"Extensive stage (N={ns[2]}) \\\\")
    return "\n".join([
        r"\begingroup", r"\small", r"\renewcommand{\arraystretch}{1.05}", r"\setlength{\tabcolsep}{4pt}",
        r"\setlength{\LTleft}{0pt}", r"\setlength{\LTright}{0pt}",
        r"\begin{longtable}{", r"  >{\raggedright\arraybackslash}p{2.1in}",
        r"  >{\centering\arraybackslash}p{1.25in}", r"  >{\centering\arraybackslash}p{1.25in}",
        r"  >{\centering\arraybackslash}p{1.25in}", r"}",
        r"\caption{Baseline characteristics of the study cohort, stratified by disease stage}",
        r"\label{tab:baseline} \\", r"\toprule", head, r"\midrule", r"\endfirsthead",
        r"\multicolumn{4}{l}{\small\textit{Table \thetable\ continued from previous page}} \\",
        r"\toprule", head, r"\midrule", r"\endhead", r"\midrule",
        r"\multicolumn{4}{r}{\small\textit{Continued on next page}} \\", r"\endfoot",
        r"\bottomrule", r"\endlastfoot", *rows, r"\end{longtable}", r"\endgroup", "",
        r"\vspace{-0.5\baselineskip}", r"\noindent\footnotesize",
        r"Values are median [IQR] of observed values or No.\ (\%). Missing: "
        + "; ".join(f"{name} {k}" for name, k in missing) + ". "
        + rf"\textsuperscript{{a}}Former and current smokers (n = {n_smokers}). "
        r"Brain metastasis, at diagnosis only. COPD = chronic obstructive pulmonary disease, CCRT = concurrent "
        r"chemoradiotherapy, DLCO = diffusing capacity for carbon monoxide, ECOG = Eastern Cooperative "
        r"Oncology Group, FEV1 = forced expiratory volume in 1 s, FVC = forced vital capacity, ILD/IPF = "
        r"interstitial lung disease/idiopathic pulmonary fibrosis, IQR = interquartile range, LDH = lactate "
        r"dehydrogenase, mMRC = modified Medical Research Council, RT = radiotherapy",
        r"\normalsize", ""])


# ---------------------------------------------------------------------------
# LaTeX 서식
# ---------------------------------------------------------------------------
def f3(x):
    return f"{x:.3f}"


def fc(arm):
    return rf"{arm['mean']:.3f} $\pm$ {arm['sd']:.3f}"


def fp(p):
    if p is None:
        return "Reference"
    return "$<$ 0.001" if p < 0.001 else f"{p:.3f}"


def fci(lo, hi, nd=2):
    return f"{lo:.{nd}f}--{hi:.{nd}f}"


def fhr(h):
    return f"{h['hr']:.2f} ({fci(h['lo'], h['hi'])})"


def bold(s, on=True):
    return rf"\textbf{{{s}}}" if on else s


PREAMBLE = r"""\documentclass[border=8pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{newtxtext,newtxmath}
\usepackage{booktabs,multirow,array,pifont}
% --- 캡처할 때 제목·주석을 빼려면 아래 두 줄을 \...false 로 바꾼다 ---
\newif\ifshowtitle \showtitletrue
\newif\ifshownotes \shownotestrue
\newsavebox\tablebox
\begin{document}
"""


TABLE_NUMBERS = {"unimodal": 2, "multimodal": 3, "fusion": 4, "risk_groups": 5, "subgroups": 6,
                 "supp_multivariable": "S1"}


def _table_word(label: str) -> str:
    return "Supplementary Table" if label.startswith("supp") else "Table"


def _tabref(text: str, floating: bool) -> str:
    """주석 속 다른 표 참조 ``@TAB{label}`` -> 단독 파일은 "Table 5", main.tex 용은 \\ref."""
    for label, num in TABLE_NUMBERS.items():
        word = _table_word(label)
        text = text.replace(f"@TAB{{{label}}}", f"{word}~\\ref{{tab:{label}}}" if floating else f"{word} {num}")
    return text


@dataclass
class TableSpec:
    """표 하나 = 제목 + tabular + 주석. 단독 문서(캡처용)와 main.tex 용 float 두 가지로 낸다."""
    label: str          # tab:<label>, TABLE_NUMBERS 의 키
    section: str        # "3.2 Unimodal performance"
    title: str
    tabular: str
    notes: str
    float_size: str = r"\small\setlength{\tabcolsep}{4pt}"   # main.tex float 안의 글자 크기·열 간격

    def standalone(self) -> str:
        num = TABLE_NUMBERS[self.label]
        return (f"% Table {num} ({self.section})\n{GENERATED}\n{PREAMBLE}"
                "\\sbox\\tablebox{%\n" + self.tabular + "}%\n"
                "\\begin{minipage}{\\wd\\tablebox}\n"
                f"\\ifshowtitle\\noindent \\textbf{{{_table_word(self.label)} {num}.}} {self.title}\\par\\vspace{{4pt}}\\fi\n"
                "\\noindent\\usebox\\tablebox\n"
                "\\ifshownotes\\par\\vspace{3pt}\\footnotesize\\noindent "
                + _tabref(self.notes, False) + "\\par\\fi\n"
                "\\end{minipage}\n\\end{document}\n")

    def float_block(self) -> str:
        supp = self.label.startswith("supp")
        # 보충표: 본문 표 번호를 소모하지 않도록 이름·번호를 float 안에서만 바꾸고 카운터를 되돌린다
        relabel = (f"\\renewcommand{{\\tablename}}{{Supplementary Table}}"
                   f"\\renewcommand{{\\thetable}}{{{TABLE_NUMBERS[self.label]}}}\n") if supp else ""
        return (f"% ===== {self.section} =====\n"
                "\\begin{table}[htbp]\n\\centering\n" + relabel +
                "\\setlength{\\belowcaptionskip}{4pt}\n"
                f"\\caption{{{self.title}}}\n\\label{{tab:{self.label}}}\n{self.float_size}\n"
                "\\begin{adjustbox}{max width=\\textwidth}\n" + self.tabular + "\\end{adjustbox}\n"
                "\\par\\smallskip\n\\begin{minipage}{\\textwidth}\\footnotesize\n"
                + _tabref(self.notes, True) + "\n\\end{minipage}\n\\end{table}\n"
                + ("\\addtocounter{table}{-1}\n" if supp else ""))


GENERATED = ("% 자동 생성: tools/make_paper_tables.py — 수치를 여기서 고치지 말고 스크립트를 다시 돌릴 것.\n"
             "% 단독 문서다(캡처용). Overleaf 새 프로젝트에서 pdfLaTeX 로 컴파일한다.\n"
             "% 기존 main.tex 에 넣을 때는 이 파일이 아니라 overleaf_tables.tex 를 쓴다.")


def table2(rows) -> str:
    def line(mod, enc, key, sel=False):
        r = rows[key]
        return f"{mod} & {enc} & {bold(fc(r['os']), sel)} & {bold(fc(r['pfs']), sel)} \\\\\n"

    tab = (r"\begin{tabular}{@{}llcc@{}}" "\n\\toprule\n"
           r"Modality & Encoder & OS C-index & PFS C-index \\" "\n\\midrule\n"
           + line("Clinical variables", "MLP", "clin", True)
           + "\\addlinespace\n"
           + line(r"\multirow{4}{*}{Radiology report}", "TF-IDF", "report_tfidf")
           + line("", "mBERT", "report_mbert")
           + line("", "KM-BERT", "report_kmbert")
           + line("", "RadBERT", "report_radbert", True)
           + "\\addlinespace\n"
           + line(r"\multirow{3}{*}{PET-CT MIP}", "ResNet-18 (ImageNet)", "img_resnet18")
           + line("", "ResNet-50 (RadImageNet)", "img_radimagenet")
           + line("", "Simple CNN", "img_simplecnn", True)
           + "\\bottomrule\n\\end{tabular}%\n")
    notes = (r"Mean $\pm$ SD of the C-index across the five test folds (n = 238). Bold, encoder used in the "
             r"multimodal models. CNN = convolutional neural network, KM-BERT = Korean medical BERT, "
             r"mBERT = multilingual BERT, MIP = maximum intensity projection, MLP = multilayer perceptron, "
             r"OS = overall survival, PFS = progression-free survival, SD = standard deviation, "
             r"TF-IDF = term frequency--inverse document frequency")
    return TableSpec("unimodal", "3.2 Unimodal performance",
                     "Prognostic Performance of Unimodal Models", tab, notes)


def table3(rows) -> str:
    ck = r"\ding{51}"

    def line(c, r_, i, fusion, key, sel=False):
        a = rows[key]
        marks = " & ".join(ck if m else "" for m in (c, r_, i))
        return (f"{marks} & {fusion} & {bold(fc(a['os']), sel)} & {fp(a['os']['p_vs_proposed'])} & "
                f"{bold(fc(a['pfs']), sel)} & {fp(a['pfs']['p_vs_proposed'])} \\\\\n")

    tab = (r"\begin{tabular}{@{}cccl cc cc@{}}" "\n\\toprule\n"
           r"Clinical & Report & Image & Model & \multicolumn{2}{c}{OS} & \multicolumn{2}{c}{PFS} \\"
           "\n" r"\cmidrule(lr){5-6}\cmidrule(l){7-8}" "\n"
           r" & & & & C-index & \textit{P} & C-index & \textit{P} \\" "\n\\midrule\n"
           # 단일모달 행은 바로 앞 표(단일모달 성능)와 중복이라 뺐다 (2026-10-12 사용자 요청)
           + line(1, 1, 0, "Joint (concatenation)", "clin_report")
           + line(1, 0, 1, "Joint (concatenation)", "clin_img")
           + line(0, 1, 1, "Joint (concatenation)", "report_img")
           + "\\addlinespace\n"
           + line(1, 1, 1, "Proposed (late fusion)", "proposed")
           + "\\bottomrule\n\\end{tabular}%\n")
    notes = (r"Mean $\pm$ SD of the C-index across the five test folds (n = 238); \textit{P}, paired \textit{t} "
             r"test versus the proposed model. Joint, one end-to-end model on concatenated features; proposed, "
             r"late fusion of the clinical--report model and the image model. OS = overall survival, "
             r"PFS = progression-free survival, SD = standard deviation")
    return TableSpec("multimodal", "3.3 Multimodal performance",
                     "Prognostic Performance According to the Combination of Modalities", tab, notes)


def table4(rows) -> str:
    def line(strategy, arms, key):
        a = rows[key]
        return (f"{strategy} & {arms} & {fc(a['os'])} & {fp(a['os']['p_vs_proposed'])} & "
                f"{fc(a['pfs'])} & {fp(a['pfs']['p_vs_proposed'])} \\\\" "\n")

    def panel(title):
        return rf"\multicolumn{{6}}{{@{{}}l}}{{\textit{{{title}}}}} \\" "\n"

    tab = (r"\begin{tabular}{@{}ll cc cc@{}}" "\n\\toprule\n"
           r"Fusion strategy & Component models & \multicolumn{2}{c}{OS} & \multicolumn{2}{c}{PFS} \\"
           "\n" r"\cmidrule(lr){3-4}\cmidrule(l){5-6}" "\n"
           r" & & C-index & \textit{P} & C-index & \textit{P} \\" "\n\\midrule\n"
           + panel("A. Level of fusion")
           + line(r"\quad Early fusion", "[C + R + I]", "early")
           + line(r"\quad Late fusion, three-way", "C, R, I", "late3")
           + line(r"\quad Late fusion, two-way (proposed)", "[C + R], I", "proposed")
           + "\\addlinespace\n" + panel("B. Grouping of modalities in two-way late fusion")
           + line(r"\quad Late fusion, two-way", "[C + I], R", "g2")
           + line(r"\quad Late fusion, two-way", "[R + I], C", "g3")
           + "\\addlinespace\n"
           + panel("C. Adaptive fusion")
           + line(r"\quad Attention fusion\textsuperscript{a}", "[C + R + I]", "attention")
           # 2026-10-12 사용자 결정: C 패널 = 게이트 융합 1개 + attention 융합 1개(다음 세션에 실행·추가).
           # 튜닝 게이트·병기/나이 게이트는 같은 게이팅 계열이라 본문 표에서 뺐다(보충표 후보).
           + line(r"\quad Gated fusion\textsuperscript{b}", "[C + R], I", "gate")
           + "\\bottomrule\n\\end{tabular}%\n")
    # 게이트가 "사실상 꺼진" fold = 시험 환자 전체에서 게이트 변화가 꺼진 상태 대비 1% 미만
    n_fixed = {t: sum(r["gate_max_rel_change"] < 0.01 for r in rows["gate"][t]["records"]) for t in TARGETS}
    notes = (r"Mean $\pm$ SD of the C-index across the five test folds (n = 238); \textit{P}, paired \textit{t} "
             r"test versus the proposed model. Brackets, modalities trained jointly in one model. "
             r"\textsuperscript{a}Modality features combined by self-attention. "
             r"\textsuperscript{b}Cox weights multiplied by patient-specific gates (from the two risk scores); identical to the proposed model when the gate parameters are zero. Inner cross-validation turned "
             rf"the gates practically off in {n_fixed['os']} of 5 folds (OS) and {n_fixed['pfs']} of 5 (PFS). "
             r"C = clinical variables, I = PET-CT image, OS = overall survival, PFS = progression-free survival, "
             r"R = radiology report, SD = standard deviation")
    return TableSpec("fusion", "3.4 Fusion method comparison", "Comparison of Fusion Strategies", tab, notes)


def table5(km) -> str:
    def block(t, label):
        k = km[t]
        out = ""
        for g, name in (("low", "Low risk"), ("high", "High risk")):
            s = k[g]
            hr = "1 (reference)" if g == "low" else fhr(k["hr"])
            p = fp(k["logrank_p"]) if g == "low" else ""
            first = rf"\multirow{{2}}{{*}}{{{label}}}" if g == "low" else ""
            out += (f"{first} & {name} & {s['n']} & {s['events']} & "
                    f"{s['median']:.1f} ({fci(*s['median_ci'], nd=1)}) & "
                    f"{100 * s['rate_12']:.1f} & {100 * s['rate_24']:.1f} & {hr} & "
                    + (rf"\multirow{{2}}{{*}}{{{p}}}" if g == "low" else "") + " \\\\\n")
        return out

    tab = (r"\begin{tabular}{@{}ll cc c cc c c@{}}" "\n\\toprule\n"
           r"Endpoint & Risk group & Patients & Events & Median, months & 1-year & 2-year & "
           r"HR (95\% CI) & \textit{P} \\" "\n"
           r" & & & & (95\% CI) & rate, \% & rate, \% & & \\" "\n\\midrule\n"
           + block("os", "OS") + "\\addlinespace\n" + block("pfs", "PFS")
           + "\\bottomrule\n\\end{tabular}%\n")
    notes = (r"High risk, risk score above the median of the training patients of each fold; the five test folds "
             r"were pooled (not an independent test set). \textit{P}, log-rank test; HR, Cox model. "
             r"CI = confidence interval, HR = hazard ratio, OS = overall survival, PFS = progression-free survival")
    return TableSpec("risk_groups", "3.5 Survival stratification",
                     "Survival According to the Risk Group Defined by the Proposed Model", tab, notes)


def table6(sub) -> str:
    def cell_c(c):
        return f"{c['c']:.3f} ({fci(c['lo'], c['hi'], nd=3)})"

    body = ""
    for entry in sub:
        if entry["variable"] == "All patients":
            lvl = entry["levels"][0]
            body += (f"All patients & {lvl['n']} & {cell_c(lvl['os']['c'])} & {fhr(lvl['os']['hr'])} & & "
                     f"{cell_c(lvl['pfs']['c'])} & {fhr(lvl['pfs']['hr'])} & \\\\\n\\addlinespace\n")
            continue
        body += (f"{entry['variable']} & & & & {fp(entry['p_int']['os'])} & & & "
                 f"{fp(entry['p_int']['pfs'])} \\\\\n")
        for lvl in entry["levels"]:
            body += (rf"\quad {lvl['name']} & {lvl['n']} & {cell_c(lvl['os']['c'])} & {fhr(lvl['os']['hr'])}"
                     f" & & {cell_c(lvl['pfs']['c'])} & {fhr(lvl['pfs']['hr'])} & \\\\\n")
    tab = (r"\begin{tabular}{@{}lc ccc ccc@{}}" "\n\\toprule\n"
           r"Subgroup & Patients & \multicolumn{3}{c}{OS} & \multicolumn{3}{c}{PFS} \\" "\n"
           r"\cmidrule(lr){3-5}\cmidrule(l){6-8}" "\n"
           r" & & C-index (95\% CI) & HR (95\% CI) & \textit{P}\textsubscript{int} & "
           r"C-index (95\% CI) & HR (95\% CI) & \textit{P}\textsubscript{int} \\" "\n\\midrule\n"
           + body + "\\bottomrule\n\\end{tabular}%\n")
    notes = (r"C-index from patient pairs within the same test fold and subgroup (95\% CI, bootstrap). HR, high "
             r"versus low risk (@TAB{risk_groups}); \textit{P}\textsubscript{int}, Cox interaction test. "
             r"CI = confidence interval, ECOG = Eastern Cooperative Oncology Group, HR = hazard ratio, "
             r"OS = overall survival, PFS = progression-free survival")
    return TableSpec("subgroups", "3.6 Subgroup analysis",
                     "Performance of the Proposed Model in Clinical Subgroups", tab, notes,
                     float_size=r"\footnotesize\setlength{\tabcolsep}{2.5pt}")


def table_supp_multivariable(mv) -> TableSpec:
    def block(t, title):
        rows = rf"\multicolumn{{5}}{{@{{}}l}}{{\textit{{{title}}}}} \\" "\n"
        for c, label in MV_COVARIATES:
            u, m = mv[t]["uni"][c], mv[t]["multi"][c]
            rows += rf"\quad {label} & {fhr(u)} & {fp(u['p'])} & {fhr(m)} & {fp(m['p'])} \\" "\n"
        return rows

    tab = (r"\begin{tabular}{@{}l cc cc@{}}" "\n\\toprule\n"
           r"Variable & \multicolumn{2}{c}{Univariable} & \multicolumn{2}{c}{Multivariable} \\" "\n"
           r"\cmidrule(lr){2-3}\cmidrule(l){4-5}" "\n"
           r" & HR (95\% CI) & \textit{P} & HR (95\% CI) & \textit{P} \\" "\n\\midrule\n"
           + block("os", f"A. Overall survival (n = {mv['os']['n']}, {mv['os']['events']} deaths)")
           + "\\addlinespace\n"
           + block("pfs", f"B. Progression-free survival (n = {mv['pfs']['n']}, {mv['pfs']['events']} events)")
           + "\\bottomrule\n\\end{tabular}%\n")
    notes = (r"Risk group as in @TAB{risk_groups}. Because the model already uses these clinical variables, the "
             r"multivariable HR shows added prognostic information, not independence from them. "
             r"CI = confidence interval, ECOG = Eastern Cooperative Oncology Group, HR = hazard ratio")
    return TableSpec("supp_multivariable", "Supplementary: multivariable Cox (3.5)",
                     "Univariable and Multivariable Cox Regression Analyses of the Risk Group Defined by "
                     "the Proposed Model", tab, notes)


# ---------------------------------------------------------------------------
# KM 그림 (pgfplots groupplots, 좌표 내장 — Overleaf 에 파일 하나만 올리면 된다)
#   2026-10-12 사용자 요청(발표 슬라이드 스타일): x축 일(days)·전체 추적기간, y축 생존확률 0–1,
#   패널 제목 2줄(제목 + log-rank P 정확값), 초록/빨강 + 신뢰대, 범례는 그림 아래 가운데 한 번.
#   검토 의견 중 유지: 위험표(이제 노출 수만), 중도절단 눈금, 그림 안 HR, 패널 글자, 폭 170 mm,
#   고위험 점선(초록·빨강은 색각이상 독자가 구분하기 어렵다).
# ---------------------------------------------------------------------------
KM_X = {"os": (3200, 500), "pfs": (2600, 500)}   # (x 최대, 눈금 간격) days — 최대 추적 OS 3,126 / PFS 2,527일
GROUP_STYLE = {"low": ("kmlowline", "kmlow", "Low risk"), "high": ("kmhighline", "kmhigh", "High risk")}
KM_AXIS_MM = (56.3, 40)                          # 축 상자 폭·높이 (mm) — 2열 전체 폭이 170 mm 가 되도록
KM_LEFT_MM = 24                                  # 축 왼쪽 여백: 패널 글자·위험표 라벨이 여기 들어간다
KM_RIGHT_MM = 5.5                                # 마지막 눈금 숫자가 축 오른쪽으로 넘치는 폭
AT_RISK_Y = (-0.33, -0.43, -0.52)                # 축 단위(0–1 = 40 mm) — 위험표 제목, 저위험, 고위험


def _steps(times, values, end):
    """계단 함수 꼭짓점 (t_i 에서 값이 바뀌고 다음 t 까지 유지)."""
    out, prev = [], None
    for t, v in zip(times, values):
        if prev is not None:
            out.append((t, prev))
        out.append((t, v))
        prev = v
    out.append((end, prev))
    return out


def km_coords(g: pd.DataFrame, xmax: float):
    """곡선 꼭짓점, 중도절단 표식, 95% 신뢰대 다각형 (x = 일, y = 생존확률)."""
    days = g["months"] * DAYS_PER_MONTH
    kmf = KaplanMeierFitter().fit(days, g["event"])
    sf = kmf.survival_function_.iloc[:, 0]
    ci = kmf.confidence_interval_
    keep = sf.index <= xmax
    times = [float(t) for t in sf.index[keep]]
    end = min(xmax, float(days.max()))
    pts = [(0.0, 1.0)] + [(t, float(v)) for t, v in zip(times, sf[keep])] + [(end, float(sf[keep].iloc[-1]))]
    upper = _steps(times, ci.iloc[:, 1][keep].tolist(), end)
    lower = _steps(times, ci.iloc[:, 0][keep].tolist(), end)
    cens = days[(g["event"] == 0) & (days <= xmax)]
    marks = [(float(t), float(kmf.predict(t))) for t in cens]
    return pts, marks, upper + lower[::-1]


def _coords(pts):
    return " ".join(f"({t:.1f},{v:.4f})" for t, v in pts)


def at_risk_row(g: pd.DataFrame, ticks):
    """눈금(일)마다 위험 노출 수 (T ≥ t)."""
    days = g["months"] * DAYS_PER_MONTH
    return [str(int((days >= t).sum())) for t in ticks]


def fmt_p_sci(p: float) -> str:
    """log-rank P 를 정확값으로 — 0.001 미만은 지수 표기(예: 3.4 × 10^{-9})."""
    if p >= 0.001:
        return f"{p:.3f}"
    m, e = f"{p:.1e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


def km_panel(df: pd.DataFrame, target: str, letter: str, title: str) -> str:
    # x축은 그 패널의 최대 추적기간까지(100일 단위 올림) — 고정 범위면 추적이 짧은 패널(확장기)의 곡선이
    # 축 중간에서 끊겨 보였다
    step = KM_X[target][1]
    xmax = int(math.ceil(float(df["months"].max() * DAYS_PER_MONTH) / 100.0) * 100)
    ticks = list(range(0, xmax + 1, step))
    stat = km_summary(df)
    head = rf"\textbf{{{title}}}\\ Log-rank \textit{{P}} = {fmt_p_sci(stat['logrank_p'])}"
    s = (f"\\nextgroupplot[xmax={xmax}, xtick={{{','.join(map(str, ticks))}}}, title={{{head}}}]\n")
    curves = {g: km_coords(df[df.high == (g == "high")], xmax) for g in ("low", "high")}
    for g, (_pts, _marks, band) in curves.items():          # 신뢰대를 먼저 깔고 그 위에 곡선
        s += (f"\\addplot[draw=none, fill={GROUP_STYLE[g][1]}, fill opacity=0.18, forget plot] "
              f"coordinates {{{_coords(band)}}} -- cycle;\n")
    for g, (pts, marks, _band) in curves.items():
        line, color, _label = GROUP_STYLE[g]
        s += f"\\addplot[const plot, {line}, forget plot] coordinates {{{_coords(pts)}}};\n"
        if marks:
            s += (f"\\addplot[only marks, mark=|, mark size=1.6pt, {color}, forget plot] "
                  f"coordinates {{{_coords(marks)}}};\n")
    h = stat["hr"]
    s += (f"\\node[anchor=north east, font=\\kmsmall, inner sep=1pt] at (rel axis cs:0.99,0.99) "
          f"{{HR {h['hr']:.2f} (95\\% CI {fci(h['lo'], h['hi'])})}};\n")
    s += (f"\\node[anchor=south west, inner sep=0pt, font=\\kmletter, xshift=-{KM_LEFT_MM}mm, yshift=7mm] "
          f"at (rel axis cs:0,1) {{{letter}}};\n")
    # 위험표: 검은 글씨 + 군 표시용 짧은 색 선분, 숫자는 x 눈금 아래에 맞춘다
    s += (f"\\node[anchor=west, inner sep=0pt, font=\\kmsmall\\bfseries, xshift=-{KM_LEFT_MM}mm] "
          f"at (axis cs:0,{AT_RISK_Y[0]}) {{No.\\ at risk}};\n")
    for g, y in zip(("low", "high"), AT_RISK_Y[1:]):
        line, _color, label = GROUP_STYLE[g]
        s += (f"\\draw[{line}] ([xshift=-{KM_LEFT_MM}mm]axis cs:0,{y}) -- "
              f"([xshift=-{KM_LEFT_MM - 3.5}mm]axis cs:0,{y});\n")
        s += (f"\\node[anchor=west, inner sep=0pt, font=\\kmsmall, xshift=-{KM_LEFT_MM - 4.5}mm] "
              f"at (axis cs:0,{y}) {{{label}}};\n")
        s += "".join(f"\\node[font=\\kmsmall] at (axis cs:{t},{y}) {{{txt}}};\n"
                     for t, txt in zip(ticks, at_risk_row(df[df.high == (g == "high")], ticks)))
    return s


# KM 축 스타일 — 단독 문서와 main.tex 프리앰블(OVERLEAF_PREAMBLE)이 같은 정의를 쓴다
KM_SETUP = r"""\usepackage{pgfplots}
\usepgfplotslibrary{groupplots}
\usetikzlibrary{calc}
\pgfplotsset{compat=1.18}
% low risk = green, high risk = red
\definecolor{kmlow}{HTML}{3A9D5D}
\definecolor{kmhigh}{HTML}{E2574C}
% KM text is set in Times (ptm) regardless of the body font: Computer Modern 7 pt (cmr7) is wide
% enough that the at-risk numbers run into each other. Keeps main.tex and the standalone PDF identical.
\providecommand{\kmsmall}{\fontfamily{ptm}\fontsize{7}{8.4}\selectfont}
\providecommand{\kmlabel}{\fontfamily{ptm}\fontsize{8}{9.6}\selectfont}
\providecommand{\kmletter}{\fontfamily{ptm}\fontsize{10}{12}\selectfont\bfseries}
\tikzset{
  kmlowline/.style={color=kmlow, line width=1pt},
  kmhighline/.style={color=kmhigh, line width=1pt}}
\pgfplotsset{
  kmaxis/.style={
    scale only axis, width=""" + f"{KM_AXIS_MM[0]}mm, height={KM_AXIS_MM[1]}mm" + r""", xmin=0, ymin=0, ymax=1.02,
    ytick={0,0.2,0.4,0.6,0.8,1}, yticklabel style={/pgf/number format/.cd, fixed, fixed zerofill, precision=1},
    xticklabel style={/pgf/number format/1000 sep={}},
    xlabel={Time (days)}, ylabel={Survival probability}, axis lines=left, axis line style={-},
    tick align=outside, clip=false, tick label style={font=\kmsmall}, label style={font=\kmlabel},
    title style={font=\kmlabel, align=center, yshift=0.5mm}}}
"""

KM_PREAMBLE = (r"""\documentclass[border=0.5mm]{standalone}
\usepackage[T1]{fontenc}
\usepackage{newtxtext,newtxmath}
""" + KM_SETUP + r"""\begin{document}
\begin{tikzpicture}
""")


def km_groupplot(panels, rows: int) -> str:
    """패널 묶음 -> groupplot + 그림 아래 가운데 공통 범례 (tikzpicture 안에 들어갈 몸통)."""
    body = "".join(km_panel(*p) for p in panels)
    legend = (f"\\coordinate (kmleg) at ($(group c1r{rows}.south)!0.5!(group c2r{rows}.south)$);\n"
              "\\draw[kmlowline] ([xshift=-34mm,yshift=-27mm]kmleg) -- ++(6mm,0) "
              "node[right, font=\\kmsmall, text=black] {Low risk (reference)};\n"
              "\\draw[kmhighline] ([xshift=6mm,yshift=-27mm]kmleg) -- ++(6mm,0) "
              "node[right, font=\\kmsmall, text=black] {High risk};\n")
    return (f"\\begin{{groupplot}}[kmaxis, group style={{group size=2 by {rows}, "
            f"horizontal sep={KM_LEFT_MM + KM_RIGHT_MM + 2}mm, vertical sep=42mm}}]\n{body}\\end{{groupplot}}\n"
            + legend)


def km_figure(header: str, groupplot: str) -> str:
    return (f"{header}\n{GENERATED}\n{KM_PREAMBLE}{groupplot}"
            "\\end{tikzpicture}\n\\end{document}\n")


def km_captions() -> dict:
    """읽는 데 필요한 것만: 무엇을 그렸나, 음영·눈금·HR·P 의 뜻, 풀링 주의. 군별 수치는 Table 6 에 있다."""
    overall = (r"Kaplan--Meier curves of (a) overall and (b) progression-free survival by the risk group of the "
               r"proposed model. Shaded bands, 95\% CI; tick marks, censored patients; HR, high versus low risk "
               r"(reference); \textit{P}, log-rank test. Risk groups come from pooled out-of-fold predictions, "
               r"not an independent test set. CI = confidence interval, HR = hazard ratio")
    stage = (r"Kaplan--Meier curves by risk group in limited-stage (a, c) and extensive-stage (b, d) disease for "
             r"overall (a, b) and progression-free (c, d) survival. Symbols as in Figure~\ref{fig:km_overall}.")
    return {"fig_km_overall": overall, "fig_km_stage": stage}


# ---------------------------------------------------------------------------
# 기존 main.tex 에 붙여 넣는 묶음 (표 float 5개 + KM 그림 float 2개)
# ---------------------------------------------------------------------------
SNIPPET_HEADER = r"""% =============================================================================
% Results 3.2–3.6 표·그림 — 기존 main.tex 에 붙여 넣는 부분
% 자동 생성: tools/make_paper_tables.py — 수치를 여기서 고치지 말고 스크립트를 다시 돌릴 것.
%
% [1] main.tex 프리앰블(\begin{document} 위)에 overleaf_preamble.tex 내용을 추가한다.
%     이미 불러온 패키지가 겹쳐도 괜찮다.
% [2] 아래 블록을 \end{document} 위(또는 각 절 본문 뒤)에 붙여 넣는다.
%     표·그림 번호는 \caption 이 자동으로 매긴다. KM 그림도 코드로 들어 있어 업로드할 파일은 없다.
% =============================================================================

"""

OVERLEAF_PREAMBLE = ("% --- Results 3.2–3.6 표·KM 그림용 (tools/make_paper_tables.py) ---\n"
                     "\\usepackage{booktabs,multirow,pifont,adjustbox}\n" + KM_SETUP)

KM_FIGURES = [  # (절, 그림 이름, label) — 캡션은 km_captions() 가 수치를 넣어 만든다
    ("3.5 Survival stratification", "fig_km_overall", "km_overall"),
    ("3.6 Subgroup analysis", "fig_km_stage", "km_stage"),
]


def section_text(mv) -> dict:
    """절 본문에 넣을 요약 문장 {절 이름: LaTeX 문단}."""
    o, p = mv["os"]["multi"]["high"], mv["pfs"]["multi"]["high"]
    return {"3.5 Survival stratification": (
        r"In multivariable Cox analysis adjusted for stage, ECOG performance status, age, sex, and liver "
        r"and brain metastases, the high-risk group remained associated with shorter overall survival "
        rf"(HR {o['hr']:.2f}, 95\% CI {fci(o['lo'], o['hi'])}) and progression-free survival "
        rf"(HR {p['hr']:.2f}, 95\% CI {fci(p['lo'], p['hi'])}) (Supplementary Table~\ref{{tab:supp_multivariable}}). "
        r"Because the model uses these clinical variables as inputs, this indicates prognostic information "
        r"beyond their linear combination rather than independence from them." "\n\n")}


def overleaf_snippet(specs, km_bodies: dict, km_caps: dict, texts: dict | None = None) -> str:
    """표 float + KM 그림 float. ``km_bodies``·``km_caps`` = {그림 이름: groupplot 몸통 / 캡션}."""
    parts = [SNIPPET_HEADER]
    for spec in specs:
        if texts and spec.section in texts:
            parts.append(f"% ----- {spec.section}: text -----\n" + texts[spec.section])
        parts.append(spec.float_block() + "\n")
        for section, name, label in KM_FIGURES:
            if section == spec.section:
                parts.append(f"% ===== {section} (KM) =====\n"
                             "\\begin{figure}[htbp]\n\\centering\n"
                             "\\begin{adjustbox}{max width=\\textwidth}\n\\begin{tikzpicture}\n"
                             + km_bodies[name] +
                             "\\end{tikzpicture}\n\\end{adjustbox}\n"
                             f"\\caption{{{km_caps[name]}}}\n\\label{{fig:{label}}}\n\\end{{figure}}\n\n")
    return "".join(parts)


# ---------------------------------------------------------------------------
def main():
    cohort = load_trimodal_cohort()
    rows = collect_cv_results()
    scores = {t: final_model_test_scores(cohort, t) for t in TARGETS}

    # 재현 대조: 결합기를 다시 적합한 test 점수의 fold 평균 C-index 가 README ★절 값과 같아야 한다
    for t in TARGETS:
        cis = [concordant_pair_counts(g["months"], g["event"], g["risk"]) for _, g in scores[t].groupby("fold")]
        fold_mean = float(np.mean([c / m for c, m in cis]))
        assert abs(fold_mean - rows["proposed"][t]["mean"]) < 1e-3, (t, fold_mean)

    km = {t: km_summary(scores[t]) for t in TARGETS}
    mv = multivariable_cox(cohort, scores)
    sub = subgroup_analysis(cohort, scores)
    clin = cohort.drop_duplicates("research_id").set_index("research_id")

    tables = {
        "tab2_unimodal": table2(rows),
        "tab3_multimodal": table3(rows),
        "tab4_fusion": table4(rows),
        "tab5_risk_groups": table5(km),
        "tab6_subgroups": table6(sub),
        "tabS1_multivariable": table_supp_multivariable(mv),     # 보충표는 맨 끝에 놓인다
    }
    km_bodies = {"fig_km_overall": km_groupplot(
        [(scores["os"], "os", "a", "Overall survival"),
         (scores["pfs"], "pfs", "b", "Progression-free survival")], rows=1)}
    stage_sets = []                                  # (df, target, 병기) — a,b = OS LS/ES, c,d = PFS LS/ES
    for t in TARGETS:
        for code, nm in ((1, "Limited stage"), (2, "Extensive stage")):
            df = scores[t]
            mask = (clin.loc[df["research_id"], "stage"] == code).to_numpy()
            stage_sets.append((df[mask], t, nm))
    km_bodies["fig_km_stage"] = km_groupplot(
        [(df, t, "abcd"[i], f"{nm}, {'OS' if t == 'os' else 'PFS'}") for i, (df, t, nm) in enumerate(stage_sets)],
        rows=2)
    km_caps = km_captions()

    outputs = {name: spec.standalone() for name, spec in tables.items()}
    outputs["fig_km_overall"] = km_figure("% KM curves for 3.5 Survival stratification",
                                          km_bodies["fig_km_overall"])
    outputs["fig_km_stage"] = km_figure("% KM curves for 3.6 Subgroup analysis (stage)",
                                        km_bodies["fig_km_stage"])
    outputs["overleaf_tables"] = overleaf_snippet(list(tables.values()), km_bodies, km_caps, section_text(mv))
    outputs["overleaf_preamble"] = OVERLEAF_PREAMBLE
    outputs["overleaf_table1"] = ("% Table 1 (기저 특성) — main.tex 의 기존 Table 1 블록(begingroup 부터 주석 끝 normalsize 까지)을 이것으로 바꾼다\n"
                                  + GENERATED.splitlines()[0] + "\n" + table1(cohort))

    for name, tex in outputs.items():
        with open(os.path.join(PAPER_DIR, f"{name}.tex"), "w", encoding="utf-8") as fh:
            fh.write(tex)
        print(f"wrote report_assets/paper/{name}.tex")

    stats_path = os.path.join(paths.ensure_outputs("paper_tables"), "stats.json")
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump({"cv": rows, "km": km, "subgroups": sub, "multivariable": mv,
                   "high_per_fold": {t: scores[t].groupby("fold")["high"].agg(["sum", "count"]).to_dict()
                                     for t in TARGETS}},
                  fh, indent=1, ensure_ascii=False)
    print(f"wrote {os.path.relpath(stats_path, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
