# -*- coding: utf-8 -*-
"""실험14 — CNN 영상 기여가 '종양'인가 '크롭 기하/전역 밝기'인가.

[왜 이 실험이 필요한가]
  실험10(exp_trivial_stats.py)이 불편한 사실을 남겼다. PNG 에서 눈감고 뽑은
  전역 통계 6개(w/h/mean/std/frac_hot/frac_dark)만으로 OS 0.6479 가 나오는데,
  CNN 은 0.6570 이다. 차이가 0.009 뿐이다. 게다가 단일 최고 변수가
  ``frac_hot``(밝은 화소 비율 = MIP 에서는 사실상 **배경 면적**, 0.6262)이고
  크롭 폭 ``w`` 단독도 0.5708 이다. 크롭 크기가 환자마다 다르므로
  (382x510, 418x510, 464x511 ...) **체격/FOV 자체가 신호일 수 있다.**

  그러나 실험10 은 "전역 통계도 잘 맞힌다"까지만 보여줄 뿐, 정작 결정적인 질문에
  답하지 않았다 — **그 교란을 통제한 뒤에도 CNN 이 더하는 것이 남는가?**
  두 축이 같은 것을 보고 있다면 통제 후 beta_img 는 0 으로 무너져야 한다.

[설계]  재학습 없음. 저장된 OOF 위험점수 + 라벨을 보지 않는 이미지 함수만 쓴다.
  T = risk_tabular (RadBERT tabular arm OOF)
  I = risk_image   (SimpleCNN OOF)
  N = w,h,mean,std,frac_hot,frac_dark (교란 6종, 전역 z-표준화)

  ① 교란보정 beta_img       [T+N+I] CoxPH 에서 I 의 계수 -- ★핵심
  ② 우도비 사다리            T vs T+I / T vs T+N / **T+N vs T+N+I** / T+I vs T+I+N
  ③ late fusion C-index 사다리 (fold-safe stack: 결합기는 train fold 로만 적합)
  ④ CNN 점수의 몇 %가 교란으로 설명되나 (fold 내 R^2, 설명가능성 상한)
  ⑤ 종양 지표(LDH/간전이/병기)와의 상관 -- 기하(w,h) 통제 전후 부분상관

[누수 방어]  교란 6종은 라벨을 전혀 보지 않는 순수 이미지 함수라 fold 밖에서 한 번
  계산해도 안전하다(실험10 과 동일 논리). 표준화도 라벨 무관이므로 전역으로 한다.
  CoxPH 결합기는 여전히 **fold 의 train 환자 OOF 로만** 적합한다.

Run:  python experiments/실험14_영상_교란보정_검정/exp_image_confound_adjusted.py
      [--out_dir outputs/image_confound] [--oof_dir outputs/late_fusion_B_radbert_tests]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy import stats as sps

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from sclc import cohort, late_fusion_tests  # noqa: E402
from sclc.dataset import INVERTED_IMAGE_IDS  # noqa: E402
from sclc.evaluation import cindex  # noqa: E402
from sclc.radiomics import TRIVIAL_NAMES, extract_trivial_stats  # noqa: E402
from sclc.train import fold_plan  # noqa: E402

TAB, IMG = "risk_tabular", "risk_image"
NUIS = list(TRIVIAL_NAMES)
#: 6개를 성질로 쪼갠다 — 이 분해가 이 실험의 결론을 가른다.
#:   GEOM  크롭 폭/높이. 종양과 무관한 **촬영/편집 아티팩트** 후보.
#:   INT   전역 밝기·대비·면적. 이 중 frac_dark(어두운 화소=고섭취 병변 면적)와
#:         std(대비)는 교란이 아니라 **대사 종양부하의 조악한 측정치**다. 따라서
#:         이걸로 보정하는 것은 "종양을 통제하고도 종양이 보이나"를 묻는
#:         과잉보정(over-adjustment)이 된다 -- 해석할 때 반드시 구분해야 한다.
GEOM = ["w", "h"]
INT = ["mean", "std", "frac_hot", "frac_dark"]
#: 실험10 ALL6 이 재현되어야 하는 값 — 교란 특징 정의가 바뀌지 않았는지 확인한다.
TRIVIAL_ALL6_KNOWN = {"os": 0.6478744561749575, "pfs": 0.6114220072230603}
#: 종양 부하 지표(영상이 '종양'을 봤다면 붙어야 하는 것) 와 숙주 지표(붙을 이유가 없는 것)
TUMOR_MARKERS = ["ldh", "liver_meta", "stage", "brain_meta"]
HOST_MARKERS = ["ecog", "age_at_diagnosis", "fev1_pre_percent_ref"]


def load_oof(oof_dir: str, target: str) -> dict[str, dict]:
    with open(os.path.join(oof_dir, f"oof_{target}.json")) as f:
        raw = json.load(f)
    return {TAB: {int(k): float(v) for k, v in raw["tabular"].items()},
            IMG: {int(k): float(v) for k, v in raw["image"].items()}}


def zscored_nuisance(image_dir: str, ids) -> pd.DataFrame:
    """교란 6종을 전역 z-표준화. 라벨 무관 변환이라 fold 밖에서 해도 누수가 아니다."""
    raw = extract_trivial_stats(image_dir, ids, INVERTED_IMAGE_IDS)
    return (raw - raw.mean()) / raw.std().replace(0.0, 1.0)


def as_risk_maps(frame: pd.DataFrame) -> dict[str, dict]:
    """DataFrame -> ``{컬럼: {research_id: 값}}`` (risk_frame 이 먹는 형태)."""
    return {c: frame[c].to_dict() for c in frame.columns}


def fit_cox(frame: pd.DataFrame, cols: list[str]) -> CoxPHFitter:
    return CoxPHFitter().fit(frame[cols + ["duration", "event"]],
                             duration_col="duration", event_col="event")


def coef_block(model: CoxPHFitter, covariate: str) -> dict:
    s = model.summary.loc[covariate]
    return {"coef": float(s["coef"]), "lo": float(s["coef lower 95%"]),
            "hi": float(s["coef upper 95%"]), "p": float(s["p"])}


def beta_per_fold(risks, labels, plan, target, cols, covariate) -> list[dict]:
    """fold 의 train 환자로만 적합한 CoxPH 에서 covariate 의 계수 블록."""
    rows = []
    for fold, ids in plan:
        train_df = late_fusion_tests.risk_frame(risks, ids["train"], labels, target, cols)
        rows.append({"fold": int(fold)} | coef_block(fit_cox(train_df, cols), covariate))
    return rows


def stack_cindices(risks, labels, plan, target, cols) -> list[float]:
    """fold-safe late fusion: 결합기를 train fold 로 적합하고 test fold 에서 평가."""
    out = []
    for _fold, ids in plan:
        train_df = late_fusion_tests.risk_frame(risks, ids["train"], labels, target, cols)
        test_df = late_fusion_tests.risk_frame(risks, ids["test"], labels, target, cols)
        model = fit_cox(train_df, cols)
        pred = model.predict_partial_hazard(test_df[cols]).to_numpy()
        out.append(cindex(test_df["duration"], pred, test_df["event"]))
    return out


def explained_r2(risks, nuis, plan) -> dict:
    """CNN 위험점수의 분산 중 교란 6종이 **펼치는(span)** 비율 — fold 안에서만 계산.

    ⚠️ fold 마다 다른 CNN 이 낸 점수라 척도가 다르다. train fold 점수로 적합해
    test fold 를 맞히면 척도 drift 가 오차로 잡혀 R^2 가 음수로 폭주한다
    (실측: fold 별 -4.57 ~ +0.55). 라벨이 전혀 개입하지 않는 질문이므로
    **각 test fold 안에서** OLS 를 적합한 기술통계 R^2 로 답한다. 이 값은
    '설명 가능성의 상한'이며, 낮으면 CNN 점수가 교란의 재조합이 아니라는 뜻이다.
    """
    per_fold = []
    for fold, ids in plan:
        te = ids["test"]
        x = nuis.loc[te, NUIS].to_numpy(dtype=float)
        y = np.array([risks[IMG][i] for i in te], dtype=float)
        design = np.column_stack([np.ones(len(x)), x])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ beta
        ss_tot = float(((y - y.mean()) ** 2).sum())
        per_fold.append({"fold": int(fold), "n": len(te),
                         "r2": 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot else None})
    vals = [r["r2"] for r in per_fold if r["r2"] is not None]
    return {"per_fold": per_fold, "mean": float(np.mean(vals)),
            "note": "fold 내 in-sample OLS (라벨 무관, 설명가능성 상한)"}


def partial_spearman(x, y, controls: np.ndarray | None) -> tuple[float, float]:
    """controls 를 순위공간에서 선형 제거한 뒤의 Spearman (controls=None 이면 보통 Spearman)."""
    xr = sps.rankdata(x).astype(float)
    yr = sps.rankdata(y).astype(float)
    if controls is not None and controls.size:
        c = np.column_stack([sps.rankdata(controls[:, j]) for j in range(controls.shape[1])])
        c = np.column_stack([np.ones(len(c)), c])
        xr = xr - c @ np.linalg.lstsq(c, xr, rcond=None)[0]
        yr = yr - c @ np.linalg.lstsq(c, yr, rcond=None)[0]
    r, p = sps.pearsonr(xr, yr)
    return float(r), float(p)


def marker_associations(risks, nuis, clinical: pd.DataFrame, ids) -> dict:
    """영상/교란 위험점수가 종양 지표·숙주 지표와 얼마나 붙는가 (기하 통제 전후)."""
    geom = nuis.loc[ids, ["w", "h"]].to_numpy(dtype=float)
    img = np.array([risks[IMG][i] for i in ids], dtype=float)
    out = {}
    for name in TUMOR_MARKERS + HOST_MARKERS:
        if name not in clinical.columns:
            continue
        v = pd.to_numeric(clinical.loc[ids, name], errors="coerce").to_numpy(dtype=float)
        m = ~np.isnan(v)
        if m.sum() < 30:
            continue
        rho_raw, p_raw = partial_spearman(img[m], v[m], None)
        rho_adj, p_adj = partial_spearman(img[m], v[m], geom[m])
        out[name] = {"n": int(m.sum()), "kind": "tumor" if name in TUMOR_MARKERS else "host",
                     "rho_raw": rho_raw, "p_raw": p_raw,
                     "rho_geom_adjusted": rho_adj, "p_geom_adjusted": p_adj}
    return out


def nuisance_marker_table(nuis, clinical: pd.DataFrame, ids) -> dict:
    """교란 6종 각각이 종양 지표와 붙는가 — '교란인가 종양 측정치인가'의 판별식.

    w/h(기하)는 종양 지표와 무관해야 하고, 실제로 그렇다. 반면 mean/std/
    frac_hot/frac_dark 는 LDH - 병기 - 간전이와 유의하게 붙는다. 즉 이 넷은
    교란이 아니라 **대사 종양부하의 조악한 측정치**이고, 이걸로 보정한 뒤
    beta_img 가 사라지는 것은 아티팩트의 증거가 아니라 과잉보정의 결과다.
    """
    out = {}
    for f in NUIS:
        x = nuis.loc[ids, f].to_numpy(dtype=float)
        row = {}
        for m in TUMOR_MARKERS + HOST_MARKERS:
            if m not in clinical.columns:
                continue
            v = pd.to_numeric(clinical.loc[ids, m], errors="coerce").to_numpy(dtype=float)
            msk = ~np.isnan(v)
            if msk.sum() < 30:
                continue
            r, pv = sps.spearmanr(x[msk], v[msk])
            row[m] = {"rho": float(r), "p": float(pv), "n": int(msk.sum())}
        out[f] = row
    return out


def run_target(target: str, oof_dir: str, labels, plan, nuis, clinical, log) -> dict:
    risks = load_oof(oof_dir, target) | as_risk_maps(nuis)
    ids_all = sorted(labels.index[labels.index.isin(nuis.index)])
    full = late_fusion_tests.risk_frame(risks, ids_all, labels, target, [TAB, IMG] + NUIS)
    res = {"target": target, "n": len(ids_all)}

    # ── 0) 교란 정의 재현 확인 (실험10 ALL6 과 같은 값이 나와야 한다) ─────────
    only_nuis = {c: risks[c] for c in NUIS}
    trivial_folds = stack_cindices(only_nuis, labels, plan, target, NUIS)
    res["trivial_all6_reproduction"] = {
        "folds": [round(v, 4) for v in trivial_folds], "mean": float(np.mean(trivial_folds)),
        "known": TRIVIAL_ALL6_KNOWN[target],
        "note": "실험10 은 penalizer=0.1 + fold내 표준화라 소수점 셋째자리 차이는 정상"}

    # ── ① 교란보정 beta_img ────────────────────────────────────────────────
    res["beta_img"] = {}
    for label, cols in (("unadjusted_T+I", [TAB, IMG]),
                        ("geom_adjusted_T+G+I", [TAB, IMG] + GEOM),
                        ("int_adjusted_T+INT+I", [TAB, IMG] + INT),
                        ("full_adjusted_T+N+I", [TAB, IMG] + NUIS)):
        rows = beta_per_fold(risks, labels, plan, target, cols, IMG)
        coefs = np.array([r["coef"] for r in rows])
        res["beta_img"][label] = {
            "per_fold": rows, "mean": float(coefs.mean()),
            "positive_folds": int((coefs > 0).sum()),
            "significant_folds": int(sum(1 for r in rows if r["p"] < 0.05)),
            "pooled": coef_block(fit_cox(full, cols), IMG)}

    # ── ② 우도비 사다리 ────────────────────────────────────────────────────
    ladder = {"T_vs_T+I": ([TAB], [TAB, IMG]),
              "T_vs_T+GEOM": ([TAB], [TAB] + GEOM),
              "T_vs_T+INT": ([TAB], [TAB] + INT),
              "T_vs_T+N": ([TAB], [TAB] + NUIS),
              "T+GEOM_vs_T+GEOM+I": ([TAB] + GEOM, [TAB] + GEOM + [IMG]),
              "T+INT_vs_T+INT+I": ([TAB] + INT, [TAB] + INT + [IMG]),
              "T+N_vs_T+N+I": ([TAB] + NUIS, [TAB] + NUIS + [IMG]),
              "T+I_vs_T+I+N": ([TAB, IMG], [TAB, IMG] + NUIS)}
    res["lrt"] = {k: late_fusion_tests.likelihood_ratio_test(full, red, fullc)
                  for k, (red, fullc) in ladder.items()}

    # ── ③ late fusion C-index 사다리 ───────────────────────────────────────
    res["stack_cindex"] = {}
    for label, cols in (("T", [TAB]), ("T+I", [TAB, IMG]),
                        ("T+GEOM", [TAB] + GEOM), ("T+GEOM+I", [TAB] + GEOM + [IMG]),
                        ("T+INT", [TAB] + INT), ("T+N", [TAB] + NUIS),
                        ("T+N+I", [TAB] + NUIS + [IMG])):
        folds = stack_cindices(risks, labels, plan, target, cols)
        res["stack_cindex"][label] = {"folds": [round(v, 4) for v in folds],
                                      "mean": float(np.mean(folds)),
                                      "std": float(np.std(folds, ddof=1))}

    # ── ③b 교란 6종 각각의 계수 — 어느 것이 신호를 나르며 부호는 어느 쪽인가 ──
    m = fit_cox(full, [TAB] + NUIS)
    res["nuisance_coefs"] = {c: coef_block(m, c) for c in NUIS}

    # ── ④ CNN 점수 중 교란으로 설명되는 비율 ────────────────────────────────
    res["image_explained_by_nuisance_r2"] = explained_r2(risks, nuis, plan)

    # ── ⑤ 종양/숙주 지표와의 (부분)상관 ────────────────────────────────────
    res["marker_associations"] = marker_associations(risks, nuis, clinical, ids_all)
    res["nuisance_marker_associations"] = nuisance_marker_table(nuis, clinical, ids_all)
    res["nuisance_intercorrelation"] = {
        a: {b: float(sps.spearmanr(nuis.loc[ids_all, a], nuis.loc[ids_all, b])[0]) for b in NUIS}
        for a in NUIS}

    report(target, res, log)
    return res


def report(target: str, r: dict, log) -> None:
    p = log
    p(f"\n{'='*72}\n  target = {target.upper()}   (n={r['n']})\n{'='*72}")
    t = r["trivial_all6_reproduction"]
    p(f"[0] 교란6 단독 재현: {t['mean']:.4f}  (실험10 기록 {t['known']:.4f})")

    p("\n[1] beta_img — 교란 보정 전/후  ★핵심")
    for label in r["beta_img"]:
        b = r["beta_img"][label]
        po = b["pooled"]
        p(f"  {label:22s} pooled {po['coef']:+.3f} [{po['lo']:+.3f}, {po['hi']:+.3f}] "
          f"p={po['p']:.4f} | fold평균 {b['mean']:+.3f} | 양수 {b['positive_folds']}/5 "
          f"| 유의 {b['significant_folds']}/5")

    p("\n[2] 우도비 검정 사다리")
    for k, v in r["lrt"].items():
        mark = "***" if v["p"] < 0.01 else ("*" if v["p"] < 0.05 else "   ")
        p(f"  {k:16s} chi2={v['stat']:7.2f}  df={v['df']}  p={v['p']:.4f} {mark}")

    p("\n[3] late fusion C-index (fold-safe)")
    base = r["stack_cindex"]["T"]["mean"]
    for k, v in r["stack_cindex"].items():
        p(f"  {k:8s} {v['mean']:.4f} +/- {v['std']:.4f}   delta_vs_T {v['mean']-base:+.4f}   {v['folds']}")

    e = r["image_explained_by_nuisance_r2"]
    p(f"\n[4] CNN 위험점수가 교란6 로 설명되는 비율: fold내 R^2 = {e['mean']:.3f}  "
      f"(fold별 {[round(x['r2'], 3) for x in e['per_fold']]})")

    p("\n[5] 영상 위험점수 vs 임상지표 — 기하(w,h) 통제 전/후 부분상관")
    for name, v in r["marker_associations"].items():
        p(f"  [{v['kind']:5s}] {name:22s} raw {v['rho_raw']:+.3f} (p={v['p_raw']:.4f})   "
          f"-> 기하보정 {v['rho_geom_adjusted']:+.3f} (p={v['p_geom_adjusted']:.4f})")

    p("\n[6] 교란 6종 vs 임상지표 — 어느 것이 '기하'이고 어느 것이 '종양 측정치'인가 (* p<0.05)")
    cols = [m for m in (TUMOR_MARKERS + HOST_MARKERS) if m in next(iter(r["nuisance_marker_associations"].values()))]
    p("  " + " " * 10 + "".join(f"{m[:11]:>13s}" for m in cols))
    for f, row in r["nuisance_marker_associations"].items():
        kind = "기하" if f in GEOM else "강도"
        line = f"  {f:8s}({kind})" if False else f"  {f:10s}"
        for m in cols:
            v = row[m]
            line += f"{v['rho']:+.3f}{'*' if v['p'] < 0.05 else ' '}".rjust(13)
        p(line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--oof_dir", default=os.path.join(PROJECT_ROOT, "outputs",
                                                      "late_fusion_B_radbert_tests"),
                    help="oof_<target>.json 이 있는 폴더 (기본: RadBERT seed42)")
    ap.add_argument("--out_dir", default=os.path.join(PROJECT_ROOT, "outputs", "image_confound"))
    ap.add_argument("--targets", default="os,pfs")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    cohort_df = cohort.load_trimodal_cohort()
    labels = cohort_df.drop_duplicates("research_id").set_index("research_id")
    plan = fold_plan(cohort_df)
    nuis = zscored_nuisance(cohort.DEFAULT_IMAGE_DIR, labels.index)
    clinical = pd.read_csv(os.path.join(PROJECT_ROOT, "data",
                                        "merged_tabular_with_reports.csv")).set_index("research_id")

    out = {"oof_dir": os.path.relpath(args.oof_dir, PROJECT_ROOT),
           "nuisance_features": NUIS, "results": {}}
    for target in args.targets.split(","):
        out["results"][target] = run_target(target, args.oof_dir, labels, plan, nuis, clinical, log)

    path = os.path.join(args.out_dir, "results.json")
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    with open(os.path.join(args.out_dir, "run.log"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
