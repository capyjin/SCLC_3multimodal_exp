# -*- coding: utf-8 -*-
"""⑦ 논문 Table 2(단일 모달)·Table 3(모달리티 조합)을 nested CV 결과로 만든다. 재학습 없음.

    출처                                          행
    outputs/gmu_nested/run_v2_{os,pfs}/           Late fusion(A_primary) · 임상+판독지 joint(tabular_only) · Simple CNN(image_only)
    outputs/gmu_nested/baselines_v1_{os,pfs}/     ⑥ run_baselines.py 의 나머지 행

규약: C-index = outer test 5 fold 값의 평균 ± 표본 SD (ddof=1, 실험16 README·report_v2 와 같음).
      P = Late fusion 대비 fold 쌍대 t 검정 (n = 5). 기존 비-nested 표(tools/make_paper_tables.py)와 섞지 않는다.
서식(TableSpec · 프리앰블 · 숫자 형식)은 tools/make_paper_tables.py 의 것을 그대로 쓴다.

Run:  python experiments/실험16_GMU_nestedCV/make_nested_tables.py
      tools/build_paper_figures.sh tab2_unimodal_nested tab3_multimodal_nested"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

import make_paper_tables as mp
from nested_fusion.config import BASELINE_TAG, RUN_TAG, guard
from sclc import paths

TARGETS = ("os", "pfs")
RUN_V2_ROWS = {"proposed": "A_primary", "clin_report": "tabular_only", "img_simplecnn": "image_only"}
BASELINE_ROWS = {"clin": "clin_only", "report_tfidf": "report_tfidf", "report_mbert": "report_mbert",
                 "report_kmbert": "report_kmbert", "report_radbert": "report_radbert", "clin_img": "clin_image",
                 "report_img": "report_image", "img_resnet18": "image_resnet18", "img_radimagenet": "image_radimagenet"}
GENERATED = ("% 자동 생성: experiments/실험16_GMU_nestedCV/make_nested_tables.py. 수치를 여기서 고치지 말고 스크립트를 다시 돌릴 것.\n"
             "% 단독 문서다(캡처용). Overleaf 새 프로젝트에서 pdfLaTeX 로 컴파일한다.\n")


def fold_values(path: str, model: str, target: str) -> list[float]:
    """fold_metrics.csv 에서 outer 1-5 C-index (반올림 전 값). outer 마다 한 줄, 시험 환자 합 238 명을 확인한다."""
    df = pd.read_csv(path)
    sub = df[df.model == model].sort_values("outer")
    guard(sub.outer.tolist() == [1, 2, 3, 4, 5], f"{target} {model}: outer {sub.outer.tolist()}")
    guard(int(sub.n.sum()) == 238, f"{target} {model}: 시험 환자 합 {int(sub.n.sum())} != 238")
    return sub.cindex.astype(float).tolist()


def collect(baselines_root: str) -> dict:
    rows = {}
    for t in TARGETS:
        run = paths.outputs("gmu_nested", f"{RUN_TAG}_{t}", "fold_metrics.csv")
        base = os.path.join(baselines_root.format(target=t), "fold_metrics.csv")
        for key, model in RUN_V2_ROWS.items():
            rows.setdefault(key, {})[t] = fold_values(run, model, t)
        for key, model in BASELINE_ROWS.items():
            rows.setdefault(key, {})[t] = fold_values(base, model, t)
        # 대조: run_v2 요약(report_v2)의 Late fusion 평균과 같아야 한다
        summ = json.load(open(paths.outputs("gmu_nested", "report_v2", "summary.json"), encoding="utf-8"))
        guard(abs(np.mean(rows["proposed"][t]) - summ[t]["cindex"]["A_primary"]["mean"]) < 1e-4,
              f"{t}: Late fusion 평균이 report_v2 와 다르다")
    out = {}
    for key, by_t in rows.items():
        out[key] = {}
        for t, folds in by_t.items():
            ref = rows["proposed"][t]
            out[key][t] = {"mean": float(np.mean(folds)), "sd": float(np.std(folds, ddof=1)), "folds": folds,
                           "p_vs_proposed": None if key == "proposed" else float(stats.ttest_rel(folds, ref).pvalue)}
    return out


def table2(rows) -> mp.TableSpec:
    def line(mod, enc, key, sel=False):
        r = rows[key]
        return f"{mod} & {enc} & {mp.bold(mp.fc(r['os']), sel)} & {mp.bold(mp.fc(r['pfs']), sel)} \\\\\n"

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
    notes = (r"Mean $\pm$ SD (sample SD) of the C-index across the five outer test folds of nested "
             r"cross-validation (n = 238). Bold, encoder used in the multimodal models. "
             r"CNN = convolutional neural network, KM-BERT = Korean medical BERT, "
             r"mBERT = multilingual BERT, MIP = maximum intensity projection, MLP = multilayer perceptron, "
             r"OS = overall survival, PFS = progression-free survival, SD = standard deviation, "
             r"TF-IDF = term frequency--inverse document frequency")
    return mp.TableSpec("unimodal", "3.2 Unimodal performance", "Prognostic Performance of Unimodal Models", tab, notes)


def table3(rows) -> mp.TableSpec:
    ck = r"\ding{51}"

    def line(c, r_, i, model, key, sel=False):
        a = rows[key]
        marks = " & ".join(ck if m else "" for m in (c, r_, i))
        return (f"{marks} & {model} & {mp.bold(mp.fc(a['os']), sel)} & {mp.fp(a['os']['p_vs_proposed'])} & "
                f"{mp.bold(mp.fc(a['pfs']), sel)} & {mp.fp(a['pfs']['p_vs_proposed'])} \\\\\n")

    tab = (r"\begin{tabular}{@{}cccl cc cc@{}}" "\n\\toprule\n"
           r"Clinical & Report & Image & Model & \multicolumn{2}{c}{OS} & \multicolumn{2}{c}{PFS} \\"
           "\n" r"\cmidrule(lr){5-6}\cmidrule(l){7-8}" "\n"
           r" & & & & C-index & \textit{P} & C-index & \textit{P} \\" "\n\\midrule\n"
           + line(1, 0, 0, "Unimodal", "clin")
           + line(0, 1, 0, "Unimodal", "report_radbert")
           + line(0, 0, 1, "Unimodal", "img_simplecnn")
           + "\\addlinespace\n"
           + line(1, 1, 0, "Joint (concatenation)", "clin_report")
           + line(1, 0, 1, "Joint (concatenation)", "clin_img")
           + line(0, 1, 1, "Joint (concatenation)", "report_img")
           + "\\addlinespace\n"
           + line(1, 1, 1, "Proposed (late fusion)", "proposed")
           + "\\bottomrule\n\\end{tabular}%\n")
    notes = (r"Mean $\pm$ SD (sample SD) of the C-index across the five outer test folds of nested "
             r"cross-validation (n = 238); \textit{P}, paired \textit{t} test versus the proposed model "
             r"(five folds). Joint, one end-to-end model on concatenated features; proposed, late fusion of the "
             r"clinical--report model and the image model. OS = overall survival, "
             r"PFS = progression-free survival, SD = standard deviation")
    return mp.TableSpec("multimodal", "3.3 Multimodal performance",
                        "Prognostic Performance According to the Combination of Modalities", tab, notes)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--baselines_root", default=paths.outputs("gmu_nested", f"{BASELINE_TAG}_{{target}}"),
                    help="⑥ 결과 폴더 ({target} 자리표시자)")
    ap.add_argument("--out_suffix", default="_nested")
    args = ap.parse_args()
    rows = collect(args.baselines_root)
    mp.GENERATED = GENERATED          # TableSpec.standalone 머리말의 생성 스크립트 표기만 바꾼다
    for name, spec in (("tab2_unimodal", table2(rows)), ("tab3_multimodal", table3(rows))):
        path = os.path.join(mp.PAPER_DIR, f"{name}{args.out_suffix}.tex")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(spec.standalone())
        print(f"wrote {os.path.relpath(path, PROJECT_ROOT)}")
    out = os.path.join(paths.ensure_outputs("gmu_nested", "paper_tables"), "stats.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)
    print(f"wrote {os.path.relpath(out, PROJECT_ROOT)}")
    for key, by_t in rows.items():
        print(f"{key:16s} " + "  ".join(f"{t} {v['mean']:.3f} ± {v['sd']:.3f} P={v['p_vs_proposed']}" for t, v in by_t.items()))


if __name__ == "__main__":
    main()
