# -*- coding: utf-8 -*-
"""[실험] 판독지 텍스트에서 뽑은 SUVmax 수치가 현재 최고 모델을 더 좋게 하는가?

동기:
  clin_report(임상+판독지 TF-IDF, 영상 브랜치 없음)가 현재 최고다 (OS 0.7076 / PFS 0.6678).
  판독지는 TF-IDF(char n-gram)로만 쓰이고 있어서, 판독지 안의 **수치**(SUVmax)는
  사실상 문자 조각으로 흩어져 버린다. SUVmax 는 SCLC 예후 인자로 알려져 있으므로
  이걸 명시적인 숫자 feature 로 뽑아 주면 이득이 있는지 3단계로 쌓아 올리며 본다.

단계 (누적):
  none                 : SUV 없음 = clin_report 그대로 (재현 기준선, 0.7076/0.6678)
  suv_max              : + 문서 내 최대 SUV
  suv_max+count        : + 값 개수(≈수치 기재된 병변 수, 결측 지시자 역할 겸함)
  suv_max+count+mean   : + 평균 SUV

고정되는 것(= 오직 SUV 열만 달라진다):
  모델 조합(clin_report), fold split, seed, epochs, batch_size, TF-IDF(400차원, 패딩 고정),
  텍스트 source(concl_find). SUV 열은 **임상 블록 뒤**에 붙어서 clinical 브랜치로
  들어간다 → report 브랜치 차원(400)은 변하지 않고 모델 구조 변경도 없다.

누수 방지 (이 저장소의 기존 규율과 동일):
  정규식 파싱은 환자별 결정론적 연산이라 전역 1회 수행해도 무방하다.
  환자들을 가로질러 계산되는 통계(결측 대치 median, StandardScaler)만 fold 별로
  **train fold 환자만** 써서 fit 한다 (suv_features.make_extra_numeric_fn).
  실제 사용된 fold별 median/표본수는 로그와 결과 JSON(leakage_audit)에 남긴다.

Run:  python exp_suv_features.py --target os
      python exp_suv_features.py --target os --steps none,suv_max
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import json

import numpy as np

from core import cohort, features
import suv_features
from core.metrics import train_val_gap
from core.model import MODALITY_CONFIGS, make_model_factory

MODEL_CONFIG = "clin_report"   # 이 실험에서 고정하는 모델 조합
STEP_ORDER = ["none", "suv_max", "suv_max+count", "suv_max+count+mean"]
STEP_DESC = {
    "none":               "SUV 없음 (clin_report 재현 기준선)",
    "suv_max":            "문서 내 최대 SUV",
    "suv_max+count":      "최대 SUV + 값 개수",
    "suv_max+count+mean": "최대 SUV + 개수 + 평균",
}
# 기존 기준선 (RESULTS.md / outputs/text_source/concl_find). 재현 확인용.
KNOWN_BASELINE = {"os": 0.7076, "pfs": 0.6678}


def preflight(table, cols, tfidf_max_features=400):
    """학습 전에 fold별 feature 행렬을 실제로 만들어 보고 (1) 텐서 폭이 기대대로인지,
    (2) SUV 열이 환자마다 실제로 변하는지(상수/전부결측이 아닌지),
    (3) 결측 대치 median 이 정말 train fold 환자만으로 계산됐는지를 눈으로 확인한다.
    학습 없이 수 초면 끝난다."""
    cohort_df = cohort.load_trimodal_cohort()
    clinical_frame = cohort_df.drop_duplicates("research_id").set_index("research_id")
    std_cols, cat_cols = features.resolve_clinical_columns(clinical_frame)
    corpus, _ = features.load_text_corpus(cohort.DEFAULT_MERGED_CSV)
    audit = []
    fn = suv_features.make_extra_numeric_fn(table, cols, audit=audit) if cols else None

    dims, checks = None, []
    for fold in sorted(int(f) for f in cohort_df["fold"].unique()):
        fdf = cohort_df[cohort_df["fold"] == fold]
        ids = {s: fdf.loc[fdf["split"] == s, "research_id"].astype(int).tolist()
               for s in ("train", "val", "test")}
        tab, clinical_dim, report_dim = features.build_fold_multimodal_tabular(
            clinical_frame.loc[ids["train"]], clinical_frame.loc[ids["val"]],
            clinical_frame.loc[ids["test"]], corpus, std_cols, cat_cols,
            tfidf_max_features=tfidf_max_features, extra_numeric_fn=fn,
        )
        dims = (clinical_dim, report_dim, tab["train"].shape[1])
        if cols:
            # SUV 블록은 [len(std)+len(cat) : clinical_dim) 구간에 있다.
            base = len(std_cols) + len(cat_cols)
            blk = tab["train"][:, base:clinical_dim]
            checks.append({"fold": fold,
                           "n_unique": [int(len(np.unique(blk[:, j]))) for j in range(blk.shape[1])],
                           "std": [round(float(blk[:, j].std()), 4) for j in range(blk.shape[1])]})
    clinical_dim, report_dim, width = dims
    n_extra = clinical_dim - (len(std_cols) + len(cat_cols))
    print(f"[preflight] clinical={len(std_cols)}+{len(cat_cols)}={len(std_cols) + len(cat_cols)} "
          f"+ suv={n_extra} -> clinical_dim={clinical_dim}; report(tfidf)={report_dim}; "
          f"tabular width={width} (expected {len(std_cols) + len(cat_cols)} + {len(cols)} + {tfidf_max_features} "
          f"= {len(std_cols) + len(cat_cols) + len(cols) + tfidf_max_features})")
    assert n_extra == len(cols) and width == clinical_dim + report_dim, "feature width mismatch"
    for c in checks:
        print(f"[preflight] fold {c['fold']} SUV columns {list(cols)}: "
              f"n_unique={c['n_unique']} std={c['std']} (상수면 1/0.0 이 찍힌다)")
    for a in audit:
        print(f"[leakage-audit] fold n_train={a['n_train']} (SUV 있는 train 환자 {a['n_train_with_suv']}명) "
              f"-> train-only median={a['train_fold_medians']} | "
              f"결측 대치된 환자 수 {a['n_imputed']} | n_val={a['n_val']} n_test={a['n_test']}")
    return audit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="os", choices=("os", "pfs"))
    ap.add_argument("--steps", default=",".join(STEP_ORDER[1:]),
                    help="쉼표 목록. 'none'(기준선) 포함 가능. 기본값은 SUV 3단계.")
    # bs32/ep60 은 이 저장소에서 확립된 설정 (RESULTS.md 9장).
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--preflight_only", action="store_true", help="학습 없이 feature 점검만")
    args = ap.parse_args()

    names = [s.strip() for s in args.steps.split(",") if s.strip()]
    unknown = [n for n in names if n not in suv_features.STEPS]
    if unknown:
        raise SystemExit(f"unknown step(s) {unknown}; expected from {STEP_ORDER}")

    # ── 1) 추출 (fold 무관, 전역 1회) + 분포 self-check ──
    table = suv_features.extract_suv_table()
    vs = table.attrs["value_stats"]
    print(f"\n=== SUV extraction self-check ===")
    print(f"reports={table.attrs['n_reports']}  legend_sentence={table.attrs['n_legend']} "
          f"({table.attrs['n_legend'] / table.attrs['n_reports'] * 100:.1f}%)  "
          f"docs_with_value={int(table['suv_available'].sum())} "
          f"({table['suv_available'].mean() * 100:.1f}%)  n_values={vs['n']}")
    print(f"values: min={vs['min']} median={vs['median']} mean={vs['mean']:.2f} "
          f"p95={vs['p95']} max={vs['max']}  outside_plausible_range={vs['n_outside_plausible_range']}")
    csum = suv_features.cohort_summary(table)
    print(f"cohort(238): suv_available={csum['n_available']} missing={csum['n_missing']}  "
          f"suv_max median={csum['suv_max']['median']} unique={csum['suv_max']['n_unique']}  "
          f"count hist={csum['suv_count_hist']}")

    out_dir = "outputs/suv_features"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"results_{args.target}.json")
    # 설정(epochs/batch_size/model)이 같을 때만 이어붙인다 — smoke test(ep2) 결과가
    # 본 실험 파일에 섞여 비교 불가능한 숫자가 한 표에 들어가는 사고를 막는다.
    prev = json.load(open(path)) if os.path.exists(path) else {}
    same = (prev.get("epochs") == args.epochs and prev.get("batch_size") == args.batch_size
            and prev.get("model_config") == MODEL_CONFIG)
    if prev and not same:
        print(f"[warn] {path} 는 다른 설정(ep={prev.get('epochs')}, bs={prev.get('batch_size')})으로 "
              "만들어졌다 -> 이어붙이지 않고 새로 시작한다.")
    results = prev.get("steps", {}) if same else {}

    from core.train import TrimodalEvaluator

    for name in names:
        cols = suv_features.STEPS[name]
        print(f"\n########## STEP: {name}  ({STEP_DESC[name]})  cols={list(cols)}  "
              f"model={MODEL_CONFIG}  target={args.target} bs={args.batch_size} ep={args.epochs} ##########")
        pre_audit = preflight(table, cols)
        if args.preflight_only:
            continue

        audit = []
        fn = suv_features.make_extra_numeric_fn(table, cols, audit=audit) if cols else None
        ev = TrimodalEvaluator(
            target=args.target, epochs=args.epochs, batch_size=args.batch_size,
            save_dir=f"{out_dir}/{name}_{args.target}",
            model_factory=make_model_factory(MODALITY_CONFIGS[MODEL_CONFIG]),
            extra_numeric_fn=fn,     # <- 이 실험에서 유일하게 바뀌는 것
        ).run()
        cis = ev.c_indices

        gaps = train_val_gap(ev.training_history)   # 과적합 정도 (정의는 core.metrics)

        results[name] = {
            "step": name, "desc": STEP_DESC[name], "suv_cols": list(cols),
            "mean": float(np.mean(cis)), "std": float(np.std(cis)),
            "folds": [round(float(c), 4) for c in cis],
            "train_val_gap_mean": float(np.mean(gaps)) if gaps else None,
            "leakage_audit": audit,          # fold별 train-only median/표본수
            "fold_records": ev.fold_records,
            "training_history": ev.training_history,
        }
        with open(path, "w") as f:   # step 하나 끝날 때마다 저장 (crash-safe)
            json.dump({
                "target": args.target, "batch_size": args.batch_size, "epochs": args.epochs,
                "model_config": MODEL_CONFIG, "flags": MODALITY_CONFIGS[MODEL_CONFIG],
                "extraction": {"n_reports": table.attrs["n_reports"], "n_legend": table.attrs["n_legend"],
                               "n_docs_with_value": int(table["suv_available"].sum()),
                               "value_stats": vs, "cohort": csum},
                "steps": results,
            }, f, indent=2)

        print(f"[SUV] {name} {args.target}: {np.mean(cis):.4f} +/- {np.std(cis):.4f} folds={results[name]['folds']}")
        if name == "none":
            known = KNOWN_BASELINE[args.target]
            ok = abs(np.mean(cis) - known) < 1e-3
            print(f"[SUV] baseline reproduction check: got {np.mean(cis):.4f} vs known {known} "
                  f"-> {'MATCH' if ok else '*** MISMATCH ***'}")

    if args.preflight_only:
        return
    base = results.get("none", {}).get("mean")
    print(f"\n================ SUV FEATURE SUMMARY (target={args.target}, model={MODEL_CONFIG}, "
          f"bs{args.batch_size}/ep{args.epochs}) ================")
    print(f"{'step':<22}{'mean':>8}{'std':>8}{'delta':>9}{'gap':>8}   folds")
    for name in STEP_ORDER:
        r = results.get(name)
        if not r:
            continue
        d = f"{r['mean'] - base:+.4f}" if base is not None and name != "none" else "-"
        gap = f"{r['train_val_gap_mean']:.4f}" if r.get("train_val_gap_mean") is not None else "-"
        print(f"{name:<22}{r['mean']:>8.4f}{r['std']:>8.4f}{d:>9}{gap:>8}   {r['folds']}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
