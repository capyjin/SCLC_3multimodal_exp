# -*- coding: utf-8 -*-
"""[재실행] 3-way late fusion(clinical·판독지·영상 각각 독립 → CoxPH 가중합)을
개선된 학습조건(batch=32, epochs=60)으로 다시 잰다.

배경:
  이 방식(late_fusion_3modal.py 의 run_clinical_only/run_report_only/run_image_only
  + core.fusion_stack.combine_weighted_sum)은 2026-07-22 에 한 번 돌았다(outputs/EXP_20260722_late_fusion_train/
  metrics/summary.json: OS 0.6703 / PFS 0.6288). 그런데 그 실행의 clinical_only·report_only
  기본값이 **batch_size=16, epochs=30** 이었다 — 이후 §4(RESULTS.md)에서 batch=16 이 Cox
  risk set 을 좁혀 tabular 를 과소학습시킨다는 게 밝혀져 batch=32/epoch=60 으로 개선됐는데,
  그 개선이 method B(late_fusion_tab_image.py, 지금 쓰는 2-way)에는 적용됐지만 이 3-way
  방식에는 한 번도 재적용된 적이 없었다. 이 스크립트가 그 재실행이다.

무엇을 새로 학습하고 무엇을 재사용하나:
  - clinical_only, report_only : late_fusion_3modal.run_clinical_only/run_report_only 를
    batch=32/epochs=60 으로 **새로 학습**한다(pycox 기반, 이 실험 고유 아키텍처를 그대로
    유지 — 실험3 ablation 의 clin_only/report_only 와는 다른 코드 경로이므로 재사용하지 않는다).
  - image_only : **재학습하지 않는다.** outputs/late_fusion_B/oof_{target}.json 의 "image"
    키에 이미 저장된 OOF 위험점수(개선된 조건, batch=16/epochs=30 — 영상은 원래도 이 조건이
    표준이었다, RESULTS.md §1.5(c) 참고)를 그대로 재사용한다. 체크포인트(.pt)는 지워졌지만
    OOF 예측값 자체는 JSON에 남아 있다.

누수 방지: run_clinical_only/run_report_only 는 각 fold 의 train 환자로만 인코더를 fit 한다
  (기존 late_fusion_3modal.py 로직 그대로, 이 스크립트는 배치·에폭만 바꿔서 호출한다).
  combine_weighted_sum 은 fold 별 train 환자의 OOF 로만 CoxPH 를 적합한다.

Run:  python 실험1_기본융합_early_late/exp_late_fusion_3modal_rerun.py --target os
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import json

import late_fusion_3modal as lf
from core import cohort
from core.fusion_stack import combine_weighted_sum, oof_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="os", choices=("os", "pfs"))
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()

    out_dir = os.path.join(PROJECT_ROOT, "outputs", "late_fusion_3modal_rerun")
    os.makedirs(out_dir, exist_ok=True)

    cohort_df = cohort.load_trimodal_cohort()  # 기본값 fix_brain_meta=True

    print(f"\n########## clinical_only  target={args.target}  bs={args.batch_size} ep={args.epochs} ##########")
    clin = lf.run_clinical_only(cohort_df, args.target, batch_size=args.batch_size, epochs=args.epochs)
    clin_ci = [r["c_index"] for r in clin["fold_records"]]
    print(f"[3WAY] clinical_only {args.target}: {sum(clin_ci)/5:.4f}  folds={[round(float(c), 4) for c in clin_ci]}")

    print(f"\n########## report_only  target={args.target}  bs={args.batch_size} ep={args.epochs} ##########")
    rep = lf.run_report_only(cohort_df, args.target, cohort.DEFAULT_MERGED_CSV,
                             batch_size=args.batch_size, epochs=args.epochs)
    rep_ci = [r["c_index"] for r in rep["fold_records"]]
    print(f"[3WAY] report_only {args.target}: {sum(rep_ci)/5:.4f}  folds={[round(float(c), 4) for c in rep_ci]}")

    print(f"\n########## image_only  target={args.target}  (재사용, 재학습 없음) ##########")
    img_oof_saved = json.load(open(os.path.join(PROJECT_ROOT, "outputs", "late_fusion_B",
                                                 f"oof_{args.target}.json")))["image"]
    img_risk = {int(k): v for k, v in img_oof_saved.items()}
    print(f"[3WAY] image_only {args.target}: (재사용) n={len(img_risk)}")

    clin_risk = oof_dict(clin["oof_predictions"])
    rep_risk = oof_dict(rep["oof_predictions"])

    print(f"\n########## COMBINE (3-way weighted sum)  target={args.target} ##########")
    combined = combine_weighted_sum(cohort_df, args.target, img_risk, clin_risk, rep_risk)
    cis = [r["c_index"] for r in combined["fold_records"]]
    coefs = [r["coefficients"] for r in combined["fold_records"]]
    mean_ci = sum(cis) / 5
    print(f"[3WAY] late_fusion_weighted_sum {args.target}: {mean_ci:.4f}  folds={[round(float(c), 4) for c in cis]}")

    result = {
        "target": args.target, "batch_size": args.batch_size, "epochs": args.epochs,
        "clinical_only": {"mean": sum(clin_ci)/5, "folds": [round(float(c), 4) for c in clin_ci]},
        "report_only": {"mean": sum(rep_ci)/5, "folds": [round(float(c), 4) for c in rep_ci]},
        "image_only": {"note": "reused from outputs/late_fusion_B (no retrain)"},
        "late_fusion_weighted_sum": {"mean": mean_ci, "folds": [round(float(c), 4) for c in cis],
                                     "coefficients_per_fold": coefs},
    }
    path = os.path.join(out_dir, f"results_{args.target}.json")
    json.dump(result, open(path, "w"), indent=2)

    known_legacy = {"os": 0.6703, "pfs": 0.6288}[args.target]  # batch16/ep30 조건, 참고용
    print(f"\n================ SUMMARY ({args.target}) ================")
    print(f"clinical_only            {sum(clin_ci)/5:.4f}")
    print(f"report_only              {sum(rep_ci)/5:.4f}")
    print(f"late_fusion_weighted_sum {mean_ci:.4f}   (batch16/ep30 조건 참고값: {known_legacy:.4f})")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
