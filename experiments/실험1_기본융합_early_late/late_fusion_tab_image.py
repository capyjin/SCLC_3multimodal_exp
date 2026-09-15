# -*- coding: utf-8 -*-
"""방식 B: 2축 LATE fusion (후기 융합) — **프로젝트 최종 채택 모델(OS)**
  = [임상+판독지 결합 tabular 위험점수] + [영상 위험점수]
  누수 없는(leakage-safe) 5-fold CoxPH 결합.

원리 및 배경 (RESULTS.md §8 참고):
  초기 융합(concat)에서 영상을 통째로 섞어 학습시켰더니 임상+판독지만 썼을
  때(0.708)보다 셋 다 썼을 때(0.678)가 오히려 낮았다.

  그래서 LATE fusion 을 쓴다: 각 축(임상+판독지, 영상)을 **자기에게 맞는
  에포크/배치로 독립 학습**한 뒤, 거기서 나온 OOF 위험점수 2개만 모아
  최적 비율로 합친다.

비교 대상 (OS/PFS 동일):
  1. tabular  = 임상+판독지 결합, 영상 제외, bs32/ep60 (가장 강한 기본 모델)
  2. image A  = SimpleCNN, bs16/ep30 (영상 arm 의 표준 조건)
  3. image B  = ImageNet 사전학습 ResNet18, bs16/ep30, 백본 lr 1e-5 + head lr 1e-3, 좌우반전 증강

안전한 결합 (CoxPH stack):
  모든 환자의 위험점수는 자신이 test 였던 fold 의 모델이 낸 값(OOF)만 쓴다.
  fold 마다 train 환자의 위험점수로 CoxPH 를 적합해 test 에 적용한다.

이 파일은 **실험 드라이버**다. 재사용되는 부품(각 축의 OOF 추출기,
CoxPH 결합기)은 다른 실험 4개도 쓰기 때문에 ``core.fusion_stack`` 에 있다.

Run:  python 실험1_기본융합_early_late/late_fusion_tab_image.py --targets os,pfs
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import json

from core import cohort
from core.fusion_stack import (cindex_stats, combine_two, get_image_oof_resnet18,
                               get_image_oof_simplecnn, get_tabular_oof, oof_dict)

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "late_fusion_B")


def run_target(target, max_folds=None, tab_epochs=60, img_epochs=30,
               resnet_epochs=30, seed=42, fix_brain_meta: bool = True):
    """한 타깃(os 또는 pfs)에 대해 3개 축을 학습하고 2가지 결합을 평가한다.

    fix_brain_meta: 기저 시점 뇌전이만 1 로 재코딩(누수 수정). 기본 True.
    """
    cohort_df = cohort.load_trimodal_cohort(fix_brain_meta=fix_brain_meta)

    # 1. Tabular (임상+판독지) 단독
    print(f"\n########## [lateB] TABULAR (clin+report joint, bs32/ep{tab_epochs}) target={target} ##########")
    tab_ev = get_tabular_oof(target, epochs=tab_epochs, max_folds=max_folds, seed=seed,
                             fix_brain_meta=fix_brain_meta, out_dir=OUT_DIR)
    tab_risk = oof_dict(tab_ev.oof_predictions)
    tab_mean, tab_std = cindex_stats(tab_ev.c_indices)
    print(f"[lateB] tabular-only {target}: {tab_mean:.4f} +/- {tab_std:.4f}  folds={[round(float(c), 4) for c in tab_ev.c_indices]}")

    # 2. SimpleCNN 영상 단독
    print(f"\n########## [lateB] IMAGE SimpleCNN (bs16/ep{img_epochs}) target={target} ##########")
    scnn_ev = get_image_oof_simplecnn(target, epochs=img_epochs, max_folds=max_folds, seed=seed, out_dir=OUT_DIR)
    scnn_risk = oof_dict(scnn_ev.oof_predictions)
    scnn_mean, scnn_std = cindex_stats(scnn_ev.c_indices)
    print(f"[lateB] image-SimpleCNN-only {target}: {scnn_mean:.4f} +/- {scnn_std:.4f}  folds={[round(float(c), 4) for c in scnn_ev.c_indices]}")

    # 3. ResNet18 영상 단독
    print(f"\n########## [lateB] IMAGE ResNet18 (pretrained, bs16/ep{resnet_epochs}) target={target} ##########")
    rn_ev = get_image_oof_resnet18(target, epochs=resnet_epochs, max_folds=max_folds, seed=seed, out_dir=OUT_DIR)
    rn_risk = oof_dict(rn_ev.oof_predictions)
    rn_mean, rn_std = cindex_stats(rn_ev.c_indices)
    print(f"[lateB] image-ResNet18-only {target}: {rn_mean:.4f} +/- {rn_std:.4f}  folds={[round(float(c), 4) for c in rn_ev.c_indices]}")

    # 4~5. Late fusion 결합 2가지
    print(f"\n########## [lateB] COMBINE tabular+SimpleCNN target={target} ##########")
    comb_scnn = combine_two(cohort_df, target, tab_risk, scnn_risk, max_folds=max_folds)
    print(f"[lateB] late-fusion tabular+SimpleCNN {target}: {comb_scnn['mean']:.4f} +/- {comb_scnn['std']:.4f} "
          f"mean_coef={comb_scnn['mean_coef']}")

    print(f"\n########## [lateB] COMBINE tabular+ResNet18 target={target} ##########")
    comb_rn = combine_two(cohort_df, target, tab_risk, rn_risk, max_folds=max_folds)
    print(f"[lateB] late-fusion tabular+ResNet18 {target}: {comb_rn['mean']:.4f} +/- {comb_rn['std']:.4f} "
          f"mean_coef={comb_rn['mean_coef']}")

    def _arm(mean, std, ev):
        return {"mean": mean, "std": std, "folds": [round(float(c), 4) for c in ev.c_indices]}

    return {
        "target": target,
        "tabular_only": _arm(tab_mean, tab_std, tab_ev),
        "image_simplecnn_only": _arm(scnn_mean, scnn_std, scnn_ev),
        "image_resnet18_only": _arm(rn_mean, rn_std, rn_ev),
        "late_simplecnn": comb_scnn,
        "late_resnet18": comb_rn,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="os,pfs")
    ap.add_argument("--smoke", action="store_true", help="빠른 테스트용 (1 fold, 2 에포크만)")
    ap.add_argument("--max_folds", type=int, default=None)
    ap.add_argument("--tab_epochs", type=int, default=60)
    ap.add_argument("--img_epochs", type=int, default=30)
    ap.add_argument("--resnet_epochs", type=int, default=30)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "results.json"))
    ap.add_argument("--no_fix_brain_meta", dest="fix_brain_meta", action="store_false",
                    help="brain_meta 누수 수정을 끄고 2026-08-02 이전 동작으로 실행(legacy 재현용).")
    args = ap.parse_args()

    if args.smoke:
        args.max_folds, args.tab_epochs, args.img_epochs, args.resnet_epochs = 1, 2, 2, 2

    os.makedirs(OUT_DIR, exist_ok=True)
    all_results = {}
    for target in [t.strip() for t in args.targets.split(",") if t.strip()]:
        all_results[target] = run_target(
            target, max_folds=args.max_folds, tab_epochs=args.tab_epochs,
            img_epochs=args.img_epochs, resnet_epochs=args.resnet_epochs,
            fix_brain_meta=args.fix_brain_meta,
        )

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[lateB] wrote {args.out}")

    print("\n================ METHOD B SUMMARY ================")
    for target, r in all_results.items():
        print(f"\n--- target={target} ---")
        for key, label in (("tabular_only", "tabular-only          "),
                           ("image_simplecnn_only", "image SimpleCNN-only  "),
                           ("image_resnet18_only", "image ResNet18-only   ")):
            print(f"  {label}: {r[key]['mean']:.4f} +/- {r[key]['std']:.4f}")
        for key, label in (("late_simplecnn", "late fusion +SimpleCNN"),
                           ("late_resnet18", "late fusion +ResNet18 ")):
            c = r[key]["mean_coef"]
            print(f"  {label}: {r[key]['mean']:.4f} +/- {r[key]['std']:.4f}"
                  f"   coef(tab,img)=({c['risk_tabular']:.3f},{c['risk_image']:.3f})")


if __name__ == "__main__":
    main()
