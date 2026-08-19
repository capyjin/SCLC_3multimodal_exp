# -*- coding: utf-8 -*-
"""[실험3] 모달리티 절제(ablation) — "영상을 더하면 좋아지나, 방해가 되나?"

────────────────────────────────────────────────────────────────────────────
[이 실험이 하는 일 — 큰 그림]

폐암(SCLC) 환자 238명의 생존 예측 모델을 만든다. 환자마다 3종류의 정보가 있다.
  1) image    : PET-CT 영상
  2) clinical : 나이·혈액수치 같은 임상(숫자) 데이터
  3) report   : 판독지 텍스트

궁금한 것: **"영상까지 넣으면 예측이 더 좋아질까, 오히려 방해가 될까?"**
그래서 재료를 하나씩 빼보는 실험(ablation, 절제)을 한다.
  예) 임상+판독지만  vs  임상+판독지+영상  → 점수 비교

공정한 비교를 위해 고정한 것 (통제 변인):
  같은 환자 238명, 같은 5-fold 분할, 같은 학습 루프, 같은 하이퍼파라미터,
  같은 시드, fusion 방식은 concat(early) 고정.
  **오직 브랜치를 켜고 끄는 것만** 다르다.

⚠️ 여기 수치는 전부 **concat fusion 한정**이다. 같은 3모달이라도 late fusion
   (실험1 method B)이면 결과가 다르다(OS 0.678 vs 0.722). 두 표를 섞지 말 것.
────────────────────────────────────────────────────────────────────────────

조합(``core.model.MODALITY_CONFIGS``):
  all / clin_report / clin_image / clin_only / report_only / image_only

모델 클래스는 ``core.model.ConcatDeepSurv`` 다 — 예전에는 이 파일이
``AblatableConcatDeepSurv`` 라는 사본을 따로 갖고 있었는데, 다른 실험 10여 개가
이 파일을 import 하는 구조가 되어 core 로 옮겼다.

Run:  python 실험3_모달리티_절제실험/ablation.py --target os --epochs 60 --batch_size 32
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import json

import numpy as np

from core.metrics import train_val_gap
from core.model import MODALITY_CONFIGS, make_model_factory
from core.train import TrimodalEvaluator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="os", choices=("os", "pfs"))   # 전체생존 / 무진행생존
    ap.add_argument("--configs", default=",".join(MODALITY_CONFIGS))   # 돌릴 조합 목록
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--tag", default="", help="결과 저장 폴더 이름에 붙일 꼬리표")
    # brain_meta 누수 수정은 기본 적용. legacy 재현이 필요할 때만 끈다.
    ap.add_argument("--no_fix_brain_meta", dest="fix_brain_meta", action="store_false")
    args = ap.parse_args()

    out_dir = f"outputs/ablation{args.tag}"
    os.makedirs(out_dir, exist_ok=True)
    results = {}

    for name in [n.strip() for n in args.configs.split(",") if n.strip()]:
        flags = MODALITY_CONFIGS[name]
        print(f"\n########## CONFIG: {name}  ({flags})  target={args.target} "
              f"bs={args.batch_size} ep={args.epochs} ##########")

        ev = TrimodalEvaluator(
            target=args.target, epochs=args.epochs, batch_size=args.batch_size,
            save_dir=f"{out_dir}/{name}_{args.target}",
            model_factory=make_model_factory(flags),
            fix_brain_meta=args.fix_brain_meta,
        ).run()

        cis = ev.c_indices                            # fold별 C-index (높을수록 좋음)
        gaps = train_val_gap(ev.training_history)     # 과적합 정도 (train − val 격차)

        results[name] = {
            "mean": float(np.mean(cis)),
            "std": float(np.std(cis)),
            "folds": [round(float(c), 4) for c in cis],
            "train_val_gap_mean": float(np.mean(gaps)) if gaps else None,
            "flags": flags,
            "fold_records": ev.fold_records,           # fold별 표본 수·사건 수
            "training_history": ev.training_history,   # epoch별 학습곡선
        }
        print(f"[ABLATION] {name} {args.target}: {np.mean(cis):.4f} +/- {np.std(cis):.4f}  "
              f"folds={[round(float(c), 4) for c in cis]}  train-val gap={np.mean(gaps):.4f}")

    print(f"\n================ ABLATION SUMMARY (target={args.target}) ================")
    print(f"{'config':<14}{'mean':>8}{'std':>8}{'gap':>8}   folds")
    for name, r in results.items():
        print(f"{name:<14}{r['mean']:>8.4f}{r['std']:>8.4f}{r['train_val_gap_mean']:>8.4f}   {r['folds']}")

    # 터미널 출력만 남기면 나중에 숫자를 다시 찾을 수 없으므로 JSON 으로도 남긴다.
    out_path = os.path.join(out_dir, f"results_{args.target}.json")
    with open(out_path, "w") as f:
        json.dump({"target": args.target, "batch_size": args.batch_size,
                   "epochs": args.epochs, "configs": results}, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
