# -*- coding: utf-8 -*-
"""[실험4-c] 발언권을 고쳐보려는 시도 A·B — 결과는 **둘 다 실패** (RESULTS.md §9.7)

시도 A: 브랜치 출력을 ``F.normalize`` 로 길이 1에 고정하는 걸 없애거나,
        학습되는 "음량 손잡이"(branch_scale)를 달아 모델이 스스로 발언권을
        줄일 수 있게 한다.
시도 B: 판독지 브랜치 출력 차원을 16 → 64 → 128 로 키운다.

기존 절제 실험(실험3)의 모델과 **완전히 같은 구조**에서 위 두 가지만 바꿔 비교한다
(norm_mode="l2", report_hidden=(32,16) 이면 실험3과 동일 = 재현 확인용 대조군).

무엇을 알아냈나:
  A → 손잡이가 6개 실험 전부 1.0 근처(1.00~1.03)에서 안 움직였다. 손잡이는 선형
      head 가중치와 **수학적으로 중복**이라(손잡이×가중치 = 곱셈 하나) 모델에게
      새로 할 수 있는 일을 주지 못했다. ★이 실패가 §9.4 진단을 확정했다 —
      줄일 권한을 새로 줘도 안 쓴다면 "줄일 수 없어서"가 아니라 "줄이고 싶지
      않아서"다(= 구조가 아니라 최적화 문제).
  B → 판독지 자리를 넓히면 **이미지가 없는 조건에서도** 떨어진다. 16차원이
      환자 171명에 이미 적정 크기였고, 넓히면 과적합만 늘었다.

모델 클래스는 ``core.model.ConcatDeepSurv`` 다 — 예전엔 이 파일이
``FixedConcatDeepSurv`` 라는 사본을 갖고 있었지만 브랜치 정의가 절제 실험 모델과
글자 단위로 같았고 ``norm_mode`` 처리만 달라서, 그 인자를 core 모델에 흡수시켰다.

Run:  python 실험4_영상_발언권독점_진단/exp_fusion_fix.py --target pfs
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import json

import numpy as np
import torch

from core.model import make_model_factory
from core.train import TrimodalEvaluator

# 판독지 브랜치 출력 차원 -> 그 폭을 만드는 hidden 구성
REPORT_HIDDEN = {16: (32, 16), 64: (128, 64), 128: (256, 128)}

VARIANTS = {
    # 이름            : (세 모달 사용?, norm_mode, report 출력차원)
    "base":         dict(tri=True,  norm_mode="l2",    report_dim_out=16),
    "nonorm":       dict(tri=True,  norm_mode="none",  report_dim_out=16),
    "scale":        dict(tri=True,  norm_mode="scale", report_dim_out=16),
    "rep64":        dict(tri=True,  norm_mode="l2",    report_dim_out=64),
    "rep128":       dict(tri=True,  norm_mode="l2",    report_dim_out=128),
    "scale_rep64":  dict(tri=True,  norm_mode="scale", report_dim_out=64),
    "scale_rep128": dict(tri=True,  norm_mode="scale", report_dim_out=128),
    # 대조군: 이미지 없이(clin+report) 판독지 차원만 키우면?
    # -> 차원 확대가 'fusion 을 고친 것'인지 '판독지 브랜치가 그냥 좋아진 것'인지 구분용
    "cr_base":      dict(tri=False, norm_mode="l2",    report_dim_out=16),
    "cr_rep64":     dict(tri=False, norm_mode="l2",    report_dim_out=64),
    "cr_rep128":    dict(tri=False, norm_mode="l2",    report_dim_out=128),
}


def factory_for(spec):
    return make_model_factory(
        {"use_image": spec["tri"], "use_clinical": True, "use_report": True},
        norm_mode=spec["norm_mode"],
        report_hidden=REPORT_HIDDEN[spec["report_dim_out"]],
    )


def learned_branch_scales(save_dir, name, target, spec, folds):
    """"scale" 모드에서 모델이 실제로 배운 손잡이 값 (발언권을 정말 줄였는지)."""
    per_fold = []
    for f in folds:
        sd = torch.load(os.path.join(save_dir, f"{name}_{target}", f"fold{f}_early_fusion_{target}.pt"),
                        map_location="cpu")
        per_fold.append(sd["branch_scale"].tolist())
    scales = np.mean(per_fold, axis=0).tolist()
    names = (["image"] if spec["tri"] else []) + ["clinical", "report"]
    print("  learned branch_scale:", {n: round(s, 3) for n, s in zip(names, scales)})
    return scales


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="pfs", choices=("os", "pfs"))
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    out_dir = "outputs/fusion_fix"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"results_{args.target}.json")
    results = json.load(open(out_path)) if os.path.exists(out_path) else {}

    for name in [v.strip() for v in args.variants.split(",")]:
        spec = VARIANTS[name]
        print(f"\n########## {name}  {spec}  target={args.target} ##########")
        ev = TrimodalEvaluator(
            target=args.target, epochs=args.epochs, batch_size=args.batch_size,
            save_dir=f"{out_dir}/{name}_{args.target}",
            model_factory=factory_for(spec),
        ).run()
        cis = ev.c_indices

        scales = None
        if spec["norm_mode"] == "scale":
            scales = learned_branch_scales(out_dir, name, args.target, spec,
                                           sorted({r["fold"] for r in ev.fold_records}))

        results[name] = {
            "mean": float(np.mean(cis)), "std": float(np.std(cis)),
            "folds": [round(float(c), 4) for c in cis], "spec": spec, "branch_scale": scales,
        }
        print(f"[FIX] {name} {args.target}: {np.mean(cis):.4f} +/- {np.std(cis):.4f}")
        with open(out_path, "w") as f:   # 중간에 끊겨도 남도록 매번 저장
            json.dump(results, f, indent=2)

    print(f"\n================ FUSION FIX SUMMARY ({args.target}) ================")
    print(f"{'variant':<15}{'mean':>8}{'std':>8}   folds")
    for n, r in results.items():
        print(f"{n:<15}{r['mean']:>8.4f}{r['std']:>8.4f}   {r['folds']}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
