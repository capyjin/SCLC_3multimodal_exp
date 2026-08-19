# -*- coding: utf-8 -*-
"""[실험4-d] ★결정적 실험 — "발언권 독점은 차원 배분 탓 아닌가?"를 검증한다.

동기:
  head 가중치를 블록별로 분해해 보면 —
    image    128차원, |w| 점유율 46.1%, 차원당 RMS 0.0430
    clinical 128차원, |w| 점유율 37.1%, 차원당 RMS 0.0346
    report    16차원, |w| 점유율 16.8%, 차원당 RMS 0.0442  ← 차원당으론 최고
  즉 이미지의 46%는 '칸이 128개라 총합이 큰 것'이고, 차원 점유율(47.1%)과 거의 같다.
  → 발언권 독점이 ``image_proj_dim=128`` 이라는 **설계 선택**의 결과일 가능성.

기존 실험의 구멍:
  §9.7.2 는 '판독지를 넓히기'만 시험했고(실패), '이미지를 줄이기'는 안 했다.
  이 스크립트가 그 대칭 실험을 채운다. 규제가 가장 약한 것(image dropout 0.2 vs
  clinical 0.5)도 설계 선택이므로 dropout 을 맞춘 조건도 같이 본다.

★결과 (RESULTS.md §9.4.2): 이미지 발언권이 46% → 21% 로 반토막 났는데도
  **성능은 변하지 않았다** (OS −0.009 p=0.78 / PFS +0.002 p=0.95).
  → "발언권 독점 때문에 성능이 깎였다"는 인과 주장을 **철회**하게 만든 실험.
    발언권은 증상이지 원인이 아니다.

Run:  python 실험4_영상_발언권독점_진단/exp_balance_dims.py --target os --variants img16
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import glob
import json

import numpy as np
import torch

from core.model import BRANCH_OUT_DIM, MODALITY_CONFIGS, make_model_factory
from core.train import TrimodalEvaluator

VARIANTS = {
    "base":         dict(image_proj_dim=128, image_dropout=0.2),   # 기존 설정
    "img32":        dict(image_proj_dim=32,  image_dropout=0.2),   # 이미지 칸 수 축소
    "img16":        dict(image_proj_dim=16,  image_dropout=0.2),   # 판독지(16)와 동급까지
    "drop05":       dict(image_proj_dim=128, image_dropout=0.5),   # 칸은 그대로, 규제만 맞춤
    "img16_drop05": dict(image_proj_dim=16,  image_dropout=0.5),   # 둘 다
}


def head_shares(save_dir, target, image_dim):
    """저장된 체크포인트에서 블록별 |w| 점유율과 '차원당 RMS' 를 계산한다."""
    dims = {"image": image_dim, "clinical": BRANCH_OUT_DIM["clinical"], "report": BRANCH_OUT_DIM["report"]}
    files = sorted(glob.glob(os.path.join(save_dir, f"fold*_early_fusion_{target}.pt")))
    if not files:
        return None
    per = {}
    for p in files:
        w = torch.load(p, map_location="cpu")["head.weight"].squeeze(0)
        offset = 0
        for n in ("image", "clinical", "report"):
            per.setdefault(n, []).append(w[offset:offset + dims[n]].norm().item())
            offset += dims[n]
    norms = {n: float(np.mean(v)) for n, v in per.items()}
    total = sum(norms.values())
    return {n: {"norm": norms[n], "share": norms[n] / total,
                "rms_per_dim": norms[n] / np.sqrt(dims[n]), "dim": dims[n]}
            for n in norms}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="os", choices=("os", "pfs"))
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    out_dir = "outputs/balance_dims"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"results_{args.target}.json")
    results = json.load(open(path)) if os.path.exists(path) else {}

    for name in [v.strip() for v in args.variants.split(",")]:
        spec = VARIANTS[name]
        save_dir = f"{out_dir}/{name}_{args.target}"
        print(f"\n########## {name}  {spec}  target={args.target} ##########")
        ev = TrimodalEvaluator(
            target=args.target, epochs=args.epochs, batch_size=args.batch_size,
            save_dir=save_dir,
            model_factory=make_model_factory(MODALITY_CONFIGS["all"], **spec),
        ).run()
        cis = ev.c_indices
        sh = head_shares(save_dir, args.target, spec["image_proj_dim"])

        results[name] = {"spec": spec, "mean": float(np.mean(cis)), "std": float(np.std(cis)),
                         "folds": [round(float(c), 4) for c in cis], "head_shares": sh}
        with open(path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"[BAL] {name} {args.target}: {np.mean(cis):.4f} +/- {np.std(cis):.4f}")
        if sh:
            for n in ("image", "clinical", "report"):
                print(f"    {n:<9} dim={sh[n]['dim']:>4}  |w| 점유율 {sh[n]['share']:>6.1%}  "
                      f"차원당 RMS {sh[n]['rms_per_dim']:.4f}")

    print(f"\n================ SUMMARY ({args.target}) ================")
    print(f"{'variant':<15}{'C-index':>10}{'image |w| 점유율':>18}")
    for n, r in results.items():
        s = f"{r['head_shares']['image']['share']:.1%}" if r.get("head_shares") else "-"
        print(f"{n:<15}{r['mean']:>10.4f}{s:>18}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
