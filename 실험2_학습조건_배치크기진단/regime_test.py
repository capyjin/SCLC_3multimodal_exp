# -*- coding: utf-8 -*-
"""[실험2-b] 스텝 수를 맞춘 regime test — risk set 효과만 분리해서 본다.

``batch_sweep.py`` 는 에포크를 고정한 채 배치만 키웠기 때문에 **스텝 수가 줄어
과소적합**이 같이 일어난다. 여기서는 배치를 키우는 만큼 에포크를 늘려 총 옵티마이저
스텝을 ~일정하게 유지한다. 그러면 남는 차이는 **Cox risk set 크기뿐**이다.

fold 의 train 은 약 190명이므로 steps/epoch = ceil(190/bs), 총 스텝 = 그 값 × epochs.
``--runs`` 는 "config:batch:epochs" 를 쉼표로 나열한다.

Run:  python 실험2_학습조건_배치크기진단/regime_test.py --target os
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import math

import numpy as np

from core.model import MODALITY_CONFIGS, make_model_factory
from core.train import TrimodalEvaluator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="os", choices=("os", "pfs"))
    ap.add_argument("--runs", default="clin_report:32:60,clin_report:64:120,clin_report:32:90",
                    help='"config:batch:epochs" 쉼표 목록')
    args = ap.parse_args()

    results = []
    for spec in args.runs.split(","):
        cfg, bs, ep = spec.split(":")
        bs, ep = int(bs), int(ep)
        steps = math.ceil(190 / bs) * ep
        print(f"\n########## {cfg} bs={bs} ep={ep} (~{steps} steps) target={args.target} ##########")
        ev = TrimodalEvaluator(
            target=args.target, epochs=ep, batch_size=bs,
            save_dir=f"outputs/regime/{cfg}_bs{bs}_ep{ep}_{args.target}",
            model_factory=make_model_factory(MODALITY_CONFIGS[cfg]),
        ).run()
        m, s = float(np.mean(ev.c_indices)), float(np.std(ev.c_indices))
        results.append((cfg, bs, ep, steps, m, s, [round(float(c), 4) for c in ev.c_indices]))
        print(f"[REGIME] {cfg} bs={bs} ep={ep}: {m:.4f} +/- {s:.4f}")

    print(f"\n======== REGIME TEST (target={args.target}) ========")
    print(f"{'config':<12}{'bs':>4}{'ep':>5}{'steps':>7}{'mean':>9}{'std':>8}")
    for cfg, bs, ep, steps, m, s, folds in results:
        print(f"{cfg:<12}{bs:>4}{ep:>5}{steps:>7}{m:>9.4f}{s:>8.4f}   {folds}")


if __name__ == "__main__":
    main()
