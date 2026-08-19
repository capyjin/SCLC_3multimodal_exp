# -*- coding: utf-8 -*-
"""[실험2-a] batch=16 미니배치 Cox 가설 검증 — 배치 크기 sweep.

배경 (RESULTS.md §4): Cox 부분우도는 **미니배치 안에서만** risk set 을 만든다.
512² 이미지 메모리 때문에 batch=16 이 강제되는데, 원본 tabular 모델은 훨씬 큰
risk set 으로 학습됐다. 그래서 tabular 브랜치가 과소학습된다는 가설.

이미지 브랜치가 없는 clin_report 조합만 쓰므로 빠르다(CNN 없음). 배치를 키울수록
C-index 가 원본 0.7083 쪽으로 올라가면 작은 risk set 이 핸디캡이었다는 뜻이다.

⚠️ 배치를 키우면 스텝 수가 줄어 과소적합이 겹친다 — 그 교란을 제거한 것이
   같은 폴더의 ``regime_test.py`` 다. 두 결과는 같이 읽어야 한다.

Run:  python 실험2_학습조건_배치크기진단/batch_sweep.py --target os
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import argparse

import numpy as np

from core.model import MODALITY_CONFIGS, make_model_factory
from core.train import TrimodalEvaluator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="clin_report", choices=sorted(MODALITY_CONFIGS))
    ap.add_argument("--target", default="os", choices=("os", "pfs"))
    ap.add_argument("--batches", default="16,32,64,128,256")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    results = {}
    for bs in [int(b) for b in args.batches.split(",")]:
        print(f"\n########## {args.config} batch_size={bs} target={args.target} ##########")
        ev = TrimodalEvaluator(
            target=args.target, epochs=args.epochs, batch_size=bs,
            save_dir=f"outputs/batch_sweep/{args.config}_bs{bs}_{args.target}",
            model_factory=make_model_factory(MODALITY_CONFIGS[args.config]),
        ).run()
        m, s = float(np.mean(ev.c_indices)), float(np.std(ev.c_indices))
        results[bs] = (m, s, [round(float(c), 4) for c in ev.c_indices])
        print(f"[SWEEP] {args.config} bs={bs}: {m:.4f} +/- {s:.4f}")

    print(f"\n======== BATCH SWEEP ({args.config}, target={args.target}) ========")
    print(f"{'batch':>6}{'mean':>9}{'std':>8}")
    for bs, (m, s, folds) in results.items():
        print(f"{bs:>6}{m:>9.4f}{s:>8.4f}   {folds}")


if __name__ == "__main__":
    main()
