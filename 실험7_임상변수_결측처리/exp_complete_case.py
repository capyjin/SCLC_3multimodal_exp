# -*- coding: utf-8 -*-
"""Complete-case 비교 -- 5개 임상 지표(LDH/WBC/FVC%/FEV1%/DLCO%) 중 하나라도
결측인 환자를 아예 빼고 학습하면 성능이 어떻게 되는가.

`exp_missing_handling.py` 는 "결측을 어떻게 채울까"(A/B/C variant)를 묻는
실험이고, 이 스크립트는 그와 다른 질문 -- "차라리 결측 있는 환자를 안 쓰면
어떨까" -- 을 묻는다. 채택 파이프라인(`original` = 전체-중앙값 대치, 238명)과
직접 비교하기 위해 나머지는 전부 동일하게 둔다: 같은 모델/Cox loss/bs32/ep60,
같은 고정 5-fold split(`splits/trimodal_common_5fold_seed42_v1.csv`)에서
결측 있는 환자 행만 제거해서 재사용한다 (fold/split 배정 자체는 새로 만들지
않음 -- protocol section 7). 남는 환자는 애초에 5개 지표가 다 관측돼 있으므로
merged CSV의 기존(전체-중앙값 대치) 컬럼을 그대로 써도 무방하다 -- 대치값이
바로 관측값이기 때문. 즉 fold_safe_features 도 필요 없다.

Run:
  python 실험7_임상변수_결측처리/exp_complete_case.py --smoke   # 1 fold, 2 epoch
  python 실험7_임상변수_결측처리/exp_complete_case.py            # seed 42, 4 runs
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import argparse
import json
import time

import numpy as np
import pandas as pd

import raw_clinical_values as rcv
from core import cohort as cohort_mod
from core.model import MODALITY_CONFIGS, make_model_factory
from core.train import TrimodalEvaluator

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "EXP_20260819_complete_case_clinical")
MISSING_COLS = ("ldh_raw_missing", "wbc_raw_missing", "fvc_raw_missing",
                "fev1_raw_missing", "dlco_raw_missing")

MODEL_CONFIGS = ("clin_only", "clin_report")


def build_complete_case_split(out_path: str) -> tuple[str, int]:
    """5개 지표 중 하나라도 결측인 환자를 뺀 split CSV를 만든다.

    fold/split 배정은 원본 frozen split을 그대로 쓰고(행만 제거), 새 split을
    생성하지 않는다.
    """
    raw = rcv.load_cohort_indicators()
    complete_ids = set(raw.index[~raw[list(MISSING_COLS)].any(axis=1)].astype(int))

    split = cohort_mod.load_split()
    filtered = split[split["research_id"].isin(complete_ids)].copy()
    assert filtered["research_id"].nunique() == len(complete_ids), \
        "일부 complete-case 환자가 frozen split에 없음 -- 코호트 정렬 확인 필요"

    filtered.to_csv(out_path, index=False)
    return out_path, len(complete_ids)


def run_one(config: str, target: str, seed: int, epochs: int, batch_size: int,
            split_csv: str, max_folds=None) -> dict:
    flags = MODALITY_CONFIGS[config]
    tag = f"{config}_complete_case_{target}_seed{seed}"
    ev = TrimodalEvaluator(
        target=target, epochs=epochs, batch_size=batch_size, seed=seed,
        split_csv=split_csv,
        save_dir=os.path.join(OUT_DIR, "checkpoints", tag),
        model_factory=make_model_factory(flags),
        max_folds=max_folds,
    ).run()

    return {
        "variant": "complete_case", "config": config, "target": target, "seed": seed,
        "epochs": epochs, "batch_size": batch_size,
        "folds": [round(float(c), 6) for c in ev.c_indices],
        "mean": float(np.mean(ev.c_indices)), "std": float(np.std(ev.c_indices)),
        "fold_records": ev.fold_records,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(MODEL_CONFIGS))
    ap.add_argument("--targets", default="os,pfs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--smoke", action="store_true",
                    help="1 fold x 2 epochs -- pipeline sanity check, NOT a result")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    configs = [c.strip() for c in args.configs.split(",")]
    targets = [t.strip() for t in args.targets.split(",")]
    epochs, batch_size, max_folds = args.epochs, args.batch_size, None
    if args.smoke:
        epochs, max_folds = 2, 1

    os.makedirs(OUT_DIR, exist_ok=True)
    split_csv, n_complete = build_complete_case_split(
        os.path.join(OUT_DIR, "complete_case_split.csv"))
    print(f"[complete-case] {n_complete}/238 환자 유지 "
          f"({238 - n_complete}명 제외) -- split: {split_csv}")

    out_path = args.out or os.path.join(
        OUT_DIR, "results_smoke.json" if args.smoke else "results_seed42.json")

    runs, started = [], time.time()
    total = len(configs) * len(targets)
    for config in configs:
        for target in targets:
            idx = len(runs) + 1
            print(f"\n{'#'*90}\n### [{idx}/{total}] config={config} target={target} "
                  f"seed={args.seed} bs={batch_size} ep={epochs}\n{'#'*90}")
            res = run_one(config, target, args.seed, epochs, batch_size,
                          split_csv, max_folds=max_folds)
            runs.append(res)
            print(f"[RESULT] {config}/{target}/seed{args.seed}: "
                  f"{res['mean']:.4f} +/- {res['std']:.4f}  folds={res['folds']}")
            with open(out_path, "w") as fh:
                json.dump({"runs": runs, "n_complete_case": n_complete,
                           "epochs": epochs, "batch_size": batch_size,
                           "smoke": args.smoke}, fh, indent=2)

    print(f"\n{'='*90}\nSUMMARY (complete-case, n={n_complete}, seed={args.seed})\n{'='*90}")
    for config in configs:
        for target in targets:
            r = next(x for x in runs if x["config"] == config and x["target"] == target)
            print(f"{config:<13}{target:<7}{r['mean']:>9.4f}  folds={r['folds']}")

    print(f"\nelapsed {(time.time()-started)/60:.1f} min -- wrote {out_path}")


if __name__ == "__main__":
    main()
