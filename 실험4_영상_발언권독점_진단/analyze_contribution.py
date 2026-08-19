# -*- coding: utf-8 -*-
"""[실험4-a] 모달리티별 "발언권" 측정 — 최종 위험점수를 누가 얼마나 흔드는가.

측정 방법: 저장된 체크포인트를 test fold 에 다시 태워, 각 모달리티 블록이
만들어낸 위험점수 기여분(`특징 × head 가중치`의 합)의 **표준편차**를 구하고
비율로 환산한다. 표준편차를 쓰는 이유는 "환자마다 점수를 얼마나 흔드는가"가
곧 그 모달리티가 최종 판단에 미치는 영향이기 때문이다.

결과(RESULTS.md §9.4): 이미지를 넣으면 이미지가 발언권의 70~80%를 가져가고,
잘하던 판독지가 54% → 17% 로 밀려난다.

⚠️ 단, 발언권을 인위적으로 낮춰도 성능은 변하지 않았다(§9.4.2, exp_balance_dims).
   즉 발언권 독점은 **증상이지 원인이 아니다.**

Run:  T=os python 실험4_영상_발언권독점_진단/analyze_contribution.py
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from core import cohort, dataset as ds, features
from core.model import BRANCH_OUT_DIM, MODALITY_CONFIGS, ConcatDeepSurv
from core.train import fold_plan

TARGET = os.environ.get("T", "pfs")
CKPT_DIR = os.path.join(PROJECT_ROOT, "outputs", "ablation_why_pfs")
CONFIGS_TO_SCAN = ["clin_report", "clin_image", "all"]
FOLDS = (1, 2, 3)   # CPU 라 시간 절약: 3개 fold 만


def main():
    cohort_df = cohort.load_trimodal_cohort()
    clinical_frame = cohort_df.drop_duplicates("research_id").set_index("research_id")
    std_cols, cat_cols = features.resolve_clinical_columns(clinical_frame)
    corpus, _ = features.load_text_corpus(cohort.DEFAULT_MERGED_CSV)
    plan = {fold: ids for fold, ids in fold_plan(cohort_df)}

    out = {}
    for cfg in CONFIGS_TO_SCAN:
        flags = MODALITY_CONFIGS[cfg]
        names = [n for n in ("image", "clinical", "report") if flags[f"use_{n}"]]
        acc = {n: [] for n in names}

        for fold in FOLDS:
            ids = plan[fold]
            test_ds, cdim, rdim = ds.build_fold_test_dataset(
                clinical_frame, corpus, std_cols, cat_cols, ids, TARGET, cohort.DEFAULT_IMAGE_DIR)

            model = ConcatDeepSurv(cdim, rdim, **flags)
            model.load_state_dict(torch.load(
                os.path.join(CKPT_DIR, f"{cfg}_{TARGET}", f"fold{fold}_early_fusion_{TARGET}.pt"),
                map_location="cpu"))
            model.eval()

            # head 가중치를 블록별로 자른다 (브랜치가 concat 된 순서 = names 순서)
            w = model.head.weight.squeeze(0)
            slices, offset = {}, 0
            for n in names:
                slices[n] = slice(offset, offset + BRANCH_OUT_DIM[n])
                offset += BRANCH_OUT_DIM[n]

            parts = {n: [] for n in names}
            with torch.no_grad():
                for img, tab, _dur, _evt in DataLoader(test_ds, batch_size=16):
                    feats = model.branch_features(img, tab)   # forward 와 같은 경로
                    for n in names:
                        parts[n].append((feats[n] * w[slices[n]]).sum(1).numpy())
            for n in names:
                acc[n].append(float(np.concatenate(parts[n]).std()))

        out[cfg] = {n: float(np.mean(v)) for n, v in acc.items()}
        total = sum(out[cfg].values())
        print(f"--- {cfg} ({TARGET}) : 위험점수 기여도 표준편차 ---")
        for n in names:
            print(f"   {n:<9} sd={out[cfg][n]:.4f}   share={out[cfg][n] / total:.1%}")

    path = os.path.join(CKPT_DIR, f"contribution_{TARGET}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
