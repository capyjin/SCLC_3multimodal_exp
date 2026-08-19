# -*- coding: utf-8 -*-
"""[실험4-b] 모달리티 간 정보 중복 측정 — 단일모달 모델들이 같은 환자를 위험하다고 보나?

단일모달 3개(image_only / clin_only / report_only)의 OOF 위험점수 사이 순위상관
(Spearman)을 잰다. 상관이 높으면 = 같은 정보를 담고 있다 = 합쳐도 새 정보가 없다.

가설 H2("이미지 정보는 이미 판독지에 들어있다") 검증용. 결과(RESULTS.md §9.5):
image↔report 가 +0.244 로 셋 중 가장 높지만 절대값이 낮아, 중복만으로 성능 하락
(−0.029)을 설명하기엔 부족하다 → **약하게만 지지**.

Run:  python 실험4_영상_발언권독점_진단/analyze_redundancy.py
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import json

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from core import cohort, dataset as ds, features
from core.model import MODALITY_CONFIGS, ConcatDeepSurv
from core.train import fold_plan

TARGET = "pfs"
SINGLES = ["image_only", "clin_only", "report_only"]
CKPT_DIR = os.path.join(PROJECT_ROOT, "outputs", "ablation_why_pfs")
PAIRS = [("image_only", "report_only"), ("image_only", "clin_only"), ("clin_only", "report_only")]


def main():
    cohort_df = cohort.load_trimodal_cohort()
    cf = cohort_df.drop_duplicates("research_id").set_index("research_id")
    std_cols, cat_cols = features.resolve_clinical_columns(cf)
    corpus, _ = features.load_text_corpus(cohort.DEFAULT_MERGED_CSV)

    risks = {c: [] for c in SINGLES}     # 모든 fold 의 test 위험점수를 이어붙임
    for _fold, ids in fold_plan(cohort_df):
        test_ds, cdim, rdim = ds.build_fold_test_dataset(
            cf, corpus, std_cols, cat_cols, ids, TARGET, cohort.DEFAULT_IMAGE_DIR)
        for cfg in SINGLES:
            m = ConcatDeepSurv(cdim, rdim, **MODALITY_CONFIGS[cfg])
            m.load_state_dict(torch.load(
                os.path.join(CKPT_DIR, f"{cfg}_{TARGET}", f"fold{_fold}_early_fusion_{TARGET}.pt"),
                map_location="cpu"))
            m.eval()
            r = []
            with torch.no_grad():
                for img, tab, _d, _e in DataLoader(test_ds, batch_size=16):
                    r.append(m(img, tab).squeeze(1).numpy())
            risks[cfg].append(np.concatenate(r))

    flat = {c: np.concatenate(v) for c, v in risks.items()}
    print(f"\n=== 단일 모달 위험점수 간 상관 ({TARGET.upper()}, 전체 {len(flat['image_only'])}명 OOF) ===")
    out = {}
    for a, b in PAIRS:
        rho, p = spearmanr(flat[a], flat[b])
        out[f"{a}~{b}"] = {"spearman": float(rho), "p": float(p)}
        print(f"  {a:<12} ~ {b:<12}  spearman = {rho:+.3f}  (p={p:.2g})")

    path = os.path.join(CKPT_DIR, f"redundancy_{TARGET}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
