# -*- coding: utf-8 -*-
"""데이터 읽기. 확정된 분할과 생존 레이블을 불러오고, 추출된 특징을 융합 모델 입력 텐서로 바꾼다."""

import json
import os

import torch

from sclc import cohort
from sclc.train import fold_plan

from .config import SPLIT_DIR


def load_labels(target: str):
    return cohort.load_trimodal_cohort().drop_duplicates("research_id").set_index("research_id")[
        [f"{target}_days", f"{target}_event"]]


def load_manifest(split_dir: str = SPLIT_DIR) -> dict:
    with open(os.path.join(split_dir, "fold_manifest.json"), encoding="utf-8") as fh:
        return json.load(fh)


def load_outer_split(refit_csv: str) -> tuple[list, list]:
    """outer 분할 파일에서 (학습 환자 T_k, 시험 환자 O_k) ID 를 읽는다."""
    _, plan = fold_plan(cohort.load_trimodal_cohort(split_csv=refit_csv), max_folds=1)[0]
    return plan["train"], plan["test"]


def labels_tensor(ids, labels, target, dev):
    sub = labels.loc[list(ids)]
    return (torch.as_tensor(sub[f"{target}_days"].to_numpy(float), dtype=torch.float32, device=dev),
            torch.as_tensor(sub[f"{target}_event"].to_numpy(float), dtype=torch.float32, device=dev))


def feats(tab_split, img_split, dev):
    """tabular·image 인코더 특징을 {clinical, report, image} dict 로 묶는다."""
    return {"clinical": tab_split["clinical"].to(dev), "report": tab_split["report"].to(dev),
            "image": img_split["image"].to(dev)}
