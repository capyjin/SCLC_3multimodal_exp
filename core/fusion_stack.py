# -*- coding: utf-8 -*-
"""Late fusion 인프라 — 단일모달 OOF 위험점수 추출 + CoxPH 가중합 결합.

[왜 core 에 있나]
  이 코드는 원래 실험 스크립트(``late_fusion_tab_image.py``,
  ``late_fusion_3modal.py``) 안에 있었는데, 실험 4개(RadBERT 융합, RadBERT
  전체, stage-aware, 임상 결측 후속)가 그 실험 파일을 서로 import 하고 있었다.
  두 실험 이상이 쓰는 코드는 실험이 아니라 인프라이므로 여기로 옮겼다.

[중복 통합 — combine_risk_scores]
  예전에는 결합 함수가 두 벌이었다.
    late_fusion_tab_image.combine_two      : 2변수(tabular, image)
    late_fusion_3modal.combine_weighted_sum: 3변수(image, clinical, report)
  fold 별 CoxPH 적합 -> test 예측 -> C-index 라는 절차가 완전히 같고 변수
  개수와 반환 키만 달랐다. ``combine_risk_scores`` 하나로 합치고, 기존
  호출부가 쓰던 반환 키(``mean_coef['risk_tabular']`` 등)는 얇은 래퍼
  ``combine_two``/``combine_weighted_sum`` 가 그대로 유지한다.

[누수 방지]  결합기는 fold 마다 **train 환자의 OOF 점수로만** 적합하고 test
  환자에 적용한다. 각 환자의 OOF 점수는 그 환자가 test 였던 fold 의 모델이
  낸 값이다(Evaluator 가 fold 별 test 예측만 모아 준다).
"""
import os

import numpy as np
import pandas as pd
import torch.optim as optim
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

from core import cohort
from core.model import MODALITY_CONFIGS, make_model_factory
from core.train import ImageOnlyEvaluator, TrimodalEvaluator, fold_plan

DEFAULT_OUT_DIR = os.path.join(cohort.PROJECT_ROOT, "outputs", "late_fusion_B")


# ---------------------------------------------------------------------------
# 공용 소도구
# ---------------------------------------------------------------------------
def oof_dict(oof_predictions: list[dict]) -> dict[int, float]:
    """Evaluator 의 OOF 레코드 목록 -> {research_id: 위험점수}."""
    return {row["research_id"]: row["risk_score"] for row in oof_predictions}


def labels_by_id(cohort_df: pd.DataFrame, target: str) -> pd.DataFrame:
    """research_id 인덱스의 (기간, 사건) 라벨 프레임."""
    return cohort_df.drop_duplicates("research_id").set_index("research_id")[
        [f"{target}_days", f"{target}_event"]
    ]


def cindex_stats(c_indices) -> tuple[float, float]:
    return float(np.mean(c_indices)), float(np.std(c_indices))


def _bf_suffix(fix_brain_meta: bool) -> str:
    """체크포인트 폴더 꼬리표. brain_meta 수정 유무로 저장 위치를 나눠서,
    수정본 재실행이 기존(legacy) 산출물을 덮어쓰지 않게 한다."""
    return "" if fix_brain_meta else "_legacy"


# ---------------------------------------------------------------------------
# 1) 각 축(axis)의 OOF 위험점수 만들기
# ---------------------------------------------------------------------------
def get_tabular_oof(target: str, batch_size: int = 32, epochs: int = 60,
                    max_folds=None, seed: int = 42, fix_brain_meta: bool = True,
                    out_dir: str = DEFAULT_OUT_DIR, text_encoder_fn=None,
                    model_config: str = "clin_report"):
    """(1) 임상+판독지 결합 모델 (영상 제외, bs32/ep60).
    OS 0.708 을 냈던 가장 강한 tabular 축이다. Evaluator 를 그대로 돌려준다."""
    return TrimodalEvaluator(
        target=target, epochs=epochs, batch_size=batch_size,
        save_dir=os.path.join(out_dir, f"tabular_{target}{_bf_suffix(fix_brain_meta)}"),
        model_factory=make_model_factory(MODALITY_CONFIGS[model_config]),
        text_encoder_fn=text_encoder_fn,
        max_folds=max_folds, seed=seed, fix_brain_meta=fix_brain_meta,
    ).run()


def get_image_oof_simplecnn(target: str, batch_size: int = 16, epochs: int = 30,
                            max_folds=None, seed: int = 42, out_dir: str = DEFAULT_OUT_DIR):
    """(2) 영상 단독 — SimpleCNN (bs16/ep30, 영상 arm 의 표준 조건)."""
    return ImageOnlyEvaluator(
        target=target, epochs=epochs, batch_size=batch_size, resize=512,
        save_dir=os.path.join(out_dir, f"image_simplecnn_{target}"),
        ckpt_tag="image_simplecnn", max_folds=max_folds, seed=seed,
    ).run()


def get_image_oof_resnet18(target: str, batch_size: int = 16, epochs: int = 30,
                           resize: int = 224, backbone_lr: float = 1e-5,
                           head_lr: float = 1e-3, weight_decay: float = 1e-4,
                           dropout: float = 0.3, max_folds=None, seed: int = 42,
                           out_dir: str = DEFAULT_OUT_DIR):
    """(3) 영상 단독 — ImageNet 사전학습 ResNet18.
    과적합 방지: 백본은 낮은 학습률, 출력 head 는 높은 학습률."""
    from core.model import ResNet18DeepSurv

    def model_factory():
        return ResNet18DeepSurv(gray_scale=True, pretrained=True, dropout=dropout)

    def optimizer_factory(model):
        return optim.Adam(
            [{"params": model.base.parameters(), "lr": backbone_lr},
             {"params": model.head.parameters(), "lr": head_lr}],
            weight_decay=weight_decay,
        )

    return ImageOnlyEvaluator(
        target=target, epochs=epochs, batch_size=batch_size, resize=resize,
        model_factory=model_factory, optimizer_factory=optimizer_factory, augment=True,
        save_dir=os.path.join(out_dir, f"image_resnet18_{target}"),
        ckpt_tag="image_resnet18", max_folds=max_folds, seed=seed,
    ).run()


# ---------------------------------------------------------------------------
# 2) CoxPH 가중합 결합 (누수 없는 stack)
# ---------------------------------------------------------------------------
def combine_risk_scores(cohort_df: pd.DataFrame, target: str, risks: dict[str, dict],
                        max_folds: int | None = None, modality: str = "late_fusion",
                        log_prefix: str = "late/combine") -> dict:
    """N개의 OOF 위험점수를 fold별 CoxPH 로 결합한다.

    ``risks`` 는 ``{공변량이름: {research_id: 위험점수}}``. 공변량 이름이 곧
    CoxPH 설계행렬의 컬럼명이자 반환되는 계수 딕셔너리의 키다.

    fold 마다: train 환자의 위험점수로 CoxPH 적합 -> test 환자에 적용 ->
    C-index. 학습된 계수가 곧 "가중합"의 가중치다.

    Returns (2·3변수 호출부가 쓰던 키를 모두 포함하는 상위집합)::

        {"fold_cindex": [...], "mean": .., "std": ..,
         "coefs_per_fold": [{name: coef}, ...], "mean_coef": {name: coef},
         "fold_records": [...], "oof_predictions": [...]}
    """
    names = list(risks)
    labels = labels_by_id(cohort_df, target)

    def _frame(ids):
        data = {name: [risks[name][i] for i in ids] for name in names}
        data["duration"] = labels.loc[ids, f"{target}_days"].to_numpy(dtype=float)
        data["event"] = labels.loc[ids, f"{target}_event"].to_numpy(dtype=float)
        return pd.DataFrame(data)

    fold_cindex, coefs_per_fold, fold_records, oof = [], [], [], []
    for fold, ids in fold_plan(cohort_df, max_folds=max_folds):
        cph = CoxPHFitter()
        cph.fit(_frame(ids["train"]), duration_col="duration", event_col="event")

        test_df = _frame(ids["test"])
        test_risk = cph.predict_partial_hazard(test_df[names]).to_numpy()
        ci = float(concordance_index(test_df["duration"], -test_risk, test_df["event"]))
        coefs = {name: float(cph.params_[name]) for name in names}

        fold_cindex.append(ci)
        coefs_per_fold.append(coefs)
        fold_records.append({"target": target, "modality": modality, "fold": fold,
                             "c_index": ci, "n": len(ids["test"]), "coefficients": coefs})
        oof.extend({"research_id": rid, "target": target, "modality": modality, "fold": fold,
                    "duration": float(d), "event": float(e), "risk_score": float(r)}
                   for rid, d, e, r in zip(ids["test"], test_df["duration"], test_df["event"], test_risk))
        coef_txt = ",".join(f"{coefs[n]:.3f}" for n in names)
        print(f"[{log_prefix}/{target}] fold {fold}: C-index={ci:.4f} coef({','.join(names)})=({coef_txt})")

    return {
        "fold_cindex": fold_cindex,
        "mean": float(np.mean(fold_cindex)),
        "std": float(np.std(fold_cindex)),
        "coefs_per_fold": coefs_per_fold,
        "mean_coef": {n: float(np.mean([c[n] for c in coefs_per_fold])) for n in names},
        "fold_records": fold_records,
        "oof_predictions": oof,
    }


def combine_two(cohort_df, target, tabular_risk, image_risk, max_folds=None) -> dict:
    """2축 결합 (tabular = 임상+판독지 합동 모델, image = 영상 단독).
    프로젝트의 채택 모델(late fusion method B)이 쓰는 조합이다."""
    return combine_risk_scores(
        cohort_df, target, {"risk_tabular": tabular_risk, "risk_image": image_risk},
        max_folds=max_folds, modality="late_fusion_tab_image", log_prefix="lateB/combine",
    )


def combine_weighted_sum(cohort_df, target, image_risk, clinical_risk, report_risk,
                         max_folds=None) -> dict:
    """3축 결합 (영상·임상·판독지를 각각 독립 학습한 뒤 가중합)."""
    return combine_risk_scores(
        cohort_df, target,
        {"risk_image": image_risk, "risk_clinical": clinical_risk, "risk_report": report_risk},
        max_folds=max_folds, modality="late_fusion_weighted_sum", log_prefix="late_fusion/combined",
    )
