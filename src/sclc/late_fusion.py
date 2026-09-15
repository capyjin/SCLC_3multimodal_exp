# -*- coding: utf-8 -*-
"""Late fusion — 축별로 따로 학습한 OOF 위험점수를 fold별 CoxPH 로 결합한다.

결합기는 fold 마다 train 환자의 OOF 점수로만 적합하고 test 환자에 적용한다(누수 방지).
"""
import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch.optim as optim
from sclc import paths
from sclc.late_fusion_tests import iter_fold_stack
from sclc.evaluation import cindex
from sclc.model import MODALITY_CONFIGS, make_model_factory
from sclc.train import ImageOnlyEvaluator, TrimodalEvaluator, fold_plan

DEFAULT_OUT_DIR = paths.outputs("late_fusion_B")


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
    """체크포인트 폴더 꼬리표 — brain_meta 수정본이 기존 산출물을 덮어쓰지 않게 분리."""
    return "" if fix_brain_meta else "_legacy"


@dataclass
class ArmResult:
    """축 하나의 실행 결과 — OOF 위험점수 + fold별 C-index."""
    tag: str
    risk: dict = field(repr=False)          # {research_id: OOF 위험점수}
    c_indices: list = field(repr=False)     # fold별 C-index
    evaluator: object = field(default=None, repr=False)

    @classmethod
    def from_evaluator(cls, tag: str, ev) -> "ArmResult":
        return cls(tag=tag, risk=oof_dict(ev.oof_predictions),
                   c_indices=list(ev.c_indices), evaluator=ev)

    @property
    def mean(self) -> float:
        return float(np.mean(self.c_indices))

    @property
    def std(self) -> float:
        return float(np.std(self.c_indices))

    def folds(self, ndigits: int = 4) -> list:
        return [round(float(c), ndigits) for c in self.c_indices]

    def as_dict(self, ndigits: int = 4) -> dict:
        """결과 JSON 에 넣는 arm 블록 {"mean","std","folds"}."""
        return {"mean": self.mean, "std": self.std, "folds": self.folds(ndigits)}


# ---------------------------------------------------------------------------
# OOF 위험점수 캐시 (outputs/<dir>/oof_<target>.json)
# ---------------------------------------------------------------------------
def oof_cache_path(target: str, out_dir: str = DEFAULT_OUT_DIR) -> str:
    return os.path.join(out_dir, f"oof_{target}.json")


def load_oof_cache(target: str, out_dir: str = DEFAULT_OUT_DIR,
                   keys=("tabular", "image")) -> dict[str, dict]:
    """``oof_<target>.json`` -> ``{축이름: {research_id(int): 위험점수(float)}}``."""
    with open(oof_cache_path(target, out_dir), encoding="utf-8") as fh:
        payload = json.load(fh)
    return {k: {int(i): float(v) for i, v in payload[k].items()} for k in keys}


def save_oof_cache(target: str, risks: dict[str, dict],
                   out_dir: str = DEFAULT_OUT_DIR) -> str:
    path = oof_cache_path(target, out_dir)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(risks, fh)
    return path


# ---------------------------------------------------------------------------
# 1) 각 축(axis)의 OOF 위험점수 만들기
# ---------------------------------------------------------------------------
def get_tabular_oof(target: str, batch_size: int = 32, epochs: int = 60,
                    max_folds=None, seed: int = 42, fix_brain_meta: bool = True,
                    out_dir: str = DEFAULT_OUT_DIR, text_encoder_fn=None,
                    model_config: str = "clin_report"):
    """tabular 축 — 임상+판독지 결합 모델 (영상 제외, bs32/ep60)."""
    return TrimodalEvaluator(
        target=target, epochs=epochs, batch_size=batch_size,
        save_dir=os.path.join(out_dir, f"tabular_{target}{_bf_suffix(fix_brain_meta)}"),
        model_factory=make_model_factory(MODALITY_CONFIGS[model_config]),
        text_encoder_fn=text_encoder_fn,
        max_folds=max_folds, seed=seed, fix_brain_meta=fix_brain_meta,
    ).run()


def get_image_oof_simplecnn(target: str, batch_size: int = 16, epochs: int = 30,
                            max_folds=None, seed: int = 42, out_dir: str = DEFAULT_OUT_DIR):
    """영상 축 — SimpleCNN (bs16/ep30, 채택 백본)."""
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
    """영상 축 대조군 — ImageNet 사전학습 ResNet18 (백본 저LR + head 고LR)."""
    from sclc.model import ResNet18DeepSurv

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


def get_image_oof_radimagenet(target: str, batch_size: int = 16, epochs: int = 30,
                              resize: int = 224, backbone_lr: float = 1e-5,
                              head_lr: float = 1e-3, weight_decay: float = 1e-4,
                              dropout: float = 0.3, pretrained: bool = True,
                              max_folds=None, seed: int = 42,
                              out_dir: str = DEFAULT_OUT_DIR):
    """영상 축 대조군 — RadImageNet 사전학습 ResNet50. ``pretrained=False`` 면 랜덤 초기화 대조군."""
    from sclc.model import RadImageNetDeepSurv

    def model_factory():
        return RadImageNetDeepSurv(pretrained=pretrained, dropout=dropout)

    def optimizer_factory(model):
        return optim.Adam(
            [{"params": model.backbone.parameters(), "lr": backbone_lr},
             {"params": model.head.parameters(), "lr": head_lr}],
            weight_decay=weight_decay,
        )

    tag = "image_radimagenet" if pretrained else "image_radimagenet_random"
    return ImageOnlyEvaluator(
        target=target, epochs=epochs, batch_size=batch_size, resize=resize,
        model_factory=model_factory, optimizer_factory=optimizer_factory, augment=True,
        save_dir=os.path.join(out_dir, f"{tag}_{target}"),
        ckpt_tag=tag, max_folds=max_folds, seed=seed,
    ).run()


# ---------------------------------------------------------------------------
# 2) CoxPH 가중합 결합 (누수 없는 stack)
# ---------------------------------------------------------------------------
def combine_risk_scores(cohort_df: pd.DataFrame, target: str, risks: dict[str, dict],
                        max_folds: int | None = None, modality: str = "late_fusion",
                        log_prefix: str = "late/combine") -> dict:
    """N개의 OOF 위험점수를 fold별 CoxPH 로 결합한다 (적합 계수 = 가중합의 가중치).

    ``risks`` 는 ``{공변량이름: {research_id: 위험점수}}``. 반환:
    ``fold_cindex · mean · std · coefs_per_fold · mean_coef · fold_records · oof_predictions``.
    """
    names = list(risks)
    labels = labels_by_id(cohort_df, target)
    plan = fold_plan(cohort_df, max_folds=max_folds)

    fold_cindex, coefs_per_fold, fold_records, oof = [], [], [], []
    for fold, ids, cph, test_df, _cols in iter_fold_stack(risks, labels, plan, target, names):
        test_risk = cph.predict_partial_hazard(test_df[names]).to_numpy()
        ci = cindex(test_df["duration"], test_risk, test_df["event"])
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
    """2축 결합 (tabular + image) — 프로젝트 채택 모델."""
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
