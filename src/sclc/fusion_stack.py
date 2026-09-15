# -*- coding: utf-8 -*-
"""Late fusion 인프라 — 단일모달 OOF 위험점수 추출 + CoxPH 가중합 결합.

[왜 src/sclc 에 있나]
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
import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch.optim as optim
from sclc import paths
from sclc.fusion_diag import iter_fold_stack
from sclc.metrics import cindex
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
    """체크포인트 폴더 꼬리표. brain_meta 수정 유무로 저장 위치를 나눠서,
    수정본 재실행이 기존(legacy) 산출물을 덮어쓰지 않게 한다."""
    return "" if fix_brain_meta else "_legacy"


@dataclass
class ArmResult:
    """축(arm) 하나의 실행 결과 — OOF 위험점수 + fold별 C-index.

    [왜 필요한가]
      ``late_fusion_tab_image`` 와 ``late_fusion_seed_sweep`` 가 축마다 똑같은
      네 줄을 반복하고 있었다::

          risk = oof_dict(ev.oof_predictions)
          mean, std = cindex_stats(ev.c_indices)
          folds = [round(float(c), 4) for c in ev.c_indices]
          print(f"[lateB] ... {mean:.4f} +/- {std:.4f} folds={folds}")

      두 파일의 소수점 자리수(4 vs 6)만 달랐고 나머지는 같았다. 결과 JSON 의
      arm 블록 모양(``{"mean","std","folds"}``)이 여기서 한 번만 정의되므로,
      ``outputs/late_fusion_B/results.json`` 을 읽는 ``tools/plot_all_figures.py``
      와의 계약이 한 곳에 고정된다.
    """
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
        """결과 JSON 에 넣는 arm 블록 (기존 스키마와 동일)."""
        return {"mean": self.mean, "std": self.std, "folds": self.folds(ndigits)}


# ---------------------------------------------------------------------------
# OOF 위험점수 캐시 (outputs/<dir>/oof_<target>.json)
# ---------------------------------------------------------------------------
# 정리 전에는 이 파일을 읽는 코드가 네 벌이었다 — 실험1의 3modal 재실행,
# 실험5의 분석 두 개, 실험7의 결합 후속. 넷 다 "JSON 을 열고 문자열 키를 int
# 로 바꾼다"는 같은 두 줄이었는데, 그중 하나만 float() 캐스팅을 빠뜨려도
# 아무 에러 없이 다른 타입이 흘러들어간다. 그래서 여기 한 곳으로 모은다.
def oof_cache_path(target: str, out_dir: str = DEFAULT_OUT_DIR) -> str:
    return os.path.join(out_dir, f"oof_{target}.json")


def load_oof_cache(target: str, out_dir: str = DEFAULT_OUT_DIR,
                   keys=("tabular", "image")) -> dict[str, dict]:
    """``oof_<target>.json`` -> ``{축이름: {research_id(int): 위험점수(float)}}``.

    파일이 없으면 ``FileNotFoundError`` 를 그대로 올린다 — 조용히 빈 dict 를
    돌려주면 "재학습 없이 캐시로 분석한다"는 전제가 깨진 채로 통계가 나온다.
    """
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
    """(4) 영상 단독 — RadImageNet(CT/MRI/초음파 130만 장) 사전학습 ResNet50.
    ResNet18(자연영상)과 대비되는 조건: 방사선영상 통계로는 사전학습됐지만
    PET 자체는 학습 데이터에 없음. ``pretrained=False`` 로 실험8/10 식 랜덤
    초기화 대조군도 만들 수 있다(같은 구조, 학습 안 함)."""
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
    plan = fold_plan(cohort_df, max_folds=max_folds)

    # fold 루프 자체(train 으로만 적합 -> test 에 적용)는 ``fusion_diag.iter_fold_stack``
    # 한 곳에만 있다. 진단 쪽(순열검정·계수검정)이 같은 절차를 다시 구현하면
    # "결합기와 검정이 같은 절차인가"를 매번 눈으로 확인해야 한다.
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
