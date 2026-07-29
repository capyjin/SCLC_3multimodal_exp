# -*- coding: utf-8 -*-
"""방식 B: 3가지 모달리티 LATE fusion (후기 융합)
  = [임상+리포트 결합 탭 데이터 위험 점수] + [영상 위험 점수]
  정보 누수가 없는(leakage-safe) 5-fold 교차 검증 생존분석(CoxPH) 결합 방식.

원리 및 배경 (RESULTS.md 참고):
  초기 융합(Early Concat Fusion)에서 영상(SimpleCNN)을 통째로 섞어 학습시켰더니,
  임상+리포트만 썼을 때(0.708)보다 3개를 다 썼을 때(0.678) 오버피팅(과적합)으로 성능이 오히려 떨어졌습니다.
  
  따라서 LATE fusion(후기 융합)을 적용합니다:
  각 모달리티(임상+리포트, 영상)를 각자에 맞는 적절한 에포크/학습률로 독립적으로 따로따로 먼저 학습시킨 후,
  여기서 나온 '예측 위험 점수(Out-of-fold risk score)' 2개만 모아서 최적의 비율로 합칩니다.

비교 대상 (OS/PFS 목표 모두 동일):
  1. tabular (숫자/글자) = 임상+리포트 결합 모델, 영상 제외, bs32/ep60
                          (OS 0.708 / PFS 0.668을 찍었던 가장 강력한 기본 모델)
  2. image A (영상 A)    = SimpleCNN 모델, bs16/ep30 (자체 적정 학습 조건)
  3. image B (영상 B)    = ResNet18 사전학습 모델, bs16/ep30,
                          낮은 학습률 백본(1e-5) + 높은 학습률 헤드(1e-3), 데이터 좌우반전 등 적용

안전한 결합 (CoxPH Stack):
  정보 누수를 막기 위해, 모든 환자의 위험 점수는 자신이 테스트 세트일 때만 계산된 점수(OOF)를 사용합니다.
  각 Fold마다 훈련 데이터(Train)의 위험 점수로 CoxPH 모델을 학습시키고,
  검증 데이터(Test)에 적용해 성능(C-index)을 평가합니다.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from torchvision.models import ResNet18_Weights, resnet18

import ablation
import cohort
from train import ImageOnlyEvaluator, TrimodalEvaluator, _fold_plan

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs", "late_fusion_B")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# ResNet18 기반 영상 전용 DeepSurv 신경망
# (흑백 PET-CT 영상을 처리하기 위해 첫번째 입력 채널을 3개(RGB)에서 1개(흑백)로 변환)
# ---------------------------------------------------------------------------
class ResNet18DeepSurv(nn.Module):
    def __init__(self, gray_scale: bool = True, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        # 사전 학습된(Pretrained) ResNet18 가져오기
        if pretrained:
            self.base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.base = resnet18()

        # 흑백 이미지(1채널)를 다루기 위한 변환
        if gray_scale:
            old_conv = self.base.conv1
            self.base.conv1 = nn.Conv2d(
                in_channels=1,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            with torch.no_grad():
                # 기존 RGB 3개 채널 가중치의 평균을 내어 1개 채널 가중치로 만듦
                self.base.conv1.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)

        in_features = self.base.fc.in_features  # ResNet18 출력 특징 수 (512개)
        self.base.fc = nn.Identity()           # 기존 분류용 마지막 층 제거
        # 드롭아웃 후 최종 위험 점수(1개)를 출력하는 예측 헤드 연결
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))

    def forward(self, x):
        return self.head(self.base(x))


# ---------------------------------------------------------------------------
# 각 모달리티별 예측 위험 점수(OOF, Out-Of-Fold) 추출 함수들
# ---------------------------------------------------------------------------
def get_tabular_oof(target: str, batch_size: int = 32, epochs: int = 60,
                    max_folds=None, seed: int = 42):
    """(1) 임상 + 리포트 결합 모델 (영상 제외, 배치크기 32 / 60에포크)
    가장 성능이 좋았던 0.708 (OS) 기준 모델의 예측 위험 점수를 얻습니다."""
    ev = TrimodalEvaluator(
        target=target, epochs=epochs, batch_size=batch_size,
        save_dir=os.path.join(HERE, "outputs", "late_fusion_B", f"tabular_{target}"),
        model_factory=ablation.make_factory(ablation.CONFIGS["clin_report"]),
        max_folds=max_folds, seed=seed,
    ).run()
    return ev


def get_image_oof_simplecnn(target: str, batch_size: int = 16, epochs: int = 30,
                            max_folds=None, seed: int = 42):
    """(2) 영상 전용 모델 - SimpleCNN 구조 사용 (배치크기 16 / 30에포크)"""
    ev = ImageOnlyEvaluator(
        target=target, epochs=epochs, batch_size=batch_size, resize=512,
        save_dir=os.path.join(HERE, "outputs", "late_fusion_B", f"image_simplecnn_{target}"),
        ckpt_tag="image_simplecnn", max_folds=max_folds, seed=seed,
    ).run()
    return ev


def get_image_oof_resnet18(target: str, batch_size: int = 16, epochs: int = 30,
                           resize: int = 224, backbone_lr: float = 1e-5,
                           head_lr: float = 1e-3, weight_decay: float = 1e-4,
                           dropout: float = 0.3, max_folds=None, seed: int = 42):
    """(3) 영상 전용 모델 - 사전학습된 ResNet18 사용
    오버피팅 방지: 신경망 뼈대는 낮은 학습률, 출력 헤드는 높은 학습률 적용"""

    def model_factory():
        return ResNet18DeepSurv(gray_scale=True, pretrained=True, dropout=dropout)

    def optimizer_factory(model):
        return optim.Adam(
            [
                {"params": model.base.parameters(), "lr": backbone_lr},
                {"params": model.head.parameters(), "lr": head_lr},
            ],
            weight_decay=weight_decay,
        )

    ev = ImageOnlyEvaluator(
        target=target, epochs=epochs, batch_size=batch_size, resize=resize,
        model_factory=model_factory, optimizer_factory=optimizer_factory, augment=True,
        save_dir=os.path.join(HERE, "outputs", "late_fusion_B", f"image_resnet18_{target}"),
        ckpt_tag="image_resnet18", max_folds=max_folds, seed=seed,
    ).run()
    return ev


# ---------------------------------------------------------------------------
# 두 개의 예측 점수(Tabular 점수 + Image 점수)를 안전하게 합치는 CoxPH 결합기
# ---------------------------------------------------------------------------
def _oof_dict(oof_predictions):
    # 환자 ID별 위험 점수를 딕셔너리 형태로 변환
    return {row["research_id"]: row["risk_score"] for row in oof_predictions}


def _labels_by_id(cohort_df, target):
    # 환자 ID별 생존 기간(days) 및 사건 발생 여부(event) 라벨 추출
    return cohort_df.drop_duplicates("research_id").set_index("research_id")[
        [f"{target}_days", f"{target}_event"]
    ]


def combine_two(cohort_df, target, tabular_risk, image_risk, max_folds=None):
    """2개 점수(risk_tabular + risk_image) 결합:
    Fold마다 훈련 데이터로 생존분석(CoxPH) 모델을 가중치 학습시키고,
    테스트 데이터에 적용하여 최종 성능(C-index)을 측정합니다."""
    labels = _labels_by_id(cohort_df, target)

    def _frame(ids):
        return pd.DataFrame({
            "risk_tabular": [tabular_risk[i] for i in ids],
            "risk_image": [image_risk[i] for i in ids],
            "duration": labels.loc[ids, f"{target}_days"].to_numpy(dtype=float),
            "event": labels.loc[ids, f"{target}_event"].to_numpy(dtype=float),
        })

    fold_cindex, coefs_per_fold = [], []
    for fold, ids in _fold_plan(cohort_df, max_folds=max_folds):
        # 1. 훈련 데이터셋 구축 및 CoxPH 모델 적응(fit)
        train_df = _frame(ids["train"])
        cph = CoxPHFitter()
        cph.fit(train_df, duration_col="duration", event_col="event")

        # 2. 테스트 데이터셋 예측 및 C-index 계산
        test_df = _frame(ids["test"])
        test_risk = cph.predict_partial_hazard(test_df[["risk_tabular", "risk_image"]]).to_numpy()
        ci = concordance_index(test_df["duration"], -test_risk, test_df["event"])
        coefs = {name: float(cph.params_[name]) for name in ("risk_tabular", "risk_image")}
        fold_cindex.append(float(ci))
        coefs_per_fold.append(coefs)
        print(f"[lateB/combine/{target}] fold {fold}: C-index={ci:.4f} "
              f"coef(tabular,image)=({coefs['risk_tabular']:.3f},{coefs['risk_image']:.3f})")

    # Fold별 가중치 평균 계산
    mean_coef = {
        "risk_tabular": float(np.mean([c["risk_tabular"] for c in coefs_per_fold])),
        "risk_image": float(np.mean([c["risk_image"] for c in coefs_per_fold])),
    }
    return {
        "fold_cindex": fold_cindex,
        "mean": float(np.mean(fold_cindex)),
        "std": float(np.std(fold_cindex)),
        "coefs_per_fold": coefs_per_fold,
        "mean_coef": mean_coef,
    }


def _cindex_stats(c_indices):
    return float(np.mean(c_indices)), float(np.std(c_indices))


# ---------------------------------------------------------------------------
# 메인 실험 집행 함수 (target = OS 또는 PFS)
# ---------------------------------------------------------------------------
def run_target(target, max_folds=None, tab_epochs=60, img_epochs=30,
               resnet_epochs=30, seed=42):
    cohort_df = cohort.load_trimodal_cohort()

    # 1. Tabular (임상+리포트) 단독 모델 학습 및 점수 추출
    print(f"\n########## [lateB] TABULAR (clin+report joint, bs32/ep{tab_epochs}) target={target} ##########")
    tab_ev = get_tabular_oof(target, epochs=tab_epochs, max_folds=max_folds, seed=seed)
    tab_risk = _oof_dict(tab_ev.oof_predictions)
    tab_mean, tab_std = _cindex_stats(tab_ev.c_indices)
    print(f"[lateB] tabular-only {target}: {tab_mean:.4f} +/- {tab_std:.4f}  folds={[round(c,4) for c in tab_ev.c_indices]}")

    # 2. SimpleCNN 영상 단독 모델 학습 및 점수 추출
    print(f"\n########## [lateB] IMAGE SimpleCNN (bs16/ep{img_epochs}) target={target} ##########")
    scnn_ev = get_image_oof_simplecnn(target, epochs=img_epochs, max_folds=max_folds, seed=seed)
    scnn_risk = _oof_dict(scnn_ev.oof_predictions)
    scnn_mean, scnn_std = _cindex_stats(scnn_ev.c_indices)
    print(f"[lateB] image-SimpleCNN-only {target}: {scnn_mean:.4f} +/- {scnn_std:.4f}  folds={[round(c,4) for c in scnn_ev.c_indices]}")

    # 3. ResNet18 영상 단독 모델 학습 및 점수 추출
    print(f"\n########## [lateB] IMAGE ResNet18 (pretrained, bs16/ep{resnet_epochs}) target={target} ##########")
    rn_ev = get_image_oof_resnet18(target, epochs=resnet_epochs, max_folds=max_folds, seed=seed)
    rn_risk = _oof_dict(rn_ev.oof_predictions)
    rn_mean, rn_std = _cindex_stats(rn_ev.c_indices)
    print(f"[lateB] image-ResNet18-only {target}: {rn_mean:.4f} +/- {rn_std:.4f}  folds={[round(c,4) for c in rn_ev.c_indices]}")

    # 4. Late Fusion: Tabular + SimpleCNN 결합 평가
    print(f"\n########## [lateB] COMBINE tabular+SimpleCNN target={target} ##########")
    comb_scnn = combine_two(cohort_df, target, tab_risk, scnn_risk, max_folds=max_folds)
    print(f"[lateB] late-fusion tabular+SimpleCNN {target}: {comb_scnn['mean']:.4f} +/- {comb_scnn['std']:.4f} "
          f"mean_coef={comb_scnn['mean_coef']}")

    # 5. Late Fusion: Tabular + ResNet18 결합 평가
    print(f"\n########## [lateB] COMBINE tabular+ResNet18 target={target} ##########")
    comb_rn = combine_two(cohort_df, target, tab_risk, rn_risk, max_folds=max_folds)
    print(f"[lateB] late-fusion tabular+ResNet18 {target}: {comb_rn['mean']:.4f} +/- {comb_rn['std']:.4f} "
          f"mean_coef={comb_rn['mean_coef']}")

    result = {
        "target": target,
        "tabular_only": {"mean": tab_mean, "std": tab_std, "folds": [round(c, 4) for c in tab_ev.c_indices]},
        "image_simplecnn_only": {"mean": scnn_mean, "std": scnn_std, "folds": [round(c, 4) for c in scnn_ev.c_indices]},
        "image_resnet18_only": {"mean": rn_mean, "std": rn_std, "folds": [round(c, 4) for c in rn_ev.c_indices]},
        "late_simplecnn": comb_scnn,
        "late_resnet18": comb_rn,
    }
    return result


# ---------------------------------------------------------------------------
# 스크립트 실행 시작 지점
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="os,pfs")
    ap.add_argument("--smoke", action="store_true", help="빠른 테스트용 (1 fold, 2 에포크만 동작)")
    ap.add_argument("--max_folds", type=int, default=None)
    ap.add_argument("--tab_epochs", type=int, default=60)
    ap.add_argument("--img_epochs", type=int, default=30)
    ap.add_argument("--resnet_epochs", type=int, default=30)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "results.json"))
    args = ap.parse_args()

    # --smoke 옵션 지정 시 테스트 모드로 설정
    if args.smoke:
        args.max_folds = 1
        args.tab_epochs = 2
        args.img_epochs = 2
        args.resnet_epochs = 2

    all_results = {}
    for target in [t.strip() for t in args.targets.split(",") if t.strip()]:
        all_results[target] = run_target(
            target, max_folds=args.max_folds, tab_epochs=args.tab_epochs,
            img_epochs=args.img_epochs, resnet_epochs=args.resnet_epochs,
        )

    # 결과를 JSON 파일로 저장
    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[lateB] wrote {args.out}")

    # 최종 실험 요약 결과 출력
    print("\n================ METHOD B SUMMARY ================")
    for target, r in all_results.items():
        print(f"\n--- target={target} ---")
        print(f"  tabular-only            : {r['tabular_only']['mean']:.4f} +/- {r['tabular_only']['std']:.4f}")
        print(f"  image SimpleCNN-only    : {r['image_simplecnn_only']['mean']:.4f} +/- {r['image_simplecnn_only']['std']:.4f}")
        print(f"  image ResNet18-only     : {r['image_resnet18_only']['mean']:.4f} +/- {r['image_resnet18_only']['std']:.4f}")
        print(f"  late fusion +SimpleCNN  : {r['late_simplecnn']['mean']:.4f} +/- {r['late_simplecnn']['std']:.4f}"
              f"   coef(tab,img)=({r['late_simplecnn']['mean_coef']['risk_tabular']:.3f},{r['late_simplecnn']['mean_coef']['risk_image']:.3f})")
        print(f"  late fusion +ResNet18   : {r['late_resnet18']['mean']:.4f} +/- {r['late_resnet18']['std']:.4f}"
              f"   coef(tab,img)=({r['late_resnet18']['mean_coef']['risk_tabular']:.3f},{r['late_resnet18']['mean_coef']['risk_image']:.3f})")


if __name__ == "__main__":
    main()