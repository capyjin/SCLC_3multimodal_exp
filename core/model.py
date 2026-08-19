# -*- coding: utf-8 -*-
"""이미지 백본 · 모달리티 브랜치 · 융합 모델.

- ``SimpleCNNBackbone``/``ConvBlock`` 은 ``clinical+image/model.py`` 에서 그대로
  가져온 것이다 (융합 지시상 이미지 인코더 구조는 변경 금지).
- ``_MLPBranch`` 는 ``clinical+report`` 의 early fusion 브랜치 형태
  (Linear -> BatchNorm -> ReLU -> Dropout 반복) 이며 earlyfusion.md 가 검증한
  구조다 (OS C-index 0.7083).
- ``generate_net``/``get_cox_ph_model`` 은 두 원본 프로젝트 공통 코드로,
  late fusion 의 clinical-only/report-only arm (pycox CoxPH + torchtuples) 이 쓴다.

[통합 이력 — ConcatDeepSurv]
  예전에는 같은 concat 융합 모델이 **세 벌** 따로 있었다.
    model.TrimodalConcatDeepSurv        : 3모달 고정
    ablation.AblatableConcatDeepSurv    : use_* 플래그로 브랜치 on/off
    exp_fusion_fix.FixedConcatDeepSurv  : 위 + norm_mode(l2/none/scale)
  세 클래스의 브랜치 정의는 글자 단위로 같았고 forward 의 정규화 처리만 달랐다.
  ``ConcatDeepSurv`` 하나로 합치면서 **속성 이름(backbone/img_proj/
  clinical_branch/report_branch/head/branch_scale)을 그대로 유지**했으므로
  기존 체크포인트(.pt)의 state_dict 키가 전부 그대로 로드된다.
    - 기본값(use_* 전부 True, norm_mode="l2") = 옛 TrimodalConcatDeepSurv
    - use_* 만 지정        = 옛 AblatableConcatDeepSurv
    - norm_mode 까지 지정  = 옛 FixedConcatDeepSurv
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtuples as tt
from pycox.models import CoxPH


def generate_net(in_features, num_nodes, out_features=1, batch_norm=True, dropout=0.1, output_bias=False):
    return tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout, output_bias=output_bias)


def get_cox_ph_model(net, optimizer, lr):
    model = CoxPH(net, optimizer)
    model.optimizer.set_lr(lr)
    return model


# ---------------------------------------------------------------------------
# 이미지 백본 -- clinical+image/model.py 에서 그대로 가져옴
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


class SimpleCNNBackbone(nn.Module):
    """4-블록 CNN -> 512차원 특징 (clinical+image 원본과 동일)."""

    def __init__(self, in_channels: int = 1, output_dim: int = 512):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding = nn.Linear(256, output_dim)
        self.output_dim = int(output_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.embedding(x)


# ---------------------------------------------------------------------------
# 임상/판독지 브랜치 -- clinical+report/earlyfusion.md 가 검증한 형태
# ---------------------------------------------------------------------------

class _MLPBranch(nn.Module):
    """[Linear -> BatchNorm1d -> ReLU -> Dropout] 를 hidden 폭마다 반복."""

    def __init__(self, in_dim: int, hidden_dims: tuple[int, ...], dropout: float):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.net = nn.Sequential(*layers)
        self.out_dim = prev

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Early(concat) fusion -- 세 브랜치를 이어붙여 Cox head 하나로
# ---------------------------------------------------------------------------

NORM_MODES = ("l2", "none", "scale")


class ConcatDeepSurv(nn.Module):
    """image / clinical / report 를 concat 하는 DeepSurv 모델.

    forward 는 ``(image, tabular)`` 를 받는다. ``tabular`` 는
    ``features.build_fold_multimodal_tabular`` 가 만든 [clinical | text] 결합
    벡터 하나이며, ``clinical_dim`` 으로 다시 두 브랜치로 잘라 쓴다. 덕분에
    이미지+tabular 용 ``PetSurvivalDataset`` 배관을 고치지 않고 그대로 쓴다.

    브랜치 형태(고정):
      image    : SimpleCNNBackbone(512) -> Linear(512,image_proj_dim)+ReLU+Dropout(0.2)
      clinical : [Linear-BN-ReLU-Dropout(0.5)] x4 @128
      report   : [Linear-BN-ReLU-Dropout(0.3)] x(32,16)
      head     : concat -> Dropout(0.3) -> Linear(fused,1,bias=False)

    ``use_*`` 로 브랜치를 켜고 끈다 (절제 실험). 최소 하나는 켜져 있어야 한다.

    ``norm_mode`` 는 브랜치 출력을 이어붙이기 **직전**의 크기 조절 방식이다.
      "l2"    기본. 모든 브랜치를 길이 1로 맞춘다.
      "none"  아무것도 안 함 (브랜치가 스스로 출력 크기를 조절 가능).
      "scale" 길이 1로 맞춘 뒤 브랜치마다 학습되는 스칼라(``branch_scale``)를 곱한다.
              -> "이 모달리티는 작게 듣자"를 모델이 배울 수 있는지 보는 장치.
    """

    def __init__(
        self,
        clinical_dim: int,
        report_dim: int,
        use_image: bool = True,
        use_clinical: bool = True,
        use_report: bool = True,
        norm_mode: str = "l2",
        gray_scale: bool = True,
        image_proj_dim: int = 128,
        image_dropout: float = 0.2,
        clinical_hidden: tuple[int, ...] = (128, 128, 128, 128),
        clinical_dropout: float = 0.5,
        report_hidden: tuple[int, ...] = (32, 16),
        report_dropout: float = 0.3,
        fusion_dropout: float = 0.3,
    ):
        super().__init__()
        if not (use_image or use_clinical or use_report):
            raise ValueError("최소 한 개의 모달리티는 켜져 있어야 한다")
        if norm_mode not in NORM_MODES:
            raise ValueError(f"unknown norm_mode {norm_mode!r}; expected one of {list(NORM_MODES)}")

        self.clinical_dim = int(clinical_dim)
        self.report_dim = int(report_dim)
        self.use_image, self.use_clinical, self.use_report = use_image, use_clinical, use_report
        self.norm_mode = norm_mode

        fused_dim = 0
        if use_image:
            self.backbone = SimpleCNNBackbone(in_channels=1 if gray_scale else 3, output_dim=512)
            self.img_proj = nn.Sequential(
                nn.Linear(self.backbone.output_dim, image_proj_dim), nn.ReLU(), nn.Dropout(image_dropout),
            )
            fused_dim += image_proj_dim
        if use_clinical:
            self.clinical_branch = _MLPBranch(self.clinical_dim, clinical_hidden, clinical_dropout)
            fused_dim += self.clinical_branch.out_dim
        if use_report:
            self.report_branch = _MLPBranch(self.report_dim, report_hidden, report_dropout)
            fused_dim += self.report_branch.out_dim

        if norm_mode == "scale":
            # 브랜치당 1개, 1.0 에서 출발해 학습으로 조절된다.
            self.branch_scale = nn.Parameter(torch.ones(sum([use_image, use_clinical, use_report])))

        self.fusion_dropout = nn.Dropout(fusion_dropout)
        self.head = nn.Linear(fused_dim, 1, bias=False)

    def _shape(self, feat: torch.Tensor, idx: int) -> torch.Tensor:
        if self.norm_mode == "none":
            return feat
        feat = F.normalize(feat, dim=1)
        return feat if self.norm_mode == "l2" else feat * self.branch_scale[idx]

    def branch_features(self, image: torch.Tensor, tabular: torch.Tensor) -> dict[str, torch.Tensor]:
        """모달리티별 (크기 조절까지 끝난) 특징. 기여도 분석이 head 가중치와
        곱해 쓸 수 있도록 forward 와 **같은 경로**로 계산해 돌려준다."""
        expected = self.clinical_dim + self.report_dim
        if tabular.ndim != 2 or tabular.size(1) != expected:
            raise ValueError(f"Expected tabular shape [B, {expected}], got {tuple(tabular.shape)}")

        feats, idx = {}, 0
        if self.use_image:
            feats["image"] = self._shape(self.img_proj(self.backbone(image)), idx)
            idx += 1
        if self.use_clinical:
            feats["clinical"] = self._shape(self.clinical_branch(tabular[:, : self.clinical_dim]), idx)
            idx += 1
        if self.use_report:
            feats["report"] = self._shape(self.report_branch(tabular[:, self.clinical_dim:]), idx)
        return feats

    def forward(self, image: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        feats = self.branch_features(image, tabular)
        return self.head(self.fusion_dropout(torch.cat(list(feats.values()), dim=1)))


# ── 모달리티 조합표 ──────────────────────────────────────────────────────────
# 절제(ablation) 실험뿐 아니라 "임상+판독지만 학습" 같은 조합을 쓰는 모든 실험이
# 여기서 가져다 쓴다. 이름이 outputs/ 폴더명·문서 표기와 1:1로 대응한다.
MODALITY_CONFIGS = {
    "all":         dict(use_image=True,  use_clinical=True,  use_report=True),   # 3모달
    "clin_report": dict(use_image=False, use_clinical=True,  use_report=True),   # 임상+판독지
    "clin_image":  dict(use_image=True,  use_clinical=True,  use_report=False),  # 임상+영상
    "clin_only":   dict(use_image=False, use_clinical=True,  use_report=False),  # 임상만
    "report_only": dict(use_image=False, use_clinical=False, use_report=True),   # 판독지만
    "image_only":  dict(use_image=True,  use_clinical=False, use_report=False),  # 영상만
}

# 각 모달리티 브랜치의 기본 출력 차원 (기여도/가중치 분해에서 head 를 블록으로 자를 때 쓴다)
BRANCH_OUT_DIM = {"image": 128, "clinical": 128, "report": 16}


def make_model_factory(flags: dict | None = None, **overrides):
    """``TrimodalEvaluator(model_factory=...)`` 에 넘길 공장 함수를 만든다.

    ``flags`` 는 ``MODALITY_CONFIGS`` 의 한 항목(use_* 3개)이고, ``overrides`` 로
    ``norm_mode``/``image_proj_dim``/``report_hidden`` 등 나머지 인자를 덮어쓴다.
    Evaluator 가 fold 마다 clinical_dim/report_dim 을 알아낸 뒤 호출한다.
    """
    kwargs = {**(flags or MODALITY_CONFIGS["all"]), **overrides}

    def factory(clinical_dim, report_dim):
        return ConcatDeepSurv(clinical_dim, report_dim, **kwargs)

    return factory


# ---------------------------------------------------------------------------
# 영상 단독 DeepSurv -- late fusion 의 이미지 arm (tabular 입력 없음)
# ---------------------------------------------------------------------------

class ImageOnlyDeepSurv(nn.Module):
    """순수 영상 DeepSurv. clinical+image/model.py 의 tabular_dim=0 경로와 동일
    (backbone -> Dropout -> Linear(512,1), 투영/L2정규화 없음)."""

    def __init__(self, gray_scale: bool = True, dropout: float = 0.2):
        super().__init__()
        self.backbone = SimpleCNNBackbone(in_channels=1 if gray_scale else 3, output_dim=512)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.backbone.output_dim, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(image))


class ResNet18DeepSurv(nn.Module):
    """ImageNet 사전학습 ResNet18 백본을 쓰는 영상 단독 DeepSurv (대조 백본).

    흑백 PET-CT 를 넣기 위해 conv1 을 1채널로 교체하고, 사전학습 RGB 필터의
    **채널 평균**을 초기값으로 쓴다(무작위 초기화보다 사전학습 정보를 보존).
    """

    def __init__(self, gray_scale: bool = True, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        self.base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1) if pretrained else resnet18()

        if gray_scale:
            old_conv = self.base.conv1
            self.base.conv1 = nn.Conv2d(
                in_channels=1, out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size, stride=old_conv.stride,
                padding=old_conv.padding, bias=old_conv.bias is not None,
            )
            with torch.no_grad():
                self.base.conv1.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)

        in_features = self.base.fc.in_features   # 512
        self.base.fc = nn.Identity()             # 분류 head 제거
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))

    def forward(self, x):
        return self.head(self.base(x))
