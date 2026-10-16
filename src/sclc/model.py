# -*- coding: utf-8 -*-
"""영상 백본(SimpleCNN 등), 임상/판독지 브랜치, 융합 모델 정의.

- SimpleCNNBackbone: 영상 인코더 (구조 변경 금지)
- _MLPBranch: 임상/판독지용 MLP 브랜치
- ConcatDeepSurv: 영상·임상·판독지를 이어붙여 Cox head 로 보내는 early fusion 모델
  (use_* 플래그로 모달리티 on/off 가능)
- ImageOnlyDeepSurv: 영상 단독 모델 (late fusion 의 영상 arm)
- generate_net / get_cox_ph_model: late fusion 의 임상·판독지 arm 용 (pycox CoxPH)
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
# 영상 백본
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
    """ConvBlock 4개(채널 32→64→128→256) → 평균풀링 → Linear → 512차원 특징."""

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
# 임상/판독지 브랜치
# ---------------------------------------------------------------------------

class _MLPBranch(nn.Module):
    """[Linear → BatchNorm → ReLU → Dropout] 을 hidden 크기 목록만큼 반복."""

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
# Early fusion -- 브랜치 출력을 이어붙여 Cox head 하나로
# ---------------------------------------------------------------------------

NORM_MODES = ("l2", "none", "scale")


class ConcatDeepSurv(nn.Module):
    """영상 / 임상 / 판독지 브랜치를 이어붙여(concat) 위험점수 1개를 내는 DeepSurv.

    입력: forward(image, tabular)
      tabular = [임상 | 판독지] 가 붙은 벡터. clinical_dim 기준으로 다시 나눠 쓴다.

    구조:
      영상   : SimpleCNN(512) → Linear(512→image_proj_dim) → ReLU → Dropout(0.2)
      임상   : [Linear-BN-ReLU-Dropout(0.5)] × 4, 폭 128
      판독지 : [Linear-BN-ReLU-Dropout(0.3)] 폭 32, 16
      head   : concat → Dropout(0.3) → Linear(→1)

    use_*   : 모달리티 on/off (최소 1개 필요)
    norm_mode : concat 직전 브랜치 출력 크기 조절 방식
      "l2"    각 브랜치 길이를 1로 맞춤 (기본)
      "none"  조절 안 함
      "scale" 길이를 1로 맞춘 뒤 브랜치별 학습 가능한 배율(branch_scale)을 곱함
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
            # 브랜치당 배율 1개 (초기값 1.0)
            self.branch_scale = nn.Parameter(torch.ones(sum([use_image, use_clinical, use_report])))

        self.fusion_dropout = nn.Dropout(fusion_dropout)
        self.head = nn.Linear(fused_dim, 1, bias=False)

    def _shape(self, feat: torch.Tensor, idx: int) -> torch.Tensor:
        if self.norm_mode == "none":
            return feat
        feat = F.normalize(feat, dim=1)
        return feat if self.norm_mode == "l2" else feat * self.branch_scale[idx]

    def branch_features(self, image: torch.Tensor, tabular: torch.Tensor) -> dict[str, torch.Tensor]:
        """모달리티별 특징(크기 조절 후)을 dict 로 반환. forward 와 같은 경로라 기여도 분석에 쓴다."""
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


# 모달리티 조합표 (이름은 outputs/ 폴더명·문서 표기와 동일)
MODALITY_CONFIGS = {
    "all":         dict(use_image=True,  use_clinical=True,  use_report=True),   # 3모달
    "clin_report": dict(use_image=False, use_clinical=True,  use_report=True),   # 임상+판독지
    "clin_image":  dict(use_image=True,  use_clinical=True,  use_report=False),  # 임상+영상
    "report_image": dict(use_image=True, use_clinical=False, use_report=True),   # 판독지+영상
    "clin_only":   dict(use_image=False, use_clinical=True,  use_report=False),  # 임상만
    "report_only": dict(use_image=False, use_clinical=False, use_report=True),   # 판독지만
    "image_only":  dict(use_image=True,  use_clinical=False, use_report=False),  # 영상만
}

# 브랜치별 기본 출력 차원 (head 가중치를 모달리티별로 자를 때 사용)
BRANCH_OUT_DIM = {"image": 128, "clinical": 128, "report": 16}


def make_model_factory(flags: dict | None = None, **overrides):
    """TrimodalEvaluator 에 넘길 모델 생성 함수를 만든다.

    flags     : MODALITY_CONFIGS 의 한 항목 (use_* 3개)
    overrides : norm_mode, image_proj_dim 등 나머지 인자
    fold 마다 clinical_dim/report_dim 이 정해진 뒤 호출된다.
    """
    kwargs = {**(flags or MODALITY_CONFIGS["all"]), **overrides}

    def factory(clinical_dim, report_dim):
        return ConcatDeepSurv(clinical_dim, report_dim, **kwargs)

    return factory


# ---------------------------------------------------------------------------
# 영상 단독 DeepSurv (late fusion 의 영상 arm)
# ---------------------------------------------------------------------------

class ImageOnlyDeepSurv(nn.Module):
    """영상 단독 DeepSurv: SimpleCNN(512) → Dropout → Linear(512→1)."""

    def __init__(self, gray_scale: bool = True, dropout: float = 0.2):
        super().__init__()
        self.backbone = SimpleCNNBackbone(in_channels=1 if gray_scale else 3, output_dim=512)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.backbone.output_dim, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(image))


class ResNet18DeepSurv(nn.Module):
    """ImageNet 사전학습 ResNet18 기반 영상 단독 DeepSurv (비교용).

    흑백 입력을 위해 conv1 을 1채널로 바꾸고, 사전학습 RGB 필터의 채널 평균으로 초기화한다.
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


class _RadImageNetBackbone(nn.Module):
    """흑백 1채널 → 3채널 복제 → RadImageNet ResNet50(분류 head 없음) → 평균풀링 → 2048차원.

    conv1 은 원본(3채널) 그대로 두고 입력 쪽을 3채널로 복제한다.
    """

    output_dim = 2048

    def __init__(self, pretrained: bool = True):
        super().__init__()
        if pretrained:
            self.resnet = torch.hub.load(
                "Warvito/radimagenet-models", "radimagenet_resnet50", trust_repo=True,
            )
        else:
            # 대조군: 같은 구조, 사전학습 가중치 없음 (실험8/10과 동일한 논리)
            module = torch.hub.load(
                "Warvito/radimagenet-models", "radimagenet_resnet50", trust_repo=True,
            ).__class__
            self.resnet = module()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 1:
            x = x.expand(-1, 3, -1, -1)
        feat = self.resnet(x)
        return self.pool(feat).flatten(1)


class RadImageNetDeepSurv(nn.Module):
    """RadImageNet(방사선영상 사전학습) ResNet50 기반 영상 단독 DeepSurv (비교용).

    ResNet18DeepSurv(ImageNet 자연영상)와 달리 의료영상으로 사전학습된 백본이다.
    head 는 Dropout → Linear(→1).
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.backbone = _RadImageNetBackbone(pretrained=pretrained)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.backbone.output_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))
