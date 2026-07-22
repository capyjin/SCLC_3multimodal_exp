# -*- coding: utf-8 -*-
"""Image backbone (unchanged), clinical/report branches, and the fusion models.

- ``SimpleCNNBackbone`` / ``ConvBlock`` are copied verbatim from
  ``clinical+image/SCLC_simple_CNN-main/model.py`` -- the fusion instructions
  require the image encoder architecture stay exactly as-is.
- ``_MLPBranch`` reproduces the clinical/TF-IDF branch shape from
  ``clinical+report/SCLC_report_unimodal_test-main/CODE/train_early_fusion_clinical_report.py``
  (Linear -> BatchNorm -> ReLU -> Dropout, repeated per hidden width), which
  is the architecture ``earlyfusion.md`` validated (OS C-index 0.7083) and
  which the user chose over the plain clinical+image branch (no BN/dropout).
- ``generate_net``/``get_cox_ph_model`` are copied from both source
  projects' ``model.py`` verbatim -- used by the clinical-only/report-only
  unimodal arms in late fusion (pycox CoxPH + torchtuples MLPVanilla).
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
# Image backbone -- copied verbatim from clinical+image/model.py
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
    """Small four-block CNN producing a 512-D feature (unchanged from clinical+image)."""

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
# Clinical / report branches -- shape validated in clinical+report/earlyfusion.md
# ---------------------------------------------------------------------------

class _MLPBranch(nn.Module):
    """[Linear -> BatchNorm1d -> ReLU -> Dropout] repeated per hidden width."""

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
# Early (concat) fusion: image + clinical + report -> one Cox head
# ---------------------------------------------------------------------------

class TrimodalConcatDeepSurv(nn.Module):
    """Image + clinical + report concat-fusion DeepSurv model.

    Forward takes ``tabular`` as the single combined [clinical | tfidf]
    vector produced by ``features.build_fold_multimodal_tabular`` and splits
    it back into the two branches using ``clinical_dim``.

    Branch shapes (fixed per earlyfusion.md / the image+clinical baseline):
    - image:    SimpleCNNBackbone (512D) -> Linear(512,128)+ReLU+Dropout(0.2) -> L2 normalize
    - clinical: [Linear-BN-ReLU-Dropout(0.5)] x4 @128 -> L2 normalize
    - report:   [Linear-BN-ReLU-Dropout(0.3)] x(32,16) -> L2 normalize
    - head:     concat(128+128+16=272) -> Dropout(0.3) -> Linear(272,1,bias=False)
    """

    def __init__(
        self,
        clinical_dim: int,
        report_dim: int,
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
        self.clinical_dim = int(clinical_dim)
        self.report_dim = int(report_dim)

        self.backbone = SimpleCNNBackbone(in_channels=1 if gray_scale else 3, output_dim=512)
        self.img_proj = nn.Sequential(
            nn.Linear(self.backbone.output_dim, image_proj_dim), nn.ReLU(), nn.Dropout(image_dropout),
        )
        self.clinical_branch = _MLPBranch(self.clinical_dim, clinical_hidden, clinical_dropout)
        self.report_branch = _MLPBranch(self.report_dim, report_hidden, report_dropout)

        fused_dim = image_proj_dim + self.clinical_branch.out_dim + self.report_branch.out_dim
        self.fusion_dropout = nn.Dropout(fusion_dropout)
        self.head = nn.Linear(fused_dim, 1, bias=False)

    def forward(self, image: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        if tabular.ndim != 2 or tabular.size(1) != self.clinical_dim + self.report_dim:
            raise ValueError(
                f"Expected tabular shape [B, {self.clinical_dim + self.report_dim}], got {tuple(tabular.shape)}"
            )
        clinical_x = tabular[:, : self.clinical_dim]
        report_x = tabular[:, self.clinical_dim:]

        img_feat = F.normalize(self.img_proj(self.backbone(image)), dim=1)
        clinical_feat = F.normalize(self.clinical_branch(clinical_x), dim=1)
        report_feat = F.normalize(self.report_branch(report_x), dim=1)

        fused = self.fusion_dropout(torch.cat([img_feat, clinical_feat, report_feat], dim=1))
        return self.head(fused)


# ---------------------------------------------------------------------------
# Image-only DeepSurv -- late-fusion image arm (no tabular input at all)
# ---------------------------------------------------------------------------

class ImageOnlyDeepSurv(nn.Module):
    """Pure-image DeepSurv, matching clinical+image/model.py's tabular_dim=0
    path (backbone -> Dropout -> Linear(512,1), no projection/L2-norm)."""

    def __init__(self, gray_scale: bool = True, dropout: float = 0.2):
        super().__init__()
        self.backbone = SimpleCNNBackbone(in_channels=1 if gray_scale else 3, output_dim=512)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.backbone.output_dim, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(image))


def parameter_counts(model: nn.Module) -> dict[str, int]:
    return {
        "total": int(sum(p.numel() for p in model.parameters())),
        "trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }
