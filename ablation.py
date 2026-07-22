# -*- coding: utf-8 -*-
"""Ablation: same 238-patient split, same training loop (TrimodalEvaluator),
only the set of active branches changes. Isolates whether adding the image
branch to clinical+report actually helps or hurts.

Configs (OS target, 5-fold):
  all         : image + clinical + report   (== the reported early_fusion)
  clin_report : clinical + report           (reproduce the 0.7083 2-modal)
  clin_image  : clinical + image
  clin_only   : clinical
  report_only : report
  image_only  : image

Run:  python ablation.py --target os
"""
import argparse
import functools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import SimpleCNNBackbone, _MLPBranch
from train import TrimodalEvaluator


class AblatableConcatDeepSurv(nn.Module):
    """Identical branch shapes to TrimodalConcatDeepSurv; concat only the
    branches enabled by the use_* flags. forward signature is unchanged
    (image, tabular) so it drops into the existing evaluator/data pipeline."""

    def __init__(self, clinical_dim, report_dim, use_image=True, use_clinical=True,
                 use_report=True, gray_scale=True, image_proj_dim=128, image_dropout=0.2,
                 clinical_hidden=(128, 128, 128, 128), clinical_dropout=0.5,
                 report_hidden=(32, 16), report_dropout=0.3, fusion_dropout=0.3):
        super().__init__()
        assert use_image or use_clinical or use_report
        self.clinical_dim = int(clinical_dim)
        self.report_dim = int(report_dim)
        self.use_image, self.use_clinical, self.use_report = use_image, use_clinical, use_report

        fused_dim = 0
        if use_image:
            self.backbone = SimpleCNNBackbone(in_channels=1 if gray_scale else 3, output_dim=512)
            self.img_proj = nn.Sequential(
                nn.Linear(self.backbone.output_dim, image_proj_dim), nn.ReLU(), nn.Dropout(image_dropout))
            fused_dim += image_proj_dim
        if use_clinical:
            self.clinical_branch = _MLPBranch(self.clinical_dim, clinical_hidden, clinical_dropout)
            fused_dim += self.clinical_branch.out_dim
        if use_report:
            self.report_branch = _MLPBranch(self.report_dim, report_hidden, report_dropout)
            fused_dim += self.report_branch.out_dim

        self.fusion_dropout = nn.Dropout(fusion_dropout)
        self.head = nn.Linear(fused_dim, 1, bias=False)

    def forward(self, image, tabular):
        feats = []
        if self.use_image:
            feats.append(F.normalize(self.img_proj(self.backbone(image)), dim=1))
        clinical_x = tabular[:, : self.clinical_dim]
        report_x = tabular[:, self.clinical_dim:]
        if self.use_clinical:
            feats.append(F.normalize(self.clinical_branch(clinical_x), dim=1))
        if self.use_report:
            feats.append(F.normalize(self.report_branch(report_x), dim=1))
        fused = self.fusion_dropout(torch.cat(feats, dim=1))
        return self.head(fused)


CONFIGS = {
    "all":         dict(use_image=True,  use_clinical=True,  use_report=True),
    "clin_report": dict(use_image=False, use_clinical=True,  use_report=True),
    "clin_image":  dict(use_image=True,  use_clinical=True,  use_report=False),
    "clin_only":   dict(use_image=False, use_clinical=True,  use_report=False),
    "report_only": dict(use_image=False, use_clinical=False, use_report=True),
    "image_only":  dict(use_image=True,  use_clinical=False, use_report=False),
}


def make_factory(flags):
    def factory(clinical_dim, report_dim):
        return AblatableConcatDeepSurv(clinical_dim, report_dim, **flags)
    return factory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="os", choices=("os", "pfs"))
    ap.add_argument("--configs", default="all,clin_report,clin_image,clin_only,report_only,image_only")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    results = {}
    for name in args.configs.split(","):
        name = name.strip()
        flags = CONFIGS[name]
        print(f"\n########## CONFIG: {name}  ({flags})  target={args.target} bs={args.batch_size} ep={args.epochs} ##########")
        ev = TrimodalEvaluator(
            target=args.target, epochs=args.epochs, batch_size=args.batch_size,
            save_dir=f"outputs/ablation{args.tag}/{name}_{args.target}",
            model_factory=make_factory(flags),
        ).run()
        cis = ev.c_indices
        results[name] = (float(np.mean(cis)), float(np.std(cis)), [round(c, 4) for c in cis])
        print(f"[ABLATION] {name} {args.target}: {np.mean(cis):.4f} +/- {np.std(cis):.4f}  folds={[round(c,4) for c in cis]}")

    print("\n================ ABLATION SUMMARY (target=%s) ================" % args.target)
    print(f"{'config':<14}{'mean':>8}{'std':>8}   folds")
    for name, (m, s, folds) in results.items():
        print(f"{name:<14}{m:>8.4f}{s:>8.4f}   {folds}")


if __name__ == "__main__":
    main()
