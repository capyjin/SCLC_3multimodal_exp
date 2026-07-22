# -*- coding: utf-8 -*-
"""Method B: 3-modal LATE fusion = [clinical+report joint tabular OOF risk]
combined with [image OOF risk] via a leakage-safe per-fold CoxPH stack.

Rationale (see RESULTS.md sections 3-6): adding the SimpleCNN image branch
*inside* early concat fusion HURTS once the tabular model is properly trained
(tri-modal 0.678 < clin+report 0.708) because concat forces one shared
end-to-end schedule and the from-scratch image CNN overfits. LATE fusion
trains each modality independently with its own regime/early-stopping, then
learns a 2-covariate weighted sum of their out-of-fold risk scores.

Arms compared (both targets os/pfs):
  tabular  = clinical+report joint early-concat, NO image branch, bs32/ep60
             (the strong 0.708 OS / 0.668 PFS model) -- computed ONCE per
             target and reused for both image arms.
  image A  = SimpleCNN ImageOnlyDeepSurv, bs16/ep30 (its own good regime).
  image B  = ResNet18DeepSurv, ImageNet-pretrained, bs16/ep~30, low-LR
             backbone (1e-5) + higher-LR head (1e-3), flip augmentation,
             resize 224, dropout 0.3, weight decay 1e-4.

The CoxPH stack is leakage-safe: every risk value is out-of-fold (each
patient is in the test split exactly once across the 5 folds), so per fold we
fit lifelines.CoxPHFitter on that fold's TRAIN research_ids' OOF risk columns
and evaluate on that fold's TEST research_ids -- adapting
late_fusion.combine_weighted_sum from 3 covariates to 2
(risk_tabular + risk_image).
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
# ResNet18 image DeepSurv -- ported from
# SCLC_resnet_experiment/SCLC/SCLC_exp4_layernorm_fusion/model.py
# (ResNet18DeepSurv). Grayscale handling: replace conv1 with a 1-channel conv
# whose weights are the channel-mean of the pretrained RGB conv1, preserving
# the learned ImageNet filters while accepting our single-channel PET-CT crops.
# ---------------------------------------------------------------------------
class ResNet18DeepSurv(nn.Module):
    def __init__(self, gray_scale: bool = True, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        if pretrained:
            self.base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.base = resnet18()

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
                # average pretrained RGB filters -> single-channel filter
                self.base.conv1.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)

        in_features = self.base.fc.in_features  # 512
        self.base.fc = nn.Identity()
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))

    def forward(self, x):
        return self.head(self.base(x))


# ---------------------------------------------------------------------------
# Per-arm OOF producers
# ---------------------------------------------------------------------------
def get_tabular_oof(target: str, batch_size: int = 32, epochs: int = 60,
                    max_folds=None, seed: int = 42):
    """clinical+report joint early-concat (NO image branch), bs32/ep60.
    Returns the fitted TrimodalEvaluator (has .oof_predictions and .c_indices).
    This is the strong 0.708 OS / 0.668 PFS tabular model."""
    ev = TrimodalEvaluator(
        target=target, epochs=epochs, batch_size=batch_size,
        save_dir=os.path.join(HERE, "outputs", "late_fusion_B", f"tabular_{target}"),
        model_factory=ablation.make_factory(ablation.CONFIGS["clin_report"]),
        max_folds=max_folds, seed=seed,
    ).run()
    return ev


def get_image_oof_simplecnn(target: str, batch_size: int = 16, epochs: int = 30,
                            max_folds=None, seed: int = 42):
    """SimpleCNN ImageOnlyDeepSurv, bs16/ep30 (image's own good regime;
    60 epochs overfits per RESULTS.md section 5)."""
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
    """ResNet18 (ImageNet-pretrained) DeepSurv image arm. Overfit control:
    low-LR backbone + higher-LR head, flip augmentation, dropout, weight decay,
    modest epochs, best-val-cindex early stopping (via fit())."""

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
# Leakage-safe 2-covariate CoxPH stack
# ---------------------------------------------------------------------------
def _oof_dict(oof_predictions):
    return {row["research_id"]: row["risk_score"] for row in oof_predictions}


def _labels_by_id(cohort_df, target):
    return cohort_df.drop_duplicates("research_id").set_index("research_id")[
        [f"{target}_days", f"{target}_event"]
    ]


def combine_two(cohort_df, target, tabular_risk, image_risk, max_folds=None):
    """2-covariate (risk_tabular + risk_image) leakage-safe CoxPH stack.
    Per fold: fit CoxPHFitter on the fold's TRAIN ids' OOF risk columns,
    evaluate on the fold's TEST ids. Returns per-fold C-index, coefficients,
    and mean/std."""
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
        train_df = _frame(ids["train"])
        cph = CoxPHFitter()
        cph.fit(train_df, duration_col="duration", event_col="event")

        test_df = _frame(ids["test"])
        test_risk = cph.predict_partial_hazard(test_df[["risk_tabular", "risk_image"]]).to_numpy()
        ci = concordance_index(test_df["duration"], -test_risk, test_df["event"])
        coefs = {name: float(cph.params_[name]) for name in ("risk_tabular", "risk_image")}
        fold_cindex.append(float(ci))
        coefs_per_fold.append(coefs)
        print(f"[lateB/combine/{target}] fold {fold}: C-index={ci:.4f} "
              f"coef(tabular,image)=({coefs['risk_tabular']:.3f},{coefs['risk_image']:.3f})")

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
# Driver
# ---------------------------------------------------------------------------
def run_target(target, max_folds=None, tab_epochs=60, img_epochs=30,
               resnet_epochs=30, seed=42):
    cohort_df = cohort.load_trimodal_cohort()

    print(f"\n########## [lateB] TABULAR (clin+report joint, bs32/ep{tab_epochs}) target={target} ##########")
    tab_ev = get_tabular_oof(target, epochs=tab_epochs, max_folds=max_folds, seed=seed)
    tab_risk = _oof_dict(tab_ev.oof_predictions)
    tab_mean, tab_std = _cindex_stats(tab_ev.c_indices)
    print(f"[lateB] tabular-only {target}: {tab_mean:.4f} +/- {tab_std:.4f}  folds={[round(c,4) for c in tab_ev.c_indices]}")

    print(f"\n########## [lateB] IMAGE SimpleCNN (bs16/ep{img_epochs}) target={target} ##########")
    scnn_ev = get_image_oof_simplecnn(target, epochs=img_epochs, max_folds=max_folds, seed=seed)
    scnn_risk = _oof_dict(scnn_ev.oof_predictions)
    scnn_mean, scnn_std = _cindex_stats(scnn_ev.c_indices)
    print(f"[lateB] image-SimpleCNN-only {target}: {scnn_mean:.4f} +/- {scnn_std:.4f}  folds={[round(c,4) for c in scnn_ev.c_indices]}")

    print(f"\n########## [lateB] IMAGE ResNet18 (pretrained, bs16/ep{resnet_epochs}) target={target} ##########")
    rn_ev = get_image_oof_resnet18(target, epochs=resnet_epochs, max_folds=max_folds, seed=seed)
    rn_risk = _oof_dict(rn_ev.oof_predictions)
    rn_mean, rn_std = _cindex_stats(rn_ev.c_indices)
    print(f"[lateB] image-ResNet18-only {target}: {rn_mean:.4f} +/- {rn_std:.4f}  folds={[round(c,4) for c in rn_ev.c_indices]}")

    print(f"\n########## [lateB] COMBINE tabular+SimpleCNN target={target} ##########")
    comb_scnn = combine_two(cohort_df, target, tab_risk, scnn_risk, max_folds=max_folds)
    print(f"[lateB] late-fusion tabular+SimpleCNN {target}: {comb_scnn['mean']:.4f} +/- {comb_scnn['std']:.4f} "
          f"mean_coef={comb_scnn['mean_coef']}")

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="os,pfs")
    ap.add_argument("--smoke", action="store_true", help="1 fold, tiny epochs, sanity only")
    ap.add_argument("--max_folds", type=int, default=None)
    ap.add_argument("--tab_epochs", type=int, default=60)
    ap.add_argument("--img_epochs", type=int, default=30)
    ap.add_argument("--resnet_epochs", type=int, default=30)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "results.json"))
    args = ap.parse_args()

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

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[lateB] wrote {args.out}")

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
