# -*- coding: utf-8 -*-
"""[임시 실험 스크립트] RESULTS.md 9.7의 3번·4번 제안을 실제로 검증한다.

3번: 브랜치마다 F.normalize로 크기를 1로 강제하는 것을 없애거나,
     학습 가능한 스케일을 붙여서 "모델이 스스로 발언권을 줄일 수 있게" 만든다.
4번: 판독지 브랜치 출력 차원을 16 -> 64 -> 128로 키운다.

기존 ablation.py의 모델과 완전히 같은 구조에서, 위 두 가지만 바꿔 비교한다.
(norm_mode="l2", report_hidden=(32,16)이면 ablation.py와 동일 = 재현 확인용)

Run:  python exp_fusion_fix.py --target pfs
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import SimpleCNNBackbone, _MLPBranch
from train import TrimodalEvaluator


class FixedConcatDeepSurv(nn.Module):
    """ablation.py의 AblatableConcatDeepSurv와 같은 구조.
    단, 브랜치 출력을 합치기 직전의 '크기 조절 방식'(norm_mode)만 바꿀 수 있다.

    norm_mode:
      "l2"    - 기존 방식. 모든 브랜치를 길이 1로 강제 (발언권을 줄일 수단이 없음)
      "none"  - 아무것도 안 함. 브랜치가 스스로 출력 크기를 키우거나 줄일 수 있음
      "scale" - 길이 1로 맞춘 뒤, 브랜치마다 '학습되는 손잡이' 하나를 곱함
                -> 안정성은 유지하면서 모델이 "이 모달리티는 작게 듣자"를 배울 수 있음
    """

    def __init__(self, clinical_dim, report_dim, use_image=True, use_clinical=True,
                 use_report=True, norm_mode="l2", gray_scale=True,
                 image_proj_dim=128, image_dropout=0.2,
                 clinical_hidden=(128, 128, 128, 128), clinical_dropout=0.5,
                 report_hidden=(32, 16), report_dropout=0.3, fusion_dropout=0.3):
        super().__init__()
        assert use_image or use_clinical or use_report
        assert norm_mode in ("l2", "none", "scale")
        self.clinical_dim, self.report_dim = int(clinical_dim), int(report_dim)
        self.use_image, self.use_clinical, self.use_report = use_image, use_clinical, use_report
        self.norm_mode = norm_mode

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

        # "scale" 모드에서 쓸 손잡이. 브랜치마다 1개씩, 1.0에서 시작해 학습으로 조절된다.
        # (이 값이 작아지면 = 모델이 그 모달리티를 덜 듣기로 배운 것)
        if norm_mode == "scale":
            n_branch = sum([use_image, use_clinical, use_report])
            self.branch_scale = nn.Parameter(torch.ones(n_branch))

        self.fusion_dropout = nn.Dropout(fusion_dropout)
        self.head = nn.Linear(fused_dim, 1, bias=False)

    def _shape(self, feat, idx):
        """브랜치 출력의 크기를 norm_mode에 따라 조절한다."""
        if self.norm_mode == "l2":
            return F.normalize(feat, dim=1)
        if self.norm_mode == "none":
            return feat
        return F.normalize(feat, dim=1) * self.branch_scale[idx]

    def forward(self, image, tabular):
        feats, idx = [], 0
        if self.use_image:
            feats.append(self._shape(self.img_proj(self.backbone(image)), idx)); idx += 1
        clinical_x = tabular[:, : self.clinical_dim]
        report_x = tabular[:, self.clinical_dim:]
        if self.use_clinical:
            feats.append(self._shape(self.clinical_branch(clinical_x), idx)); idx += 1
        if self.use_report:
            feats.append(self._shape(self.report_branch(report_x), idx)); idx += 1
        return self.head(self.fusion_dropout(torch.cat(feats, dim=1)))


# ── 실험 목록 ──
# base = 기존 방식(대조군). 나머지는 3번/4번 제안을 하나씩 또는 같이 적용한 것.
REPORT_HIDDEN = {16: (32, 16), 64: (128, 64), 128: (256, 128)}

VARIANTS = {
    # 이름            : (세 모달 사용?, norm_mode, report 출력차원)
    "base":            dict(tri=True,  norm_mode="l2",    report_dim_out=16),
    "nonorm":          dict(tri=True,  norm_mode="none",  report_dim_out=16),
    "scale":           dict(tri=True,  norm_mode="scale", report_dim_out=16),
    "rep64":           dict(tri=True,  norm_mode="l2",    report_dim_out=64),
    "rep128":          dict(tri=True,  norm_mode="l2",    report_dim_out=128),
    "scale_rep64":     dict(tri=True,  norm_mode="scale", report_dim_out=64),
    "scale_rep128":    dict(tri=True,  norm_mode="scale", report_dim_out=128),
    # 대조군: 이미지 없이(clin+report) 판독지 차원만 키우면 어떻게 되나?
    # -> 차원 확대가 'fusion을 고친 것'인지 '그냥 판독지 브랜치가 좋아진 것'인지 구분용
    "cr_base":         dict(tri=False, norm_mode="l2",    report_dim_out=16),
    "cr_rep64":        dict(tri=False, norm_mode="l2",    report_dim_out=64),
    "cr_rep128":       dict(tri=False, norm_mode="l2",    report_dim_out=128),
}


def make_factory(spec):
    def factory(clinical_dim, report_dim):
        return FixedConcatDeepSurv(
            clinical_dim, report_dim,
            use_image=spec["tri"], use_clinical=True, use_report=True,
            norm_mode=spec["norm_mode"],
            report_hidden=REPORT_HIDDEN[spec["report_dim_out"]],
        )
    return factory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="pfs", choices=("os", "pfs"))
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    out_dir = "outputs/fusion_fix"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"results_{args.target}.json")
    results = json.load(open(out_path)) if os.path.exists(out_path) else {}

    for name in [v.strip() for v in args.variants.split(",")]:
        spec = VARIANTS[name]
        print(f"\n########## {name}  {spec}  target={args.target} ##########")
        ev = TrimodalEvaluator(
            target=args.target, epochs=args.epochs, batch_size=args.batch_size,
            save_dir=f"{out_dir}/{name}_{args.target}",
            model_factory=make_factory(spec),
        ).run()
        cis = ev.c_indices

        # "scale" 모드면 모델이 배운 손잡이 값을 확인한다 (발언권을 실제로 줄였는지)
        scales = None
        if spec["norm_mode"] == "scale":
            per_fold = []
            for f in sorted({r["fold"] for r in ev.fold_records}):
                sd = torch.load(f"{out_dir}/{name}_{args.target}/fold{f}_early_fusion_{args.target}.pt",
                                map_location="cpu")
                per_fold.append(sd["branch_scale"].tolist())
            scales = np.mean(per_fold, axis=0).tolist()
            names = (["image"] if spec["tri"] else []) + ["clinical", "report"]
            print("  learned branch_scale:", {n: round(s, 3) for n, s in zip(names, scales)})

        results[name] = {
            "mean": float(np.mean(cis)), "std": float(np.std(cis)),
            "folds": [round(c, 4) for c in cis], "spec": spec, "branch_scale": scales,
        }
        print(f"[FIX] {name} {args.target}: {np.mean(cis):.4f} +/- {np.std(cis):.4f}")
        json.dump(results, open(out_path, "w"), indent=2)   # 중간에 끊겨도 남도록 매번 저장

    print(f"\n================ FUSION FIX SUMMARY ({args.target}) ================")
    print(f"{'variant':<15}{'mean':>8}{'std':>8}   folds")
    for n, r in results.items():
        print(f"{n:<15}{r['mean']:>8.4f}{r['std']:>8.4f}   {r['folds']}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
