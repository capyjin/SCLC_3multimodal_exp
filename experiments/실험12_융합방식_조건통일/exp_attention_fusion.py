# -*- coding: utf-8 -*-
"""Attention 융합(모달리티 토큰 self-attention)을 채택 모델과 **같은 조건**(brain_meta 수정 후 + RadBERT)에서 잰다.

논문 융합 비교표(Results 3.4) C 패널의 "attention fusion" 행이다(2026-10-12 사용자 결정: C = 게이트 1 + attention 1).
early fusion(실험12 exp_fusion_methods.py 의 early_concat, OS 0.6879 / PFS 0.6555)과 브랜치·학습 설정이 같고,
concat 대신 attention 으로 섞는 것만 다르다:
  영상·임상·판독지 브랜치 출력(L2 정규화) → 모달리티별 Linear 로 d=32 토큰 3개 + 학습 가능한 모달리티 임베딩
  → 1-head self-attention(잔차 + LayerNorm) → 토큰 평균 → Dropout(0.3) → Linear(32→1) Cox head.
영상의학 생존예측의 경량 토큰 attention(TMSS, MICCAI 2022 류)을 238명 규모에 맞게 줄인 형태다
(추가 파라미터 약 1.3×10⁴: 토큰 사상 8,800 + attention 4,224 + 기타). 브랜치 속성명은 ConcatDeepSurv 를 상속해 그대로 쓴다.

Run:  python experiments/실험12_융합방식_조건통일/exp_attention_fusion.py --targets os,pfs
      python experiments/실험12_융합방식_조건통일/exp_attention_fusion.py --smoke   # 1 fold × 2 epoch
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import argparse
import json

import torch
import torch.nn as nn
from sclc import cohort, features, paths
from sclc.encoders import build_encoder
from sclc.late_fusion import cindex_stats
from sclc.model import ConcatDeepSurv
from sclc.train import TrimodalEvaluator
from sclc.utils.cli import comma_list

OUT_DIR = paths.outputs("attention_fusion")


class AttentionFusionDeepSurv(ConcatDeepSurv):
    """ConcatDeepSurv 의 세 브랜치를 그대로 쓰고, concat 대신 모달리티 토큰 self-attention 으로 섞는다."""

    def __init__(self, clinical_dim: int, report_dim: int, token_dim: int = 32, **kwargs):
        super().__init__(clinical_dim, report_dim, **kwargs)
        dims = {"image": self.img_proj[0].out_features, "clinical": self.clinical_branch.out_dim,
                "report": self.report_branch.out_dim}
        self.token_proj = nn.ModuleDict({m: nn.Linear(d, token_dim) for m, d in dims.items()})
        self.modality_embed = nn.Parameter(torch.zeros(len(dims), token_dim))
        self.attn = nn.MultiheadAttention(token_dim, num_heads=1, batch_first=True)
        self.attn_norm = nn.LayerNorm(token_dim)
        self.head = nn.Linear(token_dim, 1, bias=False)        # concat 폭(272) 대신 토큰 폭

    def forward(self, image: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        feats = self.branch_features(image, tabular)
        tokens = torch.stack([self.token_proj[m](f) for m, f in feats.items()], dim=1) + self.modality_embed
        mixed, _ = self.attn(tokens, tokens, tokens)
        fused = self.attn_norm(tokens + mixed).mean(dim=1)
        return self.head(self.fusion_dropout(fused))


def run_target(target, text_encoder_fn, args) -> dict:
    print(f"\n{'#' * 12} [{target}] attention fusion (token_dim={args.token_dim}) {'#' * 12}")
    ev = TrimodalEvaluator(
        target=target, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
        save_dir=os.path.join(OUT_DIR, f"attn_{target}"),
        model_factory=lambda c, r: AttentionFusionDeepSurv(c, r, token_dim=args.token_dim),
        text_encoder_fn=text_encoder_fn, max_folds=args.max_folds,
    ).run()
    mean, std = cindex_stats(ev.c_indices)
    folds = [round(float(c), 4) for c in ev.c_indices]
    print(f"[attention/{target}] {mean:.4f} ± {std:.4f}  folds={folds}")
    return {"target": target, "token_dim": args.token_dim, "encoder": "radbert", "fix_brain_meta": True,
            "epochs": args.epochs, "batch_size": args.batch_size, "seed": args.seed,
            "mean": mean, "std": std, "folds": folds}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", type=comma_list, default=["os", "pfs"])
    ap.add_argument("--token_dim", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_folds", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="1 fold × 2 epoch 배관 점검")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.max_folds = 2, 1
    os.makedirs(OUT_DIR, exist_ok=True)

    # RadBERT 임베딩은 frozen 이라 fold/seed 와 무관 — 한 번만 계산 (캐시 재사용)
    corpus, _ = features.load_text_corpus(cohort.DEFAULT_MERGED_CSV)
    text_encoder_fn = build_encoder("radbert").build_encoder_fn(corpus)

    for t in args.targets:
        res = run_target(t, text_encoder_fn, args)
        path = os.path.join(OUT_DIR, f"results_{t}{'_smoke' if args.smoke else ''}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2, ensure_ascii=False)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
