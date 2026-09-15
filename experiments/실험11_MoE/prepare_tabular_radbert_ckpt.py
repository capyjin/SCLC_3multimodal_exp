# -*- coding: utf-8 -*-
"""MoE 게이트 실험을 RadBERT 판독지 인코더 기준으로 재현하기 위한 준비 단계.

`src/sclc/expert_embeddings.py::load_tabular_oof`가 읽는 tabular 체크포인트
(`outputs/late_fusion_B/tabular_{target}/`)는 TF-IDF로 학습된 것뿐이다(§실험1
의 기본 `text_encoder_fn=None`). RadBERT 버전 MoE 게이트를 실험하려면 같은
레시피(bs32/ep60/seed42, `실험6/exp_encoder_trimodal.py`와 동일한 RadBERT
전처리)로 tabular 축만 다시 학습해 **별도 경로**에 저장해야 한다 -- 기존
published 체크포인트(outputs/late_fusion_B/tabular_*)는 절대 안 건드린다.

image 축은 판독지 인코더와 무관하므로 재학습 불필요 -- 기존
`outputs/image_cph/emb_*.npz` 캐시를 그대로 재사용한다(src/sclc/expert_embeddings.py
::load_image_oof).

Run: python 실험11_MoE/prepare_tabular_radbert_ckpt.py --targets os,pfs
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import argparse
import json

from sclc import bert_features, cohort, features, ko2en
from sclc.late_fusion import get_tabular_oof

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "late_fusion_B_radbert")
PROTECTED_DIR = os.path.join(PROJECT_ROOT, "outputs", "late_fusion_B")

# 참고용 -- MODEL_SUMMARY.md §2-1 RadBERT 열, fix_brain_meta=True, seed=42
ANCHOR = {"os": 0.7153, "pfs": 0.6456}
ANCHOR_TOL = 1e-3


def assert_safe(out_dir: str) -> None:
    real, protected = os.path.realpath(out_dir), os.path.realpath(PROTECTED_DIR)
    if real == protected or real.startswith(protected + os.sep):
        raise RuntimeError(f"[prepare_radbert] 보호된 경로에 쓰려 한다: {real}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="os,pfs")
    ap.add_argument("--out_dir", default=OUT_DIR)
    args = ap.parse_args()
    assert_safe(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    print("[prepare_radbert] RadBERT 임베딩 계산 중 (1회, 모든 fold 공용)...")
    corpus, _ = features.load_text_corpus(cohort.DEFAULT_MERGED_CSV)
    emb = bert_features.embed_corpus(
        bert_features.DEFAULT_MODEL, {r: ko2en.translate_korean(t) for r, t in corpus.items()})
    rad_fn = bert_features.make_text_encoder_fn(emb, out_dim=400, audit=[], do_svd=False, do_scale=False)

    results = {}
    for target in [t.strip() for t in args.targets.split(",") if t.strip()]:
        print(f"\n[prepare_radbert/{target}] tabular(clinical+RadBERT) 학습 중...")
        ev = get_tabular_oof(target, epochs=60, batch_size=32, seed=42,
                             fix_brain_meta=True, out_dir=args.out_dir, text_encoder_fn=rad_fn)
        mean = float(sum(ev.c_indices) / len(ev.c_indices))
        results[target] = {"mean": mean, "folds": [round(float(c), 6) for c in ev.c_indices]}
        expected = ANCHOR.get(target)
        dev = abs(mean - expected) if expected else None
        status = "OK" if (dev is not None and dev <= ANCHOR_TOL) else ("MISMATCH" if dev is not None else "-")
        print(f"[prepare_radbert/{target}] tabular_radbert={mean:.4f}  "
              f"기준={expected}  [{status}]")

    with open(os.path.join(args.out_dir, "prepare_results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(f"\n-> {args.out_dir}")


if __name__ == "__main__":
    main()
