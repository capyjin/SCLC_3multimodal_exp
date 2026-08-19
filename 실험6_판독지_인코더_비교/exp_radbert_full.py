# -*- coding: utf-8 -*-
"""[실험6-e] 현재 OS 최고 모델의 tabular 축 인코더만 RadBERT 로 교체하면?

현재 최고: late fusion( concat[임상+판독지(TF-IDF)] , 영상 SimpleCNN ) = OS 0.7221.
concat 단계에서는 이미 RadBERT 가 앞선다(0.7153 vs 0.7076) — 그럼 영상까지 얹으면
최고 기록을 넘는가? 이 조합은 미시험이었다.

측정:
  tabular 축을 인코더 2종(TF-IDF / RadBERT)으로 각각 학습 -> 영상 축과 결합.
  두 결합 결과의 차이가 곧 "인코더 교체의 순효과"다.

누수 방지: 검증된 ``core.fusion_stack.combine_two`` 를 그대로 쓴다
(fold 마다 train 환자의 OOF 점수로만 CoxPH 적합).

Run:  python 실험6_판독지_인코더_비교/exp_radbert_full.py --target os
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import json

import numpy as np

import bert_features
import ko2en
from core import cohort, features
from core.fusion_stack import combine_two, get_image_oof_simplecnn, oof_dict
from core.model import MODALITY_CONFIGS, make_model_factory
from core.train import TrimodalEvaluator

OUT = os.path.join(PROJECT_ROOT, "outputs", "radbert_full")
KNOWN_BEST = {"os": 0.7221, "pfs": 0.6678}   # 현재 최고(late fusion + SimpleCNN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="os", choices=("os", "pfs"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    cohort_df = cohort.load_trimodal_cohort()
    corpus, _ = features.load_text_corpus(cohort.DEFAULT_MERGED_CSV)
    # 단독에서 최선이었던 설정 그대로: 한글->영어 치환 + 축소/표준화 없음(raw 768)
    emb = bert_features.embed_corpus(bert_features.DEFAULT_MODEL,
                                     {r: ko2en.translate_korean(t) for r, t in corpus.items()})
    rad_fn = bert_features.make_text_encoder_fn(emb, out_dim=400, audit=[],
                                                do_svd=False, do_scale=False)

    res, oof_by_tag = {}, {}

    # ── tabular 축: concat[임상+판독지] 를 인코더 2종으로 ──
    for tag, fn in (("tab_tfidf", None), ("tab_radbert", rad_fn)):
        print(f"\n########## TABULAR {tag}  target={args.target} ##########")
        ev = TrimodalEvaluator(
            target=args.target, epochs=args.epochs, batch_size=args.batch_size,
            save_dir=os.path.join(OUT, f"{tag}_{args.target}"),
            model_factory=make_model_factory(MODALITY_CONFIGS["clin_report"]),
            text_encoder_fn=fn,
        ).run()
        res[tag] = {"mean": float(np.mean(ev.c_indices)), "std": float(np.std(ev.c_indices)),
                    "folds": [round(float(c), 4) for c in ev.c_indices]}
        oof_by_tag[tag] = oof_dict(ev.oof_predictions)
        print(f"[TAB] {tag} {args.target}: {res[tag]['mean']:.4f}")

    # ── 영상 축 ──
    print(f"\n########## IMAGE SimpleCNN  target={args.target} ##########")
    img_ev = get_image_oof_simplecnn(args.target, epochs=30)
    img_oof = oof_dict(img_ev.oof_predictions)
    res["image_simplecnn"] = {"mean": float(np.mean(img_ev.c_indices)),
                              "std": float(np.std(img_ev.c_indices)),
                              "folds": [round(float(c), 4) for c in img_ev.c_indices]}
    print(f"[IMG] simplecnn {args.target}: {res['image_simplecnn']['mean']:.4f}")

    # ── 결합 ──
    for tag in ("tab_tfidf", "tab_radbert"):
        print(f"\n########## COMBINE {tag} + image  target={args.target} ##########")
        out = combine_two(cohort_df, args.target, oof_by_tag[tag], img_oof)
        res[f"late_{tag}+img"] = {"mean": out["mean"], "std": out["std"],
                                  "folds": [round(float(c), 4) for c in out["fold_cindex"]],
                                  "mean_coef": out["mean_coef"]}
        print(f"[LATE] {tag}+image {args.target}: {out['mean']:.4f} +/- {out['std']:.4f}")

    path = os.path.join(OUT, f"results_{args.target}.json")
    with open(path, "w") as f:
        json.dump({"target": args.target, "epochs": args.epochs,
                   "batch_size": args.batch_size, "results": res}, f, indent=2)

    known = KNOWN_BEST[args.target]
    print(f"\n================ SUMMARY ({args.target}) ================")
    for k, v in res.items():
        print(f"{k:<26}{v['mean']:>10.4f}{v['mean'] - known:>+12.4f}")
    print(f"\n현재 최고(OS late fusion+SimpleCNN) = {known:.4f}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
