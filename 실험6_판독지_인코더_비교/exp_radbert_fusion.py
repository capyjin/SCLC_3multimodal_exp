# -*- coding: utf-8 -*-
"""[실험] RadBERT 에 맞는 융합 방식 찾기.

동기 — 관측된 모순:
  판독지 **단독**으로는 RadBERT 가 TF-IDF 를 확실히 이긴다
      OS 0.6685 vs 0.6268, PFS 0.6354 vs 0.6094 (10 fold Δ=+0.034, p=0.027)
  그런데 임상변수와 **concat(조기융합)** 하면 우위가 사라진다
      Δ = -0.0006, p = 0.951
  지금까지 RadBERT 실험은 **전부 concat 이었다.** 그런데 이 프로젝트의 OS 최고 모델은
  late fusion 이다. RadBERT 는 "단독 모델로서" 더 좋으므로, 단독 성능을 그대로 쓰는
  late fusion 과 궁합이 맞을 가능성이 있는데 **한 번도 시험된 적이 없다.**

이 스크립트가 재는 것:
  (A) late fusion  : [임상 단독 모델] + [판독지 단독 모델] 의 OOF 위험점수를 CoxPH 로 결합
                     판독지 인코더를 TF-IDF / RadBERT 로 갈아 끼워 비교
  (B) concat 대조군 : 기존 clin_report 조기융합 (알려진 값과 대조)

  핵심 비교는 "같은 late fusion 틀에서 인코더만 바꿨을 때 RadBERT 가 이기는가" 이다.
  단독에서의 +0.034 가 late fusion 에서 살아남는다면, concat 이 RadBERT 를
  못 살렸다는 뜻이 된다.

누수 방지:
  - 각 단독 모델의 OOF 위험점수는 그 환자가 test 였던 fold 의 모델이 낸 값이다
    (TrimodalEvaluator 가 fold 별로 test 예측만 모아 준다).
  - CoxPH 결합은 fold 마다 train 환자의 OOF 점수로만 fit 하고 test 에 적용한다
    (late_fusion_tab_image.combine_two 와 동일한 방식 — 검증된 코드를 그대로 재사용).
  - 텍스트 인코더의 fold 별 통계(TF-IDF vocabulary 등)는 기존 파이프라인이 처리한다.
  ⚠️ 알려진 한계: 이 방식은 nested CV 가 아니라, 메타학습기의 입력이 test fold 를 본
    모델에서 나온다. 계수가 2개뿐이라 편향은 작지만 0은 아니다. 기존 late fusion
    (OS 0.7221)과 **같은 조건**이라 비교 가능성을 위해 동일하게 유지한다.

Run:  python exp_radbert_fusion.py --target os
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import json

import numpy as np

import bert_features
from core import cohort, features
from core.fusion_stack import combine_two, oof_dict
from core.model import MODALITY_CONFIGS, make_model_factory
from core.train import TrimodalEvaluator

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "radbert_fusion")


def unimodal_oof(target, config, encoder_fn, tag, epochs, batch_size):
    """단독 모달 모델을 5-fold 로 학습하고 (OOF 위험점수, fold별 C-index) 를 돌려준다."""
    ev = TrimodalEvaluator(
        target=target, epochs=epochs, batch_size=batch_size,
        save_dir=os.path.join(OUT_DIR, f"{tag}_{target}"),
        model_factory=make_model_factory(MODALITY_CONFIGS[config]),
        text_encoder_fn=encoder_fn,
    ).run()
    return oof_dict(ev.oof_predictions), ev.c_indices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="os", choices=("os", "pfs"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--model_name", default=bert_features.DEFAULT_MODEL)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    cohort_df = cohort.load_trimodal_cohort()
    corpus, _ = features.load_text_corpus(cohort.DEFAULT_MERGED_CSV)

    # ── RadBERT 임베딩 (ko2en 치환본. 영어 전용 tokenizer 라 한글은 [UNK] 가 되므로) ──
    import ko2en
    ko_corpus = {rid: ko2en.translate_korean(t) for rid, t in corpus.items()}
    emb = bert_features.embed_corpus(args.model_name, ko_corpus)
    # 단독에서 최선이었던 설정 그대로: 축소 없음 / 표준화 없음 (raw 768)
    radbert_fn = bert_features.make_text_encoder_fn(emb, out_dim=400, audit=[],
                                                    do_svd=False, do_scale=False)

    results = {}

    # ── 1) 단독 모델 3개의 OOF ──
    print(f"\n########## [1/3] 임상 단독  target={args.target} ##########")
    clin_oof, clin_ci = unimodal_oof(args.target, "clin_only", None, "clin_only",
                                     args.epochs, args.batch_size)
    print(f"\n########## [2/3] 판독지 단독 (TF-IDF)  target={args.target} ##########")
    tfidf_oof, tfidf_ci = unimodal_oof(args.target, "report_only", None, "report_tfidf",
                                       args.epochs, args.batch_size)
    print(f"\n########## [3/3] 판독지 단독 (RadBERT)  target={args.target} ##########")
    rad_oof, rad_ci = unimodal_oof(args.target, "report_only", radbert_fn, "report_radbert",
                                   args.epochs, args.batch_size)

    for name, ci in (("clin_only", clin_ci), ("report_tfidf", tfidf_ci), ("report_radbert", rad_ci)):
        results[name] = {"mean": float(np.mean(ci)), "std": float(np.std(ci)),
                         "folds": [round(float(c), 4) for c in ci]}
        print(f"[UNI] {name} {args.target}: {np.mean(ci):.4f} +/- {np.std(ci):.4f}")

    # ── 2) late fusion: 임상 + 판독지 (인코더만 갈아 끼움) ──
    for name, rep_oof in (("late_clin+tfidf", tfidf_oof), ("late_clin+radbert", rad_oof)):
        print(f"\n########## COMBINE {name}  target={args.target} ##########")
        # combine_two 는 (tabular_risk, image_risk) 두 축을 CoxPH 로 묶는 함수다.
        # 여기서는 축 이름만 빌려 쓴다: 축1=임상 단독, 축2=판독지 단독.
        out = combine_two(cohort_df, args.target, clin_oof, rep_oof)
        cis = out["fold_cindex"]
        results[name] = {"mean": out["mean"], "std": out["std"],
                         "folds": [round(float(c), 4) for c in cis],
                         "coefs_per_fold": out["coefs_per_fold"],
                         "mean_coef": out["mean_coef"]}
        print(f"[LATE] {name} {args.target}: {out['mean']:.4f} +/- {out['std']:.4f}  "
              f"coef(임상,판독지)={out['mean_coef']}")

    path = os.path.join(OUT_DIR, f"results_{args.target}.json")
    with open(path, "w") as f:
        json.dump({"target": args.target, "epochs": args.epochs,
                   "batch_size": args.batch_size, "model_name": args.model_name,
                   "results": results}, f, indent=2)

    # ── 요약 ──
    known = {"os": 0.7076, "pfs": 0.6678}[args.target]
    print(f"\n================ SUMMARY ({args.target}) ================")
    print(f"{'구성':<24}{'C-index':>10}{'concat 기준선 대비':>20}")
    for k, v in results.items():
        print(f"{k:<24}{v['mean']:>10.4f}{v['mean'] - known:>+20.4f}")
    print(f"\n기준선(concat clin_report, TF-IDF) = {known:.4f}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
