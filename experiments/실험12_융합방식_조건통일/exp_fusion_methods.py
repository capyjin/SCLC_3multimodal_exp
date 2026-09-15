# -*- coding: utf-8 -*-
"""논문 Figure 1의 세 융합 구조를 **같은 조건**(brain_meta 수정 후 + RadBERT)에서 잰다.

기존 (a)(b) 수치는 TF-IDF + brain_meta 수정 전이라 (c)와 직접 비교가 안 됐다.

  (a) early_concat   영상+임상+판독지를 concat 해 Cox head 하나로 end-to-end 학습
  (b) late_3way      세 모달리티를 각각 독립 학습 -> 위험점수 3개를 CoxPH 로 가중합
  (c) late_2way      이미 같은 조건으로 측정됨 (OS 0.7224 / PFS 0.6470) -> 재실행 안 함

영상 축은 재학습하지 않고 outputs/late_fusion_B/oof_<target>.json 을 재사용한다
(영상 모델은 임상변수를 안 보므로 brain_meta 수정과 무관 — 실측으로 확인됨).

Run:  python experiments/실험12_융합방식_조건통일/exp_fusion_methods.py --targets os,pfs
      python experiments/실험12_융합방식_조건통일/exp_fusion_methods.py --smoke   # 2 epoch 점검
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import argparse
import json

from sclc import cohort, features, paths
from sclc.encoders import build_encoder
from sclc.late_fusion import (cindex_stats, combine_weighted_sum, load_oof_cache,
                              oof_dict)
from sclc.model import MODALITY_CONFIGS, make_model_factory
from sclc.train import TrimodalEvaluator
from sclc.utils.cli import comma_list

OUT_DIR = paths.outputs("fusion_methods")

# (c) 2-way — 같은 조건(brain_meta 수정 후 + RadBERT, seed 42)으로 이미 측정된 값
KNOWN_2WAY = {"os": (0.7224, 0.0332), "pfs": (0.6470, 0.0286)}


def train_axis(name, target, model_config, text_encoder_fn, args):
    """축 하나를 5-fold 학습하고 (mean, std, folds, OOF위험점수) 를 돌려준다."""
    print(f"\n{'#' * 12} [{target}] {name} ({model_config}) {'#' * 12}")
    ev = TrimodalEvaluator(
        target=target, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
        save_dir=os.path.join(OUT_DIR, f"{name}_{target}"),
        model_factory=make_model_factory(MODALITY_CONFIGS[model_config]),
        text_encoder_fn=text_encoder_fn, max_folds=args.max_folds,
    ).run()
    mean, std = cindex_stats(ev.c_indices)
    folds = [round(float(c), 4) for c in ev.c_indices]
    print(f"[{name}/{target}] {mean:.4f} ± {std:.4f}  folds={folds}")
    return {"mean": mean, "std": std, "folds": folds}, oof_dict(ev.oof_predictions)


def run_target(target, text_encoder_fn, args) -> dict:
    cohort_df = cohort.load_trimodal_cohort()      # fix_brain_meta=True 가 기본
    out = {"target": target}

    # (a) early fusion — 세 모달리티를 하나의 모델로 concat 학습
    out["early_concat"], _ = train_axis("early_concat", target, "all", text_encoder_fn, args)

    # (b) late fusion 3-way — 축 3개를 따로 학습한 뒤 CoxPH 가중합
    clin, clin_risk = train_axis("clin_only", target, "clin_only", None, args)
    rep, rep_risk = train_axis("report_radbert", target, "report_only", text_encoder_fn, args)
    img_risk = load_oof_cache(target, keys=("image",))["image"]   # 재학습 없음
    print(f"[image/{target}] 캐시 재사용 (n={len(img_risk)})")

    combo = combine_weighted_sum(cohort_df, target, img_risk, clin_risk, rep_risk,
                                 max_folds=args.max_folds)
    out["late_3way"] = {"mean": combo["mean"], "std": combo["std"],
                        "folds": [round(float(c), 4) for c in combo["fold_cindex"]],
                        "mean_coef": combo["mean_coef"]}
    out["axes"] = {"clin_only": clin, "report_radbert": rep}
    print(f"[late_3way/{target}] {combo['mean']:.4f} ± {combo['std']:.4f}  "
          f"coef={combo['mean_coef']}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", type=comma_list, default=["os", "pfs"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_folds", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="2 epoch 플러밍 점검 (fold 는 5개 그대로)")
    args = ap.parse_args()
    if args.smoke:
        # fold 수는 줄이지 않는다: 결합 단계는 "모든 환자가 어느 fold 에선가 test"
        # 였어야 하는데(OOF 는 자기 test fold 에서만 생김), fold 를 하나만 돌리면
        # 나머지 환자의 위험점수가 없어 KeyError 가 난다. epoch 만 줄인다.
        args.epochs = 2

    os.makedirs(OUT_DIR, exist_ok=True)

    # RadBERT 임베딩은 frozen 이라 fold/seed 와 무관 — 한 번만 계산해 모든 축이 공유
    print("[fusion_methods] RadBERT 임베딩 계산 중 (1회)...")
    corpus, _ = features.load_text_corpus(cohort.DEFAULT_MERGED_CSV)
    text_encoder_fn = build_encoder("radbert").build_encoder_fn(corpus)

    results = {t: run_target(t, text_encoder_fn, args) for t in args.targets}

    path = os.path.join(OUT_DIR, "results_smoke.json" if args.smoke else "results.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 74}\n조건 통일 결과 (brain_meta 수정 후 · RadBERT · seed {args.seed})\n{'=' * 74}")
    print(f"{'구조':<28}{'OS':>18}{'PFS':>18}")
    for key, label in (("early_concat", "(a) early fusion"),
                       ("late_3way", "(b) late fusion 3-way")):
        cells = []
        for t in ("os", "pfs"):
            r = results.get(t, {}).get(key)
            cells.append(f"{r['mean']:.4f} ± {r['std']:.4f}" if r else "-")
        print(f"{label:<28}{cells[0]:>18}{cells[1]:>18}")
    c = [f"{KNOWN_2WAY[t][0]:.4f} ± {KNOWN_2WAY[t][1]:.4f}" for t in ("os", "pfs")]
    print(f"{'(c) late fusion 2-way':<28}{c[0]:>18}{c[1]:>18}   <- 기존 측정값")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
