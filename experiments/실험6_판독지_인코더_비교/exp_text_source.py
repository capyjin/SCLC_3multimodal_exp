# -*- coding: utf-8 -*-
"""[실험] "판독지의 **어느 부분**이 신호를 갖고 있나?" 를 검증한다.

동기:
  clinical+report(=clin_report)가 현재 최고 성능이다(OS 0.7076 / PFS 0.6678).
  그런데 판독지 텍스트는 두 부분으로 되어 있다.
    - conclusion(결론) : 판독의가 내린 요약·판단  (중앙값 288자 / 코호트238 기준 288.5자)
    - finding(소견)    : 영상에서 관찰한 것들의 나열 (중앙값 571자 / 코호트238 기준 556자)
  지금까지는 이 둘을 합쳐서(conclusion + "\n" + finding) 한 덩어리로 썼다.
  → 성능이 '판단'에서 오는 건지 '관찰'에서 오는 건지 구분이 안 된다.

방법:
  **텍스트 입력만** 바꾸고 나머지는 전부 고정한다.
    concl_find (기존) / concl_only / find_only  × {os, pfs}
  모델은 clin_report(임상+판독지, 영상 브랜치 없음)로 고정.

지켜지는 불변식(invariant) — 이전에 report 브랜치 차원 때문에 한 번 데였으므로 명시한다:
  - report_dim 은 항상 tfidf_max_features(=400)로 **고정**이다.
    TfidfEncoder.transform() 이 vocabulary가 400보다 적게 나오면 0으로 패딩하기 때문에,
    텍스트가 짧아져도(conclusion만 쓰면 절반 이하) 텐서 폭·브랜치 크기는 변하지 않는다.
    → 세 variant의 모델 구조는 완전히 동일하고, 오직 TF-IDF 값(내용)만 달라진다.
  - clinical 브랜치·fold split·seed·에폭 수 전부 동일.
  - 마스킹(날짜/병원명 삭제)은 세 variant 모두에 그대로 적용된다.

Run:  python exp_text_source.py --target os
      python exp_text_source.py --target pfs --variants concl_find
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import json
import statistics

import numpy as np

from core import cohort, features
from core.metrics import train_val_gap
from core.model import MODALITY_CONFIGS, make_model_factory

# 이 실험에서 고정하는 모델 조합. clin_report = 임상+판독지 (영상 브랜치 없음).
MODEL_CONFIG = "clin_report"

# variant 이름 -> features.load_text_corpus(source=...) 에 넘길 값 + 설명
VARIANTS = {
    "concl_find": "conclusion + finding (기존 동작 / 재현 기준선)",
    "concl_only": "conclusion 만 (판독의의 '판단'만)",
    "find_only":  "finding 만 (영상 '관찰'만)",
}


def text_length_stats(source: str, merged_csv: str = cohort.DEFAULT_MERGED_CSV,
                      split_csv: str = cohort.DEFAULT_SPLIT_CSV) -> dict:
    """해당 source로 실제 학습에 들어가는 텍스트의 길이 통계.

    두 가지 모집단으로 각각 잰다 — 헷갈리기 쉬워서 일부러 둘 다 기록한다.
      - median_chars      : 238명 tri-modal 공통 코호트 기준.
                            **실제로 모델에 들어가는** 텍스트라서 이게 주 지표다.
                            (concl_find 826 / concl_only 288.5 / find_only 556)
      - median_chars_all  : has_report=1 전체 248명(corpus 원본) 기준.
                            문서·이전 기록에 적힌 수치(846 / 288 / 571)가 이쪽이므로,
                            그 값과 대조하려면 이 필드를 봐야 한다.
    두 수치가 다른 이유는 코호트 238명이 248명의 부분집합이기 때문이지
    source 스위치와는 무관하다.
    """
    corpus, _ = features.load_text_corpus(merged_csv, source=source)
    all_lengths = [len(t) for t in corpus.values()]
    cohort_df = cohort.load_trimodal_cohort(merged_csv, split_csv)
    rids = sorted({int(r) for r in cohort_df["research_id"]})
    lengths = [len(corpus.get(rid, "")) for rid in rids]
    return {
        "n_patients": len(lengths),
        "median_chars": float(statistics.median(lengths)),
        "mean_chars": float(np.mean(lengths)),
        "min_chars": int(min(lengths)),
        "max_chars": int(max(lengths)),
        "n_empty": int(sum(1 for x in lengths if x == 0)),
        "n_all_reports": len(all_lengths),
        "median_chars_all": float(statistics.median(all_lengths)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="os", choices=("os", "pfs"))
    ap.add_argument("--variants", default=",".join(VARIANTS))
    # bs32/ep60 은 이 저장소에서 확립된 설정 (RESULTS.md 9장, ablation_improved).
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    names = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variant(s) {unknown}; expected from {list(VARIANTS)}")

    # ── 학습 전에 먼저 텍스트 길이를 찍어 둔다 ──
    # source 스위치가 정말 먹었는지를 몇 초 만에 확인할 수 있게 (smoke test용).
    print(f"\n=== text length sanity check ===")
    print(f"{'variant':<12}{'median':>9}{'mean':>9}{'min':>7}{'max':>8}{'med(all248)':>13}  desc")
    length_stats = {}
    for name in names:
        st = text_length_stats(name)
        length_stats[name] = st
        print(f"{name:<12}{st['median_chars']:>9.1f}{st['mean_chars']:>9.1f}"
              f"{st['min_chars']:>7}{st['max_chars']:>8}{st['median_chars_all']:>13.1f}  {VARIANTS[name]}")
    print("  (median/mean/min/max = 코호트 238명 = 실제 학습 입력, med(all248) = corpus 전체)")

    out_dir = "outputs/text_source"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"results_{args.target}.json")
    # 이미 돌린 variant가 있으면 이어서 채운다 (중간에 죽어도 앞선 결과를 잃지 않도록).
    # 단, 설정(epochs/batch_size)이 다른 결과와는 절대 섞지 않는다.
    # 예: smoke test(--epochs 2) 결과가 본 실험(60 epoch) 파일에 남아 있으면
    #     한 표 안에 서로 비교 불가능한 숫자가 섞여 잘못된 결론이 난다.
    prev = json.load(open(path)) if os.path.exists(path) else {}
    same_setting = (prev.get("epochs") == args.epochs and prev.get("batch_size") == args.batch_size
                    and prev.get("model_config") == MODEL_CONFIG)
    if prev and not same_setting:
        print(f"[warn] {path} 는 다른 설정(ep={prev.get('epochs')}, bs={prev.get('batch_size')}, "
              f"model={prev.get('model_config')})으로 만들어졌다 -> 이어붙이지 않고 새로 시작한다.")
    results = prev.get("variants", {}) if same_setting else {}

    # TrimodalEvaluator import는 여기서 (torch 로딩이 느려서 인자 검증 뒤에 하는 게 낫다).
    from core.train import TrimodalEvaluator

    for name in names:
        flags = MODALITY_CONFIGS[MODEL_CONFIG]
        save_dir = f"{out_dir}/{name}_{args.target}"
        print(f"\n########## TEXT SOURCE: {name}  ({VARIANTS[name]})  "
              f"model={MODEL_CONFIG}{flags}  target={args.target} "
              f"bs={args.batch_size} ep={args.epochs} ##########")

        ev = TrimodalEvaluator(
            target=args.target, epochs=args.epochs, batch_size=args.batch_size,
            save_dir=save_dir, model_factory=make_model_factory(flags),
            text_source=name,   # <- 이 실험에서 유일하게 바뀌는 것
        ).run()
        cis = ev.c_indices

        gaps = train_val_gap(ev.training_history)   # 과적합 정도 (정의는 core.metrics)

        results[name] = {
            "text_source": name,
            "desc": VARIANTS[name],
            "mean": float(np.mean(cis)),
            "std": float(np.std(cis)),
            "folds": [round(float(c), 4) for c in cis],   # float() 은 출력만 깔끔하게 (값 동일)
            "train_val_gap_mean": float(np.mean(gaps)) if gaps else None,
            "text_length": length_stats[name],
            "fold_records": ev.fold_records,
            "training_history": ev.training_history,
        }
        # variant 하나 끝날 때마다 저장 (crash-safe)
        with open(path, "w") as f:
            json.dump({
                "target": args.target, "batch_size": args.batch_size, "epochs": args.epochs,
                "model_config": MODEL_CONFIG, "flags": flags, "variants": results,
            }, f, indent=2)

        print(f"[TEXT_SRC] {name} {args.target}: {np.mean(cis):.4f} +/- {np.std(cis):.4f}  "
              f"folds={results[name]['folds']}  median_chars={length_stats[name]['median_chars']:.0f}")

    # ── 요약표 ──
    print(f"\n================ TEXT SOURCE SUMMARY (target={args.target}, "
          f"model={MODEL_CONFIG}, bs{args.batch_size}/ep{args.epochs}) ================")
    print(f"{'variant':<12}{'mean':>8}{'std':>8}{'gap':>8}{'median_chars':>14}   folds")
    for name, r in results.items():
        gap = f"{r['train_val_gap_mean']:.4f}" if r.get("train_val_gap_mean") is not None else "-"
        med = r.get("text_length", {}).get("median_chars", float("nan"))
        print(f"{name:<12}{r['mean']:>8.4f}{r['std']:>8.4f}{gap:>8}{med:>14.0f}   {r['folds']}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
