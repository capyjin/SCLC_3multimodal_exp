# -*- coding: utf-8 -*-
"""[실험] 판독지 인코더를 char n-gram TF-IDF -> frozen 영어 biomedical BERT(RadBERT)로
바꾸면 더 좋아지는가? 그리고 한국어 [UNK] 문제가 범인인가?

동기:
  clin_report(임상+판독지, 영상 브랜치 없음)가 현재 최고다 (OS 0.7076 / PFS 0.6678).
  판독지는 지금 char n-gram TF-IDF 로만 쓰인다 — 글자 조각 빈도라서 의미를 모르고,
  특히 부정문("no evidence of metastasis")과 긍정문을 거의 구분하지 못한다.
  의학 코퍼스로 사전학습된 BERT 는 그 차이를 안다. 그래서 **텍스트 인코더만**
  바꿔서 이득이 있는지 본다.

  걸림돌: 판독지는 영어 의학용어(~64%)와 한국어(~9%, 조사 + 정형 서술어)가 섞여
  있는데 RadBERT tokenizer 는 영어 전용이라 한국어를 전부 [UNK] 로 만든다
  (이 코퍼스 실측 평균 16.3% 토큰). BERT 가 져도 "BERT 가 나쁜 것"인지
  "[UNK] 때문"인지 구분이 안 된다. 그래서 한국어 처리를 3갈래로 갈라 원인을 분리한다.

arm:
  tfidf            : 기존 TF-IDF 400 (재현 기준선 — 리팩터가 동작을 안 바꿨음을 증명)
  bert_raw         : RadBERT, 원문 그대로 (한국어 -> [UNK])
  bert_nokr        : RadBERT, 한국어 글자 삭제 (ko2en.strip_korean)
  bert_ko2en       : RadBERT, 한국어 -> 영어 구 치환 (ko2en.translate_korean)
  tfidf_plus_bert  : TF-IDF 400 + BERT 200 이어붙이기 (대체가 아니라 '보완'인가?)

고정되는 것(= 오직 텍스트 블록만 달라진다):
  모델 조합(--model_config, 기본 clin_report), fold split, seed, epochs, batch_size, 임상 블록,
  텍스트 source(concl_find), 그리고 **report 브랜치 폭 400**.
  폭을 고정하는 게 중요하다 — 폭이 달라지면 파라미터 수와 추정 부담이 같이
  달라져서 "인코더가 좋아서"인지 "폭이 달라서"인지 구분이 안 된다
  (이 프로젝트는 폭을 늘리면 오히려 나빠지는 걸 반복 확인했다).
  BERT 임베딩 768 -> 400 축소 시 SVD 성분은 최대 n_train(171)개라 나머지는 0 패딩된다
  (TfidfEncoder 가 vocabulary 부족분을 0 패딩하는 것과 동일한 관행). 자세한 건
  bert_features._reduce_block 참고.

누수 방지 (이 저장소의 기존 규율과 동일):
  BERT forward 는 문서 하나만 보는 frozen 연산이라 전역 1회 계산해도 무방하다.
  환자를 가로질러 계산되는 통계 — SVD 기저, StandardScaler, TF-IDF vocabulary —
  는 전부 fold 별로 **train fold 환자만** 써서 fit 한다
  (bert_features.make_text_encoder_fn). 실제 쓰인 설명분산·scaler 노름·표본수는
  로그와 결과 JSON(leakage_audit)에 남긴다.

환자 정보 보호: 판독지 원문은 stdout/결과파일 어디에도 찍지 않는다. 토큰 통계는
  집계값(평균 토큰 수, [UNK] 비율 등)만 낸다.

Run:  python exp_bert_text.py --target os
      python exp_bert_text.py --target os --arms tfidf,bert_raw
      python exp_bert_text.py --target os --arms bert_raw --epochs 2   # smoke test
      python exp_bert_text.py --target os --model_config report_only   # 판독지 단독
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import json

import numpy as np

from core.metrics import paired_pvalues, train_val_gap

import bert_features
from core import cohort, features
from core.model import MODALITY_CONFIGS, make_model_factory

MODEL_CONFIG = "clin_report"   # --model_config 의 기본값 (기존 동작)
ARM_ORDER = ["tfidf", "tfidf_svd", "bert_raw", "bert_nokr", "bert_ko2en", "tfidf_plus_bert"]
ARM_DESC = {
    "tfidf":           "char n-gram TF-IDF 400 (기존 기준선)",
    "tfidf_svd":       "TF-IDF를 BERT와 같은 축소(train-only SVD->실효171)에 태운 랭크 통제군",
    "bert_raw":        "RadBERT 원문 (한국어 -> [UNK])",
    "bert_nokr":       "RadBERT + 한국어 삭제 (strip_korean)",
    "bert_ko2en":      "RadBERT + 한국어->영어 치환 (translate_korean)",
    "tfidf_plus_bert": "TF-IDF 400 + BERT 200 이어붙임 (보완 효과)",
}
BERT_ARMS = ("bert_raw", "bert_nokr", "bert_ko2en", "tfidf_plus_bert")
# 기존 기준선 (RESULTS.md / outputs/text_source/concl_find, report_only 는 ablation).
# 재현 확인용. 모델 조합마다 기준선이 다르므로 config -> target 으로 건다.
# 여기 없는 config 는 기준선 없음으로 보고 재현 검사를 건너뛴다 (죽지 않는다).
KNOWN_BASELINE = {
    "clin_report": {"os": 0.7076, "pfs": 0.6678},
    "report_only": {"os": 0.6268, "pfs": 0.6094},
}
KNOWN_BASELINE_FOLDS = {
    "clin_report": {
        "os":  [0.726, 0.7894, 0.6765, 0.654, 0.692],
        "pfs": [0.6444, 0.732, 0.6501, 0.641, 0.6717],
    },
    "report_only": {
        "os":  [0.5993, 0.6846, 0.5662, 0.6806, 0.6032],
        "pfs": [0.5921, 0.6322, 0.5832, 0.6729, 0.5666],
    },
}


def arm_corpus(arm: str, corpus: dict[int, str], plus_bert_variant: str = "raw") -> dict[int, str]:
    """arm 별로 BERT 에 먹일 텍스트를 만든다.

    ko2en 모듈은 필요한 arm 에서만 import 한다 (그 파일이 아직 없어도 tfidf/
    bert_raw arm 은 돌아가야 하므로). 원문은 반환만 하고 절대 출력하지 않는다.
    """
    if arm in ("tfidf", "tfidf_svd", "bert_raw"):
        return corpus
    if arm == "tfidf_plus_bert":
        return arm_corpus(f"bert_{plus_bert_variant}" if plus_bert_variant != "raw" else "bert_raw",
                          corpus, plus_bert_variant)
    import ko2en
    if arm == "bert_nokr":
        return {rid: ko2en.strip_korean(t) for rid, t in corpus.items()}
    if arm == "bert_ko2en":
        return {rid: ko2en.translate_korean(t) for rid, t in corpus.items()}
    raise ValueError(f"unknown arm {arm!r}")


def make_encoder_fn(arm: str, corpus: dict[int, str], embeddings, out_dim: int,
                    bert_out_dim: int, audit: list, do_svd: bool = True, do_scale: bool = True):
    """arm -> features.build_fold_multimodal_tabular 에 넘길 text_encoder_fn.
    tfidf arm 은 None (= 기존 경로 그대로 = 재현 기준선)."""
    if arm == "tfidf":
        return None
    if arm == "tfidf_svd":
        return bert_features.make_tfidf_svd_encoder_fn(
            corpus, out_dim=out_dim, tfidf_max_features=out_dim, audit=audit,
            do_svd=do_svd, do_scale=do_scale)
    if arm == "tfidf_plus_bert":
        return bert_features.make_tfidf_plus_bert_encoder_fn(
            corpus, embeddings, tfidf_max_features=out_dim,
            bert_out_dim=bert_out_dim, audit=audit)
    return bert_features.make_text_encoder_fn(embeddings, out_dim=out_dim, audit=audit,
                                              do_svd=do_svd, do_scale=do_scale)


def preflight(arm: str, encoder_fn, expected_width: int):
    """학습 전에 fold별 feature 행렬을 실제로 만들어 보고 (1) report 블록 폭이
    기대대로인지(=TF-IDF와 같은 400인지), (2) 블록 값이 환자마다 실제로 변하는지
    (상수/전부 0이 아닌지)를 눈으로 확인한다. 학습 없이 수 초면 끝난다."""
    cohort_df = cohort.load_trimodal_cohort()
    clinical_frame = cohort_df.drop_duplicates("research_id").set_index("research_id")
    std_cols, cat_cols = features.resolve_clinical_columns(clinical_frame)
    corpus, _ = features.load_text_corpus(cohort.DEFAULT_MERGED_CSV)

    dims, checks = None, []
    for fold in sorted(int(f) for f in cohort_df["fold"].unique()):
        fdf = cohort_df[cohort_df["fold"] == fold]
        ids = {s: fdf.loc[fdf["split"] == s, "research_id"].astype(int).tolist()
               for s in ("train", "val", "test")}
        tab, clinical_dim, report_dim = features.build_fold_multimodal_tabular(
            clinical_frame.loc[ids["train"]], clinical_frame.loc[ids["val"]],
            clinical_frame.loc[ids["test"]], corpus, std_cols, cat_cols,
            text_encoder_fn=encoder_fn,
        )
        dims = (clinical_dim, report_dim, tab["train"].shape[1])
        blk = tab["train"][:, clinical_dim:]
        nonzero = np.abs(blk).sum(axis=0) > 0
        checks.append({"fold": fold, "n_nonzero_cols": int(nonzero.sum()),
                       "col_std_mean": round(float(blk[:, nonzero].std(axis=0).mean()), 4) if nonzero.any() else 0.0,
                       "test_std_mean": round(float(tab["test"][:, clinical_dim:][:, nonzero].std(axis=0).mean()), 4) if nonzero.any() else 0.0})
    clinical_dim, report_dim, width = dims
    print(f"[preflight/{arm}] clinical_dim={clinical_dim}  report(text)_dim={report_dim} "
          f"(expected {expected_width})  tabular width={width}")
    assert report_dim == expected_width, (
        f"report block width {report_dim} != expected {expected_width} — "
        "폭이 달라지면 TF-IDF 와의 비교가 오염된다")
    assert width == clinical_dim + report_dim, "feature width mismatch"
    for c in checks:
        print(f"[preflight/{arm}] fold {c['fold']}: nonzero_cols={c['n_nonzero_cols']}/{report_dim} "
              f"train_col_std={c['col_std_mean']} test_col_std={c['test_std_mean']} "
              f"(0.0 이면 블록이 상수 = 뭔가 잘못된 것)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="os", choices=("os", "pfs"))
    ap.add_argument("--arms", default=",".join(ARM_ORDER), help="쉼표 목록.")
    ap.add_argument("--model_name", default=bert_features.DEFAULT_MODEL)
    ap.add_argument("--model_config", default=MODEL_CONFIG, choices=sorted(MODALITY_CONFIGS),
                    help="모델 조합 (MODALITY_CONFIGS). 기본 clin_report(임상+판독지) = 기존 동작. "
                         "report_only 면 판독지 브랜치만 남아 임상·영상 없이 텍스트 인코더 "
                         "단독 성능을 잰다.")
    # bs32/ep60 은 이 저장소에서 확립된 설정 (RESULTS.md 9장).
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out_dim", type=int, default=400,
                    help="텍스트 블록 폭. 기본 400 = TF-IDF 와 동일 폭(비교 공정성).")
    ap.add_argument("--bert_out_dim", type=int, default=200,
                    help="tfidf_plus_bert arm 의 BERT 절반 폭 (400+200=600).")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--embed_batch_size", type=int, default=16)
    ap.add_argument("--plus_bert_variant", default="raw", choices=("raw", "nokr", "ko2en"),
                    help="tfidf_plus_bert arm 의 BERT 절반이 쓸 한국어 처리.")
    ap.add_argument("--preflight_only", action="store_true", help="학습 없이 feature 점검만")
    ap.add_argument("--no_svd", action="store_true",
                    help="SVD 축소를 건너뛴다 (블록 폭 = 원래 폭). 축소 파이프라인이 "
                         "성능을 깎는지 확인하는 스위치.")
    ap.add_argument("--no_scale", action="store_true",
                    help="StandardScaler 를 건너뛴다. SVD 의 분산 정렬을 표준화가 "
                         "지워버리는지 확인하는 스위치.")
    ap.add_argument("--out_dir", default="outputs/bert_text",
                    help="결과 저장 폴더. 모델을 바꿔가며 돌릴 때 모델마다 다른 폴더를 줘야 "
                         "results_{target}.json 이 서로 덮어써지지 않는다.")
    args = ap.parse_args()

    names = [s.strip() for s in args.arms.split(",") if s.strip()]
    unknown = [n for n in names if n not in ARM_DESC]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; expected from {ARM_ORDER}")

    model_config = args.model_config
    # 이 (config, target) 조합의 알려진 기준선. 없으면 None -> 재현 검사를 건너뛴다.
    known = KNOWN_BASELINE.get(model_config, {}).get(args.target)
    known_folds = KNOWN_BASELINE_FOLDS.get(model_config, {}).get(args.target)

    corpus, corpus_stats = features.load_text_corpus(cohort.DEFAULT_MERGED_CSV)
    print(f"\n=== corpus ===  n_reports={corpus_stats['n']}  "
          f"total_chars={corpus_stats['total_chars']}  (원문은 출력하지 않는다)")

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"results_{args.target}.json")
    # 설정(epochs/batch_size/model/폭)이 같을 때만 이어붙인다 — smoke test(ep2) 결과가
    # 본 실험 파일에 섞여 비교 불가능한 숫자가 한 표에 들어가는 사고를 막는다.
    prev = json.load(open(path)) if os.path.exists(path) else {}
    same = (prev.get("epochs") == args.epochs and prev.get("batch_size") == args.batch_size
            and prev.get("model_config") == model_config and prev.get("out_dim") == args.out_dim
            and prev.get("model_name") == args.model_name)
    if prev and not same:
        print(f"[warn] {path} 는 다른 설정(ep={prev.get('epochs')}, bs={prev.get('batch_size')}, "
              f"model_config={prev.get('model_config')}, model_name={prev.get('model_name')})"
              "으로 만들어졌다 -> 이어붙이지 않고 새로 시작한다.")
    results = prev.get("arms", {}) if same else {}

    from core.train import TrimodalEvaluator

    for name in names:
        print(f"\n########## ARM: {name}  ({ARM_DESC[name]})  model={model_config}  "
              f"target={args.target} bs={args.batch_size} ep={args.epochs} ##########")

        # ── 1) arm 별 텍스트 준비 + 토큰화 통계 (한국어 처리가 먹었는지 확인) ──
        tok_stats, embeddings, ko_cov = None, None, None
        if name in BERT_ARMS:
            texts = arm_corpus(name, corpus, args.plus_bert_variant)
            if name == "bert_ko2en":
                # 사전이 한글을 얼마나 덮었는지 = 이 arm 의 결과를 해석하는 전제.
                # 커버율이 낮으면 "번역이 도움 안 됨"이 아니라 "번역이 덜 됨"이다.
                import ko2en
                ko_cov = ko2en.coverage(list(corpus.values()))
                print(f"[ko2en] dict_entries={ko_cov['dict_entries']} "
                      f"char_coverage={ko_cov['char_coverage'] * 100:.1f}% "
                      f"unique_chunks={ko_cov['unique_chunks']}")
            tok_stats = bert_features.tokenization_stats(args.model_name, texts, args.max_length)
            print(f"[tokens/{name}] mean={tok_stats['mean_tokens']:.1f} median={tok_stats['median_tokens']:.0f} "
                  f"UNK={tok_stats['unk_frac_mean'] * 100:.2f}% docs_with_UNK={tok_stats['pct_docs_with_unk']:.1f}% "
                  f"truncated={tok_stats['pct_docs_truncated']:.1f}%")
            embeddings = bert_features.embed_corpus(
                args.model_name, texts, max_length=args.max_length,
                batch_size=args.embed_batch_size)

        # ── 2) fold-safe 인코더 + 사전 점검 ──
        audit = []
        do_svd, do_scale = (not args.no_svd), (not args.no_scale)
        fn = make_encoder_fn(name, corpus, embeddings, args.out_dim, args.bert_out_dim, audit,
                             do_svd=do_svd, do_scale=do_scale)
        expected = args.out_dim + (args.bert_out_dim if name == "tfidf_plus_bert" else 0)
        if not do_svd:
            # 축소를 끄면 블록 폭이 '원래 폭'이 된다: TF-IDF 는 max_features,
            # BERT 는 임베딩 차원(768 등). preflight 의 폭 검사가 이를 알아야 한다.
            expected = (args.out_dim if name in ("tfidf", "tfidf_svd")
                        else len(next(iter(embeddings.values()))))
        preflight(name, fn, expected)
        if args.preflight_only:
            continue
        audit.clear()   # preflight 에서 쌓인 기록은 버리고 본 실행 것만 남긴다

        # ── 3) 학습/평가 ──
        ev = TrimodalEvaluator(
            target=args.target, epochs=args.epochs, batch_size=args.batch_size,
            save_dir=f"{out_dir}/{name}_{args.target}",
            model_factory=make_model_factory(MODALITY_CONFIGS[model_config]),
            text_encoder_fn=fn,      # <- 이 실험에서 유일하게 바뀌는 것
        ).run()
        cis = ev.c_indices

        gaps = train_val_gap(ev.training_history)   # 과적합 정도 (정의는 core.metrics)

        results[name] = {
            "arm": name, "desc": ARM_DESC[name],
            "mean": float(np.mean(cis)), "std": float(np.std(cis)),
            "folds": [round(float(c), 4) for c in cis],
            "train_val_gap_mean": float(np.mean(gaps)) if gaps else None,
            "tokenization": tok_stats,       # 집계값만 (원문 없음)
            "ko2en_coverage": ko_cov,        # bert_ko2en arm 에서만 채워진다
            "leakage_audit": audit,          # fold별 train-only SVD/scaler 통계
            "fold_records": ev.fold_records,
            "training_history": ev.training_history,
        }
        with open(path, "w") as f:   # arm 하나 끝날 때마다 저장 (crash-safe)
            json.dump({
                "target": args.target, "batch_size": args.batch_size, "epochs": args.epochs,
                "model_config": model_config, "flags": MODALITY_CONFIGS[model_config],
                "model_name": args.model_name, "out_dim": args.out_dim,
                "bert_out_dim": args.bert_out_dim, "max_length": args.max_length,
                "plus_bert_variant": args.plus_bert_variant,
                "known_baseline": known,   # 알려진 기준선이 없는 config 면 null
                "arms": results,
            }, f, indent=2)

        print(f"[BERT] {name} {args.target}: {np.mean(cis):.4f} +/- {np.std(cis):.4f} "
              f"folds={results[name]['folds']}")
        if name == "tfidf" and known is None:
            print(f"[BERT] baseline reproduction check: {model_config}/{args.target} 는 알려진 "
                  "기준선이 없어 재현 검사를 건너뛴다 (KNOWN_BASELINE 에 값을 넣으면 검사한다).")
        elif name == "tfidf":
            got_folds = results[name]["folds"]
            ok_mean = abs(np.mean(cis) - known) < 1e-3
            ok_folds = all(abs(a - b) < 1e-3 for a, b in zip(got_folds, known_folds))
            print(f"[BERT] baseline reproduction check: mean {np.mean(cis):.4f} vs known {known} "
                  f"-> {'MATCH' if ok_mean else '*** MISMATCH ***'}")
            print(f"       folds {got_folds}\n       known {known_folds} "
                  f"-> {'MATCH' if ok_folds else '*** MISMATCH ***'}")
            if not (ok_mean and ok_folds):
                print("*** WARNING: tfidf arm 이 기존 결과를 재현하지 못했다. text_encoder_fn "
                      "리팩터가 기본 경로 동작을 바꿨다는 뜻이므로 결과를 신뢰하면 안 된다. ***")

    if args.preflight_only:
        return

    # ── 요약표 ── 기준선은 이번에 돌린 tfidf arm, 없으면 알려진 값을 쓴다.
    base = results.get("tfidf", {})
    base_mean = base.get("mean", known)
    base_folds = base.get("folds", known_folds)
    base_src = "this run" if base else "known (RESULTS.md)"
    print(f"\n================ BERT TEXT ENCODER SUMMARY (target={args.target}, "
          f"model={model_config}, {args.model_name}, bs{args.batch_size}/ep{args.epochs}) ================")
    if base_mean is None:
        # tfidf arm 도 안 돌렸고 알려진 기준선도 없는 config -> 비교 없이 값만 나열한다.
        print("baseline = (없음: tfidf arm 을 안 돌렸고 이 config 의 알려진 기준선도 없다) "
              "-> delta/검정 칸은 '-' 로 둔다")
    else:
        print(f"baseline = tfidf {base_mean:.4f} [{base_src}]  {base_folds}")
    print(f"{'arm':<18}{'mean':>8}{'std':>8}{'delta':>9}{'impr':>7}{'t_p':>9}{'wilcox_p':>10}{'UNK%':>8}   folds")
    for name in ARM_ORDER:
        r = results.get(name)
        if not r:
            continue
        if name == "tfidf" or base_mean is None:
            d, impr, tp, wp = "-", "-", "-", "-"
        else:
            pv = paired_pvalues(r["folds"], base_folds)
            d = f"{r['mean'] - base_mean:+.4f}"
            impr = f"{pv['n_improved']}/{pv['n_folds']}"
            tp = f"{pv['ttest_p']:.4f}" if pv["ttest_p"] is not None else "-"
            wp = f"{pv['wilcoxon_p']:.4f}" if pv["wilcoxon_p"] is not None else "-"
        unk = r.get("tokenization")
        unk_s = f"{unk['unk_frac_mean'] * 100:.1f}" if unk else "-"
        print(f"{name:<18}{r['mean']:>8.4f}{r['std']:>8.4f}{d:>9}{impr:>7}{tp:>9}{wp:>10}{unk_s:>8}   {r['folds']}")
    print("  (n=5 fold 라 wilcoxon 양측 p 는 최소 0.0625 — 0.05 를 원리적으로 못 넘는다. 참고용.)")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
