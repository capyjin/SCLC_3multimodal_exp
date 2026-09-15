# -*- coding: utf-8 -*-
"""[실험6-d] 판독지의 **어느 부분**이 신호를 갖고 있나?

동기:
  clin_report(임상+판독지)가 현재 최고 성능이다. 그런데 판독지 텍스트는 두 부분이다.
    - conclusion(결론) : 판독의가 내린 요약·판단  (코호트238 중앙값 288.5자)
    - finding(소견)    : 영상에서 관찰한 것들의 나열 (코호트238 중앙값 556자)
  지금까지는 이 둘을 합쳐서 한 덩어리로 썼다 -> 성능이 '판단'에서 오는 건지
  '관찰'에서 오는 건지 구분이 안 된다.

방법:
  **텍스트 입력만** 바꾸고 나머지는 전부 고정한다 (concl_find / concl_only / find_only).
  인코더는 채택 인코더인 TF-IDF 로 고정 — 이 실험의 질문은 인코더가 아니라 입력이다.

지켜지는 불변식 — 이전에 report 브랜치 차원 때문에 한 번 데였으므로 명시한다:
  - report_dim 은 항상 tfidf_max_features(=400)로 **고정**이다.
    TfidfEncoder.transform() 이 vocabulary 가 400보다 적게 나오면 0으로 패딩하기
    때문에, 텍스트가 짧아져도 텐서 폭·브랜치 크기는 변하지 않는다.
    -> 세 variant 의 모델 구조는 완전히 동일하고, 오직 TF-IDF 값만 달라진다.
  - clinical 브랜치·fold split·seed·에폭 수 전부 동일.
  - 마스킹(날짜/병원명 삭제)은 세 variant 모두에 그대로 적용된다.

Run:  python experiments/실험6_판독지_인코더_비교/exp_text_source.py --target os
      python .../exp_text_source.py --target pfs --variants concl_find
"""
import os
import statistics
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import numpy as np

from sclc import cohort, features
from sclc.experiments.base import BaseExperiment
from sclc.utils import cli
from sclc.utils.summary import FOLDS_COLUMN, Table

# variant 이름 -> features.load_text_corpus(source=...) 에 넘길 값 + 설명
VARIANTS = {
    "concl_find": "conclusion + finding (기존 동작 / 재현 기준선)",
    "concl_only": "conclusion 만 (판독의의 '판단'만)",
    "find_only":  "finding 만 (영상 '관찰'만)",
}

KNOWN_BASELINE = {"os": 0.7076, "pfs": 0.6678}


def text_length_stats(source: str, merged_csv: str = cohort.DEFAULT_MERGED_CSV,
                      split_csv: str = cohort.DEFAULT_SPLIT_CSV) -> dict:
    """해당 source 로 실제 학습에 들어가는 텍스트의 길이 통계 (집계값만).

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


class TextSourceComparison(BaseExperiment):
    """텍스트 source 만 바꿔 가며 같은 모델을 5-fold 로 반복 학습한다."""

    name = "text_source"
    default_out_dir = "text_source"
    item_key = "variants"
    log_tag = "TEXT_SRC"
    model_config = "clin_report"
    baseline = "concl_find"
    known_baseline = KNOWN_BASELINE

    @classmethod
    def add_arguments(cls, ap) -> None:
        ap.add_argument("--variants", default=",".join(VARIANTS),
                        help=f"쉼표 목록 (기본 전부). 선택지: {list(VARIANTS)}")

    def __init__(self, args):
        super().__init__(args)
        self.names = cli.check_names(cli.comma_list(args.variants), VARIANTS)
        self.length_stats: dict[str, dict] = {}

    # ── BaseExperiment 계약 ──────────────────────────────────────────────
    def variants(self) -> list[str]:
        return self.names

    def describe(self, name: str) -> str:
        return VARIANTS[name]

    def evaluator_kwargs(self, name: str) -> dict:
        return {"text_source": name}       # <- 이 실험에서 유일하게 바뀌는 것

    def extra_record(self, name: str) -> dict:
        return {"text_source": name, "text_length": self.length_stats[name]}

    def prepare(self) -> None:
        """학습 전에 먼저 텍스트 길이를 찍어 둔다 — source 스위치가 정말 먹었는지
        몇 초 만에 확인할 수 있게 (smoke test 용)."""
        self.log.info("\n=== text length sanity check ===")
        table = Table([("variant", -12), ("median", 9), ("mean", 9), ("min", 7),
                       ("max", 8), ("med(all248)", 13), ("desc", -2)])
        for name in self.names:
            st = text_length_stats(name)
            self.length_stats[name] = st
            table.add(name, f"{st['median_chars']:.1f}", f"{st['mean_chars']:.1f}",
                      st["min_chars"], st["max_chars"], f"{st['median_chars_all']:.1f}",
                      f"  {VARIANTS[name]}")
        table.emit(self.log)
        self.log.info("  (median/mean/min/max = 코호트 238명 = 실제 학습 입력, "
                      "med(all248) = corpus 전체)")

    def summary_columns(self):
        return super().summary_columns()[:-1] + [("median_chars", 14), FOLDS_COLUMN]

    def summary_values(self, name, record, cmp):
        med = record.get("text_length", {}).get("median_chars")
        return super().summary_values(name, record, cmp)[:-1] + [
            f"{med:.0f}" if med is not None else None, f"  {record['folds']}"]


if __name__ == "__main__":
    TextSourceComparison.main()
