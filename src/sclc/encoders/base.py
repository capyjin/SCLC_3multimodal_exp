# -*- coding: utf-8 -*-
"""판독지 인코더의 추상 부모 클래스.

[이 클래스가 존재하는 이유]
  판독지 텍스트를 모델의 report 브랜치 입력으로 바꾸는 방법이 두 가지다
  (TF-IDF / RadBERT). 정리 전에는 그 둘을 고르는 코드가 실험 파일 안의
  ``make_encoder_fn(arm, ...)`` 이라는 if-사슬이었고, 같은 사슬이
  exp_radbert_fusion.py · exp_radbert_full.py · 실험1의 seed_sweep ·
  실험11의 prepare_tabular_radbert_ckpt 에 조금씩 다른 모양으로 복사돼 있었다.
  "RadBERT 를 어떻게 만드는가"의 정의가 4벌이면, 그중 하나만 고쳐도 두 표가
  서로 다른 인코더로 계산되고 아무도 모른다.

[인코더가 반드시 지켜야 하는 계약 — 두 가지]
  1) **fold-safe.** 환자들을 가로질러 계산되는 통계(TF-IDF vocabulary/idf,
     SVD 기저, StandardScaler)는 전부 train fold 환자만으로 fit 해야 한다.
     ``build_encoder_fn`` 이 돌려주는 콜러블은 ``(train_ids, val_ids, test_ids)``
     를 이 순서로 받으며, train 으로만 fit 하고 val/test 는 transform 만 한다.
  2) **폭 고정.** report 블록의 폭이 인코더마다 다르면 모델 파라미터 수와 추정
     부담이 같이 달라져서, "인코더가 좋아서" 이긴 건지 "폭이 달라서" 이긴 건지
     구분이 안 된다 (이 프로젝트는 폭을 늘리면 오히려 나빠지는 걸 반복 확인했다).
     그래서 ``expected_width()`` 를 선언하게 하고, ``preflight()`` 가 학습 전에
     실제 폭을 재서 다르면 즉시 죽인다.

[환자 정보 보호]
  ``corpus_stats()`` 는 집계값(토큰 수 평균, [UNK] 비율 등)만 돌려준다.
  판독지 원문은 어떤 구현체도 로그·결과 파일에 남기지 않는다.
"""
import os
import sys
from abc import ABC, abstractmethod

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from sclc import cohort, features


class ReportEncoder(ABC):
    """판독지 텍스트 -> report 브랜치 특징 블록.

    구현체는 ``sclc.encoders.tfidf.TfidfReportEncoder`` 와
    ``sclc.encoders.radbert.RadBertReportEncoder`` 둘뿐이다. 다른 후보
    (KM-BERT · mBERT · TF-IDF+BERT 결합 · SVD 랭크 통제군)는 실험6에서
    전부 탈락해 후보에서 제외됐다 — 근거는
    ``experiments/실험6_판독지_인코더_비교/REPORT_ENCODER_FINAL.md`` §1·§3.
    """

    #: 레지스트리 키 (``sclc.encoders.registry``)
    name: str = ""
    #: 같은 인코더 안의 세부 변형 (예: RadBERT 의 한글 처리 방식). 없으면 빈 문자열.
    variant: str = ""

    def __init__(self, out_dim: int = 400, audit: list | None = None):
        #: report 블록의 목표 폭. 기본 400 = TF-IDF 폭 (비교 공정성).
        self.out_dim = int(out_dim)
        #: fold별 train-only 통계 감사 기록이 쌓이는 리스트 (결과 JSON 에 저장된다).
        self.audit: list = [] if audit is None else audit

    # ── 서브클래스가 채우는 부분 ──────────────────────────────────────────
    @property
    @abstractmethod
    def description(self) -> str:
        """요약표/로그에 찍힐 한 줄 설명."""

    @abstractmethod
    def build_encoder_fn(self, corpus: dict[int, str]):
        """``features.build_fold_multimodal_tabular`` 의 ``text_encoder_fn`` 을 만든다.

        ``None`` 을 돌려주면 features 의 **기본 TF-IDF 경로**가 그대로 돈다
        (= 기존 결과를 비트 단위로 재현하는 기준선 경로).
        """

    @abstractmethod
    def expected_width(self, corpus: dict[int, str]) -> int:
        """이 인코더가 만들 report 블록의 폭. ``preflight`` 가 실측과 대조한다."""

    def evaluator_kwargs(self, corpus: dict[int, str]) -> dict:
        """``TrimodalEvaluator`` 에 넘길 인자. 기본은 ``text_encoder_fn`` 하나뿐이고,
        기본 TF-IDF 경로를 쓰는 구현체는 폭/ngram 인자를 대신 넘긴다."""
        return {"text_encoder_fn": self.build_encoder_fn(corpus)}

    def corpus_stats(self, corpus: dict[int, str]) -> dict | None:
        """토큰화 통계 등 **집계값만**. 기본은 없음(None)."""
        return None

    def config(self) -> dict:
        """결과 JSON 에 남길 설정 스냅샷. 나중에 "이 숫자가 어떤 설정에서 나왔나"를
        결과 파일만 보고 답할 수 있어야 한다."""
        return {"encoder": self.name, "variant": self.variant, "out_dim": self.out_dim}

    # ── 공통 구현 ────────────────────────────────────────────────────────
    @property
    def key(self) -> str:
        """``radbert@strip`` 처럼 변형까지 포함한 이름. 결과 JSON 의 항목 키다."""
        return f"{self.name}@{self.variant}" if self.variant else self.name

    def preflight(self, corpus: dict[int, str], logger, merged_csv: str | None = None,
                  split_csv: str | None = None) -> list[dict]:
        """학습 전에 fold별 feature 행렬을 실제로 만들어 보고 세 가지를 확인한다.

          (1) report 블록 폭이 ``expected_width()`` 와 같은가 (다르면 즉시 죽는다)
          (2) 전체 텐서 폭 == clinical_dim + report_dim 인가
          (3) 블록 값이 환자마다 실제로 변하는가 (전부 0/상수면 뭔가 잘못된 것)

        학습 없이 수 초면 끝나므로, 6시간짜리 실행을 시작하기 전에 항상 돌린다.
        """
        load_kw = {k: v for k, v in (("merged_csv", merged_csv), ("split_csv", split_csv))
                   if v is not None}
        cohort_df = cohort.load_trimodal_cohort(**load_kw)
        clinical_frame = cohort_df.drop_duplicates("research_id").set_index("research_id")
        std_cols, cat_cols = features.resolve_clinical_columns(clinical_frame)
        encoder_fn = self.build_encoder_fn(corpus)
        expected = self.expected_width(corpus)

        dims, checks = None, []
        for fold in sorted(int(f) for f in cohort_df["fold"].unique()):
            fdf = cohort_df[cohort_df["fold"] == fold]
            ids = {s: fdf.loc[fdf["split"] == s, "research_id"].astype(int).tolist()
                   for s in ("train", "val", "test")}
            tab, clinical_dim, report_dim = features.build_fold_multimodal_tabular(
                clinical_frame.loc[ids["train"]], clinical_frame.loc[ids["val"]],
                clinical_frame.loc[ids["test"]], corpus, std_cols, cat_cols,
                tfidf_max_features=self.out_dim, text_encoder_fn=encoder_fn,
            )
            dims = (clinical_dim, report_dim, tab["train"].shape[1])
            blk = tab["train"][:, clinical_dim:]
            nonzero = np.abs(blk).sum(axis=0) > 0
            checks.append({
                "fold": fold,
                "n_nonzero_cols": int(nonzero.sum()),
                "col_std_mean": round(float(blk[:, nonzero].std(axis=0).mean()), 4) if nonzero.any() else 0.0,
                "test_std_mean": round(float(tab["test"][:, clinical_dim:][:, nonzero].std(axis=0).mean()), 4) if nonzero.any() else 0.0,
            })

        clinical_dim, report_dim, width = dims
        logger.info(f"[preflight/{self.key}] clinical_dim={clinical_dim}  "
                    f"report(text)_dim={report_dim} (expected {expected})  tabular width={width}")
        if report_dim != expected:
            raise AssertionError(
                f"report block width {report_dim} != expected {expected} — "
                "폭이 달라지면 다른 인코더와의 비교가 오염된다")
        if width != clinical_dim + report_dim:
            raise AssertionError("feature width mismatch")
        for c in checks:
            logger.info(f"[preflight/{self.key}] fold {c['fold']}: "
                        f"nonzero_cols={c['n_nonzero_cols']}/{report_dim} "
                        f"train_col_std={c['col_std_mean']} test_col_std={c['test_std_mean']} "
                        f"(0.0 이면 블록이 상수 = 뭔가 잘못된 것)")
        return checks
