# -*- coding: utf-8 -*-
"""TF-IDF 판독지 인코더 — **이 프로젝트가 최종 채택한 인코더**.

char n-gram(2~4) TF-IDF 400차원. 글자 조각의 빈도라서 의미를 모르고, 특히
부정문("no evidence of metastasis")과 긍정문을 거의 구분하지 못한다. 그런데도
RadBERT 를 상대로 살아남았다 — 판독지 단독으로는 RadBERT 가 이기지만(+0.034,
p=0.027) 임상변수와 합치면 그 우위가 사라지기 때문이다(정보 중복). 근거는
``experiments/실험6_판독지_인코더_비교/REPORT_ENCODER_FINAL.md`` §1·§2·§4.

[구현이 얇은 이유 — 일부러 그렇다]
  ``build_encoder_fn`` 이 ``None`` 을 돌려준다. 그러면
  ``features.build_fold_multimodal_tabular`` 안의 **기본 TF-IDF 경로**가 그대로
  돈다. 여기서 TF-IDF 를 다시 구현하면 기존 결과(OS 0.7057/0.7143 등)를 비트
  단위로 재현한다는 보장이 깨진다. 이 클래스의 역할은 계산이 아니라, TF-IDF 를
  RadBERT 와 **같은 인터페이스로** 고를 수 있게 하는 것뿐이다.
"""
from sclc.encoders.base import ReportEncoder


class TfidfReportEncoder(ReportEncoder):
    """char n-gram TF-IDF (기존 기준선 경로 그대로).

    ``out_dim`` 은 ``TfidfEncoder.max_features`` 로 전달된다. vocabulary 가
    ``max_features`` 에 못 미치면 ``TfidfEncoder.transform()`` 이 0으로 패딩하므로
    **텍스트가 짧아져도 폭은 변하지 않는다** — 이 불변식 덕분에 텍스트 source 를
    바꾸는 실험(concl_only/find_only)에서도 모델 구조가 동일하게 유지된다.
    """

    name = "tfidf"

    def __init__(self, out_dim: int = 400, ngram_range=(2, 4), audit: list | None = None):
        super().__init__(out_dim=out_dim, audit=audit)
        self.ngram_range = tuple(ngram_range)

    @property
    def description(self) -> str:
        return (f"char n-gram{self.ngram_range} TF-IDF {self.out_dim} "
                "(프로젝트 채택 인코더 / 재현 기준선)")

    def build_encoder_fn(self, corpus):
        # None = features 의 기본 TF-IDF 경로. 위 docstring 참고 — 재구현 금지.
        return None

    def expected_width(self, corpus) -> int:
        return self.out_dim

    def evaluator_kwargs(self, corpus) -> dict:
        # 기본 경로를 쓰므로 text_encoder_fn 이 아니라 TF-IDF 인자를 넘긴다.
        return {"text_encoder_fn": None,
                "tfidf_max_features": self.out_dim,
                "tfidf_ngram_range": self.ngram_range}

    def config(self) -> dict:
        return {**super().config(), "ngram_range": list(self.ngram_range)}
