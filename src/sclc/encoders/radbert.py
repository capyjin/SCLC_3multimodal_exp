# -*- coding: utf-8 -*-
"""RadBERT 판독지 인코더 — frozen 영어 biomedical BERT (``StanfordAIMI/RadBERT``).

판독지 **단독**으로는 TF-IDF 를 확실히 이긴다 (OS 0.6685 vs 0.6268,
PFS 0.6354 vs 0.6094, 10 fold Δ=+0.034 p=0.027). 그런데 임상변수와 합치면
그 우위가 줄어든다 (OS +0.010, PFS -0.024). 한때 TF-IDF 를 채택했으나, 논문이 OS 를
주 지표로 두면서 **RadBERT 가 최종 채택 인코더**가 됐다 (2026-09-30, README.md 참고).
근거: ``experiments/실험6_판독지_인코더_비교/REPORT_ENCODER_FINAL.md`` §1·§2.

[한글 처리 — 이 인코더에만 있는 옵션]
  판독지는 영문 의학용어(~64%)와 한국어(~9%, 조사 + 정형 서술어)가 섞여 있는데
  RadBERT tokenizer 는 영어 전용이라 한국어를 전부 ``[UNK]`` 로 만든다
  (이 코퍼스 실측 평균 16.3% 토큰). BERT 가 져도 "BERT 가 나쁜 것"인지
  "[UNK] 때문"인지 구분이 안 되므로 한글 처리를 세 갈래로 나눠 원인을 분리한다:

      ko2en (기본) : 한국어 덩어리를 영어 구로 치환 (ko2en.translate_korean)
      strip        : 한국어 글자를 삭제하고 영어만 남김 (ko2en.strip_korean)
      raw          : 원문 그대로 (한국어 -> [UNK])

  이건 **인코더 후보가 아니라 같은 인코더의 전처리 변형**이다. 후보 비교
  (TF-IDF vs RadBERT)는 언제나 기본값 ko2en 으로 한다 — §5.2 에서 ko2en 이
  [UNK] 를 16.5% -> 2%대로 낮추고 성능도 가장 좋았기 때문이다.

[기본값이 곧 채택 레시피다]
  ``korean=ko2en, do_svd=False, do_scale=False`` (= 축소/표준화 없이 raw 768).
  MODEL_SUMMARY.md §2-1/§3-2 의 RadBERT 수치(OS 0.7153/0.7224,
  PFS 0.6456/0.6470)가 나온 바로 그 조합이다. §5.1 의 2×2 전처리 격자에서
  SVD 축소와 StandardScaler 가 둘 다 성능을 깎는 것으로 판명됐으므로,
  기본값을 그 결론에 맞춰 놨다. 격자를 다시 돌리려면 ``do_svd``/``do_scale``
  를 명시적으로 켠다.

  ⚠️ ``korean`` 이나 텍스트 source 를 바꾸면 ``embed_corpus`` 캐시 지문이
  달라져서 **에러 없이** 다른 임베딩이 나온다. 기존 체크포인트
  (``outputs/late_fusion_B_radbert`` 등)를 재현하려면 기본값을 바꾸지 마라.

[누수 없음]
  ``embed_corpus`` 는 frozen + ``no_grad`` + 문서 하나 단위 계산이라 fit 이 아예
  없다 -> fold 와 무관하게 전역 1회 계산해도 누수가 아니다. 반대로 환자를
  가로질러 계산되는 통계(SVD 기저 · StandardScaler)는 ``make_text_encoder_fn``
  안에서 train fold 로만 fit 한다. 기본값(do_svd/do_scale 둘 다 False)에서는
  그 fit 조차 호출되지 않으므로 TF-IDF 경로보다도 엄격하게 fold-safe 하다.
"""
from sclc import bert_features, ko2en
from sclc.encoders.base import ReportEncoder

#: 한글 처리 방식 -> 코퍼스 변환 함수 (None = 원문 그대로)
KOREAN_HANDLING = {
    "ko2en": ko2en.translate_korean,
    "strip": ko2en.strip_korean,
    "raw": None,
}

#: 한글 처리 방식 -> 한 줄 설명
KOREAN_DESC = {
    "ko2en": "한국어 -> 영어 구 치환 (채택 설정)",
    "strip": "한국어 글자 삭제 (대조군)",
    "raw": "원문 그대로 (한국어 -> [UNK])",
}


class RadBertReportEncoder(ReportEncoder):
    """frozen RadBERT mean-pooling 임베딩을 report 블록으로 쓴다."""

    name = "radbert"
    model_name = bert_features.DEFAULT_MODEL   # StanfordAIMI/RadBERT

    def __init__(self, korean: str = "ko2en", out_dim: int = 400,
                 do_svd: bool = False, do_scale: bool = False,
                 max_length: int = 512, embed_batch_size: int = 16,
                 cache_dir: str = bert_features.DEFAULT_CACHE_DIR,
                 audit: list | None = None):
        super().__init__(out_dim=out_dim, audit=audit)
        if korean not in KOREAN_HANDLING:
            raise ValueError(f"unknown korean handling {korean!r}; "
                             f"expected from {sorted(KOREAN_HANDLING)}")
        self.variant = korean
        self.korean = korean
        self.do_svd = bool(do_svd)
        self.do_scale = bool(do_scale)
        self.max_length = int(max_length)
        self.embed_batch_size = int(embed_batch_size)
        self.cache_dir = cache_dir
        self._embeddings: dict | None = None

    # ── 텍스트 준비 ──────────────────────────────────────────────────────
    def prepare_corpus(self, corpus: dict[int, str]) -> dict[int, str]:
        """한글 처리를 적용한 코퍼스. 반환만 하고 **절대 출력하지 않는다.**"""
        transform = KOREAN_HANDLING[self.korean]
        return corpus if transform is None else {rid: transform(t) for rid, t in corpus.items()}

    def embeddings(self, corpus: dict[int, str]) -> dict:
        """frozen 임베딩 (한 인스턴스 안에서 캐시). 디스크 캐시는 텍스트 내용의
        sha256 지문을 키로 쓰므로 한글 처리가 다르면 다른 캐시가 된다."""
        if self._embeddings is None:
            self._embeddings = bert_features.embed_corpus(
                self.model_name, self.prepare_corpus(corpus),
                max_length=self.max_length, batch_size=self.embed_batch_size,
                cache_dir=self.cache_dir)
        return self._embeddings

    # ── ReportEncoder 계약 ───────────────────────────────────────────────
    @property
    def description(self) -> str:
        pipe = []
        if self.do_svd:
            pipe.append(f"SVD->{self.out_dim}")
        if self.do_scale:
            pipe.append("StandardScaler")
        tail = " + ".join(pipe) if pipe else "축소/표준화 없음(raw 768)"
        return f"RadBERT frozen, 한글={self.korean}({KOREAN_DESC[self.korean]}), {tail}"

    def build_encoder_fn(self, corpus):
        return bert_features.make_text_encoder_fn(
            self.embeddings(corpus), out_dim=self.out_dim, audit=self.audit,
            do_svd=self.do_svd, do_scale=self.do_scale)

    def expected_width(self, corpus) -> int:
        # 축소를 끄면 _reduce_block 이 out_dim 을 임베딩 차원(768)으로 덮어쓴다.
        if self.do_svd:
            return self.out_dim
        return len(next(iter(self.embeddings(corpus).values())))

    def corpus_stats(self, corpus) -> dict:
        """집계값만 — 평균 토큰 수, [UNK] 비율, 잘린 문서 비율, ko2en 사전 커버율.

        커버율이 낮으면 "번역이 도움 안 됨"이 아니라 "번역이 덜 됨"이므로,
        ko2en arm 의 결과를 해석하려면 이 값을 같이 봐야 한다.
        """
        stats = {"tokenization": bert_features.tokenization_stats(
            self.model_name, self.prepare_corpus(corpus), self.max_length)}
        if self.korean == "ko2en":
            stats["ko2en_coverage"] = ko2en.coverage(list(corpus.values()))
        return stats

    def config(self) -> dict:
        return {**super().config(),
                "model_name": self.model_name, "korean": self.korean,
                "do_svd": self.do_svd, "do_scale": self.do_scale,
                "max_length": self.max_length}
