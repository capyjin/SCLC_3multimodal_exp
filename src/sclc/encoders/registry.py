# -*- coding: utf-8 -*-
"""인코더 이름 -> 클래스. **후보는 이 표가 전부다.**

  radbert         frozen RadBERT 768 (한글 ko2en) — ★최종 채택 (2026-09-30, README 참고)
  tfidf           char n-gram TF-IDF 400 — 비교용 (이전 채택)
  radbert@strip   RadBERT + 한국어 삭제        (§5.2 [UNK] 진단용 대조군)
  radbert@raw     RadBERT + 원문 그대로        (§5.2 [UNK] 진단용 대조군)

[제외된 후보 — 코드가 없는 이유]
  KM-BERT · mBERT      단독에서 TF-IDF 와 구분되지 않았다 (§1.2)
  TF-IDF + RadBERT 결합 600차원에서 0.6013/0.6065 로 대폭 하락 (§1.1)
  TF-IDF SVD 랭크 통제군  RadBERT 를 해석하기 위한 통제 실험이었고 역할을 끝냈다 (§5.1)
  MedCPT · BioLORD · MedEmbed · bge · MiniLM
                       가중치까지 받아 뒀지만 실행하지 않았다. 중단 사유는
                       인코더 품질이 아니라 임상변수와의 정보 중복이었다 (§2, 부록 C)

  전부 ``REPORT_ENCODER_FINAL.md`` 에 수치와 함께 남아 있다. 다시 후보로 올릴
  일이 생기면 그 문서의 판정을 먼저 뒤집어야 한다.
"""
from sclc.encoders.base import ReportEncoder
from sclc.encoders.radbert import RadBertReportEncoder
from sclc.encoders.tfidf import TfidfReportEncoder

REPORT_ENCODERS: dict[str, type[ReportEncoder]] = {
    "tfidf": TfidfReportEncoder,
    "radbert": RadBertReportEncoder,
}

#: 인코더 비교 실험의 기본 arm 순서 (기준선이 먼저 와야 delta 를 계산할 수 있다)
DEFAULT_ARMS = ["tfidf", "radbert"]


def parse_spec(spec: str) -> tuple[str, str | None]:
    """``\"radbert@strip\"`` -> ``(\"radbert\", \"strip\")``, ``\"tfidf\"`` -> ``(\"tfidf\", None)``"""
    name, _, variant = spec.partition("@")
    return name, (variant or None)


def build_encoder(spec: str, **kwargs) -> ReportEncoder:
    """이름(과 변형)으로 인코더 인스턴스를 만든다.

    변형(``@`` 뒤)은 그 인코더의 변형 인자로 넘어간다 — RadBERT 는 한글 처리
    (``korean=``)가 유일한 변형 축이다. TF-IDF 에 변형을 주면 에러다.
    """
    name, variant = parse_spec(spec)
    if name not in REPORT_ENCODERS:
        raise SystemExit(f"unknown report encoder {name!r}; expected from {available()}. "
                         "제외된 후보는 sclc/encoders/registry.py docstring 참고.")
    cls = REPORT_ENCODERS[name]
    if variant is not None:
        if cls is not RadBertReportEncoder:
            raise SystemExit(f"{name!r} 인코더에는 변형이 없다 (받은 값 {variant!r})")
        kwargs["korean"] = variant
    return cls(**kwargs)


def available() -> list[str]:
    return sorted(REPORT_ENCODERS)
