# -*- coding: utf-8 -*-
"""판독지 텍스트 인코더 — 추상 부모 ``ReportEncoder`` + 구현체 2종.

    from sclc.encoders import build_encoder
    enc = build_encoder("radbert")          # 채택 레시피 (한글 ko2en, 축소 없음)
    enc = build_encoder("tfidf")            # 최종 채택 인코더
    enc = build_encoder("radbert@strip")    # §5.2 [UNK] 진단용 대조군

후보가 왜 이 둘뿐인지는 ``registry.py`` docstring 에 근거와 함께 적혀 있다.
"""
from sclc.encoders.base import ReportEncoder
from sclc.encoders.radbert import KOREAN_HANDLING, RadBertReportEncoder
from sclc.encoders.registry import (DEFAULT_ARMS, REPORT_ENCODERS, available,
                                    build_encoder, parse_spec)
from sclc.encoders.tfidf import TfidfReportEncoder

__all__ = ["ReportEncoder", "TfidfReportEncoder", "RadBertReportEncoder",
           "KOREAN_HANDLING", "REPORT_ENCODERS", "DEFAULT_ARMS",
           "build_encoder", "parse_spec", "available"]
