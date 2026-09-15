# -*- coding: utf-8 -*-
"""실험 스크립트가 반복해서 쓰던 배관 — 로깅 · 결과 저장 · 요약표 · 공통 인자.

  logging   콘솔 + <out_dir>/run.log 동시 기록, arm 구분선
  results   ResultStore — arm 단위 crash-safe 저장 + 설정 불일치 시 섞지 않기
  summary   고정폭 요약표 + 기준선 대비 delta/쌍대검정
  cli       --target/--epochs/--batch_size/--seed/--out_dir 공통 선언
"""
from sclc.utils import cli, logging, results, summary   # noqa: F401
