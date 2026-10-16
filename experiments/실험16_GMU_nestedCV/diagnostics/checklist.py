# -*- coding: utf-8 -*-
"""진단 스크립트 공통 — PASS/FAIL 기록과 인코더 저장 규칙의 독립 재구현."""

CHECKS = []


def check(name, ok, detail=None):
    CHECKS.append({"check": name, "pass": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail is not None else ""))


def expected_saved(losses, min_delta, min_epoch=1):
    """fit(monitor="loss") 규칙의 독립 재구현: min_epoch 이후, 개선 = loss < 최저 - min_delta."""
    best, out = float("inf"), []
    for e, v in enumerate(losses, start=1):
        out.append(bool(e >= min_epoch and v < best - min_delta))
        best = v if out[-1] else best
    return out
