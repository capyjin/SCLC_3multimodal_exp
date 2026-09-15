# -*- coding: utf-8 -*-
"""실험 스크립트 공통 명령행 인자.

정리 전에는 실험마다 ``--target/--epochs/--batch_size`` 를 각자 선언했고,
기본값이 조금씩 달랐다 (어떤 파일은 epochs=30, 어떤 파일은 60). 어느 쪽이
"이 저장소의 확립된 설정"인지 파일을 열어 봐야 알 수 있었다. 여기 한 곳에
모아 두면 기본값이 하나뿐이고, 실험이 일부러 다르게 쓰면 그게 눈에 띈다.

확립된 설정: **batch_size 32 / epochs 60 / seed 42** (RESULTS.md 9장).
"""
import argparse

DEFAULT_EPOCHS = 60
DEFAULT_BATCH_SIZE = 32
DEFAULT_SEED = 42
TARGETS = ("os", "pfs")


def add_common_args(ap: argparse.ArgumentParser, *, targets=TARGETS,
                    epochs: int = DEFAULT_EPOCHS, batch_size: int = DEFAULT_BATCH_SIZE,
                    seed: int = DEFAULT_SEED) -> argparse.ArgumentParser:
    ap.add_argument("--target", default=targets[0], choices=targets,
                    help="예측 대상 (기본 %(default)s)")
    ap.add_argument("--epochs", type=int, default=epochs,
                    help="에폭 수 (기본 %(default)s — RESULTS.md 9장의 확립된 설정)")
    ap.add_argument("--batch_size", type=int, default=batch_size,
                    help="배치 크기 (기본 %(default)s — 확립된 설정)")
    ap.add_argument("--seed", type=int, default=seed, help="난수 시드 (기본 %(default)s)")
    ap.add_argument("--out_dir", default=None,
                    help="결과 폴더. 기본값은 실험 클래스의 default_out_dir "
                         "(outputs/ 아래). 설정을 바꿔 가며 돌릴 때는 반드시 다른 "
                         "폴더를 줘야 results_<target>.json 이 덮어써지지 않는다.")
    ap.add_argument("--preflight_only", action="store_true",
                    help="학습 없이 feature 점검만 하고 끝낸다 (수 초).")
    return ap


def comma_list(value: str) -> list[str]:
    """``\"a, b ,c\"`` -> ``[\"a\", \"b\", \"c\"]``"""
    return [s.strip() for s in value.split(",") if s.strip()]


def check_names(names: list[str], allowed) -> list[str]:
    """알 수 없는 이름이 있으면 바로 죽는다 — 오타 하나로 6시간짜리 실행이
    엉뚱한 arm 만 돌고 끝나는 사고를 막는다."""
    unknown = [n for n in names if n not in allowed]
    if unknown:
        raise SystemExit(f"unknown name(s) {unknown}; expected from {sorted(allowed)}")
    return names
