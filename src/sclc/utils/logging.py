# -*- coding: utf-8 -*-
"""실험 로깅 — 콘솔과 실행 로그 파일에 동시에 쓴다.

[왜 print 대신 이걸 쓰나]
  정리 전에는 모든 실험이 ``print()`` 였다. 결과 JSON 에는 숫자만 남고, 그 숫자가
  나온 과정(어느 fold 에서 scaler 노름이 얼마였나, preflight 가 뭘 봤나)은
  터미널 스크롤이 사라지면 같이 사라졌다. 여기 로거는 같은 내용을
  ``<out_dir>/run.log`` 에도 남겨서 사후에 되짚을 수 있게 한다.

[환자 정보 보호]
  판독지 원문은 콘솔에도 로그 파일에도 절대 쓰지 않는다. 이건 규율일 뿐 기술적
  강제가 아니므로, 텍스트를 다루는 코드에서는 반드시 집계값(길이·토큰 수·비율)만
  넘겨야 한다. ``preview_forbidden`` 헬퍼가 실수로 긴 원문을 넘겼을 때
  걸러 주지만, 1차 방어선은 호출부다.
"""
import logging
import os
import sys

_FORMAT = "%(message)s"


def get_logger(name: str, out_dir: str | None = None, filename: str = "run.log") -> logging.Logger:
    """이름당 하나의 로거를 돌려준다 (같은 이름으로 다시 부르면 핸들러를 늘리지 않는다).

    ``out_dir`` 을 주면 콘솔 + ``<out_dir>/<filename>`` 양쪽에 쓴다.
    """
    logger = logging.getLogger(f"sclc.{name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    has_console = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
                      for h in logger.handlers)
    if not has_console:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(console)

    if out_dir:
        path = os.path.abspath(os.path.join(out_dir, filename))
        already = any(isinstance(h, logging.FileHandler) and h.baseFilename == path
                      for h in logger.handlers)
        if not already:
            os.makedirs(out_dir, exist_ok=True)
            fh = logging.FileHandler(path, encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S"))
            logger.addHandler(fh)
    return logger


def banner(logger: logging.Logger, text: str, char: str = "#", width: int = 10) -> None:
    """``########## text ##########`` — 실험 arm 경계를 로그에서 눈으로 찾기 위한 구분선."""
    edge = char * width
    logger.info("")
    logger.info(f"{edge} {text} {edge}")


def section(logger: logging.Logger, text: str, char: str = "=", width: int = 16) -> None:
    """``================ text ================`` — 요약표 등 절 구분선."""
    edge = char * width
    logger.info("")
    logger.info(f"{edge} {text} {edge}")


def preview_forbidden(value, limit: int = 200) -> str:
    """로그에 넣기 전 마지막 방어선. 판독지 원문이 실수로 흘러드는 걸 막는다.

    ``limit`` 자를 넘는 문자열은 내용 대신 길이만 남긴다. 통계값·짧은 라벨은
    그대로 통과한다.
    """
    s = str(value)
    if len(s) > limit:
        return f"<{len(s)} chars redacted>"
    return s
