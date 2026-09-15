# -*- coding: utf-8 -*-
"""프로젝트 경로의 **단일 정의**.

정리 전에는 51개 파일이 각자
``PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))``
를 들고 있었다. 파일이 한 단계만 옮겨져도 그 51곳을 전부 고쳐야 했고,
실제로 한 번 그랬다. 경로는 여기서만 정의하고 나머지는 여기서 가져다 쓴다.

레이아웃 (이 파일 기준):
    <PROJECT_ROOT>/src/sclc/paths.py   <- 이 파일
    <PROJECT_ROOT>/configs/            설정
    <PROJECT_ROOT>/data/               환자 데이터 (gitignore, splits/ 만 추적)
    <PROJECT_ROOT>/outputs/            체크포인트·OOF·로그 (gitignore)
    <PROJECT_ROOT>/experiments/        실험 스크립트
"""
import os

SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SRC_ROOT)

CONFIGS_DIR = os.path.join(PROJECT_ROOT, "configs")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")

DEFAULT_CONFIG = os.path.join(CONFIGS_DIR, "config.yaml")


def data(*parts: str) -> str:
    """``data(\"CUT IMAGE\")`` -> ``<PROJECT_ROOT>/data/CUT IMAGE``"""
    return os.path.join(DATA_DIR, *parts)


def outputs(*parts: str) -> str:
    """``outputs(\"bert_text\")`` -> ``<PROJECT_ROOT>/outputs/bert_text``

    실험 결과 폴더는 **항상** 이 함수로 만든다. 상대경로 ``\"outputs/...\"`` 를 쓰면
    스크립트를 어느 디렉터리에서 실행했느냐에 따라 결과가 다른 곳에 떨어진다.
    """
    return os.path.join(OUTPUTS_DIR, *parts)


def ensure_outputs(*parts: str) -> str:
    """``outputs(*parts)`` 를 만들어 두고 그 경로를 돌려준다."""
    path = outputs(*parts)
    os.makedirs(path, exist_ok=True)
    return path
