# -*- coding: utf-8 -*-
"""fold-safe 임상 블록 — 실험16 이 두 번째 사용처가 되면서 ``sclc.foldsafe_clinical`` 로 옮겼다.

이 파일은 ``exp_foldsafe_final.py`` 의 ``import foldsafe_clinical as fsc`` 를 그대로
두기 위한 재수출일 뿐이다. 구현과 설명은 ``src/sclc/foldsafe_clinical.py`` 에 있다.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from sclc.foldsafe_clinical import (NEVER_SMOKER, PREIMPUTED_COLUMNS, RAW_TO_CSV,  # noqa: F401
                                    SMOKING_COL, load_raw_values, make_extra_numeric_fn,
                                    reduced_clinical_columns)
