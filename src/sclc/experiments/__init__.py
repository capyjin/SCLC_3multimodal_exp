# -*- coding: utf-8 -*-
"""실험 파이프라인의 추상 부모 클래스들.

  BaseExperiment        "조건 하나만 바꿔 가며 5-fold 반복 + 기준선 대비" 골격
  ReportCorpusMixin     판독지 코퍼스를 한 번만 읽어 self.corpus 로 노출
  LateFusionExperiment  축을 각각 학습 -> OOF -> CoxPH 결합
  BaseAnalysis          재학습 없이 산출물만 읽어 통계를 내는 분석 골격
  TargetLoopAnalysis    그중 "타깃마다 같은 계산" 모양 (+ dispatch: 서브커맨드)

실험 스크립트(experiments/실험N_.../exp_*.py)는 이 클래스를 상속해서
"이 실험에서 유일하게 바뀌는 것"만 채운다. 실행·저장·요약·재현대조 배관은
전부 여기에 있다.
"""
from sclc.experiments.analysis import (BaseAnalysis, TargetLoopAnalysis,
                                       dispatch)
from sclc.experiments.base import BaseExperiment, ReportCorpusMixin
from sclc.experiments.fusion import Axis, Combination, LateFusionExperiment

__all__ = ["BaseExperiment", "ReportCorpusMixin",
           "LateFusionExperiment", "Axis", "Combination",
           "BaseAnalysis", "TargetLoopAnalysis", "dispatch"]
