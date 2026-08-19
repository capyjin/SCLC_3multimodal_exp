# -*- coding: utf-8 -*-
"""공용 코어 라이브러리 — 모든 실험 폴더가 여기서만 import 한다.

실험 스크립트는 폴더 밖의 다른 실험 파일을 import 하지 않는다. 두 실험이
같은 코드를 쓰게 되면 그 코드는 실험이 아니라 인프라이므로 이쪽으로 옮긴다.

  cohort        코호트(238명) 로딩 · 매니페스트 · brain_meta 누수 수정
  dataset       PNG 로딩 · (image, tabular) Dataset · fold별 test set 재구성
  features      임상 인코더 · TF-IDF · 판독지 코퍼스 · fold별 결합 텐서
  model         이미지 백본 · 브랜치 · ConcatDeepSurv(모달리티 on/off) · 모달 조합표
  train         Cox 손실 · 학습 루프 · K-fold 평가기 · fold_plan/seed_everything
  fusion_stack  단일모달 OOF 위험점수 추출 + CoxPH 가중합 결합 (late fusion)
  metrics       C-index · 과적합 격차 · 쌍대 검정 등 평가 보조
  plotstyle     그림 팔레트/rcParams (한글 폰트 fallback 포함)
  reporting     프로토콜 11절 산출물 폴더 작성

자기점검(`if __name__ == "__main__"`)이 있는 모듈은 패키지로 실행한다:
    python -m core.cohort
"""
