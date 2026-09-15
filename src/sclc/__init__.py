# -*- coding: utf-8 -*-
"""공용 코어 라이브러리 — 모든 실험 폴더가 여기서만 import 한다.

실험 스크립트는 폴더 밖의 다른 실험 파일을 import 하지 않는다. 두 실험이
같은 코드를 쓰게 되면 그 코드는 실험이 아니라 인프라이므로 이쪽으로 옮긴다.

데이터·모델·학습
  paths         프로젝트 경로 단일 정의 (configs/ data/ outputs/ experiments/)
  cohort        코호트(238명) 로딩 · 매니페스트 · brain_meta 누수 수정
  dataset       PNG 로딩 · (image, tabular) Dataset · fold별 test set 재구성
  features      임상 인코더 · TF-IDF · 판독지 코퍼스 · fold별 결합 텐서
  model         이미지 백본 · 브랜치 · ConcatDeepSurv(모달리티 on/off) · 모달 조합표
  train         Cox 손실 · 학습 루프 · K-fold 평가기 · fold_plan/seed_everything
  fusion_stack  단일모달 OOF 위험점수 추출 + CoxPH 가중합 결합 (late fusion)
  fusion_arms   ★ 3-way late fusion 의 pycox 단일모달 arm (임상/판독지/영상)
  fusion_diag   ★ 재학습 없는 late fusion 진단 원자 — fold별 CoxPH stack ·
                순열 대조군 · fold 안 정규화 (결합기도 이 fold 루프를 쓴다)

판독지 텍스트
  encoders      ★ ReportEncoder(ABC) + TF-IDF/RadBERT 구현 — 인코더 후보의 유일한 정의
  bert_features frozen BERT 임베딩 · 캐시 · fold-safe 축소 블록
  ko2en         판독지 한국어 -> 영어 구 치환 사전
  expert_embeddings  앵커 문장 임베딩

실험 배관
  experiments   ★ BaseExperiment(ABC) · LateFusionExperiment — 학습 실험 골격
                ★ BaseAnalysis(ABC) · TargetLoopAnalysis — 재학습 없는 분석 골격
  utils         로깅 · 결과 저장(ResultStore) · 요약표 · 공통 CLI 인자

평가·보고
  metrics       C-index · 과적합 격차 · 쌍대 검정
  radiomics     영상 radiomics 특징
  plotstyle     그림 팔레트/rcParams (한글 폰트 fallback 포함)
  reporting     프로토콜 11절 산출물 폴더 작성

자기점검(`if __name__ == "__main__"`)이 있는 모듈은 패키지로 실행한다:
    PYTHONPATH=src python -m sclc.cohort
(`pip install -e .` 를 해 두면 PYTHONPATH 없이 그냥 `python -m sclc.cohort`)
"""
