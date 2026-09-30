# -*- coding: utf-8 -*-
"""공용 코어 라이브러리 — 모든 실험 폴더가 여기서만 import 한다.

실험 스크립트는 폴더 밖의 다른 실험 파일을 import 하지 않는다. 두 실험이 같은
코드를 쓰게 되면 그 코드는 실험이 아니라 인프라이므로 이쪽으로 옮긴다.
**모듈 이름은 그 모듈이 쓰이는 용도를 그대로 말한다** — 이름을 보고 열 파일을
고를 수 있어야 한다.

경로
  paths               프로젝트 경로 단일 정의 (configs/ data/ outputs/ experiments/)

데이터 — 모델에 들어가기 전까지
  cohort              코호트(238명) 로딩 · 매니페스트 · brain_meta 누수 수정
  dataset             PET PNG 로딩 · (image, tabular) Dataset · fold별 test set 재구성
  features            임상 인코더 · TF-IDF · 판독지 코퍼스 · fold별 결합 텐서
  radiomics           영상 radiomics 특징

판독지 텍스트
  ko2en               판독지 한국어 -> 영어 구 치환 사전 (446항목)
  bert_features       frozen BERT 임베딩 · 캐시 · fold-safe 축소 블록
  encoders/           ★ ReportEncoder(ABC) + TF-IDF / RadBERT — 인코더 후보의 유일한 정의

모델
  model               영상 백본 · 브랜치 · ConcatDeepSurv(모달리티 on/off) · 모달 조합표

학습
  train               Cox 손실 · 학습 루프 · K-fold 평가기 · fold_plan · seed_everything

평가
  evaluation          C-index 부호 관례 · fold별 C-index · 과적합 격차 · fold 쌍대 검정

late fusion (채택 모델)
  late_fusion         각 축의 OOF 위험점수 추출 + fold별 CoxPH 가중합 결합
  late_fusion_tests   재학습 없는 진단 원자 — fold별 CoxPH stack · 계수 검정 ·
                      우도비 · 순열 난수 대조 · fold 안 정규화
                      (결합기도 이 fold 루프를 쓴다 — 정의가 한 곳에만 있게)
  unimodal_arms       3축 융합의 pycox 단일모달 arm (임상 / 판독지 / 영상)

실험 배관
  experiments/        ★ BaseExperiment · LateFusionExperiment — 학습 실험 골격
                      ★ BaseAnalysis · TargetLoopAnalysis — 재학습 없는 분석 골격
  utils/              로깅 · 결과 저장(ResultStore) · 요약표 · 공통 CLI 인자
  reporting           프로토콜 11절 산출물 폴더 작성
  plotstyle           그림 팔레트/rcParams (한글 폰트 fallback 포함)

자기점검(`if __name__ == "__main__"`)이 있는 모듈은 패키지로 실행한다:
    PYTHONPATH=src python -m sclc.cohort
(`pip install -e .` 를 해 두면 PYTHONPATH 없이 그냥 `python -m sclc.cohort`)
"""
