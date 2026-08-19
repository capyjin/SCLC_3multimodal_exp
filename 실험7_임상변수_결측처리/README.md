# clinical/ — 임상변수 관련 안내

## 결측치 처리 실험 (2026-08-05)

LDH/WBC/FVC%/FEV1%/DLCO%/성별의 subgroup 분석과 결측치 처리 개선 실험.

| 문서 | 내용 |
|---|---|
| [`RESULTS_missing_data_subgroup.md`](RESULTS_missing_data_subgroup.md) | Step 1 데이터 감사 + Step 2 KM/Cox 연관성 |
| [`RESULTS_clinical_missing_model.md`](RESULTS_clinical_missing_model.md) | Step 3 모델 재학습·비교 + late fusion 후속 |

**결론: fold-safe 결측치 처리는 채택 모델의 예측성능을 높이지 않았다**(모든
B/C vs original CI가 0 포함). 단, `original`의 전체-중앙값 대치는 test fold를
포함해 계산된 값이므로, 성능이 같다면 **누수 없는 B/C 쪽이 방법론적으로 옳다.**

| 스크립트 | 역할 |
|---|---|
| `raw_clinical_values.py` | 원본 Excel `Whole` 시트에서 관측값·결측 복원 |
| `data_audit.py` | Step 1 감사표 |
| `km_cox_analysis.py` | Step 2 KM·Cox·PH 진단 |
| `fold_safe_features.py` | fold-safe 대치 + 누수/중복 assert |
| `exp_missing_handling.py` | Step 3 학습 (4 variant × 2 config × 2 target × 3 seed) |
| `analyze_paired.py` | paired bootstrap (fold-level 주 지표 + pooled 참고) |
| `plot_results.py` | forest plot·수준 비교 그림 |
| `exp_late_fusion_followup.py` | 영상 결합 후속 확인 (재학습 없음) |

---

## 그 외 임상변수 처리

임상변수의 **기본** 처리는 공용(core) 파일 안에 있고, 다른 여러 실험이 그
파일들을 그대로 가져다 쓰기 때문에 분리하지 않았다.

## 코드가 실제로 있는 곳

| 무엇 | 파일 | 위치 |
|---|---|---|
| `brain_meta` 누수 수정 로직 | `apply_brain_meta_fix()` | [`../core/cohort.py:153`](../core/cohort.py) |
| 수정 적용 스위치 (기본 켜짐) | `fix_brain_meta: bool = True` | [`../core/cohort.py:195`](../core/cohort.py) |
| 임상변수 인코딩 | `class ClinicalEncoder` | [`../core/features.py`](../core/features.py) |
| 임상 컬럼 목록 결정 | `resolve_clinical_columns()` | [`../core/features.py`](../core/features.py) |

## 전체 근거·수치는 여기에

`brain_meta` 무엇이 문제였고 어떻게 고쳤는지, 수정 전/후 성능 비교표는
전부 [`../실험6_판독지_인코더_비교/REPORT_ENCODER_FINAL.md`](../실험6_판독지_인코더_비교/REPORT_ENCODER_FINAL.md) §0에
정리돼 있다. 여기서 별도로 발췌하지 않는다 — §0은 그 문서의 §1~§4가 참조하는
"비교 규칙"을 담고 있어서, 따로 떼어내면 나머지 부분이 앞뒤가 안 맞게 된다.
