# 실험16 — Nested CV 다중모달 융합 (Late Fusion · GMU · Gated Late Fusion)

SCLC 238명, OS/PFS 각각. 임상(19변수)+판독지(RadBERT) **tabular 인코더**와 PET MIP **image 인코더**(SimpleCNN)를
nested CV 안에서 학습하고, 두 인코더의 출력을 다섯 가지 방식으로 결합해 outer test C-index 로 비교한다.

## 폴더 구조

```
실험16_GMU_nestedCV/
├── README.md
├── nested_fusion/                  ← 공통 코드 (import 전용)
│   ├── config.py                   경로·seed · 인코더 설정 · 융합 후보와 학습 규칙 · 실행 가드
│   ├── dataset.py                  확정 분할·레이블 읽기 · 융합 모델 입력 텐서
│   ├── processing.py               인코더 특징 추출 · 체크포인트 재구성 · 위험점수 표준화와 inner OOF
│   ├── model.py                    융합 모델 (아래 클래스 구조)
│   └── train.py                    인코더 학습 · 융합 모델 학습(조기 종료) · 후보 선택 규칙
│
├── make_splits.py                  ① 분할 생성          → data/splits/gmu_nested_v1/
├── run_nested_cv.py                ② 학습·평가          → outputs/gmu_nested/run_v2_{os,pfs}/
├── summarize.py                    ③ 결과 요약           → outputs/gmu_nested/report_v2/
├── analyze_gated_late_fusion.py    ④ Gated Late Fusion (사후) → outputs/gmu_nested/gated_nested/
├── analyze_gmu_linear_part.py      ⑤ GMU 선형 부분 (사후) → outputs/gmu_nested/linear_projection_train_only/
│
└── diagnostics/                    ← 학습 전·후 점검 (결과 아님, 재현 기록용)
    ├── checklist.py                PASS/FAIL 기록 공통 코드
    ├── check_data.py               학습 전 데이터 정합성 감사
    ├── check_cox_loss.py           Efron 손실 vs lifelines/sksurv 대조
    ├── check_tabular_checkpoint.py 기존 tabular 체크포인트 재현 (OS 0.7153)
    ├── check_learning_rate.py      GMU 학습률·최대 epoch 결정용 곡선
    ├── check_early_stopping.py     GMU·인코더 정지 규칙 결정용 곡선
    ├── smoke_test.py               파이프라인 배관 점검 (2 epoch)
    ├── check_smoke_outputs.py      run_nested_cv --smoke 산출물 구조 점검
    └── check_run_outputs.py        본 학습 산출물 규칙·누출 점검
```

`model.py` 의 클래스 구조 (GMU 는 `JointGMU`):

```
특징 수준    FeatureFusion ─┬─ ConcatFusion        (Concat)
                           └─ GatedFusion ─┬─ FixedGateFusion (Constant Gate)
                                           └─ JointGMU        (GMU)
위험점수 수준 nested_late_fusion (Late Fusion) · GatedLateFusion (Gated Late Fusion)
```

실행 스크립트(①~⑤)는 순서·저장·검증만 맡고, 모델·학습 코드는 모두 `nested_fusion/` 에 있다.
`run_nested_cv.py` 의 `OuterFold` 클래스가 outer 하나의 단계(inner 인코더 → 후보 탐색 → 선택 → 재학습 → Late Fusion → 평가)를
메서드로 하나씩 갖는다. 데이터 로딩·전처리·인코더 구조·Cox 손실은 저장소 공통 모듈 `src/sclc/` 를 쓴다
(`cohort`, `features`, `dataset`, `model`, `train`, `encoders`, `foldsafe_clinical`, `evaluation`, `paths`).

## 비교 모델

| | 결합 위치 | 결합 방식 | 코드 |
|---|---|---|---|
| Late Fusion | 위험점수 | 두 arm 의 z-score 를 CoxPH(Efron) 로 결합, 계수는 inner OOF 로 추정 | `model.nested_late_fusion` |
| Concat | 특징 | projection → L2 → concat → Linear | `model.ConcatFusion` |
| Constant Gate | 특징 | 환자 공통 학습 gate(sigmoid) 가중합 → Linear | `model.FixedGateFusion` |
| GMU | 특징 | 세 특징을 함께 보는 환자별 gate 가중합 → Linear | `model.JointGMU` |
| Gated Late Fusion | 위험점수 | `r = Σ b_m · 2σ(c_m·z) · z_m`, c=0 이면 Late Fusion 과 같음 | `model.GatedLateFusion` |

결과 파일에는 `A_primary`(Late Fusion) · `B_concat`(Concat) · `C_fixed_gate`(Constant Gate) · `D_gmu`(GMU) ·
`E1_learn_beta`(Gated Late Fusion) 이름으로 기록된다 (기존 결과와 맞추려고 이름을 바꾸지 않았다).

Gated Late Fusion 은 본 결과를 확인한 뒤 추가한 **탐색적 실험**이다. 인코더를 다시 학습하지 않고 `run_v2` 체크포인트로
위험점수를 재구성해 Late Fusion 과 같은 입력에 gate 를 얹고 결합계수 b 와 gate 계수 c 를 함께 학습한다.
(b 를 Late Fusion 계수로 고정하고 c 만 학습하는 대조 실험은 채택하지 않아 코드에서 뺐다 — OS 0.723 / PFS 0.674.)

⑤ 도 사후 탐색적 분석이다. GMU 최종 위험점수를 T_k 에서만 `r_D = a·z_tab + b·z_img + c` 로 OLS 회귀(레이블 미사용)하고,
고정 계수로 O_k 의 선형 부분만 남겼을 때의 C-index 를 잰다(잔차 제거). 체크포인트 재구성, 재학습 없음.

## 데이터 흐름 (outer k 하나)

```
T_k (190~191명) ── inner 4-fold: Fit / S(정지용) / V ──┐
                                                   ├─ 인코더 학습(Fit, S 로 조기종료) → 고정 → 특징·위험점수
                                                   ├─ Concat·Constant Gate·GMU × 후보 G1~G6 학습 → V C-index → 후보 선택
                                                   └─ V 위험점수 = inner OOF → Late Fusion·Gated Late Fusion 결합 계수 추정
T_k 전체로 인코더·특징 수준 모델 재학습(고정 epoch) → O_k (47~48명) 에서 모든 모델의 C-index
```

O_k 레이블은 어떤 학습·선택 단계에도 들어가지 않는다(`config.guard`, `diagnostics/check_run_outputs.py` 로 확인).

## 실행

저장소 루트에서:

```bash
python experiments/실험16_GMU_nestedCV/run_nested_cv.py --target os    # pfs 도 같은 방식, 끊기면 다시 실행하면 이어서 진행
python experiments/실험16_GMU_nestedCV/summarize.py
python experiments/실험16_GMU_nestedCV/analyze_gated_late_fusion.py
python experiments/실험16_GMU_nestedCV/analyze_gmu_linear_part.py
python experiments/실험16_GMU_nestedCV/diagnostics/check_run_outputs.py --target os
```

단위 테스트(Late Fusion·특징 수준 모델·학습 규칙, 합성 데이터): `python -m unittest discover -s tests`

분할은 `data/splits/gmu_nested_v1/` 에 확정돼 있어 다시 만들 필요가 없다.
경로 점검만 할 때는 `run_nested_cv.py --target os --smoke --out_dir <새 폴더>` 를 쓴다(수치는 결과가 아님).

## 결과 (outer test C-index, 5 fold 평균 ± SD)

| | Late Fusion | Concat | Constant Gate | GMU | Gated Late Fusion |
|---|---|---|---|---|---|
| OS  | 0.723 ± 0.026 | 0.712 ± 0.048 | 0.710 ± 0.037 | 0.712 ± 0.037 | 0.725 ± 0.028 |
| PFS | 0.674 ± 0.026 | 0.666 ± 0.047 | 0.663 ± 0.049 | 0.665 ± 0.040 | 0.674 ± 0.026 |

⑤ GMU 선형 부분(train-only 적합): OS 0.729 ± 0.034 / PFS 0.682 ± 0.040 — GMU 보다 +0.017, Late Fusion 보다 +0.006 / +0.008
(Late Fusion 대비 이득은 outer 2 한 fold 가 주도, 사후 분석이라 정식 비교 아님).

특징 수준 결합(Concat·Constant Gate·GMU)은 Late Fusion 보다 낮고, 환자별 gate(GMU vs Constant Gate,
Gated Late Fusion vs Late Fusion)의 이득은 없었다.
기존 비-nested late fusion(OS 0.7224 / PFS 0.6470)은 평가 절차가 달라 이 표와 직접 비교하지 않는다.

## 참고

- `diagnostics/check_learning_rate.py`·`check_early_stopping.py` 는 학습 규칙을 정하던 시점의 기록이다. 그 뒤 인코더 정지 규칙
  (S Efron NLL, tabular min epoch 20)이 확정됐으므로 지금 다시 돌리면 당시 수치와 다를 수 있다.
- 출력 폴더 이름(`gated_nested`, `lr_check`, `es_check`, `data_audit` 등)은 예전 스크립트 이름을 그대로 쓴다
  (기존 결과와 연결을 유지하려고 바꾸지 않았다).
- `outputs/gmu_nested/run_os/` (v1)는 tabular min epoch 도입 전 실행이라 폐기본이다. 정식 결과는 `run_v2_*`.
