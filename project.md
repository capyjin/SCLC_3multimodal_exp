# Project: GMU 기반 SCLC 다중모달 생존 예측

> 문서 상태: 실험 설계안 (구현 전 확인 사항 포함)
> 범위: 연구 목적, 데이터 분할, 모델, 학습, 비교 실험 및 결과 보고 기준
> 원칙: 아래의 '확정 정보'와 '제안 설정'을 구분한다. 확인되지 않은 사항은 코드나 데이터에서 확인하고 기록한다.

## 1. 연구 목적

본 실험은 소세포폐암(SCLC) 환자의 임상정보, 영상 판독지, 2D PET MIP 영상에서 추출한 특징을 Gated Multimodal Unit(GMU)으로 결합하여 전체생존기간(OS)과 무진행생존기간(PFS)의 예측 성능을 평가한다.

기존 모델은 Clinical+Report와 PET MIP에서 계산한 위험점수를 최종 단계에서 결합한다. 이에 비해 이번 실험은 위험점수로 압축되기 전의 특징을 결합한다. GMU가 다른 모달리티의 정보를 참고해 환자별 가중치를 계산하도록 하고, 기존 Late Fusion보다 추가적인 예후 정보를 활용하는지 확인한다.

본 실험에서 검증할 질문은 다음과 같다.

1. 특징 수준의 융합이 기존 위험점수 기반 Late Fusion에 비해 OS/PFS 예측에 도움이 되는가?
2. 환자별 Gate를 사용한 GMU가 같은 특징을 사용하는 단순 결합 모델보다 나은가?
3. 학습된 Gate가 환자의 병기, 전이 여부, ECOG 등 임상 특성과 어떤 관련을 보이는가? 이 분석은 탐색적으로 수행한다.

GMU가 반드시 성능을 개선한다고 전제하지 않는다. Gate 값도 임상적 중요도의 비율이나 인과적 기여도로 해석하지 않는다.

## 2. 데이터와 기존 연구 조건

### 2.1 확정 정보

| 항목 | 내용 |
|---|---|
| 연구 유형 | 단일기관 후향적 SCLC 연구, 환자 단위 분석 |
| 최종 환자 수 | 238명 |
| OS 사건 수 | 199건 (83.6%) |
| PFS 사건 수 | 212건 (89.1%) |
| 입력 모달리티 | Clinical 21개 변수, Radiology Report, 2D PET/CT MIP |
| Clinical Encoder | MLP, 기존 특징 출력 128차원 |
| Report Encoder | RadBERT 기반 분기, 기존 특징 출력 **16차원** (원본 768차원을 GMU에 직접 사용하지 않음) |
| Image Encoder | SimpleCNN, 최종 Cox 출력 직전 특징 사용 (차원 확인 필요) |
| 기존 학습 방식 | Clinical+Report 공동학습 분기와 독립 Image 분기, Cox 위험점수 출력 |
| 기존 교차검증 | 환자 단위 5-fold, 대략 Train 171 / Validation 19~20 / Test 47~48 |
| 과거 참고 성능 | Fixed Late Fusion OOF C-index: OS 0.7224, PFS 0.6470 |
| 기존 모델 가중치 | SimpleCNN 및 Clinical MLP 등은 SCLC 환자 데이터로 학습되었고 fold별 체크포인트가 있음 |

위 OOF 성능은 기존 실험 결과다. 새로운 Nested CV 성능과 동일한 평가 조건에서 얻은 결과로 취급하지 않는다. 정식 비교에는 Baseline을 새 평가 절차에서 다시 산출한다.

### 2.2 전처리에서 유지할 사항

- 임상변수의 결측치 대체 및 연속형 변수 표준화는 해당 **학습 fold만** 이용한다. 이전 실험에서 원자료 323명 기준으로 계산했던 중앙값을 새 CV에 그대로 적용하지 않는다.
- 범주형 변수의 인코딩 규칙, 이상치 처리, 범주 병합 규칙도 학습 fold에서 정하고 검증/시험 데이터에 그대로 적용한다.
- PET MIP는 기존 영상 방향 보정 및 전처리 규칙을 따른다. 최종 238명에 포함된 반전 MIP 3건(ID 9, 223, 293)의 처리 여부를 확인한다. 학습 기반 영상 통계량은 fold별로 계산한다.
- 판독지에서는 기존과 동일하게 Findings와 Conclusion을 사용한다. 기존에 검증한 OCR 교정과 비식별화 규칙을 유지한다.
- RadBERT의 외부 사전학습 파라미터는 고정할 수 있다. 그러나 SCLC 생존 레이블로 학습한 Report Head 또는 Clinical+Report 공동학습 파라미터는 검증/시험 fold 밖에서 재사용할 수 없다.
- 동일 환자의 중복 기록 또는 파생 이미지가 있다면 반드시 같은 fold에 배정한다.

### 2.3 구현 전에 확인할 정보

- [ ] OS/PFS의 시작 시점, 사건 정의, 검열 규칙, 추적기간의 단위가 원 데이터 및 기존 코드와 일치하는지 확인
- [ ] SimpleCNN의 Cox 출력 직전 특징 차원 `d_img` 확인
- [ ] Clinical 128차원과 Report 16차원이 어느 계층의 출력인지 확인
- [ ] 두 특징이 Clinical+Report **공동학습 모델**에서 나온 것인지, 별도 학습 모델에서 나온 것인지 체크포인트와 코드로 확인
- [ ] 환자별 event/time 누락, 잘못된 시간값, 중복 ID 검사
- [ ] 기존 fold별 체크포인트가 어떤 환자의 생존 레이블로 학습되었는지 기록

확인되지 않은 차원이나 학습 절차는 임의로 채우지 않는다.

## 3. 모델 설계

### 3.1 전체 흐름

```text
Clinical (21 variables) -> Clinical MLP -> 128D ----------> Projection -> dD --+
                                                                      |
Report text -> RadBERT + trained Report Head -> 16D -----> Projection -> dD --+--> Joint GateNet
                                                                      |       (all 3 features)
PET MIP -> SimpleCNN -> d_img features ------------------> Projection -> dD --+       |
                                                                              v
                                                               [g_clin, g_report, g_img]
                                                                              |
                                                 weighted sum of projected features (dD)
                                                                              |
                                                                    Dropout -> Linear(1)
                                                                              |
                                                                  Cox risk score (OS/PFS)
```

여기서 `d`는 세 모달리티가 공유하는 특징 차원이다. 기본 후보는 `d=16`이며, `d=32`는 사전에 정한 비교 후보에서만 사용한다. `d_img`는 실제 코드에서 확인한다.

### 3.2 Frozen Encoder의 의미

- **Frozen은 GMU를 학습하는 동안 인코더 파라미터를 업데이트하지 않는다는 뜻**이다.
- 외부 데이터로 사전학습한 RadBERT와, 현재 SCLC 코호트의 생존 레이블로 학습한 인코더는 구별한다.
- SCLC 데이터로 학습한 인코더는 각 Inner Train 또는 Outer Train에서만 학습해야 한다. 그 후 해당 범위의 Fusion 학습을 위해 고정한다.
- 기존의 5-fold 체크포인트를 Inner CV 전체에 그대로 가져오는 것은 금지한다. Inner Validation 환자의 생존정보가 체크포인트 학습에 포함되었을 수 있기 때문이다.
- 체크포인트 재사용 가능 여부는 파일명이나 `requires_grad=False`가 아니라 **해당 체크포인트 학습에 포함된 환자 ID**로 판단한다.
- Report 16D는 기존 Clinical+Report 공동학습 과정의 영향을 받았을 수 있다. 따라서 이 결과를 완전히 독립적인 Report 표현이라고 서술하지 않는다.

### 3.3 Feature Alignment

인코더 출력은 각각 다음과 같이 표기한다.

- `e_c in R^128`: Clinical 특징
- `e_t in R^16`: Report 특징
- `e_i in R^d_img`: PET MIP 특징

각 모달리티의 특징을 학습 가능한 Linear Projection으로 `d`차원에 맞춘다.

```text
u_m = Linear_m(e_m)                 # shape: [batch, d]
h_m = u_m / max(||u_m||_2, eps)    # patient-wise L2 normalization
```

필요한 경우 원래 인코더 특징에 대한 z-score를 추가할 수 있다. 이때 평균과 표준편차는 매 학습 fold에서만 계산한다. z-score와 L2 정규화는 서로 다른 연산이므로 적용 순서를 로그에 남긴다. 기본 설계는 Projection 후 L2 정규화이며, 별도의 정규화 층을 임의로 중복 추가하지 않는다.

### 3.4 Joint GMU

**현재 실험의 Gate는 세 모달리티의 특징을 동시에 입력받는다.** 각 특징을 자기 자신만으로 Gating하는 독립 Gate는 이번 주모델로 사용하지 않는다.

```text
H = concat(h_c, h_t, h_i)                    # [batch, 3*d]
a = ReLU(Linear(3*d, bottleneck)(H))
g = sigmoid(Linear(bottleneck, 3)(a))        # [batch, 3]
g = [g_c, g_t, g_i]

h_fused = g_c*h_c + g_t*h_t + g_i*h_i        # [batch, d]
risk = Linear(d, 1)(Dropout(p)(h_fused))     # [batch, 1]
```

Gate는 환자별 스칼라 3개다. Sigmoid 출력이므로 합이 1일 필요는 없다. Gate가 0.8이라고 해서 해당 모달리티가 예후를 80% 결정한다는 뜻은 아니다. `h_fused` 이후의 Head는 처음에는 단순한 선형 Cox Head로 제한한다.

이 구조는 다른 모달리티 특징에 따라 Gate가 달라지는 *조건부 상호작용*을 학습할 수 있다. 복잡한 고차 상호작용이나 임상적 설명 가능성이 보장되는 것은 아니다.

### 3.5 생존 예측과 손실함수

OS와 PFS는 **별개의 모델**로 학습한다. 각 모델은 하나의 연속 위험점수 `risk`를 출력한다. 위험점수가 클수록 높은 위험을 뜻하도록 방향을 통일한다.

- Loss: Negative Cox Partial Log-Likelihood
- 입력: 관찰시간, 사건 여부, 모델의 위험점수
- 작은 학습 코호트 규모를 고려하여 **Full-batch Cox Loss**를 우선 적용한다.
- 동일 사건시각(ties)에 대한 처리 방식(Breslow/Efron 등)은 사용 라이브러리와 함께 명시한다.
- Cox 출력은 직접적인 사망확률이나 특정 시점 생존확률이 아니다. 필요하면 별도의 기준 생존함수 추정 절차를 설계한다.

## 4. 비교 모델 및 가설 검증

동일한 환자 분할과 평가 조건으로 아래 모델을 비교한다.

| 구분 | 모델 | 목적 |
|---|---|---|
| A | 기존 2-arm Fixed Late Fusion | 기존 위험점수 수준의 결합과 비교 |
| B | 3-arm Frozen Feature Concat + 단순 Head | 특징 수준의 결합 자체가 도움이 되는지 평가 |
| C | 3-arm Frozen Feature 고정 Gate 가중합 + Cox Head | 환자별 Gate 없이 특징을 가중합한 대조군 |
| D | 3-arm Frozen Feature Joint GMU + Cox Head | 주 실험 모델 |

- B, C, D에는 가능한 한 동일한 Encoder, 특징 추출 위치, 전처리 및 학습/시험 환자 분할을 사용한다.
- B는 결합 방식 때문에 Head 입력 차원이 다르다. 따라서 B 대 D만으로 Gate의 독립적인 효과가 증명된다고 주장하지 않는다.
- C와 D는 공통 Projection과 Head를 사용하고, **고정 Gate와 환자별 Gate**의 차이를 중심으로 비교한다.
- A와 D의 차이는 단순히 Gate뿐 아니라 2-arm Risk-level Fusion과 3-arm Feature-level Fusion의 차이도 포함한다.
- 성능이 개선되지 않으면 개선된 것으로 서술하지 않는다. Gating 효과는 B/C와의 비교 결과를 보고 판단한다.

## 5. 검증 설계: Nested Cross-Validation

### 5.1 기본 원칙

- Outer 5-fold: 최종 내부 평가. 매 환자는 Outer Test에 정확히 1회 포함한다.
- Inner 4-fold: 후보 구조, 하이퍼파라미터 및 학습 설정 선택.
- 모든 비교 모델에 동일한 Outer Fold를 사용한다. 가능하면 OS/PFS에서도 동일한 환자 분할을 유지한다.
- 사건 여부와 관찰시간 분포를 확인하여 fold 간 심한 불균형이 없도록 한다. Stratification 규칙과 seed는 실행 전에 확정한다.
- Outer Test 환자는 학습, Early Stopping, 전처리 통계량 추정, Gate 학습, 결합계수 추정 및 모델 선택에 사용하지 않는다.

### 5.2 한 Outer Fold에서의 순서

1. 238명을 Outer Train(약 190명)과 Outer Test(약 47~48명)로 구분한다.
2. Outer Train에서 Inner 4-fold를 생성한다.
3. 각 Inner Train에 대해 생존 레이블을 사용한 인코더를 학습한다. Epoch 선택이 필요하면 **Inner Train 내부의 작은 정지용 분할**만 이용한다.
4. 해당 인코더를 Frozen하고 Projection, Gate, Cox Head를 학습한다. GMU Early Stopping도 Inner Train 내부 정지용 분할에서 결정한다.
5. Inner Validation에서는 후보 설정의 C-index를 계산한다. 이 데이터로 인코더, Gate 또는 Head를 업데이트하지 않는다.
6. 네 개 Inner Validation의 결과로 후보를 선택한다. 성능이 비슷하면 미리 정한 단순한 모델을 우선한다.
7. 선택한 설정으로 **Outer Train에서 인코더와 Fusion 모델을 다시 학습**한다. 학습 epoch 수를 포함한 재학습 규칙은 Inner CV 결과만으로 정한다. 예: Inner CV에서 선택된 epoch 수의 중앙값을 사용하고 Outer Train 전체로 재학습한다.
8. 완전히 분리한 Outer Test에서 예측하고, 위험점수와 평가 지표를 저장한다.
9. 이 과정을 Outer 5개 fold에 반복한다.

주의: 기존에 저장한 Encoder를 사용할 때는 이 과정을 건너뛰어도 되는지 반드시 **학습 환자 ID 집합**을 확인한다.

Inner CV가 끝난 뒤 Outer Test 성능을 보고 Projection 크기, Gate 구조 또는 후보 목록을 변경하지 않는다. 동일한 Outer Test를 반복적으로 살펴보고 새 모델을 선택하면 Nested CV를 사용하더라도 연구자 수준의 선택 편향이 남을 수 있다.

### 5.3 Early Stopping과 최종 재학습

- Early Stopping 기본값: patience 10. 감시 지표와 동점 처리 규칙을 사전에 명시한다.
- Inner Validation은 후보 선택용이다. Epoch 선택용 데이터는 Inner Train 안에서 별도로 확보한다.
- Outer Train 전체를 최종 재학습에 사용하는 경우, epoch 수는 Inner CV에서 확정한 규칙으로 결정한다. Outer Test로 epoch를 선택하지 않는다.
- 구현상 Encoder와 Fusion의 학습 단계가 분리된다면 각 단계의 정지 규칙도 기록한다.

## 6. 하이퍼파라미터와 과적합 관리

전체 탐색 범위를 무작정 조합하지 않는다. 첫 실험에서는 아래 사전 정의 후보만 사용하고, 각 후보 수는 결과 확인 전에 고정한다.

| 후보 | Projection `d` | Gate bottleneck | LR | Weight decay | Dropout |
|---|---:|---:|---:|---:|---:|
| G1 (기본) | 16 | 8 | 1e-4 | 1e-3 | 0.30 |
| G2 | 16 | 16 | 1e-4 | 1e-3 | 0.30 |
| G3 | 32 | 8 | 1e-4 | 1e-3 | 0.30 |
| G4 | 16 | 8 | 1e-3 | 1e-3 | 0.30 |
| G5 | 16 | 8 | 1e-4 | 1e-4 | 0.30 |
| G6 | 16 | 8 | 1e-4 | 1e-3 | 0.40 |

위 표는 제안 설정이다. 실제 실행 전 실험 책임자가 후보를 확정하면 그 목록을 잠근다. 원래 고려했던 bottleneck 32와 dropout 0.25는 첫 실험의 필수 탐색 대상으로 두지 않는다. 표에 없는 조합은 Outer Test 결과를 보고 임의로 추가하지 않는다.

- Optimizer: AdamW를 기본 후보로 사용한다.
- Gate와 Projection에는 Weight Decay를 적용한다. 추가적인 L2 페널티를 더할 경우 AdamW와 중복되는 역할 및 적용 계수를 별도로 명시한다.
- Encoder를 고정하는 동안에는 `requires_grad=False`를 적용하고, Frozen Encoder가 예측·특징 추출 시 학습 모드로 동작하지 않도록 관리한다.
- Seed와 환경을 기록한다: Python, NumPy, PyTorch, CUDA 및 관련 라이브러리 버전. 가능한 경우 결정론적 연산을 사용한다.
- BatchNorm 등 학습 상태에 의존하는 층이 있는지 확인하고, 검증·시험 시에는 평가 모드로 전환한다.

## 7. 성능 평가 및 통계 분석

### 7.1 주평가

- Primary metric: Harrell's C-index (OS와 PFS 각각).
- 각 Outer Test에서 C-index를 계산하고 5개 fold의 결과와 평균을 함께 보고한다.
- 같은 Outer Test 환자에 대한 모델 간 `Delta C-index`를 제시한다.
- 기존 성능 0.7224/0.6470과는 참고 비교만 수행한다. 정식 개선 주장은 동일한 Nested CV에서 재평가한 Baseline을 기준으로 한다.

### 7.2 보조평가

- 가능한 경우 12개월, 24개월 Time-dependent AUC를 계산한다. 각 시점의 추적기간 및 검열 분포가 평가에 적합한지 확인한다.
- 신뢰구간은 사용한 추정 방법을 명시한다. 환자 단위 재표본추출 시 학습·모델 선택의 변동성을 포함하는지 구분하여 보고한다.
- 서로 다른 Outer Fold에서 출력한 Cox 위험점수는 스케일이 다를 수 있다. 별도 점검 없이 모든 OOF 위험점수를 합쳐 하나의 C-index를 계산하지 않는다.
- Kaplan-Meier 및 Log-rank 분석은 보조적이다. 고위험군 Cutoff는 학습 데이터에서 사전에 결정하고 시험 결과를 이용해 최적화하지 않는다.
- 가능하면 병기 보정 Cox 분석 등 기존 임상정보 대비 추가 예후 가치도 탐색한다. 표본 수에 맞는 변수 수와 과적합 문제를 고려한다.

### 7.3 Gate 해석

- Gate 세 개의 분포를 OS/PFS별로 제시한다: 전체 분포, fold별 분포, 환자별 값.
- Stage(LS/ES), ECOG, 전이 여부, LDH 등과 Gate의 관계는 탐색적으로 분석한다. 다중 비교 및 교란 가능성을 언급한다.
- 특정 모달리티 제거 시 예측 변화나 성능 저하를 비교하는 Ablation을 수행한다. 제거 방식은 사전에 정의한다. 학습 때 없던 0 벡터를 시험 때만 넣는 방식은 분포 변화가 생길 수 있으므로 결과를 제한적으로 해석한다.
- SHAP 또는 Integrated Gradients는 선택적 보조분석이다. 잠재 특징에 대한 설명을 원래 임상변수나 판독 문구의 인과적 기여도로 바로 바꾸지 않는다.

## 8. Agent 실행 규칙 및 결과물

### 8.1 반드시 지킬 규칙

1. 코드 수정 전에 데이터 열, Endpoint, Encoder 출력 계층 및 차원을 확인한다.
2. 환자 ID 기준으로 분할한다. 데이터 변환은 학습 fold에서만 적합한다.
3. SCLC 생존 레이블로 학습한 Encoder의 재사용은 학습 환자 ID를 확인한 경우에만 허용한다.
4. Report 16차원을 사용한다. RadBERT 원본 768차원을 이번 GMU의 입력으로 몰래 교체하지 않는다.
5. Joint Gate는 세 모달리티 특징을 함께 입력받는다.
6. OS와 PFS는 각각 별도 Cox 모델로 학습한다.
7. Nested CV의 Outer Test 결과로 후보를 추가하거나 하이퍼파라미터를 수정하지 않는다.
8. 평가 코드에서 위험점수 부호, 사건 표기(1=event), 생존시간 단위와 C-index 계산 방향을 검증한다.
9. 오류가 나거나 정보가 없으면 임의로 값을 채우지 않는다. 확인한 사실과 미확인 사항을 구분해 보고한다.
10. 실험 중 발견한 설계 변경은 그 이유와 영향을 기록한다. 사후 변경 결과를 사전 지정 실험처럼 제시하지 않는다.

### 8.2 필수 산출물

- `fold_manifest`: Outer/Inner 환자 ID, 사건 분포, seed, 분할 규칙
- `encoder_manifest`: 체크포인트별 학습 환자 ID, 출력 계층 및 차원, 생성 방법
- `preprocessing_manifest`: 결측치 대체값, 스케일링 통계량, 영상·텍스트 처리 규칙
- `experiment_config`: 모델별 구조, 모든 후보 설정, optimizer/loss/epoch 규칙, 실행 환경
- `fold_metrics`: Endpoint별, Outer Fold별 C-index 및 기타 보조 지표
- `oof_predictions`: 환자 ID, Outer Fold, OS/PFS 시간·사건, 모델별 위험점수
- `gate_analysis`: 환자별 Gate 값 및 fold별 분포 그림
- `comparison_report`: A/B/C/D 비교 결과, Delta C-index, 한계 및 실패 실험 기록

파일명과 저장 형식은 기존 프로젝트 구조에 맞춘다. 단, 어떤 데이터와 모델에서 나온 결과인지 추적할 수 있어야 한다.

## 9. 결과 해석 및 보고 시 주의사항

이번 연구의 목적은 GMU의 우월성을 전제하는 것이 아니다. 특징 수준의 결합과 환자별 Gate가 기존 Late Fusion에 비해 실제로 추가 가치를 보이는지 평가하는 것이다.

- GMU 성능이 개선된 경우에도 구조 변경, 특징 사용, Gating 자체의 기여를 비교 실험으로 구분한다.
- Gate가 특정 모달리티에 높은 값을 부여했다고 해서 해당 모달리티의 임상적 기여도가 더 크다고 단정하지 않는다.
- 내부 Nested CV는 외부 검증을 대체하지 않는다. 단일기관 후향적 연구의 일반화 한계를 보고한다.
- 성능 차이가 작거나 fold 간 변동이 크면 해당 불확실성을 그대로 기술한다.
- Methods 및 Results는 실제 실행 로그에 근거해 작성한다. 제안 단계의 설정을 완료된 실험으로 기술하지 않는다.

## 10. 참고 문헌 및 방법론 근거

1. Arevalo J, et al. Gated Multimodal Units for Information Fusion. 2017. https://arxiv.org/abs/1702.01992
2. Varma S, Simon R. Bias in error estimation when using cross-validation for model selection. BMC Bioinformatics. 2006;7:91. https://doi.org/10.1186/1471-2105-7-91
3. Cawley GC, Talbot NLC. On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation. JMLR. 2010;11:2079-2107. https://www.jmlr.org/papers/v11/cawley10a.html
4. Collins GS, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ. 2024;385:e078378. https://doi.org/10.1136/bmj-2023-078378

---

**현재 단계의 결론:** Nested CV 및 경량 Joint GMU 실험은 진행할 수 있는 설계다. 단, `d_img`, Encoder 특징 추출 위치, 체크포인트별 학습 환자 ID, OS/PFS 시간·사건 정의는 실제 코드와 데이터에서 확인한 뒤 구현을 확정한다.
