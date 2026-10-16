# PPT 제작 요청 — SCLC 다중모달 생존예측: Gated Fusion 실험 보고 (실험16)

## 0. 작업 개요

- **목적**: 지도교수님께 드리는 연구 진행 보고(랩미팅). Late fusion(기존)에서 gated fusion(새 시도)으로 넘어간 이유, 방법, 결과, 논문 방향을 보고하고 교수님 의견을 받는다.
- **청중**: 지도교수님. 생존분석(Cox, C-index)과 딥러닝을 아는 분. 기초 개념 설명은 생략하고 **설계 근거·수식·숫자**에 집중한다.
- **언어**: 슬라이드 본문은 한국어. 모델명·수식 기호·지표명(C-index, Cox, Efron, nested CV, GMU)은 영어 그대로.
- **분량**: 본문 14–18장 내외 (표지·목차 제외). 발표 15–20분.
- **형식**: 16:9. 한 장에 메시지 하나. 각 슬라이드 상단에 **결론형 제목**(예: "환자별 gate는 성능을 올리지 못했다")을 쓰고, 본문은 그 근거.
- **발표자 노트**: 각 슬라이드에 2–4문장의 발표 대본을 넣는다.

## 1. 반드시 지킬 것 (정확성 규칙)

1. **아래 2–5절의 숫자만 쓴다.** 숫자를 반올림·재계산·추정해서 새로 만들지 않는다. 표에 없는 값이 필요하면 빈칸으로 두고 `[확인 필요]`라고 표시한다.
2. **기존 비-nested 결과(OS 0.7224 / PFS 0.6470)와 이번 nested CV 결과를 같은 표에 넣지 않는다.** 평가 절차가 달라 직접 비교가 안 된다. 언급이 필요하면 각주로 "평가 절차가 달라 직접 비교하지 않음"이라고만 쓴다.
3. **Gated Late Fusion(E)과 GMU 선형 부분 분석은 "사후 탐색적 분석"**이라고 슬라이드에 명시한다. 본 결과(A·B·C·D)를 본 뒤에 추가한 것이라 사전 지정 비교가 아니다.
4. **p-value나 "유의하다/유의하지 않다"를 쓰지 않는다.** outer fold가 5개뿐이라 검정을 하지 않았다. 대신 "fold별 Δ의 평균과 양수였던 fold 수(예: 1/5)"로 표현한다.
5. **Gate 값을 "모달리티 중요도 %"로 해석하지 않는다.** (gate 0.6 ≠ 해당 모달리티가 60% 기여.) sigmoid gate는 합이 1이 아니다.
6. 인용 논문은 아래 3절 목록만 쓰고, 서지 정보(저자·연도·저널)는 PubMed/arXiv에서 **확인한 뒤** 넣는다. 확인이 안 되면 `[서지 확인 필요]`로 남긴다.
7. 환자 영상(PET MIP 원본)은 슬라이드에 넣지 않는다.

## 2. 섹션 1 — 기존 방식(Late Fusion) → 새로운 방식(Gated Fusion), 왜?

### 2.1 데이터·과제 (1장, 짧게)
- SCLC 238명, 목표 = OS, PFS 각각 위험점수 예측. 지표 = Harrell C-index.
- 모달리티 3개:
  - 임상 19변수 → Clinical MLP → 128차원 특징
  - PET-CT 판독지 텍스트 → RadBERT(고정) + Report head → 16차원 특징
  - PET MIP 2D 영상 → SimpleCNN → 512차원 특징
- 임상+판독지는 한 모델(**tabular arm**)로, 영상은 별도 모델(**image arm**)로 학습한다.

### 2.2 기존 방식: Late Fusion (위험점수 수준 결합)
- 두 arm이 각각 위험점수 r_tab, r_img를 낸다 → 학습 환자 기준으로 z-score 표준화 → CoxPH로 결합.
- 수식:
  - z_m = (r_m − μ_m) / σ_m   (μ, σ는 학습 환자에서만 계산)
  - **risk = b_tab · z_tab + b_img · z_img**
  - b는 inner CV의 out-of-fold(OOF) z로 CoxPH(Efron) 적합.
- **한계 (새 방식의 동기)**: 결합 가중치 b가 **모든 환자에게 같다**. 어떤 환자는 영상이, 어떤 환자는 임상이 더 유익할 수 있는데 이를 반영 못 한다. 또 위험점수(스칼라 1개)로 압축된 뒤에 결합하므로 특징 사이의 상호작용을 쓸 수 없다.
- 참고 수치(이번 nested CV): late fusion의 영상 실효 가중치 = |b_img|·SD_img / (|b_tab|·SD_tab + |b_img|·SD_img) → OS fold별 0.17–0.39, PFS 0.20–0.52. 영상 몫이 작고 fold마다 흔들린다.

### 2.2-1 이전 gate 실험(실험11)과 이번 실험의 차이 (1장)
- 이미 만들어 둔 한 장짜리 요약 그림을 그대로 넣는다: `report_assets/paper/fig_gate_before_after.png` (또는 .svg).
- 내용: 이전 M0 게이트가 낮았던 이유 3가지(입력 24차원, 게이트를 끄는 장치 없음, in-sample 점수) → 이번에 바꾼 점 → Δ C-index (이전 OS −0.024 / PFS −0.007, 이번 +0.001 / 0.000).
- "이번 실험에서 성능이 late fusion보다 높아졌다"고 쓰지 않는다. 확인된 것은 "게이트를 넣어도 성능이 떨어지지 않게 됐다"까지다.

### 2.3 새 방식: Gated Fusion — 질문 두 개
- **Q1 (결합 위치)**: 위험점수로 압축하기 **전의 특징**을 결합하면 late fusion보다 나은가?
- **Q2 (환자별 gate)**: 환자마다 다른 모달리티 가중치(gate)를 주면, 모든 환자에게 같은 가중치보다 나은가?
- 전제: "GMU가 반드시 낫다"고 가정하지 않고, 대조군을 두어 **결합 위치 효과**와 **gate 효과**를 분리해서 본다.

### 2.4 그림 지시
- 왼쪽 "Late Fusion: [tab arm]→r_tab, [img arm]→r_img → 고정 b로 가중합", 오른쪽 "Gated Fusion: 3개 특징 → gate network → 환자별 g → 가중합 → Cox head" 를 나란히 놓은 비교 도식을 새로 그린다.
- `report_assets/paper/fig1_late_fusion_overview.png` 는 late fusion 쪽 참고용으로 쓸 수 있다.
- 섹션 3에서 쓸 구조도는 이미 있다 (같은 스타일, 그대로 삽입): GMU = `report_assets/paper/fig_gmu_fusion.png`(또는 .svg), Gated Late Fusion = `report_assets/paper/fig_gated_late_fusion.png`(또는 .svg).
- **쓰지 말 것**: `report_assets/fig15_gated_fusion_pipeline.png`, `report_assets/paper/fig_gate_fusion.png` — 둘 다 이전 설계(PCA+softmax gate, 비-nested 171명 학습)라서 이번 실험과 다르다.

## 3. 섹션 2 — 기존 gate fusion 논문들의 정석 방식

### 3.1 소개할 논문 (서지 확인 후 사용)
1. **GMU (Gated Multimodal Unit)** — Arevalo J, et al. *Gated Multimodal Units for Information Fusion.* 2017. arXiv:1702.01992. → gate fusion의 원형.
2. **Pathomic Fusion** — Chen RJ, et al. *Pathomic Fusion: An Integrated Framework for Fusing Histopathology and Genomic Features for Cancer Diagnosis and Prognosis.* IEEE Trans Med Imaging (2020 온라인 / 2022 게재, 확인 필요). → 생존분석(Cox)에 gate를 쓴 대표 의료 사례.
3. **PORPOISE** — Chen RJ, et al. *Pan-cancer integrative histology-genomic analysis via multimodal deep learning.* Cancer Cell 2022. → Pathomic Fusion의 gating을 대규모 pan-cancer 생존예측으로 확장.

### 3.2 정석 수식 (슬라이드 1–2장)
- **GMU (2개 모달리티 v, t)**:
  - h_v = tanh(W_v x_v),  h_t = tanh(W_t x_t)
  - z = σ(W_z [x_v ; x_t])   ← 두 입력을 **함께** 보고 gate 계산
  - h = z ⊙ h_v + (1 − z) ⊙ h_t   ← **원소별(vector) gate**, 두 모달리티 몫의 합 = 1 (convex)
  - 이후 분류 head, end-to-end 학습.
- **Pathomic Fusion의 gating-based attention**:
  - h_m = ReLU(W_m x_m)
  - z_m = σ(W_{→m} [x_1 ; x_2])   ← 모든 모달리티를 보고 모달리티 m의 gate 계산
  - h_m^gated = z_m ⊙ h_m   ← 모달리티마다 **독립 sigmoid** (합이 1일 필요 없음)
  - 이후 Kronecker product로 결합 → Cox 손실로 생존 예측.

### 3.3 정석 vs 우리 설계 비교표 (1장, 표로)
| 항목 | GMU (Arevalo 2017) | Pathomic Fusion | 우리 GMU (실험16 D) |
|---|---|---|---|
| gate 입력 | 모든 모달리티 | 모든 모달리티 | 모든 모달리티 (joint gate) |
| gate 형태 | 원소별 vector, z / 1−z (합=1) | 원소별 vector, 독립 sigmoid | **환자당 모달리티별 스칼라 3개**, 독립 sigmoid |
| 모달리티 수 | 2 | 2–3 | 3 (임상·판독지·영상) |
| 결합 | 가중합 | gate 후 Kronecker product | 가중합 |
| 인코더 학습 | end-to-end | 단일모달 사전학습 후 결합 | 인코더 **고정(frozen)**, 결합층만 학습 |
| 출력·손실 | 분류 | Cox | Cox (Efron) |
- 이렇게 바꾼 이유(한 줄씩): n=238로 작아 파라미터를 최소화(스칼라 gate, frozen encoder, 선형 Cox head) / 해석을 위해 모달리티 단위 gate / 과적합 평가를 위해 nested CV.

## 4. 섹션 3 — 우리 Gated Fusion의 수식과 흐름

### 4.1 비교 모델 5개 (표로 1장)
| 이름 | 결합 위치 | 결합 방식 | 이 모델이 대조하는 것 |
|---|---|---|---|
| A. Late Fusion | 위험점수 | z_tab, z_img의 CoxPH 선형결합 (inner OOF로 계수 추정) | 기준 모델 |
| B. Concat | 특징 | projection → L2 → concat → Linear | 특징 수준 결합 자체 |
| C. Constant Gate | 특징 | 모든 환자 공통 gate 3개(학습 파라미터) 가중합 → Linear | gate는 있으나 환자별이 아님 |
| D. GMU (주 모델) | 특징 | 환자별 joint gate 가중합 → Linear | **C vs D = 환자별 gate 효과** |
| E. Gated Late Fusion (사후 탐색) | 위험점수 | A에 환자별 gate를 얹음, c=0이면 A와 같음 | **A vs E = 위험점수 수준 gate 효과** |
- 주의: A vs D 차이에는 gate뿐 아니라 "위험점수 vs 특징 수준", "2-arm vs 3-modality"가 섞여 있다.

### 4.2 GMU(D) 수식 (1–2장, 수식은 큰 글씨로)
- 입력: e_c ∈ ℝ¹²⁸ (임상), e_t ∈ ℝ¹⁶ (판독지), e_i ∈ ℝ⁵¹² (영상) — 고정된 인코더의 출력
- ① Projection + L2 정규화: u_m = W_m e_m,  h_m = u_m / ‖u_m‖₂ ∈ ℝ^d   (d = 16 또는 32)
- ② Joint gate: g = σ( W₂ · ReLU( W₁ [h_c ; h_t ; h_i] ) ) ∈ (0,1)³   (bottleneck 8 또는 16)
- ③ 가중합: h = g_c·h_c + g_t·h_t + g_i·h_i ∈ ℝ^d
- ④ Cox head: risk = wᵀ Dropout(h)  (bias 없음)
- 손실: Cox 부분우도 음의 로그 (Efron 동점 처리). full-batch AdamW.
- 대조군 수식: B: risk = wᵀ[h_c ; h_t ; h_i],  C: g = σ(θ), θ ∈ ℝ³ (환자 공통, 초기값 0 → gate 0.5)

### 4.3 Gated Late Fusion(E) 수식 (1장)
- 입력: z = [z_tab, z_img] (A와 똑같은 표준화 위험점수)
- gate: g_m(z) = 2·σ(c_mᵀ z),  C ∈ ℝ^{2×2}
- **risk = b_tab · g_tab(z) · z_tab + b_img · g_img(z) · z_img**
- 손실: Efron Cox 손실 + λ‖C‖²,  λ ∈ {0.01, 0.1, 1, 10, 100} 를 inner 4-fold로 선택 (동점이면 큰 λ). L-BFGS. 학습 파라미터 6개(b 2개 + c 4개).
- **핵심 성질**: c = 0이면 g = 1 → risk = b_tab·z_tab + b_img·z_img = **Late Fusion과 정확히 같은 식**. 즉 E는 A를 특수한 경우로 포함하고, gate가 필요 없으면 λ가 커져서 스스로 꺼진다.

### 4.4 Nested CV 흐름 (그림 1장)
- Outer 5-fold: 학습 T_k 190–191명 / 시험 O_k 47–48명.
- Outer fold 하나 안에서:
  1. T_k를 inner 4-fold로 나눔 (Fit / S(조기종료용) / V(검증))
  2. 인코더를 Fit으로 학습, S의 Efron NLL로 조기종료 → 고정 → 특징·위험점수 추출
  3. B·C·D를 하이퍼파라미터 후보 G1–G6로 학습 → V C-index로 후보 선택 (0.005 이내면 단순한 후보 우선)
  4. V 위험점수(inner OOF)로 A·E의 결합 계수 추정
  5. T_k 전체로 인코더·B·C·D 재학습(inner 최적 epoch 중앙값으로 고정) → O_k에서 C-index
- **O_k 레이블은 학습·조기종료·후보 선택·계수 추정 어디에도 쓰지 않음** (자동 검사 220/220 통과).
- 하이퍼파라미터 후보 G1–G6: d ∈ {16, 32}, bottleneck ∈ {8, 16}, lr ∈ {1e-4, 1e-3}, weight decay ∈ {1e-3, 1e-4}, dropout ∈ {0.3, 0.4} 조합 6개.
- 그림: outer → inner 중첩 구조를 상자 도식으로. 데이터가 어느 단계에서 쓰이는지 색으로 구분.

## 5. 섹션 4 — 성능 결과

### 5.1 주 결과표 (outer test C-index, 5-fold 평균 ± SD)
| | A. Late Fusion | B. Concat | C. Constant Gate | D. GMU | E. Gated Late Fusion* |
|---|---|---|---|---|---|
| OS | **0.723 ± 0.026** | 0.712 ± 0.048 | 0.710 ± 0.037 | 0.712 ± 0.037 | 0.725 ± 0.028 |
| PFS | **0.674 ± 0.026** | 0.666 ± 0.047 | 0.663 ± 0.049 | 0.665 ± 0.040 | 0.674 ± 0.026 |
\* 사후 탐색적 분석

참고 — 단일 arm: tabular only OS 0.719 ± 0.023 / PFS 0.672 ± 0.022, image only OS 0.655 ± 0.060 / PFS 0.619 ± 0.052.

### 5.2 fold별 C-index (막대/점 그래프로)
| OS | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| A | 0.7195 | 0.7477 | 0.6967 | 0.6997 | 0.7522 |
| B | 0.7176 | 0.7208 | 0.6884 | 0.6501 | 0.7808 |
| C | 0.6999 | 0.7375 | 0.6884 | 0.6663 | 0.7581 |
| D | 0.7074 | 0.7430 | 0.6728 | 0.6787 | 0.7562 |
| E | 0.7195 | 0.7477 | 0.6976 | 0.6988 | 0.7591 |

| PFS | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| A | 0.6697 | 0.7131 | 0.6510 | 0.6851 | 0.6510 |
| B | 0.7067 | 0.7212 | 0.6193 | 0.6626 | 0.6210 |
| C | 0.7031 | 0.7221 | 0.6076 | 0.6570 | 0.6276 |
| D | 0.6931 | 0.7113 | 0.6266 | 0.6720 | 0.6220 |
| E | 0.6706 | 0.7131 | 0.6501 | 0.6851 | 0.6501 |
- 기존 그림 `outputs/gmu_nested/report_v2/cindex_by_fold.png` 를 그대로 쓰거나 위 표로 새로 그려도 된다.

### 5.3 핵심 비교 (Δ = fold별 차이의 평균, 괄호 = 양수 fold 수)
| 비교 | 의미 | OS | PFS |
|---|---|---|---|
| D − A | GMU vs Late Fusion | −0.012 (1/5) | −0.009 (1/5) |
| B − A | 특징 수준 결합 vs Late Fusion | −0.012 (1/5) | −0.008 (2/5) |
| D − C | **환자별 gate 효과** | +0.002 (3/5) | +0.002 (2/5) |
| E − A | 위험점수 수준 gate 효과 | +0.001 | 0.000 |
- 결론 슬라이드 문구: "① 특징 수준 결합(B·C·D)은 Late Fusion보다 약간 낮다. ② 환자별 gate의 이득은 없다(D ≈ C, E ≈ A)."

### 5.4 gate가 실제로 어떻게 학습됐나 (1–2장)
- **D(GMU) gate 평균** (전체 환자): OS 임상 0.64 / 판독지 0.62 / 영상 0.55, PFS 임상 0.59 / 판독지 0.60 / 영상 0.50.
  - fold마다 패턴이 다르다. 예: OS 영상 gate 평균이 fold 3 = 0.68, fold 4 = 0.45. 일관된 "영상이 중요한 환자군"이 보이지 않는다.
  - 그림: `outputs/gmu_nested/report_v2/gates.png`
- **C(Constant Gate)**: 모든 fold에서 0.51–0.54 → 초기값 0.5에서 거의 안 움직임.
- **E(Gated Late Fusion)**: λ 선택 결과 OS 2/5 fold, PFS 2/5 fold에서 gate가 사실상 꺼짐(모든 환자 g = 1). 켜진 fold에서도 gate 범위가 좁음(가장 넓은 OS fold 3에서 영상 gate 0.77–1.15, 나머지는 거의 1.00).
- 과적합 지표: D의 학습 환자 성능 이득과 조기종료 분할 이득의 차이가 fold별 OS 0.28–0.48, PFS 0.18–0.44 → gate network가 학습 환자에 맞춰짐.

### 5.5 GMU가 왜 안 됐나 — 선형 부분 분해 (사후 탐색, 1장)
- 방법: GMU 최종 위험점수를 학습 환자(T_k)에서만 r_D ≈ a·z_tab + b·z_img + c 로 회귀(생존 레이블 미사용) → 시험 환자에서 "선형 부분"과 "잔차(비선형·환자별 gate 부분)"로 나눠 각각 C-index.
- 결과:
| | OS | PFS |
|---|---|---|
| GMU(D) 전체 | 0.712 ± 0.037 | 0.665 ± 0.040 |
| GMU 선형 부분 | 0.729 ± 0.034 | 0.682 ± 0.040 |
| GMU 잔차 | 0.514 ± 0.043 | 0.512 ± 0.046 |
| Late Fusion(A) | 0.723 ± 0.026 | 0.674 ± 0.026 |
- 해석: GMU가 late fusion에 **더한 부분(잔차)은 시험 환자에서 무작위 수준(≈0.51)**이고, 학습 환자에서는 0.62 → 과적합 성분. 이를 빼면 GMU가 오히려 +0.017 오른다. 선형 부분은 결국 late fusion과 같은 형태(A와 순위상관 ≥ 0.96)다.
- 주의: 선형 부분이 A보다 +0.006/+0.008 높지만 대부분 outer fold 2 하나에서 나온 차이라 의미를 두지 않는다고 명시.

## 6. 섹션 5 — 논문 방향 (1–2장)

결과가 말해주는 것: "n=238에서는 단순한 Late Fusion으로 충분하고, 특징 수준 결합과 환자별 gate는 추가 이득이 없다(과적합으로 상쇄)."

선택지를 나란히 보여주고 **권고안**을 표시한다.
- **안 1 (권고)**: 주 모델은 Late Fusion 유지. Gated fusion(B·C·D)은 "사전에 정한 nested CV 비교"로 Results 또는 Supplementary에 넣는다. 메시지 = "복잡한 gating은 소규모 코호트에서 이득이 없었고, gate의 비선형 부분은 과적합 성분이었다" → late fusion 선택의 근거로 활용.
- **안 2**: nested CV 수치(OS 0.723 / PFS 0.674)를 주 성능으로 바꾸기. 장점: 선택 편향 없는 더 엄격한 평가. 확인할 점: nested late fusion의 PFS가 기존 0.647보다 높은 이유(tabular arm 상승)를 아직 분해하지 못했다.
- **안 3 (비권고)**: gated fusion을 주제로 한 방법론 논문. 성능 이득이 없어서 근거가 약하다.
- 공통 주의: E와 선형 부분 분해는 사후 분석이라 본문 주장으로 쓰지 않고 보조 설명으로만 쓴다.

## 7. 섹션 6 — 교수님 Comment (1장)

- 교수님 의견을 적을 빈 칸 슬라이드. 아래 논의 질문을 미리 띄워 둔다.
  1. 논문 주 모델을 Late Fusion으로 유지하고, gated fusion 비교는 보조 결과로 넣어도 될까요?
  2. 주 성능 수치를 기존 5-fold 결과로 둘지, 이번 nested CV 결과로 바꿀지?
  3. 추가로 필요한 실험이 있을까요? (예: nested CV seed 반복, 외부 검증 코호트)
  4. gate와 병기(LS/ES)·ECOG 등의 관계 분석은 계속할 가치가 있을까요? (이전 실험에서 LS/ES 차이는 방향만 보이고 표본 부족으로 확정 불가였음)
- 아래쪽에 "Comment:" 제목과 빈 메모 공간을 넉넉히 둔다.

## 8. 슬라이드 구성 제안

1. 표지 — "SCLC 다중모달 생존예측: Gated Fusion 실험 보고 (실험16, nested CV)"
2. 목차 (6개 섹션)
3. 데이터·과제 요약
4. 기존 Late Fusion 구조와 수식
5. Late Fusion의 한계 → 두 가지 질문(Q1, Q2)
6. 정석 GMU 수식 (Arevalo 2017)
7. 의료 생존분석의 gating (Pathomic Fusion / PORPOISE)
8. 정석 vs 우리 설계 비교표
9. 비교 모델 5개 (A–E)
10. 우리 GMU(D) 수식·구조도
11. Gated Late Fusion(E) 수식 — c=0이면 Late Fusion
12. Nested CV 흐름도
13. 주 결과표
14. fold별 C-index 그래프
15. 핵심 Δ 비교 → 결론 두 줄
16. gate 분포 분석
17. GMU 선형 부분 분해 — 왜 안 됐나
18. 논문 방향 (안 1–3, 권고 표시)
19. 교수님 Comment / 논의 질문

## 9. 참고 파일 (필요하면 직접 읽어서 확인)

- 실험 설명·결과 요약: `experiments/실험16_GMU_nestedCV/README.md`
- 모델 코드(수식 원본): `experiments/실험16_GMU_nestedCV/nested_fusion/model.py`
- 설정(후보 G1–G6, λ 목록): `experiments/실험16_GMU_nestedCV/nested_fusion/config.py`
- 결과: `outputs/gmu_nested/report_v2/` (cindex_os.csv, cindex_pfs.csv, gates_*.csv, late_fusion_A_diag_*.csv, summary.json, cindex_by_fold.png, gates.png)
- Gated Late Fusion: `outputs/gmu_nested/gated_nested/` (fold_metrics_*.csv, results.json — `E2_*` 열은 채택하지 않은 대조 실험이므로 쓰지 않는다)
- GMU 선형 부분: `outputs/gmu_nested/linear_projection_train_only/SUMMARY.md`
- 설계 문서: `project.md`
(경로는 모두 `/home/yjlab/SCLC_3_multimodal_test/` 기준)
