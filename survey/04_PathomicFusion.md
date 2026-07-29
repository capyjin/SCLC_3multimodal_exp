# Pathomic Fusion: An Integrated Framework for Fusing Histopathology and Genomic Features for Cancer Diagnosis and Prognosis

> 한 줄 요약: 조직학 CNN·세포그래프 GCN·유전체 SNN의 32차원 임베딩을 **게이팅 어텐션으로 걸러낸 뒤 Kronecker 곱(외적)으로 묶어** 모든 단일·쌍별·삼중 상호작용을 명시적으로 모델링, glioma 생존예측에서 기존 SOTA인 **concat 융합(0.781)을 0.826으로 이겼다.**

---

## 0. 메타

| 항목 | 값 |
|---|---|
| 연도 / 학회 | 2020 / **IEEE TMI** (Trans. on Medical Imaging) — 정식 저널 |
| 인용수 (2026-07-26 조회) | **659** ★ 이 계열 최다 |
| 링크 | [arXiv:1912.08937](https://arxiv.org/abs/1912.08937) · **[github.com/mahmoodlab/PathomicFusion](https://github.com/mahmoodlab/PathomicFusion)** ← **코드 공개** |
| 소속 | **Mahmood Lab**, Brigham and Women's Hospital / Harvard Medical School |
| 모달리티 | 조직 이미지(CNN) + 세포그래프(GCN) + 유전체(SNN) — **영상·텍스트·임상 없음** |
| 융합 위치 | **Intermediate** (단일모달 선학습 → 게이팅 → Kronecker) |
| **융합 아키타입** | **연산자** (Kronecker product) + Attention(게이팅) |
| 태스크 | 생존예측(Cox) + 등급분류(CE) |
| N | glioma **769명**(1,505장) / CCRCC **417명**(1,251장) |
| 검증 방식 | **15-fold Monte Carlo CV**, 총 480개 모델 학습 |
| 주요 지표 | C-index 0.826±0.009 (glioma) / 0.720±0.028 (CCRCC) |
| **내 실험 적용가능성** | **즉시적용**(게이팅) / **변형필요**(Kronecker 차원축소 필수) |

> 📌 **이 계열의 시조.** DOF(03)가 게이팅·텐서융합을 여기서 빌려갔고, M4Survive(01)는 Table 1의 최강 baseline으로 씀. 셋 다 이 논문의 후손.

---

## 1. 논리 전개 5단계 ★

| 단계 | 내용 | 원문 근거 |
|---|---|---|
| ① 문제 제기 | 종양학자는 조직학(정성)과 유전체(정량)를 **둘 다** 보고 예후를 판단한다. 그런데 딥러닝 모델은 대부분 하나만 쓴다 | §I |
| ② 기존 한계 | (a) 유전체 딥러닝[19,20]은 영상 정보를 안 씀 (b) 조직학+유전체 융합은 **거의 전부 late fusion / vector concatenation**[20,29,30,31] (c) Shao et al.[32]는 hand-crafted 피처 의존 (d) **"Beyond late fusion, there is limited work..."** (e) **"there is little work made in interpreting histology features in these multimodal deep networks"** (f) GCN을 생존예측에 쓴 연구가 없음 | §II |
| ③ 틈새 선언 | late fusion/concat **너머** + **해석가능성** + **GCN 최초 적용** — 틈새를 3개로 쪼갬 | §II 끝 |
| ④ 해법 | 게이팅 어텐션(노이즈 억제) → **Kronecker 곱**(모든 상호작용 명시화) + Grad-CAM/Integrated Gradients(해석) | §III D–E |
| ⑤ 검증 논리 | 15-fold MC CV × 6개 모달 조합 × **2개 암종** + WHO 임상 패러다임 대비 + **이전 SOTA(concat)와 동일 split 비교** + **앙상블 대조실험** + 해석 시각화 | §IV–V |

> 🔑 **③의 설계가 교과서적.** 틈새를 3개로 쪼개고 ④에서 각각에 부품을 하나씩 대응(Kronecker / IG·Grad-CAM / GCN) → **contribution bullet 3개가 그대로 나온다.** 논문 구조가 introduction에서 이미 결정됨.

**이 논문이 공격하는 "가상의 적" — 내 논문에 그대로 쓸 문장들:**
> "most deep learning-based objective outcome prediction and grading paradigms are based on histology or genomics alone and **do not make use of the complementary information in an intuitive manner**" — Abstract

> "previous works have generally relied on the ensembling of extracted feature embeddings from separately trained deep networks (**termed late fusion**)" — §II

> "Morbadersany et al. proposed a strategy for combining histology image and genomic features via **vector concatenation**" — §II

> ⚠️⚠️ "Previous works have only relied on CNNs for extracting features from histology images, and **late fusion** for integrating image features with genomic features." — §III
> **내 현재 구조(단일모달 학습 → CoxPH 가중합)가 정확히 이 문장이다.** 그리고 이 논문은 그 대안을 **같은 데이터·같은 split에서 측정했다** → 아래 §4 참조.

---

## 2. Contribution

**(a) 저자 주장 — bullet 3개**
1. **Novel Multimodal Fusion Strategy** — 게이팅된 표현들의 **Kronecker 곱**으로 모달 간 pairwise 상호작용을 모델링
2. **GCNs for Cancer Outcome Prediction** — 세포그래프 GCN을 **생존예측에 최초 적용**. CNN의 보완재
3. **Objective Image-Omic Quantitative Study with Multimodal Interpretability** — 15-fold CV, 2개 암종, WHO 패러다임 및 기존 SOTA 상회 + Grad-CAM/IG로 **모달별 해석**

> 💡 구조: **[융합 연산자] + [새 모달 표현] + [대규모 검증 + 해석가능성]**
> 세 논문 비교 — M4Survive는 3-triad(기술+성능+실용), DOF는 4-bullet(최초성+손실+구조+임상), 여기는 3-bullet(연산자+표현+검증·해석).
> **공통 법칙: 반드시 "기술 기여 1개 이상 + 검증/임상 기여 1개"를 섞는다.**

**(b) 내가 판단한 진짜 기여**
- **Kronecker 융합이 concat보다 낫다는 것을 동일 조건에서 증명한 것** — GSCNN(concat) 0.781 → Pathomic Fusion 0.826. 이 계열에서 가장 깨끗한 대조
- **게이팅 어텐션** — 사실 Kronecker보다 실용적으로 더 중요. 노이즈 모달의 기여를 다른 모달이 억제하게 함
- **해석가능성을 실제로 구현한 것** — M4Survive는 "interpretability"를 주장만 하고 실험이 없었다. 여기는 Grad-CAM(이미지) + IG(그래프·유전체)로 **모달별 기여를 실제로 시각화**하고, *다른 모달을 조건부로 주었을 때 중요도가 어떻게 이동하는지*까지 보여줌
- **코드 공개** — 이 계열 논문 중 드물게 재현 가능

**(c) 부풀려진 부분 / 사실 남의 것**
- ⚠️ **"6.31% 개선"의 기준선이 바뀐다.** 6.31%는 WHO 패러다임(0.777) 대비, 5.76%는 concat SOTA(0.781) 대비. 그런데 **유전체 SNN 단독이 이미 0.808이다.** 진짜 융합 이득은 0.808 → 0.826 = **+0.018** (±0.009~0.014). 논문은 이 비교를 전면에 내세우지 않는다
- ⚠️ **GCN(contribution 2)은 c-index에서 효과가 없다.** CNN⊗SNN 0.820 → CNN⊗GCN⊗SNN 0.826 = **+0.006**. 저자도 인정: *"Using the c-Index metric, GCNs do not add significant improvement over CNNs alone."*
  → **대신 다른 지표로 방어한다**: KM 하위군 p-value가 0.103 → 2.68e-03으로 개선. 📌 **"주 지표에서 안 나오면 부 지표를 찾는다" — 논문 쓰기 기술로서 반드시 기억할 것** (정직하되 전략적)
- Kronecker 융합 [24](Zadeh, Tensor Fusion Network), 게이팅 [60], VGG-19 [21], SNN [58], GraphSAGE [35], SAGPool [36], CPC [55], Grad-CAM [61], IG [62] — **전부 남의 것.** 새 수식은 사실상 없고 **조합과 적용이 기여**
- **glioma 데이터의 40%가 RNA-Seq 결측** — 언급만 하고 처리 방법이 본문에 불명확

---

## 3. 방법

**Step 1 — 단일모달 임베딩 (각 32차원, 개별 선학습)**
- Histology CNN: VGG19-BN (ImageNet 사전학습), 512×512 ROI @20× → `h_i ∈ R^32`
- Histology GCN: 핵분할(cGAN) → KNN(K=5) 세포그래프 → 수작업 12피처(윤곽 8 + GLCM 4) + CPC 1024차원 → GraphSAGE + SAGPool → `h_g ∈ R^32`
- Genomic SNN: SeLU + Alpha Dropout 4층 (고차원·저표본 과적합 방지) → `h_n ∈ R^32`

**Step 2 — 게이팅 어텐션 ★내가 가져갈 부분**
```
h_m       = ReLU(W_m · h_m)
z_m       = σ(W_{ign→m} · [h_i, h_g, h_n])      ← 다른 모달 전부를 보고 점수 매김
h_m,gated = z_m ⊙ h_m                            ← element-wise product
```
> **핵심 의미:** 모달 m의 각 피처 중요도를 **나머지 모달들이 함께 결정**한다.
> 목적이 명시적임 — *"To decrease the impact of noisy unimodal features"*, *"some of the captured features may have high collinearity, in which employing a gating mechanism can reduce the size of the feature space"*

**Step 3 — Kronecker 곱 (외적)**
```
h_fusion = [h_i; 1] ⊗ [h_g; 1] ⊗ [h_n; 1]
```
- 각 33차원 → **33 × 33 × 33 = 35,937차원** 텐서
- `1`을 덧붙여서 **단일모달 항 + 쌍별 항(h_i⊗h_g 등) + 삼중 항이 한 텐서 안에 모두 공존**
- 바깥 차원 = 단일·쌍별, 안쪽 = 삼중 상호작용

**Step 4 — 해석**
- 이미지: **Grad-CAM** (마지막 뉴런=hazard에 역전파)
- 그래프·유전체: **Integrated Gradients** (그래프는 노드를 배치 차원으로 취급)

**그림 1 한 문장:** 모달별로 32차원 임베딩을 만들고, 서로를 보며 게이팅해 노이즈를 죽인 뒤, 외적으로 모든 상호작용을 펼쳐 Cox head에 넣는다.

---

## 4. 실험 설정 & 숫자

| 항목 | glioma (TCGA-GBMLGG) | CCRCC (TCGA-KIRC) |
|---|---|---|
| 환자 / 이미지 | 769 / 1,505 | 417 / 1,251 |
| ROI | 1024×1024 @20× | 512×512 @40× (환자당 3장) |
| 유전체 피처 | 320 (CNV 79 + 변이 1 + RNA-Seq 240) | 357 (CNV 117 + RNA-Seq 240) |
| 결측 | **RNA-Seq 약 40% 결측** | – |
| 검증 | 15-fold Monte Carlo CV (기존연구[29]와 **동일 split**) | 15-fold MC CV |

**Table I — glioma 생존예측 (★핵심 표)**

| 모델 | C-index | 비고 |
|---|---|---|
| Cox (Age+Gender) | 0.732±0.012 | 임상 기준선 |
| Cox (Grade) | 0.738±0.013 | |
| Cox (Molecular Subtype) | 0.760±0.011 | |
| Cox (Grade+Subtype) | 0.777±0.013 | **WHO 패러다임** |
| Histology CNN | 0.792±0.014 | 영상 단독 |
| Histology GCN | 0.746±0.023 | 그래프 단독 |
| **Genomic SNN** | **0.808±0.014** | ★**최강 단일모달** |
| SCNN (조직학만) [29] | 0.754 | |
| **GSCNN (조직학+유전체, concat)** [29] | **0.781** | ★**concat 융합 SOTA** |
| Pathomic F. (GCN⊗SNN) | 0.812±0.010 | |
| Pathomic F. (CNN⊗SNN) | 0.820±0.009 | |
| **Pathomic F. (CNN⊗GCN⊗SNN)** | **0.826±0.009** | ★최고 |

**Table II — CCRCC**

| 모델 | C-index |
|---|---|
| Cox (Age+Gender) | 0.630±0.024 |
| Cox (Grade) | 0.675±0.036 |
| Histology CNN | 0.671±0.023 |
| Histology GCN | 0.646±0.022 |
| Genomic SNN | 0.684±0.025 |
| Pathomic F. (GCN⊗SNN) | 0.688±0.029 |
| Pathomic F. (CNN⊗SNN) | 0.719±0.031 |
| **Pathomic F. (CNN⊗GCN⊗SNN)** | **0.720±0.028** |

**★ 이 표에서 내가 뽑아야 할 단 하나의 숫자:**
> **동일 데이터·동일 split·동일 모달에서 concat 0.781 → Kronecker 0.826 (+0.045).**
> 교수님이 "융합을 fancy하게"라고 하신 것의 **정량적 근거가 이것.** 세 논문 중 concat 대비 개선폭이 가장 크고 가장 깨끗하게 측정됨.

**Ablation에서 실제로 밝혀진 것:**
- **유전체 단독(0.808)이 WHO 패러다임(0.777)보다 이미 낫다.** 융합의 순수 이득은 +0.018
- **GCN은 c-index를 못 올린다**(+0.006) — 저자 자인. KM 하위군 유의성으로 방어
- **앙상블 효과가 아님을 대조실험으로 증명** — *"inputting same modality twice into Pathomic Fusion resulted in overfitting"* ★내가 반드시 따라할 통제실험
- CCRCC에서는 Histology CNN(0.671)이 hazard를 거의 균일하게 예측 → *"histology alone is a poor prognostic indicator for survival in CCRCC"*. **약한 모달이 융합에서 살아나는 사례**
- 등급분류에서도 AUC +2.75%, AP +4.23%, F1 +4.27% 개선

---

## 5. 비판적 검토

- **통계 ✓✓** — 15-fold MC CV, 480개 모델, 2개 암종, 기존연구와 동일 split. 이 계열 최고 수준의 엄밀함
- **⚠️ 기준선 선택의 수사학** — "6.31% 개선"은 WHO(0.777) 대비. 최강 단일모달(유전체 0.808) 대비로 쓰면 +2.2%. **어떤 baseline을 고르느냐로 개선폭이 3배 차이 난다.** 논문 읽을 때 항상 "무엇 대비인가"를 먼저 확인할 것
- **⚠️ contribution 2(GCN)가 주 지표에서 실패** — 저자가 정직하게 인정하고 부 지표로 방어. 배울 점이자 경계할 점
- **차원 폭발** — 33³ = **35,937차원**을 769명으로 학습. l1=32로 억제하고 있지만 여전히 극단적. **내 238명엔 그대로 못 씀**
- **RNA-Seq 40% 결측** — 처리 방식 불명확. 결측 대응이 논문의 관심사가 아님
- **재현성 ✓✓** — 코드·모델 공개, TCGA 공개 데이터. 세 논문 중 유일하게 완전 재현 가능
- **내 조건과 다른 점** — **텍스트·임상 정형변수가 없다.** N이 769로 내 238의 3배. 병리 이미지는 PET-CT와 성격이 매우 다름(세포 단위 vs 대사 단위)

---

## 6. 내 SCLC 실험에 적용

### 이 논문이 내게 주는 것: "중복 문제에 대한 두 번째 처방"

DOF(03)는 중복을 **직교화(MMO loss)**로 풀었다. 이 논문은 같은 문제를 **게이팅(억제)**으로 푼다:
> *"To decrease the impact of **noisy unimodal features**... some of the captured features may have **high collinearity**, in which employing a gating mechanism can **reduce the size of the feature space**"* — §III D
>
> **내 SimpleCNN 이미지 임베딩 = 정확히 "noisy unimodal features".**
> 지금 내 concat 모델은 L2 정규화 후 그냥 이어붙인다 → 노이즈 이미지 피처가 그대로 head에 전달된다.
> 게이팅을 넣으면 **clinical/report가 이미지 피처를 눌러줄 수 있다.**

| 가져올 것 | 이유 |
|---|---|
| **게이팅 어텐션** ★최우선 | Linear 2개면 구현 끝. `z_m = σ(W·[h_img, h_clin, h_rep])`, `h_m ⊙ z_m`. **내 문제(이미지 노이즈)에 정확히 설계된 부품.** DOF의 MMO와 **동시 적용 가능**(DOF가 실제로 그렇게 함) |
| **concat vs 연산자 비교 행** | GSCNN 0.781 → Pathomic 0.826의 구조를 내 표에 복제: 내 concat 0.678 → 게이팅+연산자 융합 ?. **교수님 요구에 대한 직접 답변이 되는 표** |
| **앙상블 통제실험** ★ | *"같은 모달을 두 번 넣으면 과적합"*. 내가 성능이 올랐을 때 **"파라미터가 늘어서 오른 것 아니냐"**는 반론을 미리 차단. 리뷰어가 반드시 묻는다 |
| **15-fold Monte Carlo CV** | DOF(n=176)와 이 논문(n=769) **둘 다 채택**. 독립된 두 논문이 같은 프로토콜 → 내 5-fold에서 올릴 강한 근거 |
| **Grad-CAM(영상) + IG(정형)** | PET-CT에 Grad-CAM, clinical/report 피처에 Integrated Gradients. **"어느 모달이 실제로 일하는가"**라는 내 핵심 질문에 답하는 그림. M4Survive가 주장만 하고 못 한 것 |
| **조건부 중요도 이동 분석** | *"how feature importance shifts when conditioning on multimodal input"* — 이미지를 넣었을 때 clinical 피처 중요도가 어떻게 변하는지. **내 "이미지가 왜 기여 안 하나"에 직접 답하는 분석** |
| **[코드](https://github.com/mahmoodlab/PathomicFusion)** | PyTorch, 융합 모듈을 그대로 읽고 이식 가능 |

| 안 가져올 것 | 이유 |
|---|---|
| **Kronecker 곱 원본 크기** | 33³ = **35,937차원 / 238명**. 자살행위. 쓴다면 **l1을 8로 줄여 9³=729** 또는 **2-모달만 외적**(clinical⊗report = 33²=1,089). 아니면 아예 게이팅만 |
| GCN / 세포그래프 | 병리 이미지 없음. 게다가 **저자 스스로 c-index 개선 없다고 인정** |
| VGG-19 ImageNet | ResNet18 ImageNet이 내 PET-CT에서 이미 실패(0.633). 01번의 의료 FM 경로가 낫다 |

**당장 해볼 실험 1개:**
> 현재 tri-modal concat 모델에 **게이팅 어텐션만 추가** (Kronecker 없이, L2 정규화 자리에 게이팅 삽입).
> 가설: 이미지 피처가 눌리면서 0.678 → clin+report 수준(0.708) 이상 회복.
> **MMO(03번)와 게이팅(04번)을 각각 단독으로 먼저 돌린 뒤 조합** — 그래야 어느 쪽이 효과인지 귀속된다.

### ★ 노블티 각도: 두 처방의 대결

| 처방 | 출처 | 원리 |
|---|---|---|
| **게이팅** | Pathomic Fusion (04) | 중복·노이즈 피처를 **억제** |
| **직교화 MMO** | DOF (03) | 임베딩을 **분리** |

> **같은 문제(모달 중복)에 대한 서로 다른 두 해법이고, 아무도 둘을 직접 비교한 적이 없다.**
> 그리고 내 데이터에는 **이 세상에서 가장 중복된 모달 쌍**이 있다 — **PET-CT 영상과 그 영상의 판독지.**
> 판독지는 정의상 그 영상을 서술한 문서다. 중복 해소 기법을 시험하기에 이보다 좋은 테스트베드가 없다.
>
> **연구 질문:** *"영상과 그 판독지처럼 본질적으로 중복된 모달 쌍에서, 게이팅과 직교화 중 무엇이 효과적인가?"*
> — 소규모 데이터로 답할 수 있고, 두 논문 모두 하지 않았으며, 임상적으로도 의미 있는 질문.

---

## 7. 인용할 문장

> "most deep learning-based objective outcome prediction and grading paradigms are based on histology or genomics alone and do not make use of the complementary information in an intuitive manner" — Abstract

> "previous works have generally relied on the ensembling of extracted feature embeddings from separately trained deep networks (termed late fusion)" — §II ⚠️ *내 현재 구조*

> "Previous works have only relied on CNNs for extracting features from histology images, and late fusion for integrating image features from CNNs with genomic features." — §III

> "To decrease the impact of noisy unimodal features during multimodal training, before the Kronecker Product, we employed a gating-based attention mechanism that controls the expressiveness of features of each modality" — §III D ★*내 이미지 노이즈 문제의 처방*

> "some of the captured features may have high collinearity, in which employing a gating mechanism can reduce the size of the feature space" — §III D

> "we demonstrate that these improvements are not due to network ensembling, as inputting same modality twice into Pathomic Fusion resulted in overfitting" — §V A ★*통제실험 근거*

---

## 8. 이 논문이 가리키는 다음 논문

| 참고문헌 | 왜 읽어야 하나 |
|---|---|
| [24] **Tensor Fusion Network** (Zadeh et al., EMNLP 2017) | Kronecker 융합의 **원본**. DOF도 여기서 빌림. 차원축소 변형을 찾으려면 필수 |
| [23] **Hadamard product / low-rank bilinear pooling** (Kim et al.) | ★**Kronecker의 저차원 대안.** 35,937차원 문제를 푸는 열쇠일 가능성 — 내 238명에 결정적 |
| [29] Mobadersany et al. (PNAS 2018) | concat 융합 baseline(GSCNN)의 원본. split 출처 |
| [60] 게이팅 어텐션 원본 | 구현 참조 |
| [62] Integrated Gradients (Sundararajan et al.) | 정형 피처 해석에 사용 |
| MCAT (ICCV 2021, 같은 Mahmood Lab) | **이 팀의 다음 작품.** Kronecker → co-attention으로 진화. 내 "판독지가 영상을 유도" 아이디어의 근거 |

---

## 9. 이전에 읽은 논문과의 대조 ★

| | **01 M4Survive** | **02 LLM Notes** | **03 DOF** | **04 Pathomic (본 논문)** |
|---|---|---|---|---|
| 연도/게재 | 2025 preprint | 2025 medRxiv | 2021 MICCAI | **2020 IEEE TMI** |
| 인용 | 극소 | 극소 | 99 | **659** |
| **Novelty 축** | 어떻게(구조) | **무엇을**(정보원) | 어떻게(**손실**) | 어떻게(**연산자**) |
| 중복 문제 해법 | 없음 | 해당 없음 | **직교화(분리)** | **게이팅(억제)** |
| N | 170 | 2,708 | 176 | 769 |
| 검증 | 단일 split ✗ | 10-fold+Wilcoxon ✓ | 15-fold MC ✓ | **15-fold MC, 480모델, 2암종** ✓✓ |
| concat 대비 측정 | 없음 | 해당 없음 | 0.739→0.775 | **0.781→0.826** ★ |
| 해석가능성 | 주장만 ✗ | SHAP ✓ | 없음 | **Grad-CAM+IG ✓✓** |
| 코드 | 부분 | ✗ | ✗ | **✓** |
| 새 수식 | 0 | 0 | 1 (MMO) | 0 (조합) |

**여기서 내가 배운 것:**

1. **인용 659 vs 99 vs 극소 — 순서가 게재 수준과 정확히 일치한다.** IEEE TMI 저널 > MICCAI > preprint. 그리고 **가장 많이 인용된 이 논문에 새 수식이 0개다.** 새 수식이 필수가 아니라는 결정적 증거.
2. **대신 이 논문이 압도적인 것은 검증량이다** — 480개 모델, 2개 암종, 15-fold, 6개 조합, 앙상블 통제실험, 해석 시각화, 코드 공개. **노블티가 평범해도 검증이 압도하면 저널에 간다.** 내 238명 연구가 갈 수 있는 가장 현실적인 길.
3. **"중복 문제"에 대한 처방이 두 개 나왔다** — 게이팅(04)과 직교화(03). **둘을 비교한 논문이 없고**, 내 데이터엔 최대 중복 쌍(영상↔판독지)이 있다. 여기가 내 자리.
4. **논문 쓰기 기술 하나:** contribution이 주 지표에서 안 나오면(GCN, +0.006) **부 지표에서 방어한다**(KM p-value 0.103→0.0027). 정직하되 전략적. 내 이미지 모달이 c-index를 못 올리면 **하위군 층화나 해석가능성에서 가치를 찾는 것**이 정당한 서술이다.
5. **baseline 선택이 개선폭을 3배 바꾼다** (WHO 대비 6.31% vs 최강 단일모달 대비 2.2%). 내 표에서도 **무엇 대비인지 명시**해야 하고, 남의 논문을 읽을 때 제일 먼저 확인할 것.
6. **읽을 순서의 정답은 04 → 03 → 01이었다.** 04가 게이팅·텐서융합을 만들고, 03이 그걸 가져다 MMO를 더하고, 01이 03을 baseline으로 씀. **계보를 따라 읽으면 각 논문이 "무엇을 새로 더했는지"가 선명해진다.**
