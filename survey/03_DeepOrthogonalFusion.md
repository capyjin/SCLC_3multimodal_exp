# Deep Orthogonal Fusion: Multimodal Prognostic Biomarker Discovery Integrating Radiology, Pathology, Genomic, and Clinical Data

> 한 줄 요약: 영상·병리·유전체·임상 4개 모달을 attention-gated tensor fusion으로 합치되, **모달별 임베딩이 서로 직교하도록 강제하는 MMO 손실항**을 추가해 중복을 제거함으로써 glioma 생존예측 C-index 0.788을 얻었다 (최고 단일모달 0.718 대비 p=0.023).

---

## 0. 메타

| 항목 | 값 |
|---|---|
| 연도 / 학회 | 2021 / **MICCAI 2021** (LNCS 12905) |
| 인용수 (2026-07-26 조회) | **99** |
| 링크 | [arXiv:2107.00648](https://arxiv.org/abs/2107.00648) · [Springer](https://doi.org/10.1007/978-3-030-87240-3_64) |
| 소속 | **Tempus Labs, Inc.** (산업계 단독) |
| 모달리티 | 영상(mpMRI) + 병리(H&E) + 유전체(DNA) + **임상(정형 14변수)** ★ |
| 융합 위치 | **Intermediate** (단일모달 임베딩 → tensor fusion) |
| **융합 아키타입** | **Loss** (MMO) + 연산자(tensor fusion) — **두 개 동시** |
| 태스크 | 전체생존(OS) 예측 + 위험군 층화 |
| N | **176명** (glioma, TCGA-GBM + TCGA-LGG) ← **내 238명과 같은 급** |
| 검증 방식 | **15-fold Monte Carlo CV** (20% holdout), Mann-Whitney U test |
| 주요 지표 | C-index 0.788±0.067 |
| **내 실험 적용가능성** | **즉시적용** (MMO loss) / 변형필요 (tensor fusion) |

---

## 1. 논리 전개 5단계 ★

| 단계 | 내용 | 원문 근거 |
|---|---|---|
| ① 문제 제기 | 종양 진료는 영상·분자·조직·임상 여러 스트림으로 유도된다. 각각이 종양 생물학의 **다른 측면**을 특징짓는다 | §1 ¶1 |
| ② 기존 한계 | (a) 기존 딥 멀티모달은 대부분 **biopsy 기반 모달끼리만** 융합 [14,15,16] (b) 영상이 낀 멀티모달 연구는 대부분 **correlative**(상관분석 수준) (c) late fusion 시도들은 **hand-crafted 피처 + 단순 분류기**라 복잡한 모달 간 상호작용을 못 배움 | §1 ¶3 |
| ③ 틈새 선언 | **"To our knowledge, no study to date has combined radiology, pathology, and genomic data within a single deep learning framework."** + **난제 2개를 명시적으로 선언**: ⓐ 멀티모달 데이터셋은 작다 → 융합이 **data-efficient**해야 한다 ⓑ **모달 간 강한 상관 → 중복성이 성능을 저해한다** | §1 ¶4 |
| ④ 해법 | attention-gated **tensor fusion**(모든 모달 상호작용을 외적으로 포착) + **MMO loss**(임베딩 직교화로 중복 제거) | §2 |
| ⑤ 검증 논리 | 15-fold MC CV + **모달 조합 11개 전수 비교** + **MMO on/off 대조** + naive late fusion 대비 + **반대 철학(상관 최대화)과 대결** + KM 하위군 분석 | §3–4 |

> 🔑 **③이 이 논문의 백미.** 문제를 두 개로 쪼개 선언하고(데이터 부족 / 중복성), ④에서 각각에 부품을 하나씩 대응시킨다(tensor fusion / MMO). **난제 선언 ↔ 해법 부품이 1:1로 맞물리는 구조** — 가장 배울 만한 논문 설계다.

**이 논문이 공격하는 "가상의 적":**
> "these strategies rely on **hand-crafted feature sets and simple multimodal classifiers** that likely limit their ability to learn complex prognostic interactions between modalities" — §1

> "the presence of **strongly correlated prognostic signals between modalities** can create **redundancy and hinder model performance**" — §1

> ⚠️⚠️ "**Naive late fusion ensembles (i.e., averaging unimodal risk scores) exhibit inferior performance**" — §4
> **이 문장이 내 현재 최고 모델(CoxPH 가중합 = 단일모달 위험점수 가중평균, OS 0.722)을 정확히 지목한다.** 게다가 이 논문은 그걸 **측정까지 했다**: naive late fusion 0.739 vs deep fusion 0.775. 교수님 지적의 정량적 근거.

---

## 2. Contribution

**(a) 저자 주장 — bullet 4개, 각각 굵은 리드인 구문**
1. **Deep Fusion of Radiology, Pathology, and Omics Data** — 최초로 3종 결합. 특히 **영상을 넣었을 때 성능 증가가 가장 컸다**
2. **MMO** — 모달 간 상관을 벌하는 손실항. 상관을 *최대화*하는 기존 방식 [16]보다 우월
3. **Multi-parametric Radiology FeatureNet** — CNN 딥피처 + hand-crafted radiomics를 합치는 아키텍처
4. **Independent prognostic biomarker of OS in glioma** — 기존 임상 마커(grade, IDH) 하위군 **안에서도** 추가 층화

> 💡 **M4Survive의 3-triad와 또 다른 구조: [최초성] + [새 손실함수] + [새 아키텍처] + [임상적 검증].**
> 논문마다 contribution 개수·성격이 다르다. **공통점은 "기술 1개 + 임상 1개"를 반드시 섞는다는 것.**

**(b) 내가 판단한 진짜 기여**
- **MMO loss.** 이게 유일한 진짜 새것이고, 아이디어가 명확하다: *"모달을 더해도 안 오르는 이유는 정보가 중복되기 때문이다"* → 손실로 직교성을 강제
- **11개 모달 조합 전수 표(Table 1).** 어떤 조합이 왜 좋은지 독자가 직접 검산 가능. 투명성이 논문의 설득력을 만든다
- **반대 철학과의 직접 대결.** Cheerla & Gevaert [16]는 모달 임베딩 **상관을 최대화**한다. 정반대 철학을 같은 데이터에서 재구현해 이김(0.730 vs 0.788). 강력한 수사 전략

**(c) 부풀려진 부분 / 사실 남의 것**
- ⚠️ **헤드라인 0.788은 4-modal "full fusion"이 아니라 Rad+Path+Gen(3-modal)이다.** 임상변수를 넣으면 **0.785로 떨어진다.** 논문 제목은 4개 모달을 내세우는데 최고 성적은 3개짜리
- ⚠️ **MMO의 기여에는 p-value가 없다.** 유의성 검정(p=0.023)은 "최고 융합 vs 최고 단일모달"에만 붙었다. 정작 논문 제목인 MMO의 효과(0.764 → 0.788, ±0.067)는 **한 번도 검정되지 않았다**
- ⚠️ tensor fusion [19], MMO의 착상 [20](OLÉ), VGG-19 [21], 병리/유전체 네트워크 [14] — 전부 남의 것. **MMO는 OLÉ를 멀티모달에 옮긴 것**
- 저자 스스로 ablation(Table S5)에서 "**simplified fusion module로도 강한 결과**"라고 인정 → contribution 3(아키텍처)의 필요성이 자기 실험에 의해 약화됨

---

## 3. 방법

**Step 1 — 단일모달 임베딩** (각 모달 Φm, 임베딩 크기 l1=32, Cox loss로 선학습)

**Step 2 — attention gating (Eq. 1)**
```
h*_m = a_m ⊙ h^S_m = σ(h_m^T · W_A · H^S_m) ⊙ h^S_m
```
→ **다른 모달들의 정보로** 자기 모달의 표현력을 조절 (모달 간 게이팅)

**Step 3 — tensor fusion (Eq. 2)**
```
F = [1; h*_1] ⊗ [1; h*_2] ⊗ ... ⊗ [1; h*_M]
```
→ 한 변이 (l2+1)인 M차원 하이퍼큐브. **1을 끼워넣어** 단일모달 항·쌍별 항·삼중 항이 **모두 한 텐서 안에 공존**
→ 조합폭발 때문에 l2를 줄임: **M=2→32, M=3→16, M=4→8**

**Step 4 — MMO loss (Eq. 3) ★핵심**
```
L_MMO = (1/(M·N)) · Σ_m max(1, ‖h_m‖_*) − ‖H‖_*
```
- `‖·‖_*` = **nuclear norm** (특이값의 합)
- 개별 모달 nuclear norm의 합 − 전체를 합친 것의 nuclear norm
- 두 모달의 분산이 **합쳤을 때 줄어드는**(=겹치는) 상황을 벌함. **완전 직교일 때 최소**
- `max(1, ·)`은 임베딩이 0으로 붕괴하는 것을 방지

**Step 5 — 총 손실 (Eq. 5)**
```
L = L_pl + γ · L_MMO        (최적 γ = 0.5, 단일모달 학습 시 γ=0)
```

- **학습 순서:** 단일모달 50 epochs → 융합 시 단일모달 임베딩층을 **5 epoch 동결**(융합층만 학습) → 해동 후 joint 학습 30 epochs
- 출력: sigmoid 후 **−3~+3으로 재조정**한 위험점수

**그림 1 한 문장:** 모달별 CNN/SNN이 32차원 임베딩을 만들고, 서로를 게이팅한 뒤 외적으로 모든 상호작용 텐서를 만들어 Cox head로 보내되, 임베딩끼리는 직교하도록 벌점을 준다.

---

## 4. 실험 설정 & 숫자

| 항목 | 값 |
|---|---|
| 데이터 | TCGA-GBM + TCGA-LGG (TCIA), glioma |
| N | **176** (영상 기준) |
| 영상 | Gd-T1w + T2w-FLAIR, 방사선의 7명이 병변 분할, MNI 정합, N4 보정. 96×96×3 패치를 종양 z축 4분면에서 4장 |
| + hand-crafted | 크기·형태·강도 9개 × 2 시퀀스 × 3 요약 = **56개 radiomics** |
| 병리 | 1024×1024 H&E ROI, 환자당 1–3장, 총 372장 |
| 유전체 | 80개 (변이 + CNV) |
| 임상 | **14개** (인구학 + 치료 + 조직학적 아형) |
| 검증 | **15-fold Monte Carlo CV**, 20% holdout, 중앙값 C-index |
| 검정 | **Mann-Whitney U test** |
| 백본 | VGG-19-BN (ImageNet 사전학습) / 유전체·임상은 SNN |

**Table 1 — 전수 비교 (Cox only → MMO 추가)**

| 그룹 | 조합 | Cox만 | **+MMO** | Δ |
|---|---|---|---|---|
| 단일모달 | Rad | 0.718±0.064 | – | |
| | Path | 0.715±0.054 | – | |
| | Gen | 0.716±0.063 | – | |
| | **Clin** | **0.702±0.049** | – | |
| 2-모달 | Path+Gen | 0.711±0.055 | 0.752±0.072 | **+0.041** |
| | Rad+Gen | 0.761±0.071 | 0.766±0.067 | +0.005 |
| | Rad+Path | 0.742±0.067 | 0.752±0.072 | +0.010 |
| | Gen+Clin | 0.702±0.053 | 0.703±0.052 | +0.001 |
| | Rad+Clin | 0.746±0.068 | 0.736±0.067 | **−0.010** ✗ |
| | Path+Clin | 0.696±0.051 | 0.690±0.043 | **−0.006** ✗ |
| 3-모달 | **Rad+Path+Gen** | 0.764±0.062 | **0.788±0.067** ★ | **+0.024** |
| | Rad+Gen+Clin | 0.754±0.066 | 0.755±0.067 | +0.001 |
| | Path+Gen+Clin | 0.704±0.059 | 0.720±0.056 | +0.016 |
| | Rad+Path+Clin | 0.748±0.067 | 0.741±0.067 | **−0.007** ✗ |
| 4-모달 | Rad+Path+Gen+Clin | 0.775±0.061 | 0.785±0.077 | +0.010 |

**참조 baseline**

| 방법 | C-index |
|---|---|
| Naive late fusion (위험점수 평균) Rad+Path+Gen | 0.739±0.062 |
| Naive late fusion + Clin | 0.735±0.063 |
| Cheerla & Gevaert [16] (상관 **최대화**) Rad+Path+Gen | 0.730±0.05 |
| **DOF (상관 최소화)** | **0.788±0.067** |

**★ 내가 표에서 직접 계산한 것 (논문은 말하지 않음):**

> MMO가 **Clin 없는 4개 조합**에서: +0.041, +0.005, +0.010, +0.024 → **평균 +0.020, 전부 양수**
> MMO가 **Clin 포함 7개 조합**에서: +0.001, −0.010, −0.006, +0.001, +0.016, −0.007, +0.010 → **평균 +0.001, 음수가 3개**
>
> ⚠️ **즉 MMO가 실패한 3개 조합은 전부 임상변수를 포함한다.**
> 해석: 임상변수(나이·grade·아형)는 영상·유전체와 **본질적으로 상관이 있다**. 이건 중복이 아니라 **실재하는 생물학적 연결**이다. 여기에 직교성을 강제하면 진짜 신호를 부순다.
> **→ 내 최강 모달이 clinical이므로, 이 경고가 나에게 직접 해당된다.**

**기타 밝혀진 것:**
- 영상 CNN만 0.687 / hand-crafted만 0.653 → **둘을 합치면 0.718로 최강 단일모달.** 딥피처와 전통 피처는 상보적
- "영상을 넣는 것이 단일 최대 성능 증가" (Rad 없는 조합은 대체로 0.70대 초반)
- MMO는 γ=0.5에서 최적, 11개 중 **8개 개선 / 3개 악화**

---

## 5. 비판적 검토

- **통계 △** — 15-fold MC CV는 n=176치고 훌륭하고 Mann-Whitney U까지 씀. **하지만 논문 제목인 MMO의 효과는 검정하지 않았다.** 0.764→0.788에 ±0.067이면 유의하지 않을 가능성이 높다
- **⚠️ 표준편차가 전부 겹친다** — 최고 0.788±0.067, 최저 융합 0.690±0.043. 15-fold의 중앙값 비교라 검정력은 있지만, 개별 조합 간 차이는 대부분 노이즈 수준
- **⚠️ 제목과 최고 결과의 불일치** — 제목은 4개 모달(+Clinical)인데 최고 성적은 3개. 임상변수는 넣으면 손해(0.788→0.785)
- **비교 공정성 ✓✓** — naive late fusion, 반대 철학(상관 최대화), 11개 조합 전수를 **모두 같은 프로토콜로** 비교. 이 분야에서 드물게 정직
- **tensor fusion의 차원 폭발** — M=3, l2=16이면 17³ = **4,913차원**을 176명으로 학습. 이걸 l2 축소로 겨우 막고 있다. 저자도 Table S5에서 "단순화해도 된다"고 인정
- **재현성 ✗** — 코드 공개 없음(산업계 논문). 데이터는 TCGA라 공개
- **선학습 백본 의존** — VGG-19 ImageNet. 도메인 갭 논의 없음
- **내 조건과 다른 점** — **텍스트 모달이 없다.** 유전체가 있고 나는 없다. 다만 **N이 176으로 내 238과 같은 급** → 프로토콜이 그대로 이식 가능한 유일한 논문

---

## 6. 내 SCLC 실험에 적용 ★★★ 세 논문 중 가장 직접적

### 왜 이 논문이 내 문제의 정확한 진단인가

내 [RESULTS.md](../RESULTS.md) 결론: *"concat fusion에서 이미지가 오히려 깎음 (0.708 → 0.678)"*, *"late fusion에서는 +0.014"*.
이 논문의 §1 가설: ***"모달 간 강한 상관 → 중복성이 성능을 저해한다."***

> **가설: 내 SimpleCNN 이미지 임베딩이 clinical/report와 중복이라 기여가 없다.**
> 특히 — **PET-CT 영상과 그 영상의 판독지는 정의상 최대로 중복된다.** 판독지는 말 그대로 그 영상을 서술한 문서다.
> MMO loss는 이 중복을 정량화하고(nuclear norm) 제거하는 도구다.

| 가져올 것 | 이유 |
|---|---|
| **MMO loss** ★최우선 | 구현이 **손실항 하나** (`torch.linalg.matrix_norm(H, ord='nuc')`). 새 아키텍처·새 의존성 없음. 238명 코호트에서 **위험 대비 노블티 효율이 최고**. γ=0.5부터 시작 |
| **11개 조합 전수 비교 표** | 내 모달 3개 → 단일 3 + 쌍 3 + 삼중 1 = **7조합 × MMO on/off = 14행**. 이 표 하나로 "이미지가 언제 도움이 되나"를 체계적 결과로 전환. **지금의 "이미지가 안 됨"이라는 난처한 발견이 논문의 주 결과가 된다** |
| **naive late fusion을 명시적 baseline 행으로** | 내 0.722를 버리지 말고 **"이겨야 할 기준선"**으로 표에 올린다. 이 논문이 그 프레이밍을 정당화 |
| **15-fold Monte Carlo CV** | 지금 5-fold → 검정에 쓸 숫자가 5개뿐이라 검정력이 없다. **15-fold MC CV(20% holdout)면 15개** → Mann-Whitney/Wilcoxon이 실제로 작동. n=176에서 검증된 프로토콜 |
| **Mann-Whitney U test** | C-index 분포 비교. 02번 논문의 paired Wilcoxon과 함께 채택 |
| **딥피처 + hand-crafted 결합** | 영상 CNN 0.687 + radiomics 0.653 → 합치면 0.718. **내 PET-CT에서 SUVmax·MTV·TLG 등 정량지표를 CNN 피처와 합치는 것**이 유망. 게다가 이 값들은 **판독지에서 LLM으로 뽑을 수도 있다**(02번 논문과 결합) |
| contribution 4-bullet 형식 | [최초성] + [기술] + [아키텍처] + [임상검증]. 내 논문 구조로 차용 |

| 안 가져올 것 | 이유 |
|---|---|
| **tensor fusion 외적** | M=3, l2=16 → **4,913차원 / 238명**. 이미 SimpleCNN 과적합 문제가 있는데 더 얹는 꼴. 저자도 "단순화해도 된다"고 인정 → **MMO만 떼어 쓰는 게 합리적** |
| VGG-19 ImageNet 백본 | ResNet18 ImageNet이 내 PET-CT에서 이미 실패(0.633 < SimpleCNN 0.657). 01번 논문의 **의료 FM** 경로가 낫다 |
| 유전체 모달 | 없음 |

### ⚠️ 반드시 챙길 경고

> 위 §4에서 계산한 대로 **MMO는 임상변수가 낀 조합에서 3번 실패했다.** 내 최강 모달이 clinical이다.
> → **γ를 0으로 두는 것부터 시작해 sweep**(0, 0.1, 0.5, 1.0)하고, **모달 쌍별로 직교화를 켜고 끌 수 있게** 구현할 것.
> 특히 **image↔report 쌍에만 MMO를 걸고 clinical은 제외**하는 변형이 내 데이터에선 더 나을 수 있다.
> ★ **그리고 이 변형 자체가 novelty가 된다** — 원 논문은 모든 쌍에 일괄 적용했고, "어떤 쌍에 직교화를 걸어야 하는가"는 미해결 문제다.

**당장 해볼 실험 1개:**
> 현재 tri-modal concat 모델(0.678)에 **MMO loss만 추가**하고 γ ∈ {0, 0.1, 0.5, 1.0}을 sweep.
> 가설이 맞다면 이미지 임베딩이 clinical/report와 직교화되며 **0.678 → 0.708(clin+report) 이상**으로 올라간다.
> 아키텍처는 한 줄도 바꾸지 않으므로 **MMO의 효과만 단독 분리**된다.

### 노블티 각도

- DOF는 **텍스트 모달로 검증된 적이 없다.** 영상+정형+**자유텍스트** 조합에서 직교화가 통하는지는 미해결
- **영상과 그 판독지는 최대 중복 쌍** — MMO에게 가장 할 일이 많은 설정. 이보다 좋은 테스트베드가 없다
- SCLC + PET-CT는 이 계열(전부 glioma/MRI/H&E)에서 미개척
- **쌍별 선택적 직교화**(모든 쌍이 아니라 중복된 쌍만)는 원 논문에 없는 확장

---

## 7. 인용할 문장

> "the presence of strongly correlated prognostic signals between modalities can create redundancy and hinder model performance" — §1, p.2

> "Naive late fusion ensembles (i.e., averaging unimodal risk scores) exhibit inferior performance" — §4, p.7 ⚠️ *내 현재 최고 모델을 지목*

> "these strategies rely on hand-crafted feature sets and simple multimodal classifiers that likely limit their ability to learn complex prognostic interactions between modalities" — §1, p.2

> "DOF was also found to outperform a fusion scheme that enforces correlated representations between modalities, emphasizing that the **dissimilarity** of these clinical data streams is crucial to their collective strength." — §1, p.3

> "fusion schemes must be highly data efficient in learning complex multimodal interactions" — §1, p.2 *(n=176 → 내 238명 정당화에 사용)*

---

## 8. 이 논문이 가리키는 다음 논문

| 참고문헌 | 왜 읽어야 하나 |
|---|---|
| [20] **OLÉ: Orthogonal Low-rank Embedding** (Lezama et al.) | **MMO의 원본 착상.** 직교화 손실을 구현하려면 필수 |
| [19] **Tensor Fusion Network** (Zadeh et al., 2017) | 외적 융합의 원본. 안 쓸 거면 개념만 |
| [14] **Pathomic Fusion** (Chen et al.) | 이 논문이 병리·유전체 네트워크를 그대로 빌려온 곳. **다음 읽을 논문 1순위** |
| [16] Cheerla & Gevaert (Bioinformatics 2019) | **상관을 최대화**하는 반대 철학. 대조군으로 인용 가치 |
| [15] Mobadersany et al. (PNAS 2018) | 데이터 split 출처 |

---

## 9. 이전에 읽은 논문과의 대조 ★

| | **01 M4Survive** | **02 LLM Clinical Notes** | **03 DOF (본 논문)** |
|---|---|---|---|
| 성격 | 방법론 (preprint) | 임상 (medRxiv) | **방법론 (MICCAI 정식)** |
| **Novelty 축** | 어떻게 합치나 (아키텍처) | **무엇을** 합치나 (새 정보원) | **어떻게 합치나 (손실함수)** |
| Contribution 구조 | 3-triad: 기술+성능+실용 | 임상발견+규모+프라이버시 | **4-bullet: 최초성+손실+구조+임상검증** |
| 융합 방식 | Mamba 어댑터 | concat → RSF | attention-gate + tensor fusion + **MMO** |
| N | 170 | 2,708 | **176** ← 내 238과 동급 |
| 검증 | 단일 split, 검정 없음 | 10-fold CV + Wilcoxon ✓ | **15-fold MC CV + Mann-Whitney ✓** |
| late fusion 비교 | 없음 | 해당 없음 | **있음 (0.739 vs 0.775)** ★ |
| 새 수식 | 0개 | 0개 | **1개 (MMO)** |
| 임상 모달 포함 | ✗ | ✓ (정형) | ✓ **(그리고 MMO가 여기서 실패)** |

**여기서 내가 배운 것:**

1. **세 논문 중 유일하게 "새 수식"이 있는 게 DOF고, 그게 MICCAI 정식 게재된 유일한 논문이다.** 우연이 아니다. 방법론 학회는 새 수식 1개를 원한다.
2. **그런데 그 새 수식이 손실함수 한 줄이다.** 아키텍처를 갈아엎을 필요가 없다 — 이게 238명짜리 내 연구에 결정적으로 중요하다.
3. **novelty 두 축을 겹칠 수 있다:** 02번의 **LLM 판독지 추출**(무엇을) + 03번의 **MMO 직교화**(어떻게). 둘 다 소규모 데이터에서 성립하고, 서로 독립적으로 검증 가능하다.
4. **N이 176인 MICCAI 논문이 존재한다** → 내 238명은 변명거리가 아니다. 대신 **검증 프로토콜을 15-fold MC CV로 올려** 검정력을 확보해야 한다.
5. 세 논문 모두 **naive/static fusion을 공격**한다. 내 late fusion 0.722는 **버릴 게 아니라 표의 baseline 행으로 올려** 이겨야 할 대상으로 삼는다.
