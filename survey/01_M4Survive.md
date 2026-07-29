# M4Survive: Multi-Modal Mamba Modeling for Survival Prediction

> 한 줄 요약: 얼려둔(frozen) 의료 파운데이션 모델 임베딩을 2-layer MLP로 공통 공간에 투영한 뒤, 모달리티마다 토큰 하나씩 세워 Mamba 어댑터로 융합해 glioma 생존예측 c-index 81.27을 냈다.

---

## 0. 메타

| 항목 | 값 |
|---|---|
| 연도 / 학회 | 2025 / arXiv preprint (MICCAI 스타일 8p, 미게재) |
| 인용수 (2026-07-26 조회) | 매우 적음 (preprint, 1년차) |
| 링크 | arXiv:2503.10057 · [github.com/microsoft/healthcareai-examples](https://github.com/microsoft/healthcareai-examples) |
| 소속 | Microsoft Health AI |
| 모달리티 | MRI 4종 (T1/T1PC/T2/FLAIR) + 병리 H&E — **정형 tabular·텍스트 없음** |
| 융합 위치 | Adapter (intermediate) |
| **융합 아키타입** | **FM어댑터** (frozen foundation model + 경량 adapter) |
| 태스크 | 생존예측 (Cox ranking loss) + tumor grading |
| N | 170명 (1,698 samples / 962 patients 중 radiology-pathology 쌍 보유자) |
| 검증 방식 | 단일 split 75/5/20 — **CV 아님** |
| 주요 지표 | C-index 81.27±0.56 |
| **내 실험 적용가능성** | **변형필요** (이미지 브랜치만; Mamba는 부적합) |

---

## 1. 논리 전개 5단계 ★

| 단계 | 내용 | 원문 근거 |
|---|---|---|
| ① 문제 제기 | 종양 예후는 macro(영상)와 micro(병리)가 상보적인데 단일 모달은 이걸 못 쓴다 | §1 첫 문단 |
| ② 기존 한계 | (1) 대부분 **"complete data" 가정** — 모든 환자가 모든 모달 보유. 현실엔 결측 많음 → 버리면 데이터 부족·과적합 (2) **static fusion**(평균/concat)은 동적·cross-scale 상호작용을 못 잡음 (3) modality dropout 등 대안도 불충분 | §1 3번째 문단, [2,5,13,19] / [3] / [8,14] |
| ③ 틈새 선언 | 결측에 강하면서 **동적으로** 모달 간 상호작용을 잡는 융합이 없다 | §1 |
| ④ 해법 | frozen FM → modality-specific MLP로 joint semantic space → 각 모달 = 토큰 1개 → Mamba selective SSM으로 융합 | §2 |
| ⑤ 검증 논리 | SOTA 3개 대비 우위(Table 1) + **2축 ablation**(FM 선택 / adapter 구조)으로 성능 출처를 분해(Table 2) + KM curve로 임상적 타당성(Fig 2) | §4 |

**이 논문이 공격하는 "가상의 적" — 그대로 외워둘 표현:**
> "conventional fusion strategies that **simply combine features via element-wise averaging or concatenation**" (§2 첫 줄)

> "**static** multi-modal fusion strategies have proven **insufficient** for fully capturing the dynamic and cross-scale interactions" (§1)

> ⚠️ **주의: 이 문장이 정확히 내 late fusion(CoxPH 가중합)을 겨냥한다.** 교수님 말씀이 이 문헌 전체의 합의와 일치. 내 논문 intro에서 이 프레임을 그대로 쓰거나, 반대로 방어해야 함.

---

## 2. Contribution

**(a) 저자 주장 (원문 3개 bullet)**
1. **Adaptive Multi-Modal Fusion** — adapter로 데이터 가용성 변화에 대응하며 macro/micro 상호작용 포착
2. **Enhanced Predictive Accuracy and Interpretability** — SOTA 상회 + joint latent space가 해석력 제공
3. **Scalability and Clinical Applicability** — "학습에 **15초**, 적은 CPU/GPU 메모리"

> 💡 **전형적인 MICCAI contribution 삼각형: [기술적 새로움] + [성능·해석력] + [실용성].** 내 논문에도 이 3칸을 채우면 됨.

**(b) 내가 판단한 진짜 기여**
- "의료 FM 임베딩을 **얼린 채로** 쓰고 어댑터만 학습해도 소규모(170명) 코호트에서 생존예측이 된다"는 **실증**
- FM 선택이 adapter 구조보다 성능에 훨씬 크게 기여한다는 것을 ablation으로 **분리해 보여준 것** (Table 2)

**(c) 부풀려진 부분 / 사실 남의 것**
- FM(MedImageInsight, Prov-GigaPath, UNI2-h)은 전부 **남의 모델, weight frozen**
- Mamba도 남의 것 [11]. MLP는 표준.
- → **새로운 수학이 하나도 없음.** 기여는 "이 조합을 이 문제에 처음 적용".
- Contribution 2의 "interpretability"는 **본문에서 전혀 입증 안 됨** (해석 실험 없음). 빈 주장.
- Contribution 3의 "15초"는 FM 임베딩 사전계산 시간을 뺀 수치. 어댑터만 15초인 건 당연.

> 🔑 **초보자로서 가장 큰 배움: "fancy"가 새 수식을 요구하지 않는다.** 기존 부품의 새 조합 + 그것을 정당화하는 논리 전개 + 깔끔한 ablation이면 논문이 된다.

---

## 3. 방법

**Step 1 — joint semantic space (Eq. 2)**
```
T_Rad  = MLP_Rad(⟨F_T1, F_T1PC, F_T2, F_FLAIR⟩)   → R^(4×C)
T_Path = MLP_Path(F_Path)                          → R^(1×C)
```

**Step 2 — Mamba 어댑터 (Eq. 3–4)**
```
h_{n+1} = A_d · h_n + B(t_n) · t_n
y_n     = C(t_n) · h_n            (n = 1..5)
```
- **고정(frozen):** MedImageInsight, Prov-GigaPath / UNI2-h
- **학습:** MLP 2개 (각 2-layer) + Mamba 어댑터 + Cox head
- **손실:** Cox ranking loss (Eq. 5), hazard 범위 −3~+3

**그림 1 한 문장:** 이미지 5장 → 얼린 FM 5번 통과 → MLP로 같은 차원 맞춤 → 토큰 5개 시퀀스 → Mamba → hazard.

> ⚠️ Eq. 2~4 어디에도 "이미지라서" 성립하는 항이 없음 → **구조 자체는 모달 무관**. 텍스트에도 원리상 적용 가능.

---

## 4. 실험 설정 & 숫자

| 항목 | 값 |
|---|---|
| 데이터 | TCGA + TCIA + BraTS 통합, glioma |
| N | 170 (radiology·pathology 쌍 + 생존시간·censor 보유) |
| Split | train 75% / val **5%** / test 20%, 단일 split |
| 하이퍼파라미터 | batch 16, lr 3e-4, 30 epochs |
| 하드웨어 | MLP·Transformer = 24-core CPU / Mamba = V100 |
| FM 배포 | Azure AI Model Catalog |

**Table 1 — SOTA 비교**

| 방법 | C-Index |
|---|---|
| Pathomic Fusion [5] | 77.13±1.04 |
| Deep Orthogonal Fusion [2] | 75.19±2.13 |
| Cui et al. w/o M.D [8] | 76.54±1.32 |
| **M4Survive** | **81.27±0.56** (+5.37% 상대) |

**Table 2 — Ablation 핵심만**

| 구성 | C-Index |
|---|---|
| Radiology 단독 (BiomedCLIP + MLP) | 62.31±5.73 |
| Radiology 단독 (MedImageInsight + MLP) | 72.51±3.29 |
| Pathology 단독 (UNI2-h + MLP) | **78.16±1.45** |
| MedImageInsight + UNI2-h + MLP | 79.46±0.10 |
| MedImageInsight + UNI2-h + Transformer | 80.87±1.23 |
| MedImageInsight + UNI2-h + **Mamba** | **81.27±0.56** |

**Ablation에서 실제로 밝혀진 것:**
- **FM 선택이 압도적으로 중요** — radiology를 BiomedCLIP→MedImageInsight로 바꾸면 62.31→72.51 (**+10점**)
- **adapter 구조는 미미** — MLP→Transformer→Mamba가 79.46→80.87→81.27 (**총 +1.8점**)
- 도메인 특화가 관건: 병리에 radiology FM을 쓰면(MedImageInsight×2) 76.13으로 하락

---

## 5. 비판적 검토

- **통계적 유의성 ✗** — Mamba 81.27±0.56 vs Transformer 80.87±1.23. **표준편차 구간이 겹침.** 논문의 핵심 주장("Mamba가 일관되게 우월")이 자기 표에서 지지되지 않음.
- **modality 기여도 과장** — 병리 단독 78.16 → 2모달 81.27. **영상을 4종이나 추가해서 +3점.** "상보적 정보"라는 서사에 비해 실증적 이득이 작다.
- **val = 5%** — 170명의 5% ≈ **8~9명**. 이 크기로 체크포인트를 고른 c-index는 신뢰구간이 매우 넓다.
- **단일 split** — 5-fold CV 아님. 내 실험(5-fold)보다 검증이 약함. **81.27과 내 0.722를 직접 비교하면 안 됨.**
- **결측 대응 주장 vs 실제** — intro에서 "complete data 가정"을 강하게 비판해놓고, 정작 자기들은 "consisting **exclusively of complete** glioma tumor data"로 170명을 골라 씀 (§3). **논리 모순.** ← 이런 걸 찾아내는 게 리뷰어의 눈.
- **재현성** — MedImageInsight는 Azure 게이트 뒤. 자유롭게 받을 수 있는 BiomedCLIP은 자기 표에서 제일 약한 선택지.
- **내 조건과 다른 점** — 정형 tabular 모달리티가 **아예 없음**. 내 최강 모달(clinical 21변수)에 대응물이 없다. 텍스트도 없음.

---

## 6. 내 SCLC 실험에 적용

| 가져올 것 | 이유 |
|---|---|
| **frozen 의료 FM + 2-layer MLP** 로 SimpleCNN 대체 | 내 RESULTS.md의 "SimpleCNN scratch 과적합" 진단에 대한 정확한 처방. 238명으로 CNN 처음부터 학습은 이 문헌 기준 애초에 하면 안 되는 설정 |
| **2축 ablation 표 구성** (인코더 선택 × 융합 구조) | 성능 출처를 분해해 보여주는 서술 방식. 내 논문 표 구조로 그대로 차용 |
| intro의 "static fusion" 프레임 | 내 late fusion을 방어하거나 넘어서기 위해 반드시 다뤄야 할 논점 |

| 안 가져올 것 | 이유 |
|---|---|
| **Mamba 어댑터** | 논문은 토큰 5개, 나는 3개(image/clinical/report). 길이 3에 selective SSM은 무의미. 게다가 순환이라 **순서 의존적** — 순서 없는 모달에 잘못된 inductive bias. 논문 내 우위도 std 안에서 겹침. `mamba-ssm` CUDA 빌드 의존성까지 |
| 단일 75/5/20 split | 내 5-fold CV가 더 엄격. 후퇴할 이유 없음 |
| Cox ranking loss | 이미 동일하게 쓰는 중. 새로 얻을 것 없음 |

**당장 해볼 실험 1개:**
> SimpleCNN → frozen 의료 FM 임베딩 + 2-layer MLP로 교체하고, **late fusion OS 0.722를 넘는지만** 본다.
> 판독지 인코더(TF-IDF) 교체는 **동시에 건드리지 않는다** — 두 변수를 같이 바꾸면 원인 귀속 불가.
> 리스크: PET-CT를 학습한 의료 FM이 없음 (논문은 MRI+H&E). ImageNet ResNet18이 이미 0.633으로 실패한 전례 있음(다만 ImageNet≠의료 FM이므로 기각 근거는 아님).

---

## 7. 인용할 문장

> "conventional fusion strategies that simply combine features via element-wise averaging or concatenation" — §2, p.3

> "static multi-modal fusion strategies have proven insufficient for fully capturing the dynamic and cross-scale interactions between radiology and pathology data" — §1, p.2

> "existing methods predominantly operate under a 'complete data' paradigm, wherein every patient record is assumed to have all modalities available" — §1, p.2

> "models that are pre-trained on domain-specific data better capture the intrinsic patterns and clinical markers" — §4.2, p.7

---

## 8. 이 논문이 가리키는 다음 논문

| 참고문헌 | 왜 읽어야 하나 |
|---|---|
| [5] Pathomic Fusion (IEEE TMI 2020) | Table 1의 최강 baseline. **융합 연산자 자체를 기여로 만든** 원조 |
| [2] Deep Orthogonal Fusion (MICCAI 2021) | **clinical 데이터를 포함한** 유일한 baseline. 내 모달 구성에 가장 가까움 |
| [8] Cui et al. (MICCAI 2022) | 결측 모달리티 정면 대응 |
| [11] Mamba (Gu & Dao 2023) | 어댑터 원본 (읽을 필요 낮음 — 내 토큰 수에선 안 씀) |
