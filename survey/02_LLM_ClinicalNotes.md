# Large Language Models Improve Cancer Survival Prediction Using Real-World Clinical Notes

> 한 줄 요약: NSCLC 2,708명의 실제 EHR 노트를 자체 호스팅 Llama 4 Scout로 **zero-shot 구조화 추출**해 얻은 해석가능한 피처를 Random Survival Forest에 넣자, TNM 병기 단독(0.64)은 물론 **텍스트 임베딩(0.69)보다도 우수한 C-index 0.72**를 얻었다.

---

## 0. 메타

| 항목 | 값 |
|---|---|
| 연도 / 학회 | 2025-08-19 / **medRxiv preprint** (peer review 미통과, 임상저널 투고형) |
| 인용수 | 미확인 (S2 API 레이트리밋 / 1년차 preprint라 낮을 것) |
| 링크 | [doi.org/10.1101/2025.08.17.25333835](https://doi.org/10.1101/2025.08.17.25333835) |
| 소속 | University Hospital Essen (Kleesiek lab) + LMU Munich + Charité, 독일 |
| 모달리티 | **텍스트(EHR 노트·퇴원요약) + 정형 임상변수** — **영상 없음** |
| 융합 위치 | Early (피처 concat → RSF) |
| **융합 아키타입** | **해당 없음 — 융합이 novelty가 아님** ★ |
| 태스크 | 전체생존(OS) 예측 + 위험군 재분류 |
| N | NSCLC **2,708** + 대장암 814 |
| 검증 방식 | **10-fold CV**, paired Wilcoxon signed-rank test |
| 주요 지표 | C-index: 0.72 (NSCLC) / 0.70 (대장암) |
| **내 실험 적용가능성** | **즉시적용** (판독지 인코딩 전략) |

---

## 1. 논리 전개 5단계 ★

| 단계 | 내용 | 원문 근거 |
|---|---|---|
| ① 문제 제기 | EHR엔 의사가 직접 쓴 방대한 비정형 텍스트가 있고, 특히 **host-specific 정보**(신체 상태, 증상)는 예후에 큰 영향을 주는데 정형 필드엔 없다. 임상 노트가 **유일한 출처** | Intro ¶1 |
| ② 기존 한계 | (a) 현행 예후 시스템은 정형 데이터만 씀 (b) **"멀티모달 모델조차 텍스트를 대부분 무시한다"** [1–7] (c) 전통적 텍스트 추출은 노동집약·오류多, 전문가 검수 필요 (d) 기존 LLM 연구는 **소규모·비현실 데이터셋** [13,14] | Intro ¶1 |
| ③ 틈새 선언 | **대규모 real-world** 코호트에서, **자체 호스팅**(=프라이버시 충족) LLM으로, **설명가능하게** 예후를 개선한 사례가 없다 | Intro ¶2 |
| ④ 해법 | Llama 4 Scout zero-shot 추출(전이부위 + 동반질환 ICD + **7개 PCI** + **2개 종합점수**) / 별도로 BGE-M3 임베딩 → 둘 다 RSF에 투입 | Methods |
| ⑤ 검증 논리 | **4단계 계단식 비교**(병기 → baseline → +임베딩 → +LLM) + 10-fold CV + Wilcoxon 검정 + **두 번째 암종으로 재현** + SHAP + KM + Sankey 재분류 | Results |

**이 논문이 공격하는 "가상의 적":**
> "current prognostic systems rely on structured data, this data source remains **largely untapped**" — Intro

> "Even current multimodal models... **mostly neglect the wealth of information hidden in text data**" — Intro

> ⚠️ "most approaches are applied on **small, non-real-world datasets** limiting their generalizability" — Intro
> **이 문장은 내 연구(238명)를 겨냥한다.** 내 논문에서는 n의 작음을 정면으로 다뤄야 함 (단일기관 정밀 큐레이션·희귀암·5-fold CV·유의성 검정으로 방어).

---

## 2. Contribution

**(a) 저자 주장** — 임상논문 형식이라 bullet이 없음. Abstract/Discussion에서 추출:
1. 자체 호스팅 LLM이 **task-specific 학습 없이(zero-shot)** 임상 노트에서 예후 인자를 뽑아낸다
2. 이를 넣으면 TNM 단독 대비 OS 예측이 유의하게 향상 (0.72 vs 0.64 / 0.70 vs 0.59)
3. **텍스트 임베딩 모델보다도 우수**
4. 4개 위험군으로 재분류 → NSCLC 61.4%, 대장암 68.3% 재분류
5. host-factor가 TNM의 예후 영향을 **modulate** 한다 (SHAP)

> 💡 **M4Survive의 "기술 삼각형"과 완전히 다른 구조.** 여기 contribution은 전부 [임상적 발견] + [규모] + [실용성(프라이버시)]. 아키텍처 얘기가 **한 줄도 없다.**

**(b) 내가 판단한 진짜 기여**
- **LLM 추출 vs 텍스트 임베딩을 같은 코호트에서 head-to-head 비교한 것.** 텍스트를 쓰는 두 경로를 정면 대결시킨 연구가 드묾 — 이게 핵심 새로움
- 병원 네트워크 내부 self-hosted 배포 실증 (프라이버시 = 임상 도입의 실제 장벽)
- 두 번째 암종에서 패턴 재현 (SHAP 순위 상관 ρ=0.758)
- **임베딩 surrogate probing**(Fig 5B): 블랙박스 임베딩에서 구조화 피처를 역예측해 "무엇이 인코딩됐는지" 밝힌 것 — 방법론적으로 영리

**(c) 부풀려진 부분 / 사실 남의 것**
- ⚠️ **"surpassed models using text embeddings"는 절반만 사실.** NSCLC에서만 유의(0.72 vs 0.69, p=0.014). **대장암에선 0.70 vs 0.68, p=0.275로 유의하지 않음.** Abstract는 이 단서를 뺐다.
- ⚠️ 대장암을 "validation cohort"라 부르지만 **같은 병원, 같은 문서 관행, 다른 질환**이다. 외부 검증(external validation)이 아니다.
- ⚠️ **Abstract 61.4% vs 본문 69.6%**(915/1315) 불일치 — 서로 다른 값을 같은 이름으로 부름
- LLM(Llama 4 Scout), 임베딩(BGE-M3), 모델(RSF), 해석(SHAP) **전부 남의 것.** 새 방법 0개
- 대장암에서 baseline 0.66 vs +임베딩 0.68은 **p=0.193으로 유의하지 않음** — 텍스트의 이득이 암종에 따라 흔들림

---

## 3. 방법

**경로 1 — LLM zero-shot 추출** (Llama 4 Scout: 활성 17B / 총 109B, 16 experts)
```
EHR 노트 + 퇴원요약  --[균일 프롬프트, JSON 출력]-->
  · 전이 부위 (리스트)
  · 동반질환 (ICD-10 코드 리스트)
  · 7개 PCI (binary): 거동장애 / 통증 / B증상 / 고위험 / 호흡곤란
                      / 복잡한 경과 / 이상 신체소견
  · 종합점수 2개 (0-100): Physical Condition Score, Survival Score
```

**경로 2 — 텍스트 임베딩**
```
raw text --[BGE-M3, multilingual, chunk 512 / stride 32]--> 1024-dim --[PCA]--> 30
```

**융합 & 모델**
- 노트와 퇴원요약이 둘 다 있으면 → 범주형은 **logical OR**, 연속형은 **평균**
- Random Survival Forest (1,000 trees, min_split=5, min_leaf=5), 10-fold CV
- 5년 위험점수: T_ref=1825일에서 누적위험 → 2.5~97.25 백분위 자르고 equal-width 4구간

- **고정:** LLM·임베딩 모델 (학습 전혀 안 함, zero-shot)
- **학습:** RSF만
- **배포:** 4× NVIDIA H100, HuggingFace TGI, context 8k, 원내 KITE 플랫폼

**그림 1 한 문장:** 치료 전 임상 텍스트를 LLM으로 구조화하거나 임베딩한 뒤, 정형 변수와 합쳐 RSF로 생존을 예측하고 위험군으로 재분류한다.

---

## 4. 실험 설정 & 숫자

| 항목 | NSCLC | 대장암 |
|---|---|---|
| N | 2,708 | 814 |
| 기간 | 2017–2025 | 2010–2025 |
| 중앙 연령 | 67 (24–91) | 66 (22–97) |
| 여성 | 41.9% | 48.2% |
| 사망 이벤트 | 996 (censored 1,712) | 510 (censored 304) |
| Stage IV | 48.6% | 57.4% |
| EHR 노트 보유 | 97.1% | 96.8% |
| 퇴원요약 보유 | 71.9% | 58.2% |

**핵심 결과 — 계단식 비교 (10-fold CV median C-index)**

| 모델 | NSCLC | 대장암 |
|---|---|---|
| 병기(TNM) 단독 | 0.64 | 0.59 |
| Baseline (병기+나이+성별[+조직형]) | 0.64 | 0.66 |
| Baseline + **텍스트 임베딩** | 0.69 (p=0.002 vs 병기) | 0.68 (p=0.193 vs baseline ✗) |
| Baseline + **LLM 추출** | **0.72** (p=0.014 vs 임베딩) | **0.70** (p=0.275 vs 임베딩 ✗) |

**Cox 분석 (NSCLC, Table 2) — 단변량 HR 상위**

| LLM 추출 변수 | 단변량 HR | 다변량 HR |
|---|---|---|
| 거동장애 (Mobility Impairment) | **2.64** | **1.78** ✓ |
| 복잡한 경과 | 2.19 | 1.63 ✓ |
| 이상 신체소견 | 2.01 | 1.31 ✓ |
| B증상 | 1.99 | 1.02 ✗ |
| 통증 | 1.62 | 1.07 ✗ |
| 호흡곤란 | 1.56 | 1.07 ✗ |
| 고위험 상태 | 1.41 | **0.71** ⚠️ |

**Ablation/분석에서 실제로 밝혀진 것:**
- **LLM 추출 > 임베딩 > 정형 baseline** — 텍스트를 "이해해서 뽑는" 게 "벡터로 바꾸는" 것보다 낫다
- SHAP: 병기가 여전히 1위지만 **LLM 종합점수 2개가 top-4에 진입**. host factor가 tumor factor를 보정
- Fig 5B surrogate probing: 임베딩만으로 PCI를 balanced accuracy 0.56~0.75로 역예측 가능 → **임베딩도 같은 정보를 담고 있으나 덜 효율적으로 담고 있음**
- Stage IV에서 이득이 가장 큼 — 병기가 못 나누는 큰 집단을 갈라줌

---

## 5. 비판적 검토

- **통계 ✓ (모범적)** — 10-fold CV + **paired Wilcoxon signed-rank test**로 fold별 쌍 비교. M4Survive(단일 split, std만 보고)보다 훨씬 엄격. 다만 다중비교 보정은 없음
- **비교 공정성 ✓** — 동일 RSF·동일 fold에서 피처 세트만 바꿔 비교. 계단식 설계가 깔끔
- **⚠️ 핵심 주장 일부가 검정에서 안 살아남음** — "LLM > 임베딩"이 대장암에서 p=0.275. Abstract는 이 단서 없이 단정
- **⚠️ 외부 검증 없음** — "validation cohort"는 같은 병원 다른 질환. 타 기관 문서 관행에서의 일반화는 미검증 (저자도 limitation에서 인정)
- **⚠️ 고위험 상태 HR 부호 역전** (1.41 → 0.71) — 다변량에서 방향이 뒤집힘은 **강한 공선성/교란**의 신호. 논문은 수치만 적고 설명하지 않음
- **⚠️ Table 2 오타 의심** — Squamous 다변량 CI (0.88–1.63)가 Adenosquamous 행과 동일. 복붙 오류로 보임
- **PCA 30 (unsupervised)** — 1024→30으로 줄이며 예후 방향을 버렸을 수 있음. LLM 경로가 이긴 이유의 일부가 여기일 가능성 (공정한 비교인지 의심 여지)
- **Survival Score의 순환성** — "생존을 예측하라"고 LLM에 물어 얻은 점수를 생존예측 피처로 씀. label leakage는 아니지만(zero-shot) 개념적으로 tautological
- **재현성** — 데이터 비공개(요청 시 제한적). 프롬프트는 Supp. Tab. 4에 공개
- **내 조건과 다른 점** — n이 **11배**(2,708 vs 238). 영상 모달 없음. 텍스트가 EHR 노트/퇴원요약(서술형)이지 **영상 판독지가 아님**

---

## 6. 내 SCLC 실험에 적용 ★★ 이번 논문 최대 수확

| 가져올 것 | 이유 |
|---|---|
| **BGE-M3 임베딩** | **multilingual.** 독일어 임상 텍스트에서 작동 검증됨 → **한영 혼용 한국어 판독지에 직접 적용 가능.** 영어 biomedical 인코더가 "~로 생각됨 / 감별이 필요함" 같은 확신도 표현을 흘리는 문제의 해답 |
| **LLM zero-shot 구조화 추출** ★최우선 | 내 판독지엔 **maxSUV 수치가 명시**돼 있음 ("hypermetabolic mass **(13.9)**"). TF-IDF char n-gram은 숫자를 수치로 표현하지 **못한다.** LLM은 SUV값·전이부위·흉막침습·LN station·확신도를 구조화 필드로 뽑을 수 있음 |
| **차원 축소 효과** ★결정적 | 현재 TF-IDF **400차원 / 238명 = 최악의 비율**. LLM 추출은 **15~20개 해석가능 피처**. 차원이 20배 줄면서 해석력은 오히려 상승 — 소규모 코호트에 정확히 필요한 것 |
| **계단식 비교 표** | 병기 → baseline → +텍스트 → +LLM. 각 단계 기여를 분리 제시하는 서술 문법. 내 논문 Table 1로 그대로 차용 |
| **paired Wilcoxon signed-rank test** | fold별 C-index 쌍 비교. **지금 내 RESULTS.md엔 유의성 검정이 전혀 없다.** 0.708 vs 0.722가 유의한지 아무도 모름 → 즉시 보완 필요 |
| **SHAP + Sankey 재분류 그림** | 임상 저널용 figure 문법. "성능 표" 말고 "임상적 의미" 그림 |
| **limitation 서술 방식** | 후향적 real-world 연구의 한계를 정직하게 나열하되 "이것이 곧 실제 진료의 현실"로 전환하는 화법 |

| 안 가져올 것 | 이유 |
|---|---|
| **4× H100 self-hosted Llama 4 Scout (109B)** | 내 환경(RTX 4070 Ti SUPER)에 불가. 더 작은 모델(Qwen2.5, Llama 3.1 8B 등) 또는 원내 서버 필요 |
| **환자 텍스트를 외부 API로 전송** | ⚠️ **IRB/개인정보 문제.** 이 논문이 self-hosting을 강조한 이유가 정확히 이것. 반드시 로컬 실행 |
| PCA 30 | unsupervised라 예후 방향 손실 위험. 238명에선 더 위험 |
| RSF 단독 | 내 이미지 모달과 결합하려면 딥러닝 필요. 단 tabular+text만이면 238명엔 RSF가 오히려 나을 수 있음 — 비교 대상으로 넣을 가치는 있음 |

**당장 해볼 실험 1개:**
> 판독지를 **로컬 LLM으로 구조화 추출**(maxSUV 값, 전이부위, 흉막/종격동 침습, LN station, 증상 기술, 확신도) → 15~20개 피처 → 기존 clinical과 concat → **현재 clin+report 0.708과 비교.**
> TF-IDF 400차원을 대체하는 것이므로 이미지는 건드리지 않는다(변수 하나만 변경).
> 동시에 **BGE-M3 임베딩 경로도 같이 돌려** 이 논문처럼 head-to-head 비교표를 만든다.

**왜 이게 novelty가 되나:**
- 이 논문은 EHR 노트를 썼지 **영상 판독지**를 쓰지 않았다 → 판독지 특유의 정량정보(SUV)를 LLM으로 뽑는 건 미개척
- 대상이 NSCLC/대장암 → **SCLC는 다뤄지지 않음** (희귀·예후 불량·연구 적음)
- 한국어-영어 혼용 임상 텍스트에서의 검증 사례 부족
- 여기에 **영상까지 더한 tri-modal**은 이 논문이 아예 안 한 영역

---

## 7. 인용할 문장

> "Even current multimodal models, which aim to incorporate comprehensive patient information and show great promise in cancer treatment guidance and biomarker discovery, mostly neglect the wealth of information hidden in text data." — Introduction

> "most approaches are applied on small, non-real-world datasets limiting their generalizability" — Introduction ⚠️ *내 연구가 이 비판의 사정권*

> "LLM-derived host-level characteristics play a modulatory role, contextualizing tumor-specific information" — Discussion

> "While embedding models allow straightforward use of unstructured text, LLMs can extract interpretable information at scale, transforming fragmented narratives into actionable features" — Discussion

---

## 8. 이 논문이 가리키는 다음 논문

| 참고문헌 | 왜 읽어야 하나 |
|---|---|
| [15] **BGE-M3** (Chen et al., arXiv:2402.03216) | 내가 쓸 임베딩 모델 원본. **한국어 지원 범위 확인 필수** |
| [19] SHAP (Lundberg & Lee) | 해석 figure의 표준 |
| [24–26] LLM 정보추출 검증 연구 | 추출 정확도를 어떻게 검증하는지 — 내가 한국어로 할 때 필요 |
| [27–29] LLM clinical risk inference | "LLM이 임상 위험을 추론한다"는 주장의 근거 계보 |

---

## 9. 이전 논문과의 대조 ★

| | **M4Survive** (01) | **본 논문** (02) |
|---|---|---|
| 성격 | 방법론 논문 (MICCAI형) | **임상 논문** (medRxiv→임상저널형) |
| **Novelty 축** | **아키텍처** (융합 구조) | **데이터·피처 소스** (새 정보를 끌어옴) |
| Contribution 구조 | 기술새로움 + 성능 + 실용성 | 임상발견 + 규모 + 프라이버시 |
| 융합 방식 | Mamba 어댑터 (fancy) | **그냥 concat → RSF (전혀 fancy 아님)** |
| N | 170 | 2,708 |
| 검증 | 단일 split, 유의성 검정 없음 | 10-fold CV + Wilcoxon **✓** |
| 새 수식 | 0개 | 0개 |

> 🔑 **가장 중요한 교훈:** 이 논문의 융합은 **교수님이 "뻔하다"고 한 바로 그 방식**(피처 concat)이다. 그런데도 좋은 논문이다.
> → **노블티는 두 축 중 하나면 된다: [어떻게 합치나] 또는 [무엇을 합치나].**
> M4Survive는 앞을 택했고, 이 논문은 뒤를 택했다.
> **내 상황에선 뒤가 유리하다** — 238명으로 화려한 구조를 학습시키긴 어렵지만, "한국어 PET-CT 판독지에서 LLM으로 SUV·침습·확신도를 뽑아 SCLC 예후에 쓴다"는 **아무도 안 한 것**이고 데이터가 작아도 성립한다.
> 이상적으로는 **둘 다**: 새 피처(LLM 추출) + 그 피처를 영상과 합치는 새 구조(co-attention 등).
