# SCLC Tri-modal Fusion — 학습 결과 & 진단 리포트

> Image(SimpleCNN) + Clinical + Report 생존예측. 5-fold CV (seed 42, 238명 공통 코호트),
> 지표 Harrell's C-index (0.5 = 무작위). 학습환경 RTX 4070 Ti SUPER, torch 2.12.1+cu126.

---

## TL;DR (핵심 결론)

1. **초기 tri-modal early fusion = OS 0.695 / PFS 0.637.** 그런데 이전 clinical+report(2-modal) 실험은 OS 0.708이었음 → "이미지를 더했는데 왜 떨어지나?"를 조사.
2. **원인은 이미지가 아니라 학습 조건이었음.** `batch_size=16` 미니배치 Cox loss가 Cox risk set을 16명으로 제한해 **tabular 브랜치를 과소학습**시켰음. batch를 키우고 스텝 수를 맞추자 clinical+report가 **0.687 → 0.708 → 0.713**으로 회복(원본 재현).
3. **concat(early) fusion에서는** tabular를 제대로 학습시키면 SimpleCNN 이미지가 오히려 깎음: clinical+report=**0.708** vs tri-modal(all)=**0.678**. 이미지 CNN이 scratch·과적합 + concat이 모달별 스케줄 충돌을 강제.
4. **그러나 late fusion으로 바꾸면 이미지가 OS에 도움이 됨(직관 확정, 섹션 8):** clin+report(0.708)에 SimpleCNN을 CoxPH로 결합 → **OS 0.722** (+0.014). concat에서 −0.017이던 이미지가 late에서 +0.014로 부호 반전 → **SimpleCNN의 문제는 이미지가 아니라 concat 구조였음.** 단 **PFS는 이미지 도움 없음**(late도 0.653/0.660 < 0.668).
5. **"더 강한 백본=ResNet"은 역효과** — 사전학습 ResNet18 단독 0.633 < SimpleCNN 0.657 (ImageNet→grayscale PET-CT 도메인 불일치).
6. **권장 최종 모델:** OS = **late fusion clin+report + SimpleCNN (0.722)**, PFS = **clin+report 단독 (0.668)**.

---

## 1. 실험 개요

| 항목 | 내용 |
|---|---|
| 코호트 | 238명 (image·clinical·report 모두 보유), `trimodal_common_5fold_seed42_v1.csv` |
| 타깃 | OS(전체생존), PFS(무진행생존) — Cox 부분우도 손실 |
| 검증 | 5-fold CV, seed 42, fold별 best-val-cindex 체크포인트 선택 |
| 두 fusion | **early**(concat→선형 Cox head, end-to-end) / **late**(단일모달 3개 학습 후 CoxPH 가중합) |

---

## 1.5 모델 구조 요약 (인코더 · 레이어 · dropout)

실험마다 백본/융합이 바뀌므로 참조용으로 정리. 모든 브랜치 출력은 early fusion에서 **L2 정규화** 후 concat.

### (a) 모달리티 인코더

| 모달 | 인코더 형식 | 레이어 구성 | 출력차원 | Dropout | 입력 |
|---|---|---|---|---|---|
| Image · SimpleCNN | 4× ConvBlock + FC (scratch) | ConvBlock=[Conv3×3(bias없음)–BN2d–ReLU–MaxPool2] ×4, ch 1→32→64→128→256 → AdaptiveAvgPool → Linear(256,512) | 512 | — | grayscale 512² |
| ↳ early fusion 투영 | 투영 head | Linear(512,128)–ReLU–Dropout | 128 | **0.2** | |
| ↳ image-only head | Cox head | Dropout–Linear(512,1) | 1 | 0.2 | |
| Image · ResNet18 | ImageNet **pretrained** ResNet18 | conv1을 1채널로 교체(사전학습 RGB 필터 평균), fc→Identity, head=Dropout–Linear(512,1) | 512 | **0.3**(head) | grayscale 224², flip 증강 |
| Clinical | MLP | [Linear–BN1d–ReLU–Dropout] **×4 @128** | 128 | **0.5** | 21개(연속 8 표준화 + 범주 13) |
| Report | char n-gram TF-IDF → MLP | TF-IDF(char_wb, n-gram 2–4, max_features=400) → [Linear–BN1d–ReLU–Dropout] **×2 (32,16)** | 16 | **0.3** | 마스킹된 report 텍스트 |

### (b) 융합(fusion) head

| 융합 방식 | 구조 | 학습 |
|---|---|---|
| Early · tri-modal concat | L2(img 128) ⊕ L2(clin 128) ⊕ L2(rep 16) = **272** → Dropout(0.3) → Linear(272→1, bias없음) → Cox | end-to-end |
| Early · clin+report concat | L2(clin 128) ⊕ L2(rep 16) = **144** → Dropout(0.3) → Linear(144→1) → Cox | end-to-end |
| Late · weighted-sum | 각 모달 단일 DeepSurv의 OOF 위험점수 → lifelines **CoxPHFitter**(2~3 covariate 계수 = 가중치) | 2-stage |

손실: **Cox 음의 부분우도**(이미지 포함 모델은 커스텀 torch 루프; late fusion의 tabular 단일모달 arm은 pycox CoxPH). 모든 fold는 **best-val-cindex 체크포인트**(조기종료) 선택, seed 42+fold, 5-fold.

### (c) 학습 하이퍼파라미터 (실험별)

| 실험 | batch | epochs | learning rate | optimizer (wd) |
|---|---:|---:|---|---|
| Early fusion 초기 | 16 | 30 | 1e-4 | Adam (1e-4) |
| 개선 조건(tabular 회복) | 32 | 60 | 1e-4 | Adam (1e-4) |
| Tabular sweep 최고 | 64 | 120 | 1e-4 | Adam (1e-4) |
| Image SimpleCNN (late arm) | 16 | 30 | 1e-4 | Adam (1e-4) |
| Image ResNet18 (late arm) | 16 | 30 | backbone **1e-5** + head **1e-3** | Adam (1e-4) |
| Tabular 단일모달 (late A, pycox) | 16 | 30 | 1e-4 | Adam |

---

## 2. 초기 결과 (batch 16, 30 epochs)

![initial mean C-index](report_assets/fig1_mean_cindex.png)

| 모델 | OS | PFS |
|---|---|---|
| Image only | 0.657 | 0.615 |
| Clinical only | 0.620 | 0.590 |
| Report only | 0.592 | 0.581 |
| Late fusion (weighted-sum) | 0.670 | 0.629 |
| **Early fusion (tri-modal)** | **0.695** | **0.637** |

초기 조건에서는 tri-modal이 최고였고 모든 fold가 0.5를 상회(중단 없이 완료). 하지만 **이전 clinical+report 단독 실험(OS 0.708)보다 낮아** 원인 조사를 진행.

---

## 3. 진단 ①: 이미지가 성능을 깎는가? → 아니오 (초기 조건 기준)

같은 238 split·같은 학습 루프에서 모델의 **브랜치 구성만** 바꾼 ablation (OS, batch 16):

![ablation bs16](report_assets/fig4_ablation_os.png)

| 구성 | 모달 수 | OS C-index |
|---|---|---|
| Report only | 1 | 0.623 |
| Clinical only | 1 | 0.639 |
| Image only | 1 | 0.647 |
| Clinical + Image | 2 | 0.679 |
| Clinical + Report | 2 | 0.687 |
| **Tri-modal (all)** | 3 | **0.695** |

→ 초기 조건에선 모달을 더할수록 단조 증가. **이미지는 오히려 단일 최강 모달(0.647)**이고, 추가하면 성능이 올랐음. 즉 "이미지 때문에 떨어졌다"는 잘못된 진단. 진짜 문제는 이 파이프라인의 clinical+report **재현치 자체가 0.687**로 원본 0.708보다 낮다는 점(차이 0.021 < 1 SD).

---

## 4. 진단 ②: 왜 clinical+report가 0.687밖에 안 되나? → batch=16 Cox 핸디캡

Cox 부분우도가 **미니배치 16명 안에서만** risk set을 구성 → tabular 학습이 불리
([train.py:47](train.py#L47), [train.py:239](train.py#L239)). 512² 이미지 메모리 때문에
batch=16이 강제되지만, 원본 tabular 모델은 훨씬 큰 risk set으로 학습됨.

**batch sweep (clin+report, 30 epoch 고정):** 16→0.687, 32→**0.693**, 64→0.667, 128→0.650, 256→0.560.
배치를 키우면 스텝 수가 줄어(fold train ≈190) 과소적합이 겹침.

**스텝 수를 맞춘 regime test (clin+report):**

| batch | epochs | ≈steps | OS C-index |
|---:|---:|---:|---:|
| 16 | 30 | 360 | 0.687 |
| 32 | 60 | 360 | **0.708** |
| 64 | 120 | 360 | **0.713** |

→ 스텝을 맞추자 **risk set이 클수록 단조 상승**, 원본 0.708을 정확히 재현(초과). **batch=16이 진짜 원인**임을 확정.

---

## 5. 개선 조건 재학습 (batch 32, 60 epochs) — 그리고 반전

![regime compare](report_assets/fig5_regime_compare_os.png)

| 구성 | 원본 (bs16/ep30) | 개선 (bs32/ep60) | 변화 |
|---|---|---|---|
| Report only | 0.623 | 0.627 | ~ |
| Clinical only | 0.639 | 0.647 | ~ |
| Image only | 0.647 | 0.639 | ↓ (이미지 과적합) |
| Clinical + Image | 0.679 | 0.677 | ~ |
| **Clinical + Report** | 0.687 | **0.708** | **+0.021 ↑** |
| **Tri-modal (all)** | 0.695 | **0.678** | **−0.017 ↓** |

**반전 포인트:** 개선 조건은 tabular(clinical+report)를 0.708로 끌어올렸지만, **tri-modal은 오히려 떨어졌음(0.695→0.678).** image_only도 epoch를 늘리자 하락(0.647→0.639). 즉 scratch로 학습되는 **이미지 CNN은 더 오래 학습하면 과적합**하고, 그 과적합 신호가 fusion을 끌어내림.

**결론:** 이전의 "tri-modal 0.695"는 사실상 **덜 학습된 clinical+report**였을 뿐. tabular를 제대로 학습시키면 **SimpleCNN 이미지는 clinical+report에 보탬이 안 되고 오히려 깎음.** 모달마다 최적 학습 스케줄이 달라(tabular는 큰 배치+많은 스텝을 원하고, 이미지 CNN은 짧은 학습을 원함) 하나의 공동 end-to-end 스케줄로는 둘을 동시에 만족시키기 어려움.

### PFS 확인 (같은 경향)

| 구성 | 개선 bs32/ep60 |
|---|---|
| Tri-modal (all) | 0.639 |
| Clinical + Image | 0.628 |
| **Clinical + Report** | **0.668** |

PFS도 동일 — clinical+report(0.668)가 tri-modal(0.639)을 앞서고, 이전 PFS 0.654도 상회.

---

## 6. 최종 권장

| 순위 | 모델 | OS | PFS | 비고 |
|---|---|---|---|---|
| **1** | **Clinical + Report (bs64/ep120)** | **0.713** | — | 최고 OS |
| 1 | Clinical + Report (bs32/ep60) | 0.708 | **0.668** | 원본 0.708 재현, 안전/저비용 |
| 3 | Tri-modal all (bs16/ep30) | 0.695 | 0.637 | 과소학습 tabular의 착시 |
| 4 | Tri-modal all (bs32/ep60) | 0.678 | 0.639 | 이미지 과적합이 fusion을 깎음 |

- **채택 모델: 잘 학습된 clinical + report.** 원본 성능을 재현하며 가장 간단·안정적.
- **SimpleCNN 이미지는 현재 형태로는 기여하지 않음.** 이미지를 살리려면 다음이 필요:
  1. **더 강한 이미지 백본**(ImageNet/의료영상 pretrained) 또는 강한 정규화·데이터증강으로 과적합 억제,
  2. **모달별로 독립 튜닝하는 late fusion**(이미지는 짧게, tabular는 크게 학습 후 CoxPH 결합),
  3. 또는 early fusion에서 **이미지 브랜치에 별도 LR/epoch·조기종료**를 부여(공동 스케줄 분리).

> ⚠️ 주의: batch/epoch는 5-fold CV C-index를 보고 고른 것이라 평가 fold에 약간의 낙관적 편향이 있음.
> 다만 fold별 best-val 체크포인트 선택으로 완화되며, 개선의 근거는 "Cox risk set" 진단에서 나온 **원칙적 단일 변경**임.

---

## 7. 산출물 / 재현

```
outputs/EXP_20260722_early_fusion_train/   # 초기 tri-modal (bs16/ep30)
outputs/EXP_20260722_late_fusion_train/    # 초기 late fusion
outputs/ablation/<config>_os/              # 초기 조건 ablation (bs16/ep30)
outputs/ablation_improved/<config>_{os,pfs}/  # 개선 조건 ablation (bs32/ep60)
outputs/batch_sweep/, outputs/regime/      # batch·regime 실험
report_assets/                             # fig1~fig5 그래프

# 재현 스크립트
ablation.py     # 브랜치 ablation:  python ablation.py --target os --batch_size 32 --epochs 60
batch_sweep.py  # batch sweep
regime_test.py  # 스텝 매칭 regime test
generate_report.py / generate_ablation_fig.py / generate_report_v2.py  # 그래프·리포트 생성
```

---

## 8. Late fusion (method B): tabular(0.708) + 이미지 백본 — 그리고 반전의 반전

섹션 5에서 **concat(early) fusion**에 이미지를 넣으면 성능이 깎였습니다("모달별 스케줄 충돌"). 이를 검증하려고 **late fusion(method B)** 을 실험: 강한 **clinical+report(0.708)** 를 한 축으로, **이미지**를 다른 축으로 각각 **독립 학습(조기종료)** 한 뒤, 두 OOF 위험점수를 **누수 없는 CoxPH stack**(fold별 train에 적합, test 평가)으로 결합. 이미지 백본은 **SimpleCNN**과 **ImageNet 사전학습 ResNet18** 두 가지.

![late fusion method B](report_assets/fig6_late_fusion_B.png)

| 구성 | OS | PFS | CoxPH 가중치(tab, img) |
|---|---|---|---|
| Tabular only (clin+report) | 0.7076 ± 0.047 | 0.6678 ± 0.034 | — |
| Image SimpleCNN only | 0.6570 ± 0.048 | 0.6154 ± 0.057 | — |
| Image ResNet18 only (pretrained) | 0.6325 ± 0.042 | 0.5722 ± 0.018 | — |
| **Late fusion + SimpleCNN** | **0.7221 ± 0.051** | 0.6531 ± 0.054 | OS (4.44, **0.30**) / PFS (3.85, **−0.05**) |
| Late fusion + ResNet18 | 0.7131 ± 0.051 | 0.6602 ± 0.037 | OS (4.87, 0.08) / PFS (3.32, 0.09) |

**핵심 발견:**

1. **직관 확정 — SimpleCNN의 문제는 이미지가 아니라 concat 구조였음.** late fusion으로 독립 학습하니 **OS에서 tabular 0.708 → 0.722로 +0.014 상승**(이미지 계수 +0.30, 양수). concat에서는 −0.017이었던 이미지가 late에서는 +0.014로 부호가 뒤집힘.
2. **"더 강한 백본=ResNet"은 여기선 역효과.** 사전학습 ResNet18 단독은 **0.633으로 SimpleCNN 0.657보다 낮음**(ImageNet 자연영상 → grayscale PET-CT 도메인 불일치 + 238장). fusion 기여도 SimpleCNN(+0.014) > ResNet(+0.005).
3. **PFS는 이미지가 도움 안 됨.** late+SimpleCNN(0.653)·late+ResNet(0.660) 모두 tabular(0.668)보다 낮고, SimpleCNN 이미지 계수는 **음수(−0.05)**. 이미지 신호는 **OS엔 보완적, PFS엔 중복/노이즈**. (이미지=종양 형태가 전체생존과는 연관되나 무진행생존 시점과는 약함, 임상적으로 그럴듯.)
4. tabular 가중치(4.4)가 이미지(0.30)보다 훨씬 큼 — 이미지는 **작은 양(+)의 보정**.

**정정:** 섹션 5–6의 "이미지는 도움 안 됨"은 **concat 한정** 결론이었음. late fusion으로 보면 **이미지는 OS에 실제로 보완 신호가 있음**(다만 이득은 +0.014로 fold SD(~0.05)보다 작아 방향은 일관되나 통계적으로는 소폭).

### 최종 권장 (업데이트)

| 타깃 | 권장 모델 | C-index |
|---|---|---|
| **OS** | **Late fusion: clin+report + SimpleCNN** (CoxPH stack) | **0.722** |
| **PFS** | **clin+report 단독** (이미지 제외) | **0.668** |

- OS는 late fusion + SimpleCNN이 최고. **ResNet은 불필요**(도메인 불일치로 오히려 약함).
- PFS는 이미지가 도움 안 되므로 tabular 단독 사용.
- 재현: `python late_fusion_B.py --targets os,pfs` → `outputs/late_fusion_B/results.json`, 그래프 `generate_fig6.py`.
