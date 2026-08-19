# 모달리티 조합별 최고 성능 모델 정리

**공통 조건**: 238명, 5-fold CV(seed 42), Cox 부분우도, Harrell C-index, epochs 60/30,
batch 32/16. 전부 `brain_meta` 누수 수정 후(2026-08-02 이후) 수치다.
※ 표에 "수정 전 참고값"이라 적힌 칸은 예외 — brain_meta 수정 후로 다시 돌리지 않은 것이며,
현재 파이프라인 기본값(`fix_brain_meta=True`)으로 재실행하면 값이 바뀔 수 있다.

---

## 한눈에 보는 표

| 구분 | 조합 | 최고 모델 | OS | PFS |
|---|---|---|---|---|
| 단일모달 | clinical | MLP | 0.6388 | 0.6223 |
| 단일모달 | 판독지 | RadBERT(frozen) | **0.6685** | **0.6354** |
| 단일모달 | 영상 | SimpleCNN+DeepSurv | 0.6570 | 0.6154 |
| 2-모달 | clinical+판독지 | concat(TF-IDF) | 0.7057 | **0.6696** |
| 2-모달 | clinical+영상 | concat | (빈칸 — 수정 후 미측정) | (빈칸) |
| 2-모달 | 영상+판독지 | — | (빈칸 — 실험 자체가 없음) | (빈칸) |
| 3-모달 | clinical+판독지+영상 | late fusion 2-way(TF-IDF) | **0.7143** | **0.6621** |
| 3-모달 | clinical+판독지+영상 | late fusion 3-way(각각 독립) | 0.6811 | 0.6313 |

**굵은 글씨 = 그 줄의 최고값.** 3-모달 late fusion(OS 0.7143)이 현재 이 프로젝트의 최종 채택 모델(OS 기준)이고, PFS는 2-모달 clinical+판독지(0.6696)가 최종 채택 모델이다(영상이 PFS엔 도움 안 됨).

---

## 1. 단일모달(uni-modal)

### 1-1. clinical 단독

| | |
|---|---|
| **성능** | OS 0.6388 ± 0.044 / PFS 0.6223 ± 0.037 |
| **모델 구조** | 임상변수 21차원 → Linear→BatchNorm→ReLU→Dropout(0.5) 를 4번 반복(128→128→128→128) → Cox head(Linear, bias 없음) |
| **코드 파일** | `실험3_모달리티_절제실험/ablation.py` (`CONFIGS["clin_only"]`) 또는 `실험6_판독지_인코더_비교/exp_radbert_fusion.py` 실행 시 부산물로 같이 나옴 |
| **실행 명령** | `python 실험3_모달리티_절제실험/ablation.py --target os --configs clin_only --epochs 60 --batch_size 32` |
| **fold별 (OS)** | `[0.6291, 0.6558, 0.6544, 0.5605, 0.694]` |

### 1-2. 판독지 단독 — **RadBERT가 최고**

| | TF-IDF (기존) | **RadBERT (최고)** |
|---|---|---|
| **성능 OS** | 0.6268 ± 0.047 | **0.6685 ± 0.037** |
| **성능 PFS** | 0.6094 ± 0.038 | **0.6354 ± 0.042** |
| **모델 구조** | char n-gram(2~4자) 400차원 → Linear→BN→ReLU→Dropout(0.3) 2번(400→32→16) → Cox head | `StanfordAIMI/RadBERT`(frozen) 768차원 mean-pooling → 위와 같은 Cox head. **전처리 없이(SVD·표준화 X)** 원본 768차원 그대로 사용, 한글은 446항목 사전으로 영어 치환 |
| **코드 파일** | `실험3_모달리티_절제실험/ablation.py` (`CONFIGS["report_only"]`) | `실험6_판독지_인코더_비교/exp_bert_text.py` |
| **실행 명령** | `python 실험3_모달리티_절제실험/ablation.py --target os --configs report_only` | `python 실험6_판독지_인코더_비교/exp_bert_text.py --target os --arms bert_ko2en --model_config report_only --no_svd --no_scale` |

⚠️ **다만 최종 채택된 판독지 인코더는 TF-IDF다.** 단독으로는 RadBERT가 이기지만, clinical과 합치면(§2-1) 그 우위가 사라진다 — 자세한 이유는 `실험6_판독지_인코더_비교/REPORT_ENCODER_FINAL.md` §2 참고.

### 1-3. 영상 단독 — **DeepSurv(SimpleCNN)가 최고**

| | **DeepSurv (최고)** | 고전 CoxPH |
|---|---|---|
| **성능 OS** | **0.6570 ± 0.048** | 0.6418~0.6557 (변형별, 전부 DeepSurv 이하) |
| **성능 PFS** | **0.6154 ± 0.057** | 0.5901~0.6315 |
| **모델 구조** | 4× ConvBlock(Conv3×3–BN–ReLU–MaxPool2, 1→32→64→128→256) → AdaptiveAvgPool → Linear(256,512) → Dropout → Linear(512,1), Cox 부분우도로 end-to-end 학습 | 같은 백본으로 특징만 뽑고 PCA/능형회귀 등으로 축소 후 lifelines CoxPH로 위험점수 산출(12가지 변형 다 시도) |
| **코드 파일** | `실험1_기본융합_early_late/late_fusion_tab_image.py` (`get_image_oof_simplecnn`) | `실험8_영상단독_고전CoxPH/exp_image_cph.py` |
| **비고** | — | 2026-08-03에 "고전 CoxPH로 바꾸면 나아지나?"를 검증했으나 **전부 DeepSurv보다 낮거나 같음**. 오히려 학습 안 한 랜덤 CNN과 성능이 구분 안 됨(§ `실험8_영상단독_고전CoxPH/RESULTS_image_cph.md`) — 이 브랜치가 실제로 종양 특징을 배운 건지에 대한 의문이 남아있음 |

---

## 2. 2-모달(bi-modal)

### 2-1. clinical + 판독지 — **완성된 조합, 최종 채택**

| | **TF-IDF (채택, 최고)** | RadBERT |
|---|---|---|
| **성능 OS** | 0.7057 ± 0.048 | **0.7153** ± 0.033 |
| **성능 PFS** | **0.6696 ± 0.041** | 0.6456 ± ? |
| **모델 구조** | clinical 브랜치(128×4) + 판독지 브랜치(400→32→16) → L2 정규화 후 concat(144차원) → Cox head | 판독지 인코더만 RadBERT(768차원, 전처리 없음)로 교체, 나머지 구조 동일 |
| **코드 파일** | `실험3_모달리티_절제실험/ablation.py` (`CONFIGS["clin_report"]`) | `실험6_판독지_인코더_비교/exp_radbert_fusion.py`, `실험6_판독지_인코더_비교/exp_radbert_full.py` |
| **실행 명령** | `python 실험3_모달리티_절제실험/ablation.py --target os --configs clin_report` | `python 실험6_판독지_인코더_비교/exp_radbert_full.py --target os` |

**왜 OS에서 더 높은 RadBERT를 안 쓰고 TF-IDF를 채택했나** — PFS에서 TF-IDF(0.6696)가 RadBERT(0.6456)를 크게 앞서고, 두 타깃을 같이 보면 TF-IDF가 낫다고 판단했다. 상세 근거는 `실험6_판독지_인코더_비교/REPORT_ENCODER_FINAL.md` §4 참고.

### 2-2. clinical + 영상 — **빈칸 (재측정 필요)**

| | |
|---|---|
| **성능** | **(빈칸)** — `brain_meta` 수정 후로 다시 돌린 적 없음 |
| **수정 전 참고값** | OS 0.6774 / PFS 0.6275 (2026-08-02 이전, 그대로 신뢰하면 안 됨) |
| **모델 구조** | clinical 브랜치(128×4) + 영상 브랜치(SimpleCNN→128) → L2 정규화 후 concat → Cox head |
| **코드 파일** | `실험3_모달리티_절제실험/ablation.py` (`CONFIGS["clin_image"]`) — 코드는 있고 실행만 안 함 |
| **재현 명령** | `python 실험3_모달리티_절제실험/ablation.py --target os --configs clin_image --epochs 60 --batch_size 32` 그리고 `--target pfs` |

### 2-3. 영상 + 판독지 — **빈칸 (실험한 적 자체가 없음)**

| | |
|---|---|
| **성능** | **(빈칸)** — 이 조합 자체를 실험 설정에 넣은 적이 없다 |
| **비고** | `실험3_모달리티_절제실험/ablation.py`의 `CONFIGS` 딕셔너리에 `all`(3모달) / `clin_report` / `clin_image` / `clin_only` / `report_only` / `image_only` 6개만 있고, "영상+판독지"(clinical 제외) 조합은 정의돼 있지 않다 |
| **재현하려면** | `실험3_모달리티_절제실험/ablation.py`의 `CONFIGS`에 `dict(use_image=True, use_clinical=False, use_report=True)` 항목을 추가해야 함(코드 수정 필요, 1줄) |

---

## 3. 3-모달(tri-modal)

### 3-1. concat(조기융합) — 참고용, 채택 안 됨

| | |
|---|---|
| **성능** | **(빈칸)** — `brain_meta` 수정 후 미측정 |
| **수정 전 참고값** | OS 0.6775 / PFS 0.6390 (2026-08-02 이전) |
| **모델 구조** | clinical(128) + 영상(128) + 판독지(16) 세 브랜치를 L2 정규화 후 이어붙여(272차원) 하나의 Cox head로 학습 |
| **코드 파일** | `실험3_모달리티_절제실험/ablation.py` (`CONFIGS["all"]`) |
| **비고** | 이 방식은 영상이 최종 위험점수의 70~80%를 독점해서 성능이 오히려 깎이는 문제가 있었다(`RESULTS.md` §9). 그래서 아래 late fusion으로 대체됐고, 최종 후보에서 제외됐다 |

### 3-2. late fusion 2-way (tabular=clinical+판독지 하나로 학습 + 영상) — **최종 채택 모델 (OS 기준)**

| | **TF-IDF (채택, 최고)** | RadBERT |
|---|---|---|
| **성능 OS** | **0.7143 ± 0.051** | 0.7224 ± 0.033 |
| **성능 PFS** | **0.6621 ± 0.040** | 0.6470 ± ? |
| **모델 구조** | ① clinical+판독지를 §2-1 구조로 **함께** 학습해 위험점수 1개를 뽑고 ② 영상(SimpleCNN)을 §1-3 구조로 따로 학습해 위험점수 1개를 뽑은 뒤 ③ 두 위험점수를 fold별로 CoxPH(lifelines)에 넣어 가중합 계수를 적합 |
| **코드 파일** | `실험1_기본융합_early_late/late_fusion_tab_image.py` (`combine_two`, `run_target`) | `실험6_판독지_인코더_비교/exp_radbert_full.py` |
| **실행 명령** | `python 실험1_기본융합_early_late/late_fusion_tab_image.py --targets os,pfs` | `python 실험6_판독지_인코더_비교/exp_radbert_full.py --target os` 그리고 `--target pfs` |

**RadBERT 버전이 OS에서 더 높은데(0.7224) 왜 TF-IDF를 채택했나** — 2-1과 같은 이유. PFS에서 TF-IDF가 크게 앞선다(0.6621 vs 0.6470). 두 타깃 동등 가중이면 TF-IDF, OS만 우선한다면 RadBERT가 방어 가능 — 이 판단 기준은 `실험6_판독지_인코더_비교/REPORT_ENCODER_FINAL.md` §4.6에 정리돼 있다.

### 3-3. late fusion 3-way (clinical·판독지·영상 각각 독립 학습 후 결합)

| | 성능 |
|---|---|
| **OS** | 0.6811 ± ? |
| **PFS** | 0.6313 ± ? |
| **모델 구조** | clinical 단독(§1-1), 판독지 단독 TF-IDF(§1-2), 영상 단독(§1-3)을 **각각 완전히 독립적으로** 학습해 위험점수 3개를 뽑은 뒤, fold별로 CoxPH(lifelines)에 3개 covariate(`risk_image`, `risk_clinical`, `risk_report`)를 넣어 가중합 계수를 적합 |
| **코드 파일** | `실험1_기본융합_early_late/late_fusion_3modal.py` (`run_clinical_only`, `run_report_only`, `combine_weighted_sum`) — 재실행 스크립트는 `실험1_기본융합_early_late/exp_late_fusion_3modal_rerun.py` |
| **실행 명령** | `python 실험1_기본융합_early_late/exp_late_fusion_3modal_rerun.py --target os` 그리고 `--target pfs` |
| **재현성** | image 축은 재학습 없이 `outputs/late_fusion_B/oof_{target}.json`의 저장된 OOF 위험점수를 재사용. clinical·report는 batch32/epoch60으로 새로 학습(2026-07-22의 첫 실행은 batch16/epoch30 이었음, 참고값 OS 0.6703/PFS 0.6288) |

**3-way가 2-way보다 뚜렷이 낮다(OS −0.033, PFS −0.031).** clinical과 판독지를 **각각 따로** 학습시키면 둘 다 약한데(OS 0.63대/0.62대), **하나로 묶어서 같이** 학습시키면 0.71까지 오른다 — §2-1에서 확인한 "clinical과 판독지는 함께 학습해야 서로 보완 효과가 난다"는 결과와 정확히 같은 현상이다. 판독지가 부족한 부분을 임상변수가 메워주는 상호작용은 **모델을 합쳐야만** 학습되고, 점수만 사후 결합하는 late fusion(3-way)은 이 상호작용을 못 잡는다.

---

## 부록: 빈칸(재측정 필요) 항목 재실행 가이드

전부 `실험3_모달리티_절제실험/ablation.py` 하나로 돌릴 수 있다. 한 번에 다 돌리려면:

```bash
python 실험3_모달리티_절제실험/ablation.py --target os  --configs clin_image,all --epochs 60 --batch_size 32
python 실험3_모달리티_절제실험/ablation.py --target pfs --configs clin_image,all --epochs 60 --batch_size 32
```

"영상+판독지" 조합은 `실험3_모달리티_절제실험/ablation.py`의 `CONFIGS` 딕셔너리에 새 항목을 먼저 추가해야 돌릴 수 있다(코드 수정 필요).

**요청하신 대로 위 세 칸은 학습을 돌리지 않고 빈칸으로만 남겨뒀다.**
