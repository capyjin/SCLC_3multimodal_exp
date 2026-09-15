# 영상 유니모달: DeepSurv 헤드 vs 고전 CoxPH (2026-08-03)

스크립트: `실험8_영상단독_고전CoxPH/exp_image_cph.py` · 산출물: `outputs/image_cph/results.json`, `results_{os,pfs}.json`
재현: `python 실험8_영상단독_고전CoxPH/exp_image_cph.py --targets os,pfs`

## 1. 무엇을 물었나

지금까지 영상 단독 성능(OS 0.6570 / PFS 0.6154)은 전부 **DeepSurv**(SimpleCNN 백본 →
Dropout→Linear(512,1), Cox 부분우도 end-to-end)로만 쟀다. 백본은 그대로 두고
**마지막 위험 예측만 고전 CoxPH(lifelines)** 로 바꾸면 어떻게 되는가?
(= 라디오믹스 논문들이 쓰는 "CNN feature → PCA → CoxPH" 파이프라인)

고정: 238명 코호트, `trimodal_common_5fold_seed42_v1` split, 512 resize + train fold 픽셀
mean/std, `model.SimpleCNNBackbone` 구조, 백본 학습 조건(bs16/ep30/lr1e-4/wd1e-4, val
C-index 최고 체크포인트 = `outputs/late_fusion_B/image_simplecnn_{target}` 재사용).
달라지는 것: **위험 점수를 내는 방법만.**

`deepsurv_head` 가 기존 값 **0.6570 / 0.6154 를 소수점 4자리까지 정확히 재현**하므로
두 팔은 같은 모델·같은 fold 위에서 비교된다.

## 2. 결과 (5-fold 평균 C-index)

| 방법 | OS | PFS |
|---|---|---|
| **deepsurv_head** (기존 기준선) | **0.6570 ± 0.048** | **0.6154 ± 0.057** |
| **cph_selected** (inner-CV로 고른 고전 CoxPH) | 0.6418 ± 0.034 | 0.5901 ± 0.030 |
| ctrl_random_cnn_pca16 (**학습 안 한** 백본 + CoxPH) | 0.6557 ± 0.039 | 0.6315 ± 0.048 |

탐색용(exploratory — test에서 최대값을 고르면 낙관 편향이 생기므로 대표값으로 쓰지 말 것):

| 변형 | OS | PFS |
|---|---|---|
| cph_pca2 / pca4 | 0.6500 / 0.6551 | 0.6095 / 0.6038 |
| cph_pca8 / pca16 / pca32 / pca64 | 0.6410 / 0.6462 / 0.6416 / 0.6278 | 0.5851 / 0.6098 / 0.6012 / 0.5513 |
| cph_ridge_full λ=0.1 / 1 / 10 | 0.6414 / 0.6473 / 0.6544 | 0.6027 / 0.5939 / 0.6014 |
| cph_unicox_top8 / top16 | 0.6422 / 0.6431 | 0.6007 / 0.5948 |

**deepsurv_head 대비 페어드 부트스트랩** (fold 내 환자 리샘플 2000회, 같은 리샘플에서 두 방법 동시 계산):

| 비교 | OS Δ [95% CI] | PFS Δ [95% CI] |
|---|---|---|
| cph_selected − deepsurv | −0.0155 [−0.0387, +0.0078] | −0.0253 [−0.0537, +0.0012] |
| ctrl_random_cnn − deepsurv | −0.0019 [−0.0378, +0.0334] | **+0.0160** [−0.0110, +0.0435] |

동일 파이프라인(PCA16+CoxPH)에서 **백본만** 바꾼 직접 비교:

| | OS | PFS |
|---|---|---|
| 학습된 백본 | 0.6462 | 0.6098 |
| 랜덤 초기화 백본 | 0.6557 | 0.6315 |
| Δ(학습−랜덤) | −0.0097 [−0.0386, +0.0189] | −0.0217 [−0.0568, +0.0135] |

## 3. 해석

1. **고전 CoxPH로 바꿔서 얻는 이득은 없다.** 12개 변형 중 어느 것도 DeepSurv 헤드를 넘지
   못했고, 정직한 대표값(inner-CV 선택)은 OS −0.0155 / PFS −0.0253 로 오히려 낮다. OS는
   CI가 0을 포함해 "차이 없음"에 가깝고, PFS는 P(Δ>0)=0.032 로 열세 쪽이 유의에 근접.
   → **영상 유니모달 보고 수치는 기존 DeepSurv 값(0.6570/0.6154)을 그대로 쓰면 된다.**

2. **⚠️ 핵심 발견: 학습 안 한 랜덤 CNN 특징이 학습된 것과 구분되지 않는다.**
   랜덤 초기화 백본(BN 통계만 train fold로 채움) + CoxPH 가 OS 0.6557 (학습 0.6462),
   PFS 0.6315 (학습 0.6098). 차이는 통계적으로 유의하지 않지만 **부호가 학습 쪽에 불리**하며,
   "Cox 손실로 학습한 백본이 랜덤 컨볼루션보다 낫다"는 증거가 전혀 없다.
   즉 이 영상 브랜치의 예측력 상당 부분은 학습된 종양 특징이 아니라 **랜덤 사영이 요약한
   저수준 영상 통계**(밝기/대비/면적류)로 설명 가능하다. 238장으로 512차원 CNN을 scratch
   학습하는 것의 한계이며, §9(영상이 concat에서 70~80% 지분을 가져감)·§10(PFS에 정보 없음)과
   일관된 그림이다.

3. **inner-CV 점수(0.67~0.75)가 test(0.62~0.71)보다 크게 높다.** inner-val 환자들이 이미
   백본 학습에 쓰인 사람들이라 임베딩이 과분리돼 있기 때문(아래 한계 참조). 그래서 선택 절차
   자체가 왜곡되어 `cph_selected`가 최고 exploratory 변형보다도 낮게 나왔다.

## 4. 누수 점검 (설계 감사 반영 사항)

- StandardScaler / PCA / 단변량 스크리닝 / CoxPH는 **전부 train fold 171명만**으로 fit.
- 이미지 픽셀 mean/std도 train fold만(학습 때와 동일). val/test는 split CSV에서 배타 확인.
- **하이퍼파라미터 선택도 test를 안 본다**: `cph_selected`는 outer train 안 inner 5-fold CV로
  후보를 고르고 171명 전체로 재적합. 12개 변형 표에서 최대값을 고르는 것은 폴드 σ≈0.05에서
  0.02~0.04 낙관 편향이 생기므로 금지 → `exploratory: true` 로 표시.
- 임베딩 캐시(npz)에 체크포인트 sha1·resize·seed·split sha1을 함께 저장, 하나라도 다르면 자동 재추출.
- 기준 체크포인트 디렉터리는 **읽기 전용**(학습은 `--retrain` 시 `outputs/image_cph/ckpt_*`로만).
- penalizer는 fold 간 통일(다르면 최대값으로 전 fold 재적합), 단변량 스크리닝 실패 수 기록(전부 0).
- 랜덤 대조군은 eval()에서 BN이 항등함수가 되지 않도록 train fold로 BN 통계 워밍업.

**남은 편향(누수 아님, 논문에 명시 필요):** CoxPH는 CNN이 이미 학습한 그 171명의 임베딩 위에서
β를 추정한다. test 정보는 전혀 안 들어가므로 누수가 아니라 편향이며, 방향은 **CoxPH에게 불리**
(in-sample 임베딩이 과분리 → β 과대추정 → test 임베딩에서 miscalibrated). 실제로 in-sample
C-index는 CoxPH 쪽이 훨씬 높다(PFS 0.724 vs DeepSurv 0.669). 완전히 없애려면 outer train 안에서
백본을 inner-CV로 재학습해 **out-of-fold 임베딩**을 만들어야 한다(타깃당 백본 25회 재학습, 약 1시간).
그 실험 전까지 위 1번 결론은 "고전 CoxPH가 더 낫다는 증거가 없다" 수준으로만 읽어야 한다.
반면 2번(랜덤 백본 대조군)은 두 팔이 **완전히 같은 파이프라인**이라 이 편향의 영향을 받지 않는다.
