# Late Fusion vs MoE 게이트 계열 — 파이프라인·아키텍처·성능 종합

5-fold CV, `splits/trimodal_common_5fold_seed42_v1.csv`(238명 고정) 기준. 모든 수치는
C-index(concordance index), 5-fold 평균. Δ 판정선은 프로젝트 관례상 **0.016**
(5-fold paired 비교의 최소 유의차, `RESULTS_TABLE_final.md`).

---

## 0. 공통 백본 (모든 실험이 공유하는 얼린 부품)

### 이미지 백본 — `SimpleCNNBackbone`
```
Conv2d(1→32)-BN-ReLU-MaxPool  x1
Conv2d(32→64)-BN-ReLU-MaxPool
Conv2d(64→128)-BN-ReLU-MaxPool
Conv2d(128→256)-BN-ReLU-MaxPool
AdaptiveAvgPool2d(1,1) -> Linear(256, 512)          # 512차원 임베딩
```
영상 단독 DeepSurv(`ImageOnlyDeepSurv`) = `backbone -> Dropout -> Linear(512,1)`,
Cox 부분우도로 end-to-end 학습(bs16/ep30/lr1e-4/wd1e-4/seed42, 512 resize, grayscale).
→ **OS 0.6570 / PFS 0.6154** (RadImageNet/ResNet18 백본 대조군은 이보다 낮아 채택 안 함).

### Tabular(clinical+report) 백본 — `ConcatDeepSurv(use_image=False)`
```
clinical(21차원) -> [Linear-BN-ReLU-Dropout(0.5)] x4 @128
report(TF-IDF char(2,4), max_features=400, train fold로만 fit) -> [Linear-BN-ReLU-Dropout(0.3)] x(32,16)
각 브랜치 출력 L2-정규화 -> concat(128+16=144) -> Dropout(0.3) -> Linear(144,1,bias=False)
```
Cox 부분우도로 end-to-end 학습(bs32/ep60). → **OS 0.7076 / PFS 0.6678**.

두 백본 모두 fold마다 독립적으로 학습되고(train 171명), OOF(out-of-fold) risk score를
test 47~48명에 대해 낸다. MoE 계열 실험은 전부 이 두 백본을 **재학습 없이 그대로
얼려서** 그 위에 새 결합/게이트 층만 얹는다(`src/sclc/expert_embeddings.py`가 캐시 읽기
+ 체크포인트 forward로 임베딩·risk만 뽑아낸다).

---

## 1. Late Fusion (현재 채택된 최종안)

**파이프라인**: 이미지·tabular 백본을 각각 수렴할 때까지 독립 학습 → 두 OOF risk score를
fold별 **2변수 CoxPH(lifelines)** 로 결합(`src/sclc/late_fusion.py::combine_risk_scores`) →
train fold에서 학습된 선형계수를 test에 적용.

```
risk = β_tabular * risk_tabular + β_image * risk_image     (β 는 CoxPH가 fold마다 추정)
```

| Target | tabular 단독 | image 단독 | **Late fusion** | mean_coef (tabular, image) |
|---|---:|---:|---:|---|
| OS  | 0.7076 (±0.047) | 0.6570 (±0.048) | **0.7221** (±0.051) | 4.442, 0.297 |
| PFS | 0.6678 (±0.034) | 0.6154 (±0.057) | **0.6531** (±0.054) | 3.849, **−0.052** |

- 두 endpoint 모두 tabular 단독보다 유의미하게 개선(OS +0.0145, PFS 는 재구성 오차 감안 시
  아래 §5 참고).
- PFS는 image 계수가 fold마다 부호가 뒤집힐 만큼 불안정 — 이 프로젝트 전반에서 반복
  확인된 "PFS의 image 정보는 약하고 불안정하다"는 패턴.
- **지금까지 시도한 융합 방식 중 유일하게 unimodal baseline을 이긴 방법.**

---

## 2. MoE 게이트 계열 (실험11_MoE/) — late fusion의 "고정 선형결합"을 "환자별 학습 게이트"로 대체하는 시도

공통 설계: 백본은 전부 동결. 새로 학습되는 건 게이트 뿐. Cox 부분우도, full-batch(fold당
171명), `torch.manual_seed`로 5-seed 반복. 게이트가 상수로 붕괴하면 late fusion의 CoxPH
결합과 수학적으로 동치이므로, "상수게이트 대조군과의 차이"가 게이트가 실제로 기능하는지를
바로 보여준다.

### 2.1 M0 — 임베딩 PCA 게이트 (`exp_gate_m0.py`)

```
s_k        = 고정. image_risk, tabular_risk (train 통계로 z-score)
gate_input = expert별 PCA-8(train 171로만 fit) 이어붙임, 24차원
g          = softmax(Linear(24 -> 2))
risk       = alpha * (g_image*image_risk + g_tabular*tabular_risk)
```

| Target | CoxPH 재현 | 상수게이트 | **학습게이트** | Δ(학습−상수) |
|---|---:|---:|---:|---:|
| OS  | 0.7228 | 0.7217 | 0.7120 | **−0.0097** |
| PFS | 0.6715 | 0.6711 | 0.6736 | **+0.0025** |

**FAIL** (판정선 0.016 미달, 둘 다). 진단: 게이트는 환자마다 실제로 값이 변하지만
(g_image std 0.29~0.36) 그 변화가 일반화되지 않음 — 신호라기보다 171명 규모의 노이즈.

### 2.2 M0-tuned — 안전한 inner-CV 하이퍼파라미터 탐색 (`exp_gate_m0_tuned.py`)

M0가 과적합 패턴을 보였으므로 "레이어를 늘리는" 대신 "입력을 줄이고 정칙화를 세게":
`gate_input ∈ {risk_only(2d), pca2(6d), pca4(12d)} × weight_decay ∈ {0.1,0.03,0.01}` 9개
후보를 outer-train(171명) 안 inner 5-fold로 탐색(실험8의 `select_by_inner_cv` 패턴,
outer-test는 최종 1회만 확인).

| Target | CoxPH 재현 | **튜닝된 게이트** | Δ |
|---|---:|---:|---:|
| OS  | 0.7228 | 0.7205 | −0.0024 |
| PFS | 0.6715 | 0.6683 | −0.0032 |

**FAIL.** fold마다 선택된 "최적" 후보가 전부 달라서(`risk_only`/`pca2`/`pca4` 뒤섞임),
안전하게 탐색해도 뚜렷한 최적점이 없다는 뜻 — 탐색면 자체가 평평한(노이즈) 것으로 해석.

### 2.3 M1 — attention 기반 게이트 (`exp_gate_m1.py`) — ⚠ 1-fold smoke test만, 완주 안 함

M0/M0-tuned와 근본적으로 다른 구조: risk 스칼라를 재가중하는 게 아니라 **원본 임베딩
(image 512 / clinical 128 / report 16, 3개 전문가)** 에 직접 cross-attention + 새 head를
얹는다 — 원래 헤드가 놓쳤을 조합을 원칙적으로 학습 가능.

```
tok_k  = LayerNorm(Linear(d_k -> proj_dim)) + 전문가별 학습 토큰임베딩(3, proj_dim)
tok'   = tok + Dropout(MultiheadAttention(tok,tok,tok, nhead=2))
s_k    = shared Linear(proj_dim -> 1)  (토큰마다 같은 head)
g      = softmax(Linear(flatten(tok') -> 3))
risk   = alpha * sum_k g_k * s_k
```

| Target | fold | CoxPH 재현 | **M1 게이트** | Δ |
|---|---|---:|---:|---:|
| OS | 1 (smoke test) | 0.7139 | **0.5953 ± 0.0154** | **−0.1185** |

**FAIL, 처참한 수준.** 파라미터가 많은 만큼(≈11K) 171명 규모에서 심하게 과적합한 것으로
보임. 이 결과가 너무 나빠서 나머지 4-fold + PFS는 돌리지 않고 중단.

### 2.4 Prior-guided Gating — 임상 prior(stage+age) 게이트 (`exp_gate_m0_clinical.py`)

사용자 제안(4개 아이디어 중 1번)을 Opus가 정교 설계, Sonnet이 구현. 동기: `실험9_병기별_
stage_aware_융합`가 "CoxPH 결합계수가 병기(LS/ES)에 따라 다른가"를 이미 테스트해
NO_EVIDENCE였지만 OS 5/5·PFS 4/5 fold에서 ES>LS 방향 이미지가중치가 일관됐던 것(LS
n=72로 검정력 부족)에 착안 — 더 저노이즈·저차원인 입력을 게이트에 주면 그 신호를
gradient descent가 스스로 찾아낼 수 있는지 검증.

```
gate_input(주실험) = [stage(±0.5 고정 인코딩, z-score 안 함), age_z(train fit)]   # 2차원
   stage는 fold별 LS 비율이 27~36%로 흔들려 중심화하면 "0" 기준점이 fold마다 달라짐
   -> ±0.5 고정 인코딩이면 weight_decay로 수축될 때 정확히 상수게이트(late fusion) 로 수렴
g = softmax(Linear(2 -> 2)),  risk = alpha*(g_image*image_risk + g_tabular*tabular_risk)
tumor size는 이 코호트에 컬럼 자체가 없어 제외(2D MIP, 세그멘테이션 마스크 없음)
```

2차(비교용): `{prior_only, risk_prior(=risk+prior 5차원)} × weight_decay` inner-CV 탐색.

| Target | CoxPH 재현 | 상수게이트 | **prior 게이트** | Δ(prior−CoxPH) | Δ(prior−상수) | 2차(튜닝) |
|---|---:|---:|---:|---:|---:|---:|
| OS  | 0.7228 | 0.7217 | 0.7200 | −0.0028 | −0.0017 | 0.7214 |
| PFS | 0.6715 | 0.6711 | 0.6684 | −0.0031 | −0.0027 | 0.6688 |

**FAIL.** 진단 — 실험9의 방향성이 재현되는지:

| | stage계수>0 fold | age계수>0 fold | g_image(ES−LS) 평균 |
|---|---|---|---:|
| OS  | 3/5 | 4/5 | **−0.0278** (실험9와 반대 방향) |
| PFS | 2/5 | 4/5 | +0.0303 (방향은 같으나 미약) |

실험9가 CoxPH 결합계수에 직접 stage-interaction term을 넣어 봤던 방향성(ES>LS)이,
Cox loss만으로 학습되는 신경망 게이트에서는 **재현되지 않았다**(OS는 부호까지 반대).
정보를 정확히 넣어줬는데도(stage/age 그대로) 안 됐다는 점에서, "게이트에 입력이
부족해서"가 아니라 **이 표본 크기에서 gradient descent가 미약한 신호와 노이즈를
구분 못한다**는 쪽에 더 무게가 실림.

### 2.5 잔차(residual) 게이트 — 중단, 결과 없음

CoxPH 해에서 정확히 출발(마지막 layer 0-초기화)해 "초기화가 나빠서 실패했나 vs 애초에
신호가 없어서 실패했나"를 분리하려던 실험. 실행 도중 사용자 지시로 중단, 저장된 결과 없음.

---

## 3. 종합 비교

| 방법 | 입력 유연성 | OS C-index | PFS C-index | 기존 late fusion 대비 |
|---|---|---:|---:|---|
| tabular 단독 | — | 0.7076 | 0.6678 | (baseline 성분) |
| image 단독 | — | 0.6570 | 0.6154 | (baseline 성분) |
| **Late fusion (CoxPH 2변수)** | 고정 선형 | **0.7221** | **0.6531** | — (현재 채택안) |
| MoE M0 (PCA8 게이트) | 학습형, 저유연 | 0.7120 | 0.6736 | OS −0.0101 / PFS +0.0205* |
| MoE M0-tuned (inner-CV) | 학습형, 정칙화 탐색 | 0.7205(재현대비) | 0.6683(재현대비) | 재현 기준 각각 −0.0024 / −0.0032 |
| MoE M1 (attention, 3-expert) | 학습형, 고유연 | 0.5953(1-fold만) | 미실행 | −0.1185 (처참) |
| MoE Prior-gate (stage+age) | 학습형, 저노이즈 입력 | 0.7200 | 0.6684 | 재현 기준 각각 −0.0028 / −0.0031 |

\* PFS의 "late fusion 저장값(0.6531) 대비" 개선은 착시임 — §5 참고. 공정한 기준(이번에
재현한 CoxPH 0.6715)으로는 모든 MoE 변형이 late fusion을 넘지 못했다.

---

## 4. 분석

**입력 유연성을 낮은 것(2차원 stage+age)부터 높은 것(3-expert attention)까지 체계적으로
바꿔가며 시도했지만, 어느 방향으로도 CoxPH 2변수 선형결합을 이기지 못했다.** 특히:

1. **저유연(M0, prior-gate)**: 게이트가 환자마다 실제로 다른 값을 내는데도(분산 존재)
   held-out 성능엔 도움이 안 됨 — "배우긴 하는데 일반화 안 되는 노이즈"라는 동일 패턴이
   4개 서로 다른 게이트 입력(PCA임베딩, risk_only, prior)에서 반복 관찰됨.
2. **고유연(M1)**: 파라미터를 늘리자 즉시 크게 악화 — 171명/fold 규모에서 과적합이
   지배적임을 확인.
3. **inner-CV로 안전하게 하이퍼파라미터를 탐색해도(M0-tuned, prior-gate 2차)** 결과가
   안 바뀜 — fold마다 다른 후보가 선택되는 불안정성 자체가 "탐색면이 평평하다(신호가
   거의 없다)"는 증거로 해석됨.
4. **이미 알려진 방향성(실험9의 stage-interaction)조차 신경망 게이트가 재현하지 못함**
   — 정보를 정확히 줬어도 이 표본 크기에서 gradient 기반 학습이 CoxPH의 명시적 MLE보다
   그 미약한 신호를 못 찾음.

**결론**: late fusion의 성능 상한은 "결합 방식이 선형이라서 생기는 손실"이 아니라
**표본 크기(fold당 171명, 특히 PFS/LS 서브그룹은 더 적음)** 에 의해 정해지는 것으로
보인다. 더 유연한 결합 모델을 넣는 방향은 이 데이터에서 소진된 것으로 판단되며,
**late fusion(OS 0.7221 / PFS 0.6531)을 최종안으로 유지하는 것이 타당**하다.

---

## 5. PFS 재구성 관련 주의사항 (반복 확인된 함정)

MoE 실험들이 자체적으로 재구성한 2변수 CoxPH 기준(PFS 0.6715)은 late fusion 저장값
(0.6531)과 0.018 차이가 난다. 두 코드의 결합 로직은 동일함을 대조 확인했고, PFS의
image 계수가 fold/seed 마다 부호가 뒤집힐 만큼 원래 불안정해서, 아주 작은 재구성
오차(개별 성분은 0.002 이내로 일치)가 결합 후 크게 흔들린 것으로 보인다. **그래서 "게이트가
도움됐는지"는 항상 그 실험이 자체 재현한 CoxPH 기준과 비교해야 하며, 저장된 historical
값과 직접 비교하면 착시가 생길 수 있다.**

---

## 6. RadBERT 판독지 인코더로 재현 (2026-09-01)

프로젝트가 OS 기준 판독지 인코더를 TF-IDF → RadBERT로 전환하면서(§1 late fusion
seed sweep에서 OS는 RadBERT가 더 좋고 더 안정적, PFS는 TF-IDF 유지로 결정), M0 /
M0-tuned / Prior-gate 세 게이트 실험을 RadBERT tabular 체크포인트
(`outputs/late_fusion_B_radbert/`, `실험11_MoE/prepare_tabular_radbert_ckpt.py`가
`실험6/exp_encoder_trimodal.py`와 동일 레시피로 생성) 기준으로 그대로 재실행했다.

Opus 설계 검토를 거쳐 `src/sclc/expert_embeddings.py::load_gate_inputs(..., report_encoder=
"tfidf"|"radbert")` 하나로 (ckpt_dir, text_encoder_fn) 쌍을 원자적으로 고르게 만들었고
(개별 인자로 노출하면 폭이 우연히 같아지는 인코더가 생겼을 때 조용히 틀릴 수 있어서),
RadBERT 인코더 조립 레시피는 `src/sclc/bert_features.py::build_radbert_report_encoder()`
하나로 승격했다(기존 실험1/6/11 세 곳의 인라인 중복 제거). 재현 검증: 재구성한 fold별
tabular C-index가 학습 시점 값과 소수점 6자리까지 정확히 일치, TF-IDF/RadBERT 체크포인트를
서로 바꿔 넣으면 `report_branch` 폭(400 vs 768) 불일치로 즉시 `RuntimeError` — 두 경우
모두 확인 완료.

| 실험 | Target | TF-IDF Δ | RadBERT Δ | 판정 |
|---|---|---:|---:|---|
| M0 (학습−상수게이트) | OS | −0.0097 | **−0.0238** | FAIL / FAIL |
| M0 (학습−상수게이트) | PFS | +0.0025 | −0.0073 | FAIL / FAIL |
| M0-tuned (튜닝−CoxPH재현) | OS | −0.0024 | −0.0029 | FAIL / FAIL |
| M0-tuned (튜닝−CoxPH재현) | PFS | −0.0032 | −0.0024 | FAIL / FAIL |
| Prior-gate (prior−CoxPH재현) | OS | −0.0028 | −0.0033 | FAIL / FAIL |
| Prior-gate (prior−CoxPH재현) | PFS | −0.0031 | +0.0003 | FAIL / FAIL(상수게이트 대비는 −0.0007) |

**결론: 인코더를 바꿔도 결론은 그대로다.** RadBERT tabular는 CoxPH 재현 자체가
TF-IDF보다 높지만(OS 0.7299 vs 0.7139 — RadBERT가 원래 더 강한 tabular 축이라는
§1의 seed sweep 결과와 일치), 그 위에 얹은 게이트 3종은 여전히 상수게이트/CoxPH
재현을 못 넘는다. M0(고정 하이퍼파라미터)는 RadBERT에서 오히려 더 크게 밀렸다
(OS Δ −0.0238, TF-IDF의 2배 이상) — 더 강한 tabular 신호 위에서 게이트가 그 강한
신호를 오히려 더 심하게 흐트러뜨린 것으로 보인다. **"게이트가 배울 patient-adaptive
신호가 없다"는 §4의 결론은 판독지 인코더 선택과 무관하게 성립한다.**

---

## 7. 추가 진단 — M0-tuned의 "1등 후보"는 우연이었나 (`diag_inner_cv_seed_stability.py`)

§2.2에서 fold마다 선택된 후보가 매번 달랐던 것을 "탐색면이 평평하다(신호 없음)"고
해석했는데, 이걸 직접 검증했다. inner-CV가 171명을 5조각(모의고사)으로 쪼갤 때 쓰는
난수 시드를 42/43/44/45로 바꿔가며 같은 fold에서 "1등 후보"가 매번 같게 나오는지 봤다
(진짜 신호라면 어떻게 쪼개도 같은 후보가 이겨야 한다).

**결과: OS·PFS 합쳐 10개 fold 중 4개 시드 전부 같은 후보를 고른 경우는 0개였다.**
PFS의 2개 fold는 시드 4번 돌려서 1등이 4번 다 달랐다. (seed=42는 기존 공식 결과와
정확히 일치해 재현성 자체엔 문제없음을 확인.)

**결론: fold마다 "1등"이 바뀐 건 진짜 최적값이 달라서가 아니라, 애초에 후보들
사이에 실력 차이가 없어서 모의고사를 어떻게 쪼개느냐(운)에 따라 아무나 1등이
된 것이다.** M0/M0-tuned/Prior-gate 전체의 "게이트가 배울 신호가 없다"는 결론을
가장 직접적으로 뒷받침하는 근거.
