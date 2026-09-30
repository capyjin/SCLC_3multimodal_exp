# 실험15 — 채택 모델을 fold-safe 결측 대치로 재학습 (2026-09-30)

**질문:** 채택 모델(late fusion 2-way, RadBERT)의 연속형 임상변수 7개는 분할 전
전체 데이터 중앙값으로 채워져 있다(test fold 환자 포함 → 미세 누수). 훈련 fold
중앙값으로만 다시 채우면 성능이 바뀌는가?

**방법:** `foldsafe_clinical.py` 가 원본 엑셀 관측값에서 7열(흡연량·Hb·WBC·LDH·
FVC·FEV1·DLCO)을 다시 만들고, fold 마다 train 환자 중앙값으로 대치 + 표준화한다.
비흡연자 흡연량 0 은 관측값으로 둔다. 영상 축은 임상변수를 안 쓰므로
`outputs/late_fusion_B/oof_{target}.json` 을 재사용했다. 나머지 조건은 채택 모델과 동일
(RadBERT·brain_meta 수정·seed 42·bs32/ep60).

**결과** (`outputs/foldsafe_final/results_{os,pfs}.json`, tab_orig 는 기존 값 fold 단위 재현 MATCH):

| | OS 기존 | OS fold-safe | PFS 기존 | PFS fold-safe |
|---|---|---|---|---|
| tabular 축 | 0.7153 | 0.7105 | 0.6456 | 0.6423 |
| late fusion (채택) | 0.7224 ± 0.033 | 0.7145 ± 0.043 | 0.6470 ± 0.029 | 0.6457 ± 0.050 |
| Δ late (paired t p, 개선 fold) | | −0.008 (p=0.46, 1/5) | | −0.001 (p=0.94, 3/5) |

**해석:** 차이는 fold 표준편차(0.03~0.05)보다 훨씬 작고 유의하지 않다. 영상 계수도
유지된다(OS β_img 0.29 → 0.37, PFS 0.14 → 0.16). 즉 대치 누수가 채택 모델 성능을
만들어 낸 것은 아니다. 단, fold-safe 7열이 입력 텐서 뒤쪽으로 옮겨 붙어 초기값 대응이
달라지는 불가피한 차이가 있다 — 시드를 바꿔 반복하면 그 흔들림 크기를 잴 수 있다.

Run:
```
python experiments/실험15_결측대치_foldsafe_최종모델/exp_foldsafe_final.py --target os
python experiments/실험15_결측대치_foldsafe_최종모델/exp_foldsafe_final.py --target pfs
```
