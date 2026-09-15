# 실험9 — 병기별(stage-aware) late fusion

**⚠️ 폐기. 채택 모델은 stage 무관 단일 가중치의 late fusion 2-way 다.**
유지보수하지 않는다. 기록과 재현을 위해 파일은 남긴다.

## 물었던 것

SCLC 는 LS(제한기, n=72)와 ES(확장기, n=166)의 치료법·진행 양상이 다르다.
채택 모델은 tabular 위험도와 영상 위험도를 CoxPH 로 결합하면서 **모든 환자에게
같은 가중치**를 쓴다. 병기별로 가중치를 다르게 학습시키면 나아지는가?

| 모델 | 공변량 | 역할 |
|---|---|---|
| `M0_shared` | zT, zI | 현재 모델 |
| `M1_stage_main` | + sc | **주 비교 기준선** |
| `M2_stage_aware` | + sc·zT, sc·zI | 제안 모델 |

M1 이 기준선인 이유가 이 실험 설계의 핵심이다. `stage` 는 이미 clinical
범주형 13개 중 하나라 tabular 축의 **입력**이고, 결합 단계에 stage 를 그냥
변수로 하나 더 넣기만 해도 성능이 오른다. M0 와 비교하면 "상호작용의 효과"가
아니라 "stage 를 한 번 더 넣은 효과"를 재게 된다.

## 결과 — NO_EVIDENCE

**OS·PFS 모두 근거 없음.** M2 가 M1 보다 나아진다는 통계적 근거를 얻지 못했다.

다만 **효과가 0이라는 뜻은 아니다.** 영상 가중치의 stage 의존성은 방향이
일관되게 관측된다(OS 5/5 fold, PFS 4/5 fold). LS 표본이 n=72 로 작아 검정력이
부족한 것으로 본다. 코호트가 커지면 다시 볼 만한 질문이다.

같은 방향을 게이트 네트워크로 다시 판 것이 실험11 의 `exp_gate_stage_subgroup.py`
인데, 그쪽은 방향조차 target 마다 뒤집혀 더 약한 결과였다.

| 파일 | 내용 |
|---|---|
| `exp_stage_aware_fusion.py` | M0/M1/M2 학습·비교 |
| `diag_stage_aware_optimization.py` | 최적화 진단 |
| `RESULTS_stage_aware_fusion.md` | 본 리포트 |
| `RESULTS_stage_aware_diagnosis.md` | 진단 리포트 |

관련: `../실험11_MoE/RESULTS_ALL.md`
