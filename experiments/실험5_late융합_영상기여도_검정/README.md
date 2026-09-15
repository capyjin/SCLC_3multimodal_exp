# 실험5 — late fusion 영상 기여도 검정 (RESULTS.md §10)

**이 폴더의 스크립트는 2026-09-10 정리에서 실험1 폴더로 통합됐다.**
결과와 해석은 그대로 유효하다.

| 옛 파일 | 지금 |
|---|---|
| `analyze_late_fusion_pfs.py` | `python experiments/실험1_기본융합_early_late/analyze_late_fusion.py contribution` |
| `verify_shuffle_sanity.py` | `python experiments/실험1_기본융합_early_late/analyze_late_fusion.py shuffle-sanity` |

산출물 경로는 바뀌지 않았다 — `outputs/late_fusion_B/pfs_diagnosis.json`.
정리 당시 기존 JSON 과 재귀 비교해 **차이 0건**(200회 순열검정의 draws 200개
포함)임을 확인했다.

## 왜 옮겼나
late fusion 코드가 실험1·5·7 세 폴더에 9개 파일로 흩어져 있었고, 그중
"fold별 C-index" 와 "fold별 CoxPH stack" 을 파일마다 다시 구현하고 있었다.
계산 원자는 `src/sclc/late_fusion_tests.py` 로, 분석 스크립트의 공통 배관은
`src/sclc/experiments/analysis.py` 의 `BaseAnalysis` 로 올리고, 분석 5개를
실험1 폴더의 `analyze_late_fusion.py` 한 파일에 서브커맨드로 모았다.
자세한 내용은 저장소 루트의 `코드_구조.md` §"3차 정리".
