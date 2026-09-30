# 실험6 — 판독지 인코더 비교

판독지 텍스트를 모델 입력으로 바꾸는 방법(인코더)을 비교하고, 그 밖에 판독지에서
더 뽑아낼 게 있는지(텍스트의 어느 부분 / SUV 수치) 확인한 실험 묶음이다.

**결론과 근거는 [REPORT_ENCODER_FINAL.md](REPORT_ENCODER_FINAL.md) 에 있다.**
이 README 는 "어느 파일이 무엇을 하고 어떻게 돌리나"만 다룬다.

---

## 인코더 후보는 둘뿐이다

| 인코더 | 성격 | 상태 |
|---|---|---|
| `radbert` | frozen `StanfordAIMI/RadBERT` 768차원 (한글 ko2en 치환) | ✅ **최종 채택** (2026-09-30) — 단독 최고, 결합 시 OS 우세·PFS 열세 |
| `tfidf` | char n-gram(2~4) TF-IDF 400차원 | 비교용 (이전 채택) · 코드 기본 경로 |

**최종 채택 모델 재현:** `python exp_encoder_trimodal.py --target os` / `--target pfs`
→ `outputs/radbert_full/results_{os,pfs}.json` 의 `late_tab_radbert+img` (OS 0.7224 / PFS 0.6470).

정의는 [`src/sclc/encoders/`](../../src/sclc/encoders/) 한 곳에만 있다
(`ReportEncoder` 추상 부모 + 구현 2개). 실험 스크립트는 인코더를 직접 만들지 않고
`build_encoder("tfidf")` / `build_encoder("radbert")` 로 가져다 쓴다.

RadBERT 의 한글 처리는 후보가 아니라 **같은 인코더의 전처리 변형**이라
`@` 뒤에 붙인다 — `radbert`(=ko2en) / `radbert@strip` / `radbert@raw`.
§5.2 의 `[UNK]` 진단을 재실행할 때만 쓴다.

### 후보에서 제외돼 코드가 삭제된 것

| 제외된 후보 | 판정 근거 |
|---|---|
| KM-BERT · mBERT | 단독에서 TF-IDF 와 구분되지 않음 (§1.2) |
| TF-IDF + RadBERT 결합(600차원) | 0.6013/0.6065 로 대폭 하락 (§1.1) |
| TF-IDF SVD 랭크 통제군 | RadBERT 해석용 통제 실험이었고 역할을 끝냄 (§5.1) |
| MedCPT · BioLORD · MedEmbed · bge · MiniLM | 가중치만 받고 미실행. 병목이 인코더가 아님이 규명됨 (§2, 부록 C) |

수치는 전부 `REPORT_ENCODER_FINAL.md` 에 남아 있고, 산출물도
`outputs/reportonly_kmbert/`, `outputs/reportonly_mbert/` 등에 그대로 있다.
다시 후보로 올리려면 그 문서의 판정을 먼저 뒤집어야 한다.

---

## 파일

| 파일 | 무엇을 재나 | 산출물 |
|---|---|---|
| `exp_encoder_compare.py` | 인코더만 갈아 끼우고 나머지 전부 고정 (concat / 판독지 단독) | `outputs/bert_text/` |
| `exp_encoder_fusion.py` | 임상 단독 + 판독지 단독을 CoxPH 로 묶고 인코더만 교체 | `outputs/radbert_fusion/` |
| `exp_encoder_trimodal.py` | concat[임상+판독지] + 영상 SimpleCNN 결합, 인코더만 교체 | `outputs/radbert_full/` |
| `exp_text_source.py` | 판독지의 어느 부분(conclusion / finding)에 신호가 있나 | `outputs/text_source/` |
| `exp_suv_features.py` | 판독지에서 뽑은 SUVmax 숫자 열이 도움이 되나 | `outputs/suv_features/` |
| `suv_features.py` | 위 실험 전용 라이브러리 — 판독지 정규식 SUV 파싱 + fold-safe 대치 | — |

`suv_features.py` 만 이 폴더에 남아 있는 이유: 쓰는 실험이 하나뿐이라
`src/sclc/` 승격 기준(두 실험 이상)에 미달한다.

---

## 실행

```bash
# 인코더 비교 (concat, 실제 채택 대상)
python experiments/실험6_판독지_인코더_비교/exp_encoder_compare.py --target os

# 인코더 자체 실력 (판독지 단독)
python experiments/실험6_판독지_인코더_비교/exp_encoder_compare.py \
    --target os --model_config report_only

# [UNK] 진단 재실행 (§5.2)
python experiments/실험6_판독지_인코더_비교/exp_encoder_compare.py \
    --target os --model_config report_only \
    --encoders radbert,radbert@strip,radbert@raw

# 전처리 2×2 격자 재실행 (§5.1) — 기본은 둘 다 꺼짐이 채택 설정
python experiments/실험6_판독지_인코더_비교/exp_encoder_compare.py \
    --target os --encoders radbert --svd --scale

# late fusion / 삼중 결합
python experiments/실험6_판독지_인코더_비교/exp_encoder_fusion.py   --target os
python experiments/실험6_판독지_인코더_비교/exp_encoder_trimodal.py --target os

# 학습 없이 feature 만 점검 (수 초)
python experiments/실험6_판독지_인코더_비교/exp_encoder_compare.py --target os --preflight_only
```

모든 스크립트에 `--target/--epochs/--batch_size/--seed/--out_dir/--preflight_only`
가 공통으로 있다 (`src/sclc/utils/cli.py`). 기본값은 이 저장소의 확립된 설정인
**bs32 / ep60 / seed42** 다.

---

## 정리 전 명령과의 대응

파일 이름과 arm 이름이 바뀌었다. 과거 문서(`RESULTS.md`, `MODEL_SUMMARY.md`,
`REPORT_ENCODER_FINAL.md`)에 적힌 옛 명령은 아래로 읽으면 된다.

| 옛 명령 | 새 명령 |
|---|---|
| `exp_bert_text.py --arms tfidf` | `exp_encoder_compare.py --encoders tfidf` |
| `exp_bert_text.py --arms bert_ko2en --no_svd --no_scale` | `exp_encoder_compare.py --encoders radbert` |
| `exp_bert_text.py --arms bert_nokr --no_svd --no_scale` | `exp_encoder_compare.py --encoders radbert@strip` |
| `exp_bert_text.py --arms bert_raw --no_svd --no_scale` | `exp_encoder_compare.py --encoders radbert@raw` |
| `exp_bert_text.py --arms bert_ko2en` (축소 켠 채) | `exp_encoder_compare.py --encoders radbert --svd --scale` |
| `exp_bert_text.py --arms tfidf_svd` | (삭제 — §5.1 에서 역할 종료) |
| `exp_bert_text.py --arms tfidf_plus_bert` | (삭제 — §1.1 에서 탈락) |
| `exp_bert_text.py --model_name madatnlp/km-bert ...` | (삭제 — §1.2 에서 탈락) |
| `exp_radbert_fusion.py` | `exp_encoder_fusion.py` |
| `exp_radbert_full.py` | `exp_encoder_trimodal.py` |

⚠️ **기본값 하나가 바뀌었다.** 옛 `exp_bert_text.py` 는 RadBERT 에 SVD 축소와
StandardScaler 를 **켠 채로** 시작했고, 채택 레시피를 쓰려면 `--no_svd --no_scale`
을 매번 붙여야 했다. 지금은 §5.1 의 결론(둘 다 성능을 깎는다)에 맞춰 **꺼진 것이
기본**이다. 즉 `--encoders radbert` 는 MODEL_SUMMARY.md §2-1/§3-2 의 공식
RadBERT 수치를 낸 바로 그 설정이다. 축소 파이프라인을 다시 보려면 `--svd --scale`
을 명시적으로 켠다.

산출물 폴더 이름(`outputs/bert_text/`, `radbert_fusion/`, `radbert_full/`)과
체크포인트 폴더 이름은 **바꾸지 않았다** — 기존 결과와 문서의 링크를 그대로
살리기 위해서다.
