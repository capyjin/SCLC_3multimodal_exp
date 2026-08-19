# SCLC Tri-modal Fusion: Image + Clinical + Report

PET-CT 영상 · 임상변수 · 판독지 텍스트를 결합한 소세포폐암(SCLC) 생존예측 프로젝트.
`clinical+image/SCLC_simple_CNN-main`(영상 백본 + 임상 브랜치)과
`clinical+report/SCLC_report_unimodal_test-main`(임상 + TF-IDF 판독지 브랜치)을
`DATA/SCLC_EXPERIMENT_PROTOCOL.md` 에 따라 하나로 합친 것이다.

| 문서 | 내용 |
|---|---|
| **[코드_구조.md](코드_구조.md)** | **어떤 실험이 어느 폴더에 있는지 · 공용 코드가 무엇인지 (여기부터 읽으면 된다)** |
| [RESULTS.md](RESULTS.md) | 전체 실험 서사와 진단 (가장 긴 문서) |
| [MODEL_SUMMARY.md](MODEL_SUMMARY.md) | 모달리티 조합별 최고 성능 모델 정리 |
| [RESULTS_TABLE_final.md](RESULTS_TABLE_final.md) | 보고용 최종 성능 표 |

## 최종 채택 모델

| 타깃 | 모델 | C-index |
|---|---|---|
| **OS** | late fusion (임상+판독지 concat + 영상 SimpleCNN) | **0.7143** |
| **PFS** | 임상+판독지 concat (영상 제외 — 영상이 PFS엔 도움 안 됨) | **0.6696** |

## 코호트

- 3모달 공통 코호트 **238명** (영상·임상·판독지 전부 보유). 이는 `report_common`
  코호트와 정확히 일치하므로(모든 report_common 환자가 영상도 갖고 있다),
  기존 `report_common_5fold_seed42_v1.csv` split 을
  `splits/trimodal_common_5fold_seed42_v1.csv` 로 **그대로 재사용**한다(5-fold, seed 42).
- `core/cohort.py` 가 매니페스트를 만들고 로드 시점에 split 파일과 교차검증한다.
  모델마다 새 split 을 만들지 않는다(프로토콜 7절).

## 구조

- **Image**: `SimpleCNNBackbone` (4× ConvBlock, 512D) → Linear(512,128)+ReLU+Dropout(0.2) → L2 정규화
- **Clinical**: 21개 변수(연속 8 표준화 + 범주 13) → [Linear-BN-ReLU-Dropout(0.5)] ×4 @128 → L2 정규화
- **Report**: char n-gram(2,4) TF-IDF (max_features=400, fold별 fit) →
  [Linear-BN-ReLU-Dropout(0.3)] ×(32,16) → L2 정규화

### Early(concat) fusion — `core.model.ConcatDeepSurv`, `core.train.TrimodalEvaluator` 로 실행

concat(128+128+16=272) → Dropout(0.3) → Linear(272,1,bias=False) → Cox 위험점수.
Cox 음의 부분우도로 end-to-end 학습. `use_image`/`use_clinical`/`use_report` 플래그로
브랜치를 켜고 끌 수 있다(절제 실험이 이걸 쓴다).

### Late(출력 수준, 가중합) fusion — `core.fusion_stack`

단일모달 모델을 각각 독립 학습해(같은 코호트/split/seed) fold별 OOF 위험점수를 모은
뒤, fold 마다 `lifelines.CoxPHFitter` 를 적합한다 — 학습된 계수가 곧 "가중합"이다.
2축(tabular + 영상)이 최종 채택 모델이고, 3축(영상·임상·판독지 각각 독립)도
비교용으로 있다(`실험1_기본융합_early_late/`).

## 실행

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # GPU면 CUDA 버전에 맞는 torch/torchvision 을 먼저 설치

# 기본 파이프라인
python main.py --experiment early_fusion --mode smoke_test    # 수 초, torch 불필요
python main.py --experiment early_fusion --mode batch_smoke   # 1 fold × 2 epoch — 배관 점검
python main.py --experiment early_fusion --mode train         # 5-fold × 30 epoch × OS/PFS

python main.py --experiment late_fusion  --mode smoke_test
python main.py --experiment late_fusion  --mode batch_smoke   # 결합 단계는 건너뜀 (main.py docstring 참고)
python main.py --experiment late_fusion  --mode train

# 최종 채택 모델 (late fusion method B)
python 실험1_기본융합_early_late/late_fusion_tab_image.py --targets os,pfs

# 그림
python 도구/plot_all_figures.py
```

- `smoke_test`: torch 없이 경로·코호트·매니페스트·split 일치와 판독지 코퍼스 로딩만 확인.
- `batch_smoke`: 1 fold × 2 epoch 로 실제 학습 배관(dataset → model → loss → backward →
  optimizer step → C-index)이 도는지 확인. **결과가 아니라 점검용 숫자다.**
- `train`: 전체 5-fold 실행. 영상 CNN이 들어가면 CPU에서 매우 느리므로 GPU 머신에서 돌릴 것.

개별 실험 실행법은 각 실험 폴더의 스크립트 docstring 과 [코드_구조.md](코드_구조.md) 참고.

산출물은 프로토콜 11절에 따라 `outputs/EXP_<date>_<experiment>_<mode>/` 아래에 쌓인다
(resolved_config.yaml, data_manifest.csv, splits.csv, environment.txt, checkpoints/,
metrics/, experiment_report.md).

### Windows MAX_PATH 주의

이 저장소 경로가 이미 깊고(`...\4)3multimodal_fusion\clinical+image+report\...`),
torch 의 설치 트리(`torch-*.dist-info/licenses/third_party/...`)가 120자쯤 더 붙는다.
venv 를 프로젝트 폴더 안에 만들면 `pip install` 이 `WinError 206`(파일명 너무 김)으로
실패할 수 있다. 그럴 땐 얕은 경로에 venv 를 만들고(예: `python -m venv C:\venv_sclc_trimodal`)
이 디렉터리를 작업 디렉터리로 해서 그 인터프리터로 실행하면 된다.

## 프로토콜 주의 (4.4절)

판독지 모달을 추가하면서 3모달 코호트(238)가 원래 영상+임상 baseline 코호트(257)보다
작아졌다. 엄밀한 비교를 위해서는 같은 238명 코호트에서 영상+임상 2모달 baseline 을
다시 평가해야 한다 — `실험3_모달리티_절제실험/ablation.py --configs clin_image` 로
돌릴 수 있으며, `brain_meta` 수정 후로는 아직 실행하지 않았다
(MODEL_SUMMARY.md §2-2 의 빈칸).

## 출처

원본 프로젝트(`clinical+image/SCLC_simple_CNN-main`,
`clinical+report/SCLC_report_unimodal_test-main`)는 git 체크아웃이 아니라 GitHub
아카이브 추출본이라, 통합하면서 잃은 히스토리는 없다.
