# Methods — Data processing (draft)

> **⚠️ 상태 (2026-09-30):** 초안이다. 최종 모델은 README.md "★ 최종 채택 모델"(RadBERT 2-way,
> OS·PFS 공통)을 따른다. 제외 사유 (3)의 8명은 날짜가 아니라 판독지 내용(치료반응·재발평가
> 문구)으로 제외됐다(4명은 판독일이 진행일보다 앞선다) — 본문과 Fig. 1 흐름도를 이에 맞춰 고쳤다.

> Items in **[brackets]** are facts that are not recorded in this repository and must be
> filled in by the authors. Every number outside brackets was recomputed from the code and
> data on 2026-09-30 (sources are listed in the notes at the end of this file).

---

## Study population

This retrospective study was approved by the Institutional Review Board of **[institution]**
(approval no. **[XXXX]**), and the requirement for informed consent was waived owing to the
retrospective design. The study was conducted in accordance with the Declaration of Helsinki.

We reviewed **[consecutive]** patients with **[histologically or cytologically confirmed]**
SCLC who were treated at **[institution]** and had clinical and follow-up data available
(n = 321). The inclusion criteria were as follows: (1) a pretreatment [¹⁸F]FDG PET-CT
examination; (2) the radiology report of that examination; and (3) complete overall survival
(OS) and progression-free survival (PFS) information. Patients were excluded for the
following reasons: (1) no PET-CT image was available (n = 64); (2) no usable radiology
report was available (n = 9); and (3) the report could not be confirmed as a baseline
document, because the available report described treatment response or recurrence rather
than initial staging (n = 8) or its date was unavailable (n = 2). Such reports were excluded
because they may contain post-baseline information about the outcome. Finally, 238 patients diagnosed between
January 2014 and January 2025 were included (Fig. 1). Disease stage was classified as
limited-stage (LS; n = 72, 30.3%) or extensive-stage (ES; n = 166, 69.7%) according to the
**[Veterans Administration Lung Study Group]** system.

## Clinical variables and survival endpoints

Twenty-one baseline clinical variables were retrieved from the electronic medical records.
The eight continuous variables were age at diagnosis, smoking pack-years, haemoglobin, white
blood cell count, lactate dehydrogenase (LDH), and the percent-predicted values of forced
vital capacity (FVC), forced expiratory volume in 1 s (FEV1) and diffusing capacity for
carbon monoxide (DLCO). The 13 categorical variables were sex, Eastern Cooperative Oncology
Group performance status, modified Medical Research Council dyspnoea grade, smoking status,
diabetes mellitus, hypertension, history of tuberculosis, chronic lung disease (COPD, asthma
or interstitial lung disease), disease stage, liver metastasis, brain metastasis,
atezolizumab use and type of thoracic radiotherapy. Laboratory values were those measured on
the day of, or immediately before, the first cycle of first-line chemotherapy.

Brain metastasis was coded as present only when it was detected at the time of diagnosis.
Brain metastases detected during follow-up (n = 34) were coded as absent, because a
metastasis first observed after diagnosis can only be recorded in patients who survive long
enough to develop it; treating it as a baseline covariate would therefore introduce immortal
time bias. Two patients with an unrecorded timing of brain metastasis were also coded as
absent.

Missing values of the continuous variables (DLCO, n = 52 [21.8%]; LDH, n = 39 [16.4%];
pack-years, n = 33 [13.9%]; FVC, n = 27 [11.3%]; FEV1, n = 26 [10.9%]; white blood cell
count and haemoglobin, n = 3 each [1.3%]) were imputed with the median of the whole cohort.
Missing categorical values (history of tuberculosis, n = 1; chronic lung disease, n = 31;
type of thoracic radiotherapy, n = 21) were encoded as a separate "unknown" category.
Because whole-cohort median imputation uses covariate values (but no outcome information)
from patients who are later assigned to the test folds, we performed a sensitivity analysis.
In that analysis, medians were estimated within each training fold, with and without
missing-indicator variables. The C-index did not change materially (Supplementary Table
**[SX]**). Categorical variables were entered as integer codes. Continuous variables were
standardised to zero mean and unit variance using the mean and standard deviation of the
training partition of each fold.

OS was defined as the time from **[the date of diagnosis]** to death from any cause. PFS was
defined as the time from **[the date of diagnosis]** to **[radiological progression according
to RECIST 1.1]** or death from any cause, whichever occurred first. Patients without an
event were censored at the date of last follow-up. **[Patients were followed up with chest
CT every X months.]** The data cut-off date was **[date]**.

## PET-CT acquisition and image preprocessing

All patients fasted for at least **[6]** h before the examination, and blood glucose was
**[below X mg/dL]** at the time of injection. PET-CT was performed **[60]** min after the
intravenous injection of **[X]** MBq/kg [¹⁸F]FDG on **[scanner, vendor]**. PET images were
reconstructed with **[an ordered-subset expectation maximisation algorithm (X iterations,
X subsets)]**, and low-dose CT was used for attenuation correction. The acquisition
parameters are summarised in Supplementary Table **[SX]**.

For each patient, a two-dimensional whole-body maximum-intensity projection (MIP) of the
pretreatment PET was **[generated/exported from the workstation]** and stored as an 8-bit
greyscale image. Five MIP images that had been stored with inverted greyscale values were
inverted back before analysis. The whole MIP was used as the network input without tumour
segmentation. Each image was resized to 512 × 512 pixels using bilinear interpolation,
scaled to [0, 1], and normalised to zero mean and unit variance using the pixel mean and
standard deviation of the training partition of each fold. During training only, images
were randomly flipped horizontally with a probability of 0.5. No other augmentation was
applied.

## Radiology report preprocessing

We used the radiology report of the same baseline PET-CT examination. Of the 238 reports,
232 were available as word-processor (DOCX) files and 6 as scanned images (JPG). Text was
extracted directly from the DOCX files. The JPG reports were converted to text by optical
character recognition, and the output was manually verified against the original images.
Each report was divided into its Findings, Conclusion and Recommendation sections. The
Conclusion and Findings sections were concatenated in that order; the Recommendation
section was excluded because it consisted largely of templated text. Dates and hospital
names were replaced with the placeholder tokens [DATE] and [HOSPITAL] using regular
expressions.

The reports were written predominantly in English (63.7% of characters). The Korean text
(8.8% of characters) consisted mainly of grammatical particles and fixed expressions of
negation and diagnostic certainty (e.g., "not observed", "suggestive of", "differential
diagnosis required"). The vocabulary of the English-language text encoder does not contain
Korean characters, so these expressions were mapped to [UNK] tokens (mean, 16.5% of tokens
per report). As a result, negated and affirmed findings became indistinguishable. We
therefore applied a fixed 446-entry Korean-to-English dictionary to all reports and removed
any Korean characters that remained. After mapping, no [UNK] tokens remained. The dictionary
was compiled from the report text alone, without reference to any outcome, and was applied
identically to all patients. No report text was transmitted to external services.

The processed text was encoded with RadBERT (StanfordAIMI/RadBERT), a BERT model pretrained
on radiology reports, with its weights frozen. Reports were tokenised with a maximum length
of 512 tokens, and longer reports (9.7%) were truncated. The last-layer token embeddings
were mean-pooled, weighted by the attention mask, to obtain a 768-dimensional report
representation. This representation was used without further dimensionality reduction or
scaling.

## Data partitioning and leakage control

The 238 patients were divided into five folds at the patient level. The split was stratified
by OS and PFS event status (random seed 42). In each fold, the test set comprised 47–48
patients. The remaining patients were divided into a training set (n = 171) and a validation
set (n = 19–20); the validation set was used only to select the checkpoint with the highest
validation C-index. Every patient appeared in the test set exactly once. The same fold
assignment was used for all modalities, models and both endpoints, so all comparisons are
paired at the patient level. All data-dependent preprocessing parameters — the
clinical-variable scaler and the image normalisation statistics — were estimated on the
training set of each fold and then applied unchanged to its validation and test sets. The
text encoder and the Korean-to-English dictionary were fixed in advance and were not fitted
to the study data. For late fusion, each unimodal arm produced out-of-fold risk scores, and
the fold-specific Cox combiner was fitted on the out-of-fold scores of that fold's training
patients only.

All analyses were performed with Python 3.12.3, PyTorch 2.12.1 (CUDA 12.6), scikit-learn
1.9.0, Hugging Face Transformers 5.14.1, lifelines 0.30.3 and pycox 0.3.0 on a single NVIDIA
GeForce RTX 4070 Ti SUPER GPU. **[Code availability statement.]**

---

## 작성 메모 (원고에 넣지 않음)

**본문 수치 출처**
- 코호트·제외 수: `sclc.cohort.build_manifest()`
- 병기·결측 수: `load_trimodal_cohort()`
- 뇌전이 재코딩: `apply_brain_meta_fix` docstring
- 판독지 통계(63.7%, 8.8%, [UNK] 16.5%, 446항목, 512토큰 초과 9.7%): `실험6/REPORT_ENCODER_FINAL.md` §5.2
- 영상 전처리: `dataset.create_dataset`
- 결측 민감도분석: `실험7/RESULTS_clinical_missing_model.md`. late fusion에서 fold 중앙값 + missing indicator(B)는 OS +0.010 [−0.004, +0.025], PFS +0.012 [−0.012, +0.038]이었다.
- 라이브러리 버전: 현재 환경에서 확인한 값

**채워야 할 항목:** [bracket] 부분 전부. 특히 다음 항목이다.
- OS/PFS 시작일
- 진행 판정 기준
- PET 획득 파라미터
- MIP 생성 방법
- 병기 체계
- 데이터 컷오프일

**리뷰어가 짚을 수 있는 지점**
1. **전체 코호트 중앙값 대치.** 본문에서는 누수 가능성을 먼저 인정하고 민감도분석으로 방어하게 썼다. 이 중앙값이 321명 기준인지 238명 기준인지는 확인하지 못했다(`data/Clinical/fillna_tabular_data_260619.csv` 생성 경위).
2. **범주형 정수코딩.** smoking_status, lung_disease, rt_type처럼 순서가 없는 변수도 one-hot 없이 정수로 들어간다. 본문에는 사실대로만 적었다.
3. **치료 변수.** atezolizumab 사용과 흉부 RT 유형은 기저 이후에 결정되는 변수라, 뇌전이와 같은 논리로 지적받을 수 있다. 최소한 Discussion 한계에 적는 것을 권한다.
4. **판독지 인코더 표기.** ~~MODEL_SUMMARY.md의 채택 모델은 TF-IDF이므로 맞춰야 한다~~ → **해결(2026-09-30):** 최종 채택이 RadBERT 로 확정돼 문서 전체를 RadBERT 기준으로 맞췄다.
5. **57·280번 환자.** 본문에는 "report date unavailable"로 썼다. 실제로는 고정 split 생성 이후 적격이 된 환자이므로 사유를 확인해야 한다.
6. **Results에 넣을 수치.** 추적관찰 중앙값은 역 KM으로 1,633일, OS 중앙값은 356일, PFS 중앙값은 183일이다.
