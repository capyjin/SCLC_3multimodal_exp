# 논문 그림 캡션 초안 (영문)

빌드: `tools/build_paper_figures.sh` → 같은 폴더의 `.pdf` / `.svg` / `.png`.
소스는 TikZ(`fig1_late_fusion_overview.tex`, `fig2_fusion_strategies.tex`, 공통 스타일 `figstyle.tex`).
시각 언어는 Mixture-of-Recursions(MoR) 논문 Fig. 1/2 를 따랐다.

---

**Figure 1: Overview of the late-fusion survival model.**
(*Left*) The tabular arm jointly encodes the 21 clinical variables and the PET-CT
radiology report. Reports are de-identified and Korean phrases are mapped to English
before a frozen RadBERT produces a 768-d mean-pooled embedding; the clinical MLP
(4 × 128) and the report MLP (32 → 16) are ℓ₂-normalised, concatenated (144-d) and fed
to a single Cox head, yielding r_tab. Each Linear layer is followed by BatchNorm, ReLU and
dropout. This arm corresponds to the gray "Tabular arm" box in the middle.
(*Middle*) The full model. The image arm is a four-block CNN trained from scratch on the
2D PET-CT maximum-intensity projection (512 × 512, grayscale), yielding r_img. The two
arms are trained independently with the Cox partial likelihood under their own schedule
(batch 16 / 30 epochs vs. batch 32 / 60 epochs, best-validation-C-index checkpoint).
A two-covariate Cox proportional-hazards combiner then forms
risk = β_tab·r_tab + β_img·r_img.
(*Right*) Cross-validation pattern (n = 238, 5 folds, seed 42). In Stage 1 each arm model
k predicts only its held-out fold, so every patient receives an out-of-fold (OOF) risk
score from a model that never saw them. In Stage 2 the combiner for fold k is fitted on
the OOF scores of the other four folds' patients and evaluated on fold k; the reported
C-index is the mean over the five folds.

> ⚠️ 저자 확인 필요 (PFS 문장): TF-IDF 시절 §10 검정에서는 PFS 영상 계수가 0과 구분되지
> 않아 "PFS는 영상 축 제외"였다. 그러나 **RadBERT 고정 시** `outputs/radbert_full/results_pfs.json`
> 의 late fusion 영상 계수는 +0.143(양수), C-index 는 late 0.6470 vs tabular 0.6456 (Δ +0.001,
> 판정선 0.015 안). RadBERT 기준으로 §10 검정(β 신뢰구간·난수 대조)을 다시 돌린 뒤
> "For PFS the image arm is omitted…" 문장을 넣을지 결정할 것.

---

**Figure 2: Fusion strategies compared.** The three input pictograms are, from left to
right, the 2D PET-CT maximum-intensity projection (an actual de-identified study), the
21 clinical variables (blue/purple table) and the radiology report (pink document); the
same colour code is used for their branches. Dashed boxes mark models that are trained
independently. (*a*) Early fusion (concat): the CNN, clinical MLP and RadBERT–MLP branches
are ℓ₂-normalised and concatenated (128 ⊕ 128 ⊕ 16 = 272) into one Cox head, so all three
modalities share a single loss and training schedule. (*b*) Late fusion, three-way: each
modality is trained as its own Cox model and the three OOF risk scores are combined by a
Cox proportional-hazards model with three coefficients. (*c*) Late fusion, two-way (ours):
clinical and report are trained jointly as one tabular arm (concat 128 ⊕ 16 → Cox head),
the image arm is trained separately, and only two risk scores are combined. Joint training
of clinical and report preserves their interaction, while keeping the image arm separate
prevents it from dominating the shared representation.
