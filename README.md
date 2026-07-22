# SCLC Tri-modal Fusion: Image + Clinical + Report

Consolidates `clinical+image/SCLC_simple_CNN-main` (image backbone + clinical
branch) and `clinical+report/SCLC_report_unimodal_test-main` (clinical + TF-IDF
report branch) into one project with two fusion experiments, per
`DATA/SCLC_EXPERIMENT_PROTOCOL.md`.

## Cohort

- Tri-modal common cohort: **238 patients** with image + clinical + report all
  available. This equals the `report_common` cohort exactly (every
  report_common patient also has an image), so the existing
  `report_common_5fold_seed42_v1.csv` split is reused unchanged as
  `splits/trimodal_common_5fold_seed42_v1.csv` (5-fold, seed 42).
- `cohort.py` builds a documentation manifest (`cohort.build_manifest()`)
  cross-checked against the split file at load time.

## Architecture

- **Image**: `SimpleCNNBackbone` (4x ConvBlock, 512D) unchanged from
  `clinical+image/model.py` -> Linear(512,128)+ReLU+Dropout(0.2) -> L2 norm.
- **Clinical**: 21 features (8 standardized continuous + 13 categorical) ->
  [Linear-BatchNorm-ReLU-Dropout(0.5)] x4 @128 -> L2 norm. This is the
  BatchNorm+Dropout variant from
  `clinical+report/CODE/train_early_fusion_clinical_report.py`
  (validated in `earlyfusion.md`, OS C-index 0.7083), not the plain
  Linear+ReLU version in the original `clinical+image/model.py`.
- **Report**: char n-gram(2,4) TF-IDF (max_features=400, fit per train fold)
  -> [Linear-BatchNorm-ReLU-Dropout(0.3)] x(32,16) -> L2 norm.

### Early (concat) fusion -- `model.TrimodalConcatDeepSurv`, run via `train.TrimodalEvaluator`

concat(128+128+16=272) -> Dropout(0.3) -> Linear(272,1,bias=False) -> Cox risk
score. Trained end-to-end with the Cox negative partial log-likelihood loss.

### Late (out-level, weighted-sum) fusion -- `late_fusion.py`

Trains image-only / clinical-only / report-only unimodal DeepSurv models
independently (same cohort/split/seed), collects fold-wise OOF risk scores,
then fits `lifelines.CoxPHFitter` on the 3 OOF risk scores per fold -- the
learned coefficients are the "weighted sum". Extends
`clinical+report/main.py`'s 2-covariate `_run_late_fusion` to 3 modalities.

## Running

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
pip install -r requirements.txt  # for GPU: install torch/torchvision for your CUDA version first

python main.py --experiment early_fusion --mode smoke_test   # seconds, no torch needed
python main.py --experiment early_fusion --mode batch_smoke  # 1 fold, 2 epochs -- pipeline sanity check
python main.py --experiment early_fusion --mode train         # full 5-fold x 30 epochs x OS/PFS

python main.py --experiment late_fusion  --mode smoke_test
python main.py --experiment late_fusion  --mode batch_smoke   # combine step skipped (see main.py docstring)
python main.py --experiment late_fusion  --mode train
```

**Windows MAX_PATH note**: this repo path is already deeply nested
(`...\4)3multimodal_fusion\clinical+image+report\...`), and torch's installed
package tree (license files under `torch-*.dist-info/licenses/third_party/...`)
adds another ~120 characters of nested folders. `pip install` can fail with
`WinError 206` (filename too long) if the venv lives inside this project
folder. If that happens, create the venv somewhere shallow instead, e.g.
`python -m venv C:\venv_sclc_trimodal`, and run `main.py` by pointing at that
interpreter (`C:\venv_sclc_trimodal\Scripts\python.exe main.py ...`) with this
directory as the working directory. This is how the smoke/batch_smoke
verification below was actually run.

Full `--mode train` runs (image CNN x 5 folds x 30 epochs x 2 targets, twice
for both experiments) are expensive on CPU -- run on a GPU machine. Outputs
land in `outputs/EXP_<date>_<experiment>_<mode>/` per
`SCLC_EXPERIMENT_PROTOCOL.md` section 11 (resolved_config.yaml,
data_manifest.csv, splits.csv, environment.txt, checkpoints/, metrics/,
experiment_report.md).

## Verification status (this build)

Both experiments were smoke-tested end-to-end on CPU in this environment:

- `smoke_test` (both experiments): PASS -- 238-patient cohort, 5-fold split
  with no train/val/test overlap, report corpus loads.
- `batch_smoke` (both experiments, target=os, 1 fold x 2 epochs): PASS --
  real forward/backward/optimizer-step training ran for image+clinical+report
  jointly (early_fusion) and for each unimodal arm (late_fusion); artifacts
  written to `outputs/EXP_20260722_early_fusion_batch_smoke/` and
  `outputs/EXP_20260722_late_fusion_batch_smoke/`. **These are pipeline
  sanity-check numbers only (1 fold, 2 epochs) -- not real results.** Full
  `--mode train` (30 epochs x 5 folds x OS/PFS) has not been run; do that on a
  GPU machine before drawing any conclusions.

## Protocol note (section 4.4)

Adding the report modality means the tri-modal cohort (238) is smaller than
the original image+clinical baseline cohort (257). Per protocol, a fair
comparison re-evaluates the image+clinical (2-modal) baseline on this same
238-patient cohort before concluding fusion helped -- that re-run isn't
included in this build (out of the two requested experiments' scope) but the
same `cohort.py`/`train.TrimodalEvaluator` machinery (skip the report branch)
would produce it if needed.

## Provenance

Source projects (`clinical+image/SCLC_simple_CNN-main`,
`clinical+report/SCLC_report_unimodal_test-main`) were GitHub-archive
extracts, not git checkouts -- no history was lost by consolidating. They are
kept until this project's smoke tests are confirmed working, then removed.
