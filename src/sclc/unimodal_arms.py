# -*- coding: utf-8 -*-
"""단일모달 **pycox** arm — 3-way late fusion 이 쓰는 임상/판독지/영상 축.

[왜 src/sclc 에 있나]
  이 코드는 실험 폴더 안의 ``실험1_기본융합_early_late/late_fusion_3modal.py``
  였는데, 실험 파일이면서 동시에 **라이브러리**로 쓰이고 있었다:

      exp_late_fusion_3modal_rerun.py :  import late_fusion_3modal as lf
      main.py                          :  sys.path.insert(... 실험1 폴더 ...)
                                          import late_fusion_3modal as lf

  ``main.py`` 가 실험 폴더를 sys.path 에 끼워 넣어야만 돌아간다는 건, 그 코드가
  실험이 아니라 인프라라는 뜻이다. 저장소 규칙("두 곳 이상이 쓰는 코드는
  ``src/sclc`` 로 올린다")대로 여기로 옮겼다. 함수 이름·시그니처·기본값은
  전부 그대로라서 호출부는 import 경로만 바뀐다.

[이 arm 들이 ``sclc.train`` 의 평가기와 다른 이유 — 재사용하면 안 된다]
  여기 clinical/report arm 은 **pycox CoxPH + torchtuples MLPVanilla** 경로다.
  실험3(ablation)의 ``clin_only``/``report_only`` 는 커스텀 torch Cox 루프이고,
  같은 모달리티라도 코드 경로가 다르다. 수치가 서로 달라도 버그가 아니며,
  그래서 서로 재사용하지 않는다. 이 파일이 존재하는 유일한 목적은 2026-07-22
  3-way 실행과 **같은 아키텍처**를 유지하는 것이다.

  - clinical-only : MLPVanilla num_nodes=[128]*4, dropout=0.5
  - report-only   : MLPVanilla num_nodes=[32,16], dropout=0.3 (TF-IDF 400)
  - image-only    : ``sclc.train.ImageOnlyEvaluator`` (PNG 를 그때그때 읽어야 해
                    pycox 를 쓸 수 없다 -> 커스텀 Cox 루프)

[누수 방지]
  fold 마다 인코더(임상 스케일러/TF-IDF)를 **train 환자로만 fit** 한다.
  ``x_by_split_fn`` 콜러블이 그 계약을 지는 유일한 지점이다.
"""
import pandas as pd
import torch.optim as optim
from lifelines.utils import concordance_index

from sclc import cohort, features
from sclc.late_fusion import labels_by_id
from sclc.model import generate_net, get_cox_ph_model
from sclc.train import ImageOnlyEvaluator, fold_plan, seed_everything


def run_pycox_arm(cohort_df, target, modality, x_by_split_fn, num_nodes, dropout,
                  lr, epochs, batch_size, max_folds, seed) -> dict:
    """clinical-only / report-only 가 공유하는 fold 루프.

    두 함수는 **인코더만 다르고** (임상 인코더 vs TF-IDF) 나머지 —
    fold 순회, 라벨 꺼내기, pycox 모델 생성/학습, OOF 기록 — 가 글자 단위로
    같았다. 달라지는 부분만 ``x_by_split_fn(ids)`` 콜러블로 받는다.
    이 콜러블은 **train id 로만 인코더를 fit** 해야 한다(누수 방지).
    """
    labels = labels_by_id(cohort_df, target)
    fold_records, oof = [], []

    for fold, ids in fold_plan(cohort_df, max_folds=max_folds):
        seed_everything(seed + fold)
        x_train, x_val, x_test = x_by_split_fn(ids)

        def _y(split):
            return (labels.loc[ids[split], f"{target}_days"].to_numpy("float32"),
                    labels.loc[ids[split], f"{target}_event"].to_numpy("float32"))

        y_train, y_val = _y("train"), _y("val")
        y_test_dur, y_test_evt = _y("test")

        net = generate_net(in_features=x_train.shape[1], num_nodes=list(num_nodes), dropout=dropout)
        model = get_cox_ph_model(net, optim.Adam, lr)
        model.fit(x_train, y_train, batch_size, epochs, verbose=False, val_data=(x_val, y_val))

        risk_test = model.predict(x_test).flatten()
        ci = concordance_index(y_test_dur, -risk_test, y_test_evt)
        fold_records.append({"target": target, "modality": modality, "fold": fold,
                             "c_index": float(ci), "n": len(ids["test"])})
        oof.extend({"research_id": rid, "target": target, "modality": modality, "fold": fold,
                    "duration": float(d), "event": float(e), "risk_score": float(r)}
                   for rid, d, e, r in zip(ids["test"], y_test_dur, y_test_evt, risk_test))
        print(f"[late_fusion/{modality}/{target}] fold {fold}: C-index={ci:.4f}")

    return {"fold_records": fold_records, "oof_predictions": oof}


def run_clinical_only(cohort_df: pd.DataFrame, target: str, num_nodes=(128, 128, 128, 128),
                      dropout=0.5, lr=1e-4, epochs=30, batch_size=16, max_folds=None,
                      seed=42) -> dict:
    """임상변수 21개만 쓰는 단일모달 arm (pycox CoxPH)."""
    clinical_frame = cohort_df.drop_duplicates("research_id").set_index("research_id")
    standardize_cols, categorical_cols = features.resolve_clinical_columns(clinical_frame)

    def x_by_split(ids):
        enc = features.ClinicalEncoder(clinical_frame, standardize_cols, categorical_cols)
        return (enc.fit_transform(ids["train"]), enc.transform(ids["val"]), enc.transform(ids["test"]))

    return run_pycox_arm(cohort_df, target, "clinical_only", x_by_split, num_nodes, dropout,
                         lr, epochs, batch_size, max_folds, seed)


def run_report_only(cohort_df: pd.DataFrame, target: str,
                    merged_csv: str = cohort.DEFAULT_MERGED_CSV, num_nodes=(32, 16),
                    dropout=0.3, tfidf_max_features=400, tfidf_ngram_range=(2, 4),
                    lr=1e-4, epochs=30, batch_size=16, max_folds=None, seed=42) -> dict:
    """판독지 TF-IDF 만 쓰는 단일모달 arm (pycox CoxPH)."""
    corpus, _ = features.load_text_corpus(merged_csv)

    def x_by_split(ids):
        enc = features.TfidfEncoder(max_features=tfidf_max_features, ngram_range=tfidf_ngram_range)
        return (enc.fit_transform([corpus.get(rid, "") for rid in ids["train"]]),
                enc.transform([corpus.get(rid, "") for rid in ids["val"]]),
                enc.transform([corpus.get(rid, "") for rid in ids["test"]]))

    return run_pycox_arm(cohort_df, target, "report_only", x_by_split, num_nodes, dropout,
                         lr, epochs, batch_size, max_folds, seed)


def run_image_only(target: str, merged_csv, image_dir, split_csv, epochs=30, batch_size=16,
                   resize=512, gray_scale=True, lr=1e-4, weight_decay=1e-4,
                   save_dir="checkpoints/late_fusion_image_only", max_folds=None, seed=42,
                   num_workers=4, device=None) -> dict:
    """영상 단독 arm (커스텀 Cox 루프)."""
    ev = ImageOnlyEvaluator(
        target=target, merged_csv=merged_csv, image_dir=image_dir, split_csv=split_csv,
        resize=resize, gray_scale=gray_scale, lr=lr, weight_decay=weight_decay,
        epochs=epochs, batch_size=batch_size, save_dir=save_dir, max_folds=max_folds,
        device=device, num_workers=num_workers, seed=seed,
    ).run()
    return {"fold_records": ev.fold_records, "oof_predictions": ev.oof_predictions,
            "training_history": ev.training_history}
