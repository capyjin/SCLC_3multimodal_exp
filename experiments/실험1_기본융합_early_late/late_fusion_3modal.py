# -*- coding: utf-8 -*-
"""Late fusion 3-way — 세 모달리티를 **각각 독립으로** 학습한 뒤 가중합.

같은 238명/5-fold 코호트에서 단일모달 3개를 따로 학습하고, fold별 OOF 위험점수
3개를 CoxPH 로 선형 결합한다(계수 = 가중치).

- image-only    : ``core.train.ImageOnlyEvaluator`` (이미지는 PNG 를 그때그때
                  읽어야 해서 pycox 를 쓸 수 없어 커스텀 Cox 루프를 쓴다)
- clinical-only : pycox CoxPH + torchtuples MLPVanilla, num_nodes=[128]*4,
                  dropout=0.5 — earlyfusion.md 의 clinical-only 절제와 같은 용량
- report-only   : pycox CoxPH + torchtuples MLPVanilla, num_nodes=[32,16],
                  dropout=0.3 — early fusion 의 판독지 브랜치와 동일 형태

⚠️ 이 파일의 clinical/report arm 은 **pycox 경로**라, 실험3(ablation)의
``clin_only``/``report_only``(커스텀 torch 루프)와는 다른 코드 경로다. 수치가
서로 달라도 버그가 아니며, 그래서 서로 재사용하지 않는다.

결합 함수(``combine_weighted_sum``)와 OOF 헬퍼는 ``core.fusion_stack`` 에 있다
— 2-way(실험1 method B)와 절차가 같아 한 곳으로 합쳤다.
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import torch.optim as optim
from lifelines.utils import concordance_index

from core import features
from core.fusion_stack import labels_by_id
from core.model import generate_net, get_cox_ph_model
from core.train import ImageOnlyEvaluator, fold_plan, seed_everything


def _run_pycox_arm(cohort_df, target, modality, x_by_split_fn, num_nodes, dropout,
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

    return _run_pycox_arm(cohort_df, target, "clinical_only", x_by_split, num_nodes, dropout,
                          lr, epochs, batch_size, max_folds, seed)


def run_report_only(cohort_df: pd.DataFrame, target: str, merged_csv: str, num_nodes=(32, 16),
                    dropout=0.3, tfidf_max_features=400, tfidf_ngram_range=(2, 4),
                    lr=1e-4, epochs=30, batch_size=16, max_folds=None, seed=42) -> dict:
    """판독지 TF-IDF 만 쓰는 단일모달 arm (pycox CoxPH)."""
    corpus, _ = features.load_text_corpus(merged_csv)

    def x_by_split(ids):
        enc = features.TfidfEncoder(max_features=tfidf_max_features, ngram_range=tfidf_ngram_range)
        return (enc.fit_transform([corpus.get(rid, "") for rid in ids["train"]]),
                enc.transform([corpus.get(rid, "") for rid in ids["val"]]),
                enc.transform([corpus.get(rid, "") for rid in ids["test"]]))

    return _run_pycox_arm(cohort_df, target, "report_only", x_by_split, num_nodes, dropout,
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
