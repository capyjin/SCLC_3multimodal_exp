# -*- coding: utf-8 -*-
"""[실험16-3] 저장된 seed42 tabular 체크포인트로 기존 OS 0.7153 을 재현한다 — 학습 없음, 추론만.

체크포인트 : outputs/late_fusion_seed_sweep_radbert/ckpt/seed42/tabular_<target>/fold{k}_early_fusion_<target>.pt
             (README 의 outputs/radbert_full/ 에는 tab_radbert 체크포인트가 없다 — 첫 점검 보고 참고)
설정       : 기존 그대로 — 21개 임상변수(CSV 전체 중앙값 대치), brain_meta 수정, lung_disease 원래 코드,
             RadBERT(ko2en, raw 768), 기존 5-fold 분할의 train fold 로만 scaler 적합.
대조       : (1) fold별 C-index 가 runs.jsonl 의 seed42 기록과 같은가
             (2) 환자별 위험점수가 oof/seed42_<target>.json 의 tabular 값과 같은가
체크포인트는 읽기만 한다. 결과는 outputs/gmu_nested/repro_tabular_<target>.json.

Run:  python experiments/실험16_GMU_nestedCV/diagnostics/check_tabular_checkpoint.py --target os
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))                 # diagnostics/
EXP_DIR = os.path.dirname(HERE)                                   # 실험16 폴더 (nested_fusion 패키지)
PROJECT_ROOT = os.path.dirname(os.path.dirname(EXP_DIR))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, HERE)

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from sclc import cohort, features, paths
from sclc import dataset as ds
from sclc.encoders import build_encoder
from sclc.evaluation import cindex
from sclc.model import MODALITY_CONFIGS, make_model_factory
from sclc.train import fold_plan

SWEEP = paths.outputs("late_fusion_seed_sweep_radbert")


def known_record(target: str) -> dict:
    with open(os.path.join(SWEEP, "runs.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec["seed"] == 42 and rec["target"] == target:
                return rec
    raise KeyError(target)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="os", choices=("os", "pfs"))
    args = ap.parse_args()
    target = args.target
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cdf = cohort.load_trimodal_cohort()                      # 기존 기본값 그대로
    clin = cdf.drop_duplicates("research_id").set_index("research_id")
    std_cols, cat_cols = features.resolve_clinical_columns(clin)
    corpus, _ = features.load_text_corpus(cohort.DEFAULT_MERGED_CSV)
    text_fn = build_encoder("radbert").evaluator_kwargs(corpus)["text_encoder_fn"]
    factory = make_model_factory(MODALITY_CONFIGS["clin_report"])
    with open(os.path.join(SWEEP, "oof", f"seed42_{target}.json"), encoding="utf-8") as fh:
        stored = {int(k): float(v) for k, v in json.load(fh)["tabular"].items()}

    folds, max_abs_risk_diff = [], 0.0
    for fold, ids in fold_plan(cdf):
        tab, clin_dim, rep_dim = features.build_fold_multimodal_tabular(
            clin.loc[ids["train"]], clin.loc[ids["val"]], clin.loc[ids["test"]], corpus,
            std_cols, cat_cols, text_encoder_fn=text_fn)
        train_s = ds.preprocess_data(cohort.DEFAULT_IMAGE_DIR, clin.loc[ids["train"]].reset_index(),
                                     target, True)
        test_s = ds.preprocess_data(cohort.DEFAULT_IMAGE_DIR, clin.loc[ids["test"]].reset_index(),
                                    target, True)
        _, test_ds = ds.create_dataset(train_s, test_s, 512, False, True, False,
                                       train_tabular=tab["train"], test_tabular=tab["test"])
        model = factory(clin_dim, rep_dim).to(device)
        ckpt = os.path.join(SWEEP, "ckpt", "seed42", f"tabular_{target}", f"fold{fold}_early_fusion_{target}.pt")
        model.load_state_dict(torch.load(ckpt, map_location=device), strict=True)
        model.eval()
        risks, dur, evt = [], [], []
        with torch.no_grad():
            for img, t, d, e in DataLoader(test_ds, batch_size=32, shuffle=False):
                risks += model(img.to(device), t.to(device)).squeeze(1).cpu().tolist()
                dur += d.tolist()
                evt += e.tolist()
        diff = max(abs(r - stored[i]) for i, r in zip(ids["test"], risks))
        max_abs_risk_diff = max(max_abs_risk_diff, diff)
        folds.append(round(cindex(dur, risks, evt), 6))
        print(f"fold {fold}: C-index {folds[-1]:.6f}  max |risk - stored OOF| = {diff:.2e}")

    known = known_record(target)["tabular_only"]
    out = {"target": target, "folds": folds, "mean": float(np.mean(folds)),
           "known_folds": known["folds"], "known_mean": known["mean"],
           "max_abs_fold_diff": float(max(abs(a - b) for a, b in zip(folds, known["folds"]))),
           "max_abs_risk_diff_vs_stored_oof": max_abs_risk_diff}
    out["reproduced"] = out["max_abs_fold_diff"] < 1e-4 and abs(out["mean"] - known["mean"]) < 1e-4
    path = paths.outputs("gmu_nested", f"repro_tabular_{target}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
