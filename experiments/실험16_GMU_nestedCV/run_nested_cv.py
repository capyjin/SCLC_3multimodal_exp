# -*- coding: utf-8 -*-
"""② 학습·평가: Late Fusion 과 특징 수준 융합 모델(Concat · Constant Gate · GMU)을 nested CV 로 학습하고
outer 시험 환자에서 평가한다. outer fold 마다 인코더와 융합 모델 후보를 inner CV 로 고른 뒤 다시 학습한다.
끊겼다가 다시 실행하면 끝난 outer 는 건너뛴다.
결과 파일의 모델 이름: A_primary = Late Fusion, B_concat = Concat, C_fixed_gate = Constant Gate, D_gmu = GMU.

Run:  python experiments/실험16_GMU_nestedCV/run_nested_cv.py --target os   (--smoke: 짧은 경로 점검)"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import glob
import json
import platform
import subprocess
import time

import numpy as np
import pandas as pd
import torch

from nested_fusion.config import (CANDIDATES, COMMON_RULE, ENCODERS, LR_RULES, MODALITIES, RUN_TAG, SEED,
                                  SELECTION_TOLERANCE, SIMPLICITY_ORDER, SMOKE, SPLIT_DIR, guard)
from nested_fusion.dataset import feats, labels_tensor, load_labels, load_manifest, load_outer_split
from nested_fusion.model import FUSION_MODELS, build, nested_late_fusion
from nested_fusion.processing import arm_records
from nested_fusion.train import (effective_candidates, refit_epochs, rule_for, run_encoders, select_candidate,
                                 train_fusion)
from sclc import cohort, features, paths
from sclc import foldsafe_clinical as fsc
from sclc.encoders import build_encoder
from sclc.evaluation import cindex
from sclc.train import cox_efron_loss, seed_everything

OUTER_FILES = ("inner_results.csv", "fold_metrics.csv", "oof_predictions.csv", "encoder_history.csv",
               "selection.json", "encoder_manifest.json", "preprocessing_manifest.json",
               "late_fusion_A.json", "gate_analysis.json")


def environment() -> dict:
    def git(*a):
        try:
            return subprocess.check_output(["git", *a], cwd=PROJECT_ROOT, text=True).strip()
        except Exception:
            return None
    import lifelines
    import sklearn
    return {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "lifelines": lifelines.__version__, "sklearn": sklearn.__version__,
            "git_commit": git("rev-parse", "HEAD"), "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain"))}


def gmu_rule(cfg: dict, smoke: bool) -> dict:
    return SMOKE["gmu"] if smoke else rule_for(cfg)


def experiment_config(args) -> dict:
    """실행 설정 기록 (이어서 실행할 때 저장된 설정과 같은지 비교한다)."""
    return {
        "experiment": "실험16 GMU nested CV", "target": args.target, "smoke": args.smoke,
        "cohort": {"n": 238, "source": "data/merged_tabular_with_reports.csv + data/splits/trimodal_common_5fold_seed42_v1.csv",
                   "brain_meta_fix": True, "lung_disease": "원본 엑셀 범주로 재코딩 (cohort.recode_lung_disease)",
                   "labels": "수정 없음"},
        "clinical_input": {"n": 19, "excluded": list(fsc.TREATMENT_COLUMNS),
                           "imputation": "fold-safe 7열 (학습 환자 중앙값) — sclc.foldsafe_clinical",
                           "scaling": "학습 환자로 StandardScaler", "categorical": "정수 코드 그대로 (fit 없음)"},
        "report": "RadBERT frozen mean-pooling 768 (ko2en, SVD·스케일 없음), Findings+Conclusion, 날짜·병원 마스킹",
        "image": "PET MIP 512x512 흑백, 반전 MIP 보정, 학습 환자 영상 mean/std 로 정규화, 증강 없음",
        "loss": "Efron Cox 부분우도 (sclc.train.cox_efron_loss; 인코더는 미니배치, 융합은 full-batch)",
        "splits": {"dir": os.path.relpath(args.split_dir, PROJECT_ROOT),
                   "outer": "기존 5-fold test 분할", "inner": "4-fold, OS 사건 층화, seed 42+100k",
                   "stop": "Tr 의 15%, OS 사건 층화, seed 42+100k+j"},
        "encoders": {k: {**v, "max_epochs": SMOKE["encoder_epochs"] if args.smoke else v["max_epochs"],
                         "min_epoch": 1 if args.smoke else v["min_epoch"]}
                     for k, v in ENCODERS.items()},
        "encoder_min_epoch_note": "tabular 20 / image 1 — 2026-10-14 첫 fold(v1) 점검 후 사용자 결정",
        "encoder_refit_epochs": "inner 최적 epoch 중앙값 (.5 올림), val 없이 고정 학습",
        "fusion_features": {"clinical": "128 (L2 후)", "report": "16 (L2 후)", "image": "512 (z-score 없음, N6)",
                            "projection": "Linear -> d, 이후 L2", "normalization_order": "projection -> L2"},
        "fusion_models": {"A": "2-arm late fusion, CoxPH(Efron) on inner OOF z-scores (nested_late_fusion)",
                          "B": "concat(projected) -> Dropout -> Linear", "C": "환자 공통 학습 gate (sigmoid)",
                          "D": "joint GMU gate (3d -> bottleneck -> 3, sigmoid)"},
        "candidates": CANDIDATES,
        "candidates_per_model": {m: list(effective_candidates(m)) for m in FUSION_MODELS},
        "candidate_note": "B·C 는 bottleneck 을 안 써서 G2 == G1 -> 본 실행 전(2026-10-14) G2 제외로 확정",
        "gmu_training": {"rules_by_lr": {str(k): v for k, v in LR_RULES.items()}, **COMMON_RULE,
                         "smoke_override": SMOKE["gmu"] if args.smoke else None,
                         "optimizer": "AdamW", "batch": "full-batch (1 epoch = 1 step)",
                         "monitor": "정지용 분할 S 의 Efron NLL (eval 모드)",
                         "best_epoch_rule": "epoch >= min_epoch 중 마지막 개선 epoch; 개선 = NLL < best - min_delta; 동점은 이른 epoch"},
        "selection": {"metric": "inner V C-index 4-fold 평균", "tolerance": SELECTION_TOLERANCE,
                      "simplicity_order": list(SIMPLICITY_ORDER)},
        "fusion_refit_epochs": "선택 후보의 inner 최적 epoch 중앙값 (.5 올림)",
        "late_fusion_A": {"primary": "모델별 자기 학습 환자 점수로 표준화", "sensitivity": ["ensemble", "raw"],
                          "diagnostics": ["calibration_slopes", "sd_ratio_outer_over_inner", "effective_image_weight"]},
        "evaluation": "outer test 별 Harrell C-index (sclc.evaluation.cindex; 큰 점수 = 높은 위험). OOF 를 합친 C-index 는 계산하지 않음",
        "seeds": {"encoder": "seed_everything(42 + 100k + j) (refit: 42 + 100k + k)",
                  "gmu": "torch.manual_seed(42*1000 + 100k + 10j + G 번호-1); refit 42*1000 + 100k + 90 + 모델 순번"},
        "reference_only": {"late_fusion_os": 0.7224, "late_fusion_pfs": 0.6470,
                           "note": "기존 비-nested 결과 — 정식 비교 기준 아님"},
        "environment": environment(),
    }


def null_nll(lab: tuple) -> float:
    return float(cox_efron_loss(torch.zeros_like(lab[0]), lab[0], lab[1]))


def encoder_manifest(store, ev, save_dir):
    rec = {}
    by_fold = {r["fold"]: r for r in ev.fold_records}
    for fold, s in store.items():
        rec[int(fold)] = {"train_ids": sorted(s["ids"]["train"]), "stop_ids": sorted(s["ids"]["val"]),
                          "eval_ids": sorted(s["ids"]["test"]),
                          "best_epoch": by_fold[fold].get("best_epoch"),
                          "epochs_run": sum(1 for h in ev.training_history if h["fold"] == fold),
                          "checkpoint_dir": os.path.relpath(save_dir, PROJECT_ROOT),
                          "state_sha256": s["hash"], "image_norm": s["image_norm"],
                          "feature_dims": {m: int(v.shape[1]) for m, v in s["splits"]["train"].items()
                                           if m in ("clinical", "report", "image")}}
    return rec


def history_frame(ev, kind, phase) -> pd.DataFrame:
    h = pd.DataFrame(ev.training_history)
    return h.assign(encoder=kind, phase=phase)


class OuterFold:
    """outer fold 하나의 전체 절차. run() 이 아래 메서드를 순서대로 부른다."""

    def __init__(self, k, args, ctx, out):
        self.k, self.args, self.ctx, self.out = k, args, ctx, out
        self.target, self.dev, self.labels = args.target, ctx["device"], ctx["labels"]
        self.om = next(o for o in ctx["manifest"]["outer"] if o["k"] == k)
        self.inner_csv = os.path.join(args.split_dir, f"outer{k}_inner.csv")
        self.refit_csv = os.path.join(args.split_dir, f"outer{k}_refit.csv")
        self.t_ids, self.o_ids = load_outer_split(self.refit_csv)
        self.seed_k = SEED + 100 * k
        self.audit_inner, self.audit_refit = [], []
        self.ck = os.path.join(out, "ckpt", f"outer{k}")
        self.inner = self.tab_s = self.img_s = self.folds = None
        self.refit = self.RT = self.RI = None

    def run(self) -> dict:
        self.train_inner_encoders()
        inner_df, hist_rows, stats_inner = self.search_candidates()
        selection, enc_refit_ep = self.select(inner_df)
        self.refit_encoders(enc_refit_ep)
        preds, gate_rows, stats_outer, refit_hist = self.refit_fusion(selection)
        a_res = self.late_fusion(preds)
        return self.outputs(inner_df, hist_rows, stats_inner, selection, enc_refit_ep,
                            preds, gate_rows, stats_outer, refit_hist, a_res)

    def train_inner_encoders(self) -> None:
        """1. inner fold 마다 인코더를 학습하고 특징을 뽑는다."""
        args = self.args
        enc_ep = {kind: (SMOKE["encoder_epochs"] if args.smoke else ENCODERS[kind]["max_epochs"]) for kind in ENCODERS}
        self.inner = {kind: run_encoders(kind, self.target, self.inner_csv, enc_ep[kind], ENCODERS[kind]["patience"],
                                         self.seed_k, os.path.join(self.ck, f"{kind}_inner"), self.o_ids, self.ctx,
                                         self.audit_inner, min_epoch=1 if args.smoke else None)
                      for kind in ENCODERS}
        self.tab_s, self.img_s = self.inner["tabular"][0], self.inner["image"][0]
        self.check_splits()
        self.folds = sorted(self.tab_s)

    def search_candidates(self):
        """2. 특징 수준 융합 모델 × 후보를 inner 학습 분할로 학습하고 검증 C-index 를 기록한다."""
        k, target, dev, labels = self.k, self.target, self.dev, self.labels
        rows, hist_rows, stats_inner = [], [], {}
        for j in self.folds:
            T, I, ids = self.tab_s[j]["splits"], self.img_s[j]["splits"], self.tab_s[j]["ids"]
            tr_lab, st_lab = labels_tensor(ids["train"], labels, target, dev), labels_tensor(ids["val"], labels, target, dev)
            tr = (feats(T["train"], I["train"], dev), *tr_lab)
            st = (feats(T["val"], I["val"], dev), *st_lab)
            nulls = {"train": null_nll(tr_lab), "stop": null_nll(st_lab)}
            v_feats = feats(T["test"], I["test"], dev)
            v_lab = labels.loc[ids["test"]]
            for name in FUSION_MODELS:
                for cand, cfg in effective_candidates(name).items():
                    ci = list(CANDIDATES).index(cand)
                    torch.manual_seed(SEED * 1000 + 100 * k + 10 * j + ci)
                    model = build(name, cfg).to(dev)
                    res = train_fusion(model, tr, st, cfg, cox_efron_loss, **gmu_rule(cfg, self.args.smoke))
                    with torch.no_grad():
                        v_risk = model(v_feats).squeeze(1).cpu().numpy()
                    guard(bool(np.isfinite(v_risk).all()), f"outer {k} inner {j} {name}/{cand}: V 위험점수 비유한값")
                    at = res["history"][res["best_epoch"] - 1]
                    rows.append({"outer": k, "inner": j, "model": name, "candidate": cand,
                                 "cindex_V": cindex(v_lab[f"{target}_days"], v_risk, v_lab[f"{target}_event"]),
                                 "best_epoch": res["best_epoch"], "epochs_run": res["epochs_run"],
                                 "rule_max_epochs": gmu_rule(cfg, self.args.smoke)["max_epochs"],
                                 "train_nll_at_best": at["train_loss_eval"], "stop_nll_at_best": at["stop_loss"],
                                 "train_gain_at_best": nulls["train"] - at["train_loss_eval"],
                                 "stop_gain_at_best": nulls["stop"] - at["stop_loss"],
                                 "null_train": nulls["train"], "null_stop": nulls["stop"]})
                    hist_rows.append(pd.DataFrame(res["history"]).assign(outer=k, inner=j, model=name, candidate=cand))
                    if cand == "G1":
                        stats_inner[f"{name}/inner{j}/V"] = model.feature_stats(v_feats)
        return pd.DataFrame(rows), hist_rows, stats_inner

    def select(self, inner_df):
        """3. 모델별 후보와 재학습 epoch 을 정한다."""
        selection = {}
        for name in FUSION_MODELS:
            sub = inner_df[inner_df.model == name]
            means = sub.groupby("candidate")["cindex_V"].mean().to_dict()
            chosen = select_candidate(means)
            chosen_rows = sub[sub.candidate == chosen]
            guard(len(chosen_rows) == len(self.folds), f"outer {self.k} {name}: 선택 후보의 inner 기록 수 이상")
            selection[name] = {"inner_mean_cindex": {c: round(v, 6) for c, v in means.items()},
                               "selected": chosen, "inner_best_epochs": chosen_rows["best_epoch"].tolist(),
                               "refit_epochs": refit_epochs(chosen_rows["best_epoch"])}
        enc_refit_ep = {kind: refit_epochs([r.get("best_epoch") for r in self.inner[kind][1].fold_records])
                        for kind in ENCODERS}
        return selection, enc_refit_ep

    def refit_encoders(self, enc_refit_ep) -> None:
        """4. outer 학습 환자 전체로 인코더를 다시 학습한다."""
        k = self.k
        self.refit = {kind: run_encoders(kind, self.target, self.refit_csv, enc_refit_ep[kind], None, self.seed_k,
                                         os.path.join(self.ck, f"{kind}_refit"), None, self.ctx, self.audit_refit,
                                         min_epoch=1)
                      for kind in ENCODERS}
        self.check_audits()
        for kind in ENCODERS:
            r = self.refit[kind][0][k]["ids"]
            guard(sorted(r["train"]) == sorted(self.t_ids) and not r["val"] and sorted(r["test"]) == sorted(self.o_ids),
                  f"outer {k} {kind} refit: T_k/O_k 불일치")
        self.RT, self.RI = self.refit["tabular"][0][k]["splits"], self.refit["image"][0][k]["splits"]

    def refit_fusion(self, selection):
        """5. 선택된 특징 수준 융합 모델을 다시 학습해 outer 시험 환자를 예측한다."""
        k, dev, out = self.k, self.dev, self.out
        RT, RI = self.RT, self.RI
        t_feats, o_feats = feats(RT["train"], RI["train"], dev), feats(RT["test"], RI["test"], dev)
        t_lab = labels_tensor(self.t_ids, self.labels, self.target, dev)
        preds, gate_rows, stats_outer, refit_hist = {}, {}, {}, []
        os.makedirs(os.path.join(out, "fusion_state", f"outer{k}"), exist_ok=True)
        for mi, name in enumerate(FUSION_MODELS):
            cfg = CANDIDATES[selection[name]["selected"]]
            torch.manual_seed(SEED * 1000 + 100 * k + 90 + mi)
            model = build(name, cfg).to(dev)
            res = train_fusion(model, (t_feats, *t_lab), None, cfg, cox_efron_loss,
                                  max_epochs=selection[name]["refit_epochs"])
            refit_hist.append(pd.DataFrame(res["history"]).assign(outer=k, model=name, candidate=selection[name]["selected"]))
            torch.save(model.state_dict(), os.path.join(out, "fusion_state", f"outer{k}", f"{name}.pt"))
            with torch.no_grad():
                preds[name] = model(o_feats).squeeze(1).cpu().numpy()
                g = model.gates(o_feats)
            if g is not None:
                gate_rows[name] = g.cpu().numpy()
            stats_outer[name] = {"T_k": model.feature_stats(t_feats), "O_k": model.feature_stats(o_feats)}
        return preds, gate_rows, stats_outer, refit_hist

    def late_fusion(self, preds) -> dict:
        """6. Late Fusion 예측을 preds 에 더한다."""
        k, RT, RI = self.k, self.RT, self.RI
        inner_recs, outer_rec = arm_records(self.tab_s, self.img_s, RT, RI, self.o_ids)
        a_res = nested_late_fusion(inner_recs, outer_rec, self.t_ids, self.labels.loc[self.t_ids], self.target)
        for v, p in a_res["pred"].items():
            preds[f"A_{v}"] = p
        preds["tabular_only"] = RT["test"]["risk"].numpy()
        preds["image_only"] = RI["test"]["risk"].numpy()
        for m, p in preds.items():
            guard(len(p) == len(self.o_ids) and bool(np.isfinite(p).all()), f"outer {k} {m}: O_k 예측 길이/비유한값 이상")
        return a_res

    def outputs(self, inner_df, hist_rows, stats_inner, selection, enc_refit_ep,
                preds, gate_rows, stats_outer, refit_hist, a_res) -> dict:
        """outer 시험 C-index 와 저장할 파일 내용."""
        k, target, o_ids, ck = self.k, self.target, self.o_ids, self.ck
        inner, refit = self.inner, self.refit
        o_lab = self.labels.loc[o_ids]
        metrics = [{"outer": k, "model": m, "n": len(o_ids), "events": int(o_lab[f"{target}_event"].sum()),
                    "cindex": cindex(o_lab[f"{target}_days"], p, o_lab[f"{target}_event"])}
                   for m, p in preds.items()]
        oof = pd.DataFrame({"research_id": o_ids, "outer": k, "target": target,
                            "duration": o_lab[f"{target}_days"].to_numpy(), "event": o_lab[f"{target}_event"].to_numpy(),
                            **{f"risk_{m}": p for m, p in preds.items()},
                            **{f"gate_{name}_{mod}": g[:, i] for name, g in gate_rows.items()
                               for i, mod in enumerate(MODALITIES)}})
        enc_hist = pd.concat([history_frame(inner[kind][1], kind, "inner") for kind in ENCODERS]
                             + [history_frame(refit[kind][1], kind, "refit") for kind in ENCODERS], ignore_index=True)

        return {
            "inner_results.csv": inner_df, "fold_metrics.csv": pd.DataFrame(metrics), "oof_predictions.csv": oof,
            "encoder_history.csv": enc_hist,
            "gmu_inner_history.csv.gz": pd.concat(hist_rows, ignore_index=True),
            "gmu_refit_history.csv": pd.concat(refit_hist, ignore_index=True),
            "selection.json": {**selection, "encoders_refit_epochs": enc_refit_ep},
            "encoder_manifest.json": {kind: {"inner": encoder_manifest(inner[kind][0], inner[kind][1],
                                                                       os.path.join(ck, f"{kind}_inner")),
                                             "refit": encoder_manifest(refit[kind][0], refit[kind][1],
                                                                       os.path.join(ck, f"{kind}_refit"))}
                                      for kind in ENCODERS},
            "preprocessing_manifest.json": {"foldsafe_audit_inner": self.audit_inner,
                                            "foldsafe_audit_refit": self.audit_refit},
            "late_fusion_A.json": {"coef_z": a_res["coef_z"], "coef_raw": a_res["coef_raw"],
                                   "diagnostics": a_res["diagnostics"]},
            "gate_analysis.json": {"inner_G1": stats_inner, "outer_selected": stats_outer},
        }

    def check_splits(self) -> None:
        """인코더가 본 환자가 확정 분할과 같은지, 시험 환자가 섞이지 않았는지 검사한다."""
        k, om, tab_s, img_s, t_ids, o_ids = self.k, self.om, self.tab_s, self.img_s, self.t_ids, self.o_ids
        o_set = set(o_ids)
        guard(sorted(o_ids) == om["outer_test_ids"], f"outer {k}: O_k 가 확정 분할과 다르다")
        guard(not (set(t_ids) & o_set) and len(t_ids) + len(o_ids) == 238, f"outer {k}: T_k/O_k 가 겹치거나 238명이 아니다")
        for j, f in enumerate(om["inner"], start=1):
            for name, s in (("tabular", tab_s), ("image", img_s)):
                guard(j in s, f"outer {k} {name}: inner {j} 결과가 없다")
                ids = s[j]["ids"]
                guard(sorted(ids["train"]) == f["ids"]["fit"] and sorted(ids["val"]) == f["ids"]["stop"]
                      and sorted(ids["test"]) == f["ids"]["val"], f"outer {k} {name} inner {j}: Fit/S/V 불일치")
                guard(not (set(ids["train"] + ids["val"] + ids["test"]) & o_set),
                      f"outer {k} {name} inner {j}: O_k 환자가 inner 분할에 들어 있다 (누출)")
            guard(tab_s[j]["ids"] == img_s[j]["ids"], f"outer {k} inner {j}: 두 인코더의 분할이 다르다")

    def check_audits(self) -> None:
        """결측 대치 중앙값이 학습 환자로만 계산됐는지 검사한다."""
        k, audit_inner, audit_refit, om, n_o = self.k, self.audit_inner, self.audit_refit, self.om, len(self.o_ids)
        guard(len(audit_inner) == 8, f"outer {k}: inner 대치 감사 기록 수 {len(audit_inner)} != 8")
        for j, f in enumerate(om["inner"]):
            a, b = audit_inner[2 * j], audit_inner[2 * j + 1]
            guard(a["n"] == {"train": len(f["ids"]["fit"]), "val": len(f["ids"]["stop"]), "test": len(f["ids"]["val"])},
                  f"outer {k} inner {j + 1}: 대치 감사 n 불일치 {a['n']}")
            guard(b["n"]["test"] == n_o and b["train_medians"] == a["train_medians"],
                  f"outer {k} inner {j + 1}: O_k extra 전처리가 Fit 중앙값과 다르다")
        guard(len(audit_refit) == 1 and audit_refit[0]["n"]["val"] == 0 and audit_refit[0]["n"]["test"] == n_o,
              f"outer {k}: refit 대치 감사 이상 {[a['n'] for a in audit_refit]}")


def save_outer(out, k, res) -> None:
    """임시 폴더에 다 쓴 뒤 outer{k}/ 로 옮긴다 (중단돼도 반쯤 쓴 결과가 남지 않게)."""
    d = os.path.join(out, f"outer{k}")
    tmp = d + ".partial"
    os.makedirs(tmp, exist_ok=True)
    for fname, obj in res.items():
        path = os.path.join(tmp, fname)
        if isinstance(obj, pd.DataFrame):
            obj.to_csv(path, index=False)
        else:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(obj, fh, ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, d)


def aggregate(out) -> None:
    """끝난 outer 결과를 파일별로 합친다."""
    done = sorted(int(os.path.basename(p)[5:]) for p in glob.glob(os.path.join(out, "outer[0-9]")))
    for fname in OUTER_FILES:
        parts = [os.path.join(out, f"outer{k}", fname) for k in done]
        if fname.endswith(".csv"):
            pd.concat([pd.read_csv(p) for p in parts], ignore_index=True).to_csv(os.path.join(out, fname), index=False)
        else:
            payload = {k: json.load(open(p, encoding="utf-8")) for k, p in zip(done, parts)}
            with open(os.path.join(out, fname), "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=1, default=str)
    oof = pd.read_csv(os.path.join(out, "oof_predictions.csv"))
    guard(not oof.research_id.duplicated().any(), "합본 OOF 에 중복 환자")
    if done == [1, 2, 3, 4, 5]:
        guard(oof.research_id.nunique() == 238, "합본 OOF 가 238명이 아니다")
    print(f"[aggregate] outers {done} -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--target", required=True, choices=("os", "pfs"))
    ap.add_argument("--outers", default="1,2,3,4,5")
    ap.add_argument("--smoke", action="store_true", help="outer 1, 짧은 epoch — 경로 점검 전용")
    ap.add_argument("--split_dir", default=SPLIT_DIR)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    outers = [1] if args.smoke else [int(x) for x in args.outers.split(",")]
    out = args.out_dir or paths.outputs("gmu_nested", ("pathcheck_" if args.smoke else f"{RUN_TAG}_") + args.target)
    os.makedirs(out, exist_ok=True)
    cfg_path = os.path.join(out, "experiment_config.json")
    config = experiment_config(args)
    if os.path.exists(cfg_path):
        prev = json.load(open(cfg_path, encoding="utf-8"))
        same = {k: prev.get(k) for k in config if k != "environment"} == \
               {k: json.loads(json.dumps(v, default=str)) for k, v in config.items() if k != "environment"}
        if not same:
            raise SystemExit(f"{cfg_path} 의 설정이 지금과 다르다 — 같은 폴더에 섞지 않는다.")
    else:
        with open(cfg_path, "w", encoding="utf-8") as fh:
            json.dump(config, fh, ensure_ascii=False, indent=1, default=str)

    seed_everything(SEED)
    corpus, _ = features.load_text_corpus(cohort.DEFAULT_MERGED_CSV)
    ctx = {"device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
           "raw": fsc.load_raw_values(), "text_kwargs": build_encoder("radbert").evaluator_kwargs(corpus),
           "labels": load_labels(args.target), "manifest": load_manifest(args.split_dir)}
    for k in outers:
        if os.path.exists(os.path.join(out, f"outer{k}", "fold_metrics.csv")):
            print(f"[run_nested_cv] outer {k} 은 이미 끝났다 — 건너뛴다 (덮어쓰지 않음)")
            continue
        t0 = time.time()
        res = OuterFold(k, args, ctx, out).run()
        save_outer(out, k, res)
        print(f"[run_nested_cv] target={args.target} outer {k} done in {time.time() - t0:.0f}s")
    aggregate(out)


if __name__ == "__main__":
    main()
