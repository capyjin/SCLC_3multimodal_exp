# -*- coding: utf-8 -*-
"""⑥ 비교 모델(논문 Table 2·3): 단일 모달(임상 · 판독지 인코더 4종 · 영상 백본 2종)과 영상이 묶인 joint 모델 2종을
실험16 인코더와 같은 nested 절차로 학습하고 outer 시험 환자에서 평가한다.

  outer k 마다  inner 4-fold: Fit 으로 학습, S(정지용) Efron NLL 로 조기 종료 → fold 별 최적 epoch
               T_k 전체로 최적 epoch 중앙값만큼 고정 학습 → O_k 위험점수 · C-index
분할 · seed · 정지 규칙 · 전처리(임상 19변수, fold-safe 대치)는 run_nested_cv.py 의 인코더 학습과 같다.
끊겼다가 다시 실행하면 끝난 (모델, outer) 는 건너뛴다.

--verify: 실험16 본 실행의 두 arm(clin_report = tabular, image_simplecnn = image)을 이 경로로 다시 학습해
          run_v2 의 가중치 해시 · O_k 위험점수 · C-index 와 같은지 확인한다 (이 경로가 인코더 학습과 같다는 증거).

Run:  python experiments/실험16_GMU_nestedCV/run_baselines.py --target os --verify --outers 1
      python experiments/실험16_GMU_nestedCV/run_baselines.py --target os
      python experiments/실험16_GMU_nestedCV/run_baselines.py --target os --smoke --out_dir <새 폴더>"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import glob
import json
import time

import numpy as np
import pandas as pd
import torch

from nested_fusion.config import (BASELINE_TAG, BASELINES, ENCODERS, REPORT_BERT_MODELS, RUN_TAG, SEED, SMOKE,
                                  SPLIT_DIR, guard)
from nested_fusion.dataset import load_labels, load_manifest, load_outer_split
from nested_fusion.processing import report_encoder
from nested_fusion.train import refit_epochs, run_baseline
from run_nested_cv import environment
from sclc import cohort, features, paths
from sclc import foldsafe_clinical as fsc
from sclc.evaluation import cindex
from sclc.train import seed_everything

TABLE_BASELINES = tuple(n for n, s in BASELINES.items() if not s.get("verify_only"))
VERIFY_BASELINES = {"clin_report": "tabular", "image_simplecnn": "image"}    # 이름 -> run_v2 의 arm 이름
OUTER_FILES = ("fold_metrics.json", "oof_predictions.csv", "encoder_manifest.json", "history.csv",
               "preprocessing_manifest.json")


def experiment_config(args, names) -> dict:
    return {
        "experiment": "실험16 ⑥ 비교 모델 (nested)", "target": args.target, "smoke": args.smoke,
        "baselines": {n: BASELINES[n] for n in names},
        "rules_by_trainer": {k: {**v, "max_epochs": SMOKE["encoder_epochs"] if args.smoke else v["max_epochs"],
                                 "min_epoch": 1 if args.smoke else v["min_epoch"]} for k, v in ENCODERS.items()},
        "report_bert_models": REPORT_BERT_MODELS,
        "splits": os.path.relpath(args.split_dir, PROJECT_ROOT),
        "procedure": "inner 4-fold (Fit 학습 / S Efron NLL 조기 종료) → 최적 epoch 중앙값(.5 올림)으로 T_k 재학습 → O_k C-index",
        "seeds": "seed_everything(42 + 100k + fold), run_nested_cv 인코더와 같음",
        "clinical_input": "19변수 (치료 변수 제외), fold-safe 대치, run_nested_cv 와 같음",
        "evaluation": "outer test 별 Harrell C-index (sclc.evaluation.cindex)",
    }


def report_kwargs(names, corpus) -> dict:
    """모델이 쓰는 판독지 인코더별 TrimodalEvaluator 인자 (임베딩은 한 번만 계산)."""
    needed = sorted({BASELINES[n]["report"] for n in names if BASELINES[n]["trainer"] == "tabular"})
    out = {}
    for enc_name in needed:
        enc = report_encoder(enc_name)
        out[enc_name] = enc.evaluator_kwargs(corpus)
        if enc_name not in ("tfidf",):
            emb = enc.embeddings(corpus)
            dims = {len(v) for v in emb.values()}
            guard(dims == {768}, f"{enc_name}: 임베딩 차원 {dims} != 768")
            guard(set(cohort.load_trimodal_cohort().research_id.astype(int)) <= set(emb),
                  f"{enc_name}: 임베딩이 없는 환자가 있다")
        print(f"[run_baselines] report encoder {enc_name}: {enc.description}")
    return out


class BaselineOuter:
    """비교 모델 하나 × outer fold 하나."""

    def __init__(self, name, k, args, ctx, out):
        self.name, self.k, self.args, self.ctx = name, k, args, ctx
        self.spec, self.target, self.labels = BASELINES[name], args.target, ctx["labels"]
        self.om = next(o for o in ctx["manifest"]["outer"] if o["k"] == k)
        self.inner_csv = os.path.join(args.split_dir, f"outer{k}_inner.csv")
        self.refit_csv = os.path.join(args.split_dir, f"outer{k}_refit.csv")
        self.t_ids, self.o_ids = load_outer_split(self.refit_csv)
        self.seed_k = SEED + 100 * k
        self.ck = os.path.join(out, "ckpt", name, f"outer{k}")
        self.audit_inner, self.audit_refit = [], []

    def run(self) -> dict:
        rule = ENCODERS[self.spec["trainer"]]
        max_ep = SMOKE["encoder_epochs"] if self.args.smoke else rule["max_epochs"]
        s_in, ev_in = run_baseline(self.name, self.target, self.inner_csv, max_ep, rule["patience"], self.seed_k,
                                   os.path.join(self.ck, "inner"), self.o_ids, self.ctx, self.audit_inner,
                                   min_epoch=1 if self.args.smoke else None)
        self.check_inner(s_in)
        best = [r.get("best_epoch") for r in sorted(ev_in.fold_records, key=lambda r: r["fold"])]
        guard(all(isinstance(b, int) and b >= 1 for b in best), f"{self.tag}: inner 최적 epoch 이 비었다 {best}")
        ep = refit_epochs(best)
        s_re, ev_re = run_baseline(self.name, self.target, self.refit_csv, ep, None, self.seed_k,
                                   os.path.join(self.ck, "refit"), None, self.ctx, self.audit_refit, min_epoch=1)
        self.check_refit(s_re, ev_re, ep)
        self.check_audits()
        pred = self.outer_predictions(ev_re)
        o_lab = self.labels.loc[self.o_ids]
        ci = cindex(o_lab[f"{self.target}_days"], pred, o_lab[f"{self.target}_event"])
        guard(abs(ci - float(ev_re.c_indices[0])) < 1e-9, f"{self.tag}: C-index 재계산 불일치 {ci} vs {ev_re.c_indices[0]}")
        inner_v = {int(r["fold"]): float(r["c_index"]) for r in ev_in.fold_records}
        metrics = {"outer": self.k, "model": self.name, "n": len(self.o_ids),
                   "events": int(o_lab[f"{self.target}_event"].sum()), "cindex": ci,
                   "inner_best_epochs": best, "refit_epochs": ep, "inner_V_cindex": inner_v}
        manifest = {"inner": {int(j): {"train_ids": sorted(s["ids"]["train"]), "stop_ids": sorted(s["ids"]["val"]),
                                       "eval_ids": sorted(s["ids"]["test"]), "state_sha256": s["hash"],
                                       "image_norm": s["image_norm"]} for j, s in s_in.items()},
                    "refit": {int(j): {"train_ids": sorted(s["ids"]["train"]), "eval_ids": sorted(s["ids"]["test"]),
                                       "state_sha256": s["hash"], "image_norm": s["image_norm"]}
                              for j, s in s_re.items()}}
        hist = pd.concat([pd.DataFrame(ev_in.training_history).assign(phase="inner"),
                          pd.DataFrame(ev_re.training_history).assign(phase="refit")], ignore_index=True)
        oof = pd.DataFrame({"research_id": self.o_ids, "outer": self.k, "target": self.target,
                            "duration": o_lab[f"{self.target}_days"].to_numpy(),
                            "event": o_lab[f"{self.target}_event"].to_numpy(), f"risk_{self.name}": pred})
        return {"fold_metrics.json": metrics, "oof_predictions.csv": oof, "encoder_manifest.json": manifest,
                "history.csv": hist.assign(model=self.name, outer=self.k),
                "preprocessing_manifest.json": {"foldsafe_audit_inner": self.audit_inner,
                                                "foldsafe_audit_refit": self.audit_refit}}

    @property
    def tag(self) -> str:
        return f"{self.name} outer {self.k}"

    def check_inner(self, s_in) -> None:
        """inner 분할이 확정 분할(Fit/S/V)과 같고 O_k 가 섞이지 않았는지."""
        o_set = set(self.o_ids)
        guard(sorted(self.o_ids) == self.om["outer_test_ids"], f"{self.tag}: O_k 가 확정 분할과 다르다")
        guard(not (set(self.t_ids) & o_set) and len(self.t_ids) + len(self.o_ids) == 238,
              f"{self.tag}: T_k/O_k 가 겹치거나 238명이 아니다")
        guard(sorted(s_in) == [1, 2, 3, 4], f"{self.tag}: inner fold {sorted(s_in)}")
        for j, f in enumerate(self.om["inner"], start=1):
            ids = s_in[j]["ids"]
            guard(sorted(ids["train"]) == f["ids"]["fit"] and sorted(ids["val"]) == f["ids"]["stop"]
                  and sorted(ids["test"]) == f["ids"]["val"], f"{self.tag} inner {j}: Fit/S/V 불일치")
            guard(not (set(ids["train"] + ids["val"] + ids["test"]) & o_set), f"{self.tag} inner {j}: O_k 누출")
            guard(not s_in[j]["training_flag"], f"{self.tag} inner {j}: 평가 시 train 모드")

    def check_refit(self, s_re, ev_re, ep) -> None:
        guard(list(s_re) == [self.k], f"{self.tag}: refit fold {list(s_re)}")
        r = s_re[self.k]["ids"]
        guard(sorted(r["train"]) == sorted(self.t_ids) and not r["val"] and sorted(r["test"]) == sorted(self.o_ids),
              f"{self.tag} refit: T_k/O_k 불일치")
        h = [row for row in ev_re.training_history]
        guard(len(h) == ep and h[-1]["saved"] and sum(bool(x["saved"]) for x in h) == 1,
              f"{self.tag} refit: 고정 {ep} epoch 학습·마지막 epoch 저장이 아니다")

    def check_audits(self) -> None:
        """결측 대치 중앙값이 학습 환자로만 계산됐는지 (임상 입력이 있는 trainer 만)."""
        if self.spec["trainer"] != "tabular":
            guard(not self.audit_inner and not self.audit_refit, f"{self.tag}: 영상 모델에 임상 대치 기록")
            return
        n_o = len(self.o_ids)
        guard(len(self.audit_inner) == 8, f"{self.tag}: inner 대치 감사 기록 수 {len(self.audit_inner)} != 8")
        for j, f in enumerate(self.om["inner"]):
            a, b = self.audit_inner[2 * j], self.audit_inner[2 * j + 1]
            guard(a["n"] == {"train": len(f["ids"]["fit"]), "val": len(f["ids"]["stop"]), "test": len(f["ids"]["val"])},
                  f"{self.tag} inner {j + 1}: 대치 감사 n 불일치 {a['n']}")
            guard(b["n"]["test"] == n_o and b["train_medians"] == a["train_medians"],
                  f"{self.tag} inner {j + 1}: O_k extra 전처리가 Fit 중앙값과 다르다")
        guard(len(self.audit_refit) == 1 and self.audit_refit[0]["n"]["val"] == 0
              and self.audit_refit[0]["n"]["test"] == n_o, f"{self.tag}: refit 대치 감사 이상")

    def outer_predictions(self, ev_re) -> np.ndarray:
        """refit evaluator 의 test(O_k) 위험점수를 O_k 순서로. 레이블이 확정 레이블과 같은지도 확인한다."""
        df = pd.DataFrame(ev_re.oof_predictions).set_index("research_id")
        guard(sorted(df.index) == sorted(self.o_ids) and not df.index.duplicated().any(), f"{self.tag}: O_k 예측 ID 이상")
        df = df.loc[self.o_ids]
        lab = self.labels.loc[self.o_ids]
        guard(np.allclose(df["duration"].to_numpy(float), lab[f"{self.target}_days"].to_numpy(float))
              and np.array_equal(df["event"].to_numpy(float), lab[f"{self.target}_event"].to_numpy(float)),
              f"{self.tag}: evaluator 레이블이 확정 레이블과 다르다")
        pred = df["risk_score"].to_numpy(float)
        guard(bool(np.isfinite(pred).all()) and pred.std() > 0, f"{self.tag}: O_k 위험점수 비유한값/상수")
        return pred


def verify_against_run_v2(name, k, target, res) -> dict:
    """--verify: 같은 경로로 다시 학습한 arm 이 run_v2 와 같은가."""
    arm = VERIFY_BASELINES[name]
    run = paths.outputs("gmu_nested", f"{RUN_TAG}_{target}", f"outer{k}")
    man = json.load(open(os.path.join(run, "encoder_manifest.json"), encoding="utf-8"))[arm]
    sel = json.load(open(os.path.join(run, "selection.json"), encoding="utf-8"))
    ref = pd.read_csv(os.path.join(run, "oof_predictions.csv")).set_index("research_id")
    mine = res["encoder_manifest.json"]
    oof = res["oof_predictions.csv"].set_index("research_id")
    ref_pred = ref.loc[oof.index, f"risk_{arm}_only"].to_numpy(float)
    out = {"inner_hash_equal": {j: mine["inner"][j]["state_sha256"] == man["inner"][str(j)]["state_sha256"]
                                for j in mine["inner"]},
           "refit_hash_equal": mine["refit"][k]["state_sha256"] == man["refit"][str(k)]["state_sha256"],
           "refit_epochs": [res["fold_metrics.json"]["refit_epochs"], sel["encoders_refit_epochs"][arm]],
           "max_abs_pred_diff": float(np.abs(oof[f"risk_{name}"].to_numpy(float) - ref_pred).max()),
           "cindex": [res["fold_metrics.json"]["cindex"],
                      cindex(ref.loc[oof.index, "duration"], ref_pred, ref.loc[oof.index, "event"])]}
    out["pass"] = bool(all(out["inner_hash_equal"].values()) and out["refit_hash_equal"]
                       and out["refit_epochs"][0] == out["refit_epochs"][1] and out["max_abs_pred_diff"] < 1e-5
                       and abs(out["cindex"][0] - out["cindex"][1]) < 1e-9)
    return out


def check_or_write(path, obj, skip=()) -> None:
    """없으면 원자적으로 쓰고, 있으면 skip 키를 뺀 내용이 같은지 확인한다 (다르면 같은 폴더에 섞지 않는다)."""
    norm = lambda d: {k: v for k, v in json.loads(json.dumps(d, default=str)).items() if k not in skip}
    if os.path.exists(path):
        if norm(json.load(open(path, encoding="utf-8"))) != norm(obj):
            raise SystemExit(f"{path} 의 설정이 지금과 다르다. 같은 폴더에 섞지 않는다.")
        return
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({**obj, **({"environment": environment()} if "environment" in skip else {})},
                  fh, ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, path)


def save_outer(d, res) -> None:
    """임시 폴더에 다 쓴 뒤 옮긴다 (중단돼도 반쯤 쓴 결과가 남지 않게)."""
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
    """폴더에 있는 모든 (모델, outer) 결과를 합쳐 fold 별 C-index 표를 만든다 (다른 프로세스가 끝낸 모델 포함)."""
    rows, oofs = [], {}
    for p in sorted(glob.glob(os.path.join(out, "*", "outer[0-9]", "fold_metrics.json"))):
        r = json.load(open(p, encoding="utf-8"))
        rows.append(r)
        oofs.setdefault(r["model"], []).append(pd.read_csv(os.path.join(os.path.dirname(p), "oof_predictions.csv")))
    if not rows:
        return
    m = pd.DataFrame(rows)
    for name, g in m.groupby("model"):
        ids = pd.concat([o["research_id"] for o in oofs[name]], ignore_index=True)
        guard(not ids.duplicated().any(), f"{name}: 합본 O_k 에 중복 환자")
        if sorted(g.outer) == [1, 2, 3, 4, 5]:
            guard(ids.nunique() == 238, f"{name}: 합본 O_k 가 238명이 아니다")
    wide = m.pivot(index="model", columns="outer", values="cindex")
    done = wide.notna().all(axis=1) & (wide.shape[1] == 5)
    wide["mean"] = wide.iloc[:, :5].mean(axis=1).where(done)
    wide["sd"] = wide.iloc[:, :5].std(axis=1, ddof=1).where(done)
    for fname, df, idx in (("fold_metrics.csv", m.drop(columns=["inner_V_cindex"]), False),
                           ("cindex_by_fold.csv", wide.round(4), True)):
        tmp = os.path.join(out, f".{fname}.{os.getpid()}.tmp")
        df.to_csv(tmp, index=idx)
        os.replace(tmp, os.path.join(out, fname))
    print(f"[aggregate] {out}\n{wide.round(4)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--target", required=True, choices=("os", "pfs"))
    ap.add_argument("--baselines", default=None, help=f"쉼표 목록 (기본: {','.join(TABLE_BASELINES)})")
    ap.add_argument("--outers", default="1,2,3,4,5")
    ap.add_argument("--verify", action="store_true", help="run_v2 두 arm 재현 검증 (clin_report, image_simplecnn)")
    ap.add_argument("--smoke", action="store_true", help="2 epoch 경로 점검 전용 (--out_dir 필수)")
    ap.add_argument("--split_dir", default=SPLIT_DIR)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    if args.verify:
        names = list(VERIFY_BASELINES)
        guard(args.baselines is None and not args.smoke, "--verify 는 --baselines·--smoke 와 같이 쓰지 않는다")
    else:
        names = args.baselines.split(",") if args.baselines else list(TABLE_BASELINES)
        bad = [n for n in names if n not in TABLE_BASELINES]
        guard(not bad, f"표에 넣을 비교 모델이 아니다: {bad}")
    guard(not args.smoke or args.out_dir is not None, "--smoke 는 결과 폴더와 섞이지 않게 --out_dir 를 따로 준다")
    guard(len(set(names)) == len(names), f"중복된 모델 이름: {names}")
    outers = [int(x) for x in args.outers.split(",")]
    sub = "verify" if args.verify else BASELINE_TAG
    out = args.out_dir or paths.outputs("gmu_nested", f"{sub}_{args.target}")
    os.makedirs(out, exist_ok=True)

    # 설정 기록: 공통 설정은 결과 폴더에 한 번, 모델별 설정은 모델 폴더에 따로 둔다 (여러 프로세스가 서로 다른
    # 모델을 같은 폴더에 동시에 써도 기록이 덮어써지지 않게). 이미 있으면 지금 설정과 같은지만 확인한다.
    config = experiment_config(args, names)
    common = {k: v for k, v in config.items() if k != "baselines"}
    check_or_write(os.path.join(out, "experiment_config.json"), common, skip=("baselines", "environment"))
    for n in names:
        os.makedirs(os.path.join(out, n), exist_ok=True)
        check_or_write(os.path.join(out, n, "baseline_spec.json"), BASELINES[n])

    seed_everything(SEED)
    corpus, _ = features.load_text_corpus(cohort.DEFAULT_MERGED_CSV)
    ctx = {"device": torch.device("cuda" if torch.cuda.is_available() else "cpu"), "raw": fsc.load_raw_values(),
           "report_kwargs": report_kwargs(names, corpus), "labels": load_labels(args.target),
           "manifest": load_manifest(args.split_dir)}
    verify = {}
    for name in names:
        for k in outers:
            d = os.path.join(out, name, f"outer{k}")
            if os.path.exists(os.path.join(d, "fold_metrics.json")):
                print(f"[run_baselines] {name} outer {k} 은 이미 끝났다. 건너뛴다")
                continue
            # 여러 프로세스가 같은 폴더를 나눠 돌릴 때 같은 (모델, outer) 를 두 번 잡지 않게 잠근다 (mkdir 는 원자적).
            # 프로세스가 비정상 종료하면 <outer>.lock 이 남으므로 그때는 지우고 다시 실행한다.
            lock = d + ".lock"
            try:
                os.makedirs(os.path.dirname(lock), exist_ok=True)
                os.mkdir(lock)
            except FileExistsError:
                print(f"[run_baselines] {name} outer {k} 는 다른 프로세스가 돌리는 중이다. 건너뛴다")
                continue
            t0 = time.time()
            res = BaselineOuter(name, k, args, ctx, out).run()
            if args.verify:
                v = verify_against_run_v2(name, k, args.target, res)
                verify[f"{name}/outer{k}"] = v
                print(f"[verify] {name} outer {k}: {v}")
            save_outer(d, res)
            os.rmdir(lock)
            print(f"[run_baselines] target={args.target} {name} outer {k}: C-index {res['fold_metrics.json']['cindex']:.4f} "
                  f"(refit {res['fold_metrics.json']['refit_epochs']} ep) in {time.time() - t0:.0f}s")
    if args.verify:
        path = os.path.join(out, "verify.json")
        prev = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({**prev, **verify}, fh, ensure_ascii=False, indent=1)
        guard(all(v["pass"] for v in verify.values()), f"run_v2 재현 실패: {verify}")
    aggregate(out)


if __name__ == "__main__":
    main()
