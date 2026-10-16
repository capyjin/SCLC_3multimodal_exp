# -*- coding: utf-8 -*-
"""[실험16-4] Smoke test — outer 1 / inner fold 1 x 2 epoch 로 파이프라인 배관만 점검한다.

**여기서 나오는 C-index 는 성능이 아니다.** 연구 결과·모델 선택·하이퍼파라미터 선택에 쓰지 않는다.
outer test(O_1)에서는 C-index 를 아예 계산하지 않는다.

점검 항목 (각각 PASS/FAIL 을 smoke_report.json 에 남긴다)
  split    : 분할 파일의 환자 수·불변식, fold_plan 이 읽은 Fit/S/V 가 매니페스트와 같은가
  prep     : 임상 폭 19(치료 변수 제외), fold-safe 중앙값 = Fit 환자 원자료 중앙값,
             scaler 가 Fit 으로만 적합(Fit 평균 0·SD 1), lung_disease 재코딩 반영,
             레이블이 CSV 와 같은가(수정 없음), 영상 정규화 통계 = Fit 영상 통계
  encoder  : Efron 손실로 2 epoch 학습·손실 유한, 추출 특징 차원 128/16/512,
             추출 특징에 고정 head 를 다시 씌우면 모델 위험점수와 같은가, 추출이 결정론적인가
  gmu      : B/C/D forward 출력 차원, gate 범위·형태, 첫 step 그래디언트(유한·0 아님),
             2 epoch 학습 손실 유한, 인코더 state_dict 해시 불변(고정)
  refit    : val 없는 outer 재학습 경로(고정 2 epoch, 마지막 epoch 저장)와 O_1 특징 추출

모든 산출물은 outputs/gmu_nested/smoke_<target>/ — 기존 체크포인트·결과를 덮어쓰지 않는다.
(2026-10-14 첫 OS 실행은 outputs/gmu_nested/smoke/ 에 그대로 남아 있다.)

Run:  python experiments/실험16_GMU_nestedCV/diagnostics/smoke_test.py --target pfs
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
import time

import numpy as np
import pandas as pd
import torch

from checklist import CHECKS, check

from nested_fusion.config import CANDIDATES
from nested_fusion.model import FUSION_MODELS, build
from nested_fusion.processing import ImageFeatureHook, TabularFeatureHook, state_hash
from nested_fusion.train import train_fusion
from sclc import cohort, features, paths
from sclc import foldsafe_clinical as fsc
from sclc.encoders import build_encoder
from sclc.evaluation import cindex
from sclc.model import MODALITY_CONFIGS, make_model_factory
from sclc.train import (ImageOnlyEvaluator, TrimodalEvaluator, cox_efron_loss, fold_plan,
                        seed_everything)

TARGET = "os"   # main() 에서 --target 으로 바뀐다
OUTER = 1
EPOCHS = 2
SPLIT_DIR = paths.outputs("gmu_nested", "splits_draft")
OUT = None
def labels_of(ids, clin):
    sub = clin.loc[ids]
    return (torch.as_tensor(sub[f"{TARGET}_days"].to_numpy(float), dtype=torch.float32),
            torch.as_tensor(sub[f"{TARGET}_event"].to_numpy(float), dtype=torch.float32))


def main() -> None:
    global TARGET, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="os", choices=("os", "pfs"))
    TARGET = ap.parse_args().target
    OUT = paths.outputs("gmu_nested", f"smoke_{TARGET}")
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    seed_everything(42)
    inner_csv = os.path.join(SPLIT_DIR, f"outer{OUTER}_inner.csv")
    refit_csv = os.path.join(SPLIT_DIR, f"outer{OUTER}_refit.csv")
    with open(os.path.join(SPLIT_DIR, "fold_manifest.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    om = next(o for o in manifest["outer"] if o["k"] == OUTER)
    inner1 = om["inner"][0]["ids"]

    # ── split ─────────────────────────────────────────────────────────────
    cdf_inner = cohort.load_trimodal_cohort(split_csv=inner_csv, recode_lung_disease=True)
    _, ids = fold_plan(cdf_inner, max_folds=1)[0]
    check("split: fold_plan(inner j=1) == manifest Fit/S/V",
          sorted(ids["train"]) == inner1["fit"] and sorted(ids["val"]) == inner1["stop"]
          and sorted(ids["test"]) == inner1["val"],
          {k: len(v) for k, v in ids.items()})
    check("split: outer invariants (manifest)", all(om["invariants"].values()), om["invariants"])
    check("split: O_1 disjoint from Fit/S/V",
          not (set(om["outer_test_ids"]) & set(ids["train"] + ids["val"] + ids["test"])))

    clin_all = cohort.load_trimodal_cohort(recode_lung_disease=True).drop_duplicates(
        "research_id").set_index("research_id")
    raw = fsc.load_raw_values()
    audit: list = []
    corpus, _ = features.load_text_corpus(cohort.DEFAULT_MERGED_CSV)
    text_kwargs = build_encoder("radbert").evaluator_kwargs(corpus)
    tab_common = dict(
        target=TARGET, epochs=EPOCHS, batch_size=32, seed=42, max_folds=1,
        model_factory=make_model_factory(MODALITY_CONFIGS["clin_report"]),
        clinical_columns_fn=fsc.baseline_clinical_columns,
        extra_numeric_fn=fsc.make_extra_numeric_fn(raw, audit),
        recode_lung_disease=True, loss_fn=cox_efron_loss, patience=10, num_workers=2,
        **text_kwargs)

    # ── tabular encoder (inner 1) ─────────────────────────────────────────
    tab_store = {}
    ev_tab = TrimodalEvaluator(split_csv=inner_csv, save_dir=os.path.join(OUT, "ckpt", "tab_inner1"),
                               fold_end_hook=TabularFeatureHook(tab_store), **tab_common).run()
    tab = {**tab_store[1], **tab_store[1]["splits"]}
    hist = ev_tab.training_history
    check("encoder/tab: Efron loss finite over 2 epochs",
          len(hist) == EPOCHS and all(np.isfinite(r["train_loss"]) for r in hist),
          [round(r["train_loss"], 4) for r in hist])
    check("encoder/tab: best_epoch recorded", ev_tab.fold_records[0].get("best_epoch") in (1, 2),
          ev_tab.fold_records[0].get("best_epoch"))
    model_tab = tab["model"]
    check("prep: clinical width 19 (treatment vars excluded), report width 768",
          model_tab.clinical_dim == 19 and model_tab.report_dim == 768,
          (model_tab.clinical_dim, model_tab.report_dim))
    std_cols, cat_cols = fsc.baseline_clinical_columns(clin_all)
    check("prep: rt_type/atezolizumab not in clinical columns",
          not set(fsc.TREATMENT_COLUMNS) & set(std_cols + cat_cols), std_cols + cat_cols)
    a0 = audit[0]
    expect_med = np.nanmedian(raw.reindex(ids["train"])[list(fsc.PREIMPUTED_COLUMNS)].to_numpy(float), axis=0)
    got_med = np.array([a0["train_medians"][c] for c in fsc.PREIMPUTED_COLUMNS])
    check("prep: fold-safe medians == Fit-only raw medians",
          a0["n"] == {"train": 120, "val": 22, "test": 48} and np.allclose(expect_med, got_med),
          dict(zip(fsc.PREIMPUTED_COLUMNS, got_med.round(2).tolist())))
    n_std, n_cat = len(std_cols), len(cat_cols)
    tr_tab = tab["train"]["tabular"].numpy()
    cont = np.concatenate([tr_tab[:, :n_std], tr_tab[:, n_std + n_cat: n_std + n_cat + 7]], axis=1)
    check("prep: continuous block standardized on Fit only (Fit mean~0, sd~1)",
          np.allclose(cont.mean(0), 0, atol=1e-5) and np.allclose(cont.std(0), 1, atol=1e-3),
          {"max|mean|": float(np.abs(cont.mean(0)).max()), "sd range": [float(cont.std(0).min()),
                                                                         float(cont.std(0).max())]})
    lung_col = n_std + cat_cols.index("lung_disease")
    recoded = cohort.recode_lung_disease_from_excel(
        pd.read_excel(cohort.DEFAULT_SOURCE_XLSX).assign(
            research_id=lambda x: pd.to_numeric(x["연구번호"], errors="coerce")).dropna(
            subset=["research_id"]).astype({"research_id": int}).set_index("research_id"))
    check("prep: lung_disease recoded values reach the model input",
          np.array_equal(tr_tab[:, lung_col], recoded.loc[ids["train"]].to_numpy(float)),
          sorted(set(tr_tab[:, lung_col].tolist())))
    csv = cohort.load_merged_clinical().set_index("research_id")
    ds_dur = [s["duration"] for s in tab["datasets"]["train"].samples]
    ds_evt = [s["event"] for s in tab["datasets"]["train"].samples]
    check("prep: labels unchanged vs merged CSV (Fit)",
          np.array_equal(ds_dur, csv.loc[ids["train"], f"{TARGET}_days"].to_numpy())
          and np.array_equal(ds_evt, csv.loc[ids["train"], f"{TARGET}_event"].to_numpy()))
    check("encoder/tab: eval mode at extraction", tab["training_flag"] is False)
    check("encoder/tab: feature dims clinical 128 / report 16",
          tab["train"]["clinical"].shape[1] == 128 and tab["train"]["report"].shape[1] == 16,
          {s: (tuple(tab[s]["clinical"].shape), tuple(tab[s]["report"].shape)) for s in ("train", "val", "test")})
    check("encoder/tab: L2-normalized features (norm=1)",
          all(torch.allclose(tab["train"][m].norm(dim=1), torch.ones(1), atol=1e-4)
              for m in ("clinical", "report")))
    check("encoder/tab: head(features) reproduces model risk",
          all(torch.allclose(tab[s]["risk"], tab[s]["head_risk"], atol=1e-5) for s in ("train", "val", "test")))

    # ── image encoder (inner 1) ───────────────────────────────────────────
    img_store = {}
    ev_img = ImageOnlyEvaluator(target=TARGET, epochs=EPOCHS, batch_size=16, seed=42, max_folds=1,
                                split_csv=inner_csv, save_dir=os.path.join(OUT, "ckpt", "img_inner1"),
                                ckpt_tag="smoke_image", loss_fn=cox_efron_loss, patience=10,
                                num_workers=2, fold_end_hook=ImageFeatureHook(img_store)).run()
    img = {**img_store[1], **img_store[1]["splits"]}
    check("encoder/img: Efron loss finite over 2 epochs",
          all(np.isfinite(r["train_loss"]) for r in ev_img.training_history),
          [round(r["train_loss"], 4) for r in ev_img.training_history])
    check("encoder/img: feature dim 512", img["train"]["image"].shape[1] == 512,
          {s: tuple(img[s]["image"].shape) for s in ("train", "val", "test")})
    check("encoder/img: head(features) reproduces model risk",
          all(torch.allclose(img[s]["risk"], img[s]["head_risk"], atol=1e-5) for s in ("train", "val", "test")))
    norm = img["datasets"]["train"].transform.transforms[2]
    from sclc.dataset import _compute_mean_std_from_samples
    m_fit, s_fit = _compute_mean_std_from_samples(img["datasets"]["train"].samples)
    check("prep: image normalization stats == Fit images",
          np.allclose(norm.mean, m_fit) and np.allclose(norm.std, s_fit)
          and img["datasets"]["test"].transform.transforms[2].mean == norm.mean,
          {"mean": [round(x, 5) for x in norm.mean], "std": [round(x, 5) for x in norm.std]})
    check("split: image and tabular arms saw the same patients",
          img["ids"] == tab["ids"])

    # 추출 결정론성 (같은 모델로 다시 추출)
    again = {}
    ImageFeatureHook(again)(fold=1, ids=img["ids"], model=img["model"], device=next(img["model"].parameters()).device,
                            datasets={"test": img["datasets"]["test"]})
    check("encoder: feature extraction deterministic",
          torch.equal(again[1]["splits"]["test"]["image"], img["test"]["image"]))

    # ── GMU B/C/D ─────────────────────────────────────────────────────────
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hashes = {"tab": state_hash(model_tab), "img": state_hash(img["model"])}
    for mdl in (model_tab, img["model"]):
        for p in mdl.parameters():
            p.requires_grad_(False)

    def feats(split):
        return {"clinical": tab[split]["clinical"].to(dev), "report": tab[split]["report"].to(dev),
                "image": img[split]["image"].to(dev)}

    data = {s: (feats(s), *[x.to(dev) for x in labels_of(ids[s], clin_all)])
            for s in ("train", "val", "test")}
    cfg = CANDIDATES["G1"]
    gmu_summary = {}
    for name in FUSION_MODELS:
        torch.manual_seed(42)
        model = build(name, cfg).to(dev)
        f, d, e = data["train"]
        out = model(f)
        check(f"gmu/{name}: forward shape [n,1]", tuple(out.shape) == (len(ids["train"]), 1), tuple(out.shape))
        g = model.gates(f)
        if g is not None:
            ok = tuple(g.shape) == (len(ids["train"]), 3) and bool(((g > 0) & (g < 1)).all())
            if name == "C_fixed_gate":
                ok = ok and bool((g == g[0]).all())
            check(f"gmu/{name}: gates shape [n,3] in (0,1)" + (" and patient-common" if name == "C_fixed_gate" else ""),
                  ok, g[:2].detach().cpu().numpy().round(3).tolist())
        model.zero_grad()
        cox_efron_loss(out, d, e).backward()
        grads = {n: p.grad for n, p in model.named_parameters()}
        finite = all(gr is not None and torch.isfinite(gr).all() for gr in grads.values())
        nonzero = {n: float(gr.abs().sum()) > 0 for n, gr in grads.items()}
        check(f"gmu/{name}: first-step grads finite and non-zero for every parameter",
              finite and all(nonzero.values()), {n: v for n, v in nonzero.items() if not v} or "all non-zero")
        model.zero_grad()
        res = train_fusion(model, data["train"], data["val"], cfg, cox_efron_loss,
                              max_epochs=EPOCHS, min_epoch=1)
        check(f"gmu/{name}: 2-epoch full-batch train, losses finite",
              all(np.isfinite(r["train_loss"]) and np.isfinite(r["stop_loss"]) for r in res["history"]),
              res["history"])
        with torch.no_grad():
            v_risk = model(data["test"][0]).squeeze(1).cpu().numpy()
        gmu_summary[name] = {"best_epoch": res["best_epoch"],
                             "inner_V_cindex_SMOKE_ONLY": round(cindex(data["test"][1].cpu(), v_risk,
                                                                      data["test"][2].cpu()), 4)}
    check("gmu: encoders frozen (state_dict hash unchanged)",
          hashes == {"tab": state_hash(model_tab), "img": state_hash(img["model"])}
          and hashes["tab"] == tab["hash"] and hashes["img"] == img["hash"])

    # ── outer 재학습 경로 (val 없음, 고정 2 epoch) ──────────────────────────
    cdf_refit = cohort.load_trimodal_cohort(split_csv=refit_csv)
    _, rids = fold_plan(cdf_refit, max_folds=1)[0]
    check("refit: T_1 / O_1 sizes, no val", (len(rids["train"]), len(rids["val"]), len(rids["test"])) == (190, 0, 48),
          {k: len(v) for k, v in rids.items()})
    audit.clear()
    rtab_store, rimg_store = {}, {}
    ev_rt = TrimodalEvaluator(split_csv=refit_csv, save_dir=os.path.join(OUT, "ckpt", "tab_refit"),
                              fold_end_hook=TabularFeatureHook(rtab_store), **tab_common).run()
    ev_ri = ImageOnlyEvaluator(target=TARGET, epochs=EPOCHS, batch_size=16, seed=42, max_folds=1,
                               split_csv=refit_csv, save_dir=os.path.join(OUT, "ckpt", "img_refit"),
                               ckpt_tag="smoke_image", loss_fn=cox_efron_loss, num_workers=2,
                               fold_end_hook=ImageFeatureHook(rimg_store)).run()
    rtab, rimg = rtab_store[OUTER]["splits"], rimg_store[OUTER]["splits"]
    for tag, ev in (("tab", ev_rt), ("img", ev_ri)):
        h = ev.training_history
        check(f"refit/{tag}: fixed epochs, val None, last epoch saved",
              len(h) == EPOCHS and all(r["val_cindex"] is None for r in h) and h[-1]["saved"]
              and not any(r["saved"] for r in h[:-1]), [(r["epoch"], r["saved"]) for r in h])
    check("refit: fold-safe medians fitted on T_1 (n=190)", audit[0]["n"] == {"train": 190, "val": 0, "test": 48})
    torch.manual_seed(42)
    gmu = build("D_gmu", cfg).to(dev)
    rfeat = {s: {"clinical": rtab[s]["clinical"].to(dev), "report": rtab[s]["report"].to(dev),
                 "image": rimg[s]["image"].to(dev)} for s in ("train", "test")}
    rd, re_ = (x.to(dev) for x in labels_of(rids["train"], clin_all))
    res = train_fusion(gmu, (rfeat["train"], rd, re_), None, cfg, cox_efron_loss, max_epochs=EPOCHS)
    with torch.no_grad():
        o_risk = gmu(rfeat["test"])
        o_gate = gmu.gates(rfeat["test"])
    check("refit/gmu: fixed 2 epochs, O_1 risk [48,1] & gates [48,3] finite (no C-index computed)",
          len(res["history"]) == EPOCHS and tuple(o_risk.shape) == (48, 1)
          and tuple(o_gate.shape) == (48, 3) and bool(torch.isfinite(o_risk).all()))

    n_fail = sum(not c["pass"] for c in CHECKS)
    report = {"status": "SMOKE TEST ONLY — not a performance result",
              "target": TARGET, "outer": OUTER, "inner": 1, "epochs": EPOCHS,
              "n_checks": len(CHECKS), "n_fail": n_fail, "checks": CHECKS,
              "gmu_smoke_numbers": gmu_summary, "elapsed_s": round(time.time() - t0, 1)}
    with open(os.path.join(OUT, "smoke_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1, default=str)
    print(f"\n=== smoke: {len(CHECKS) - n_fail}/{len(CHECKS)} PASS, {report['elapsed_s']}s -> {OUT}")


if __name__ == "__main__":
    main()
