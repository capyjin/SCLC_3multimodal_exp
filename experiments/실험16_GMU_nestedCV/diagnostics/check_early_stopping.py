# -*- coding: utf-8 -*-
"""[실험16-8] 정지 규칙 진단 — outer 1 / inner 1 의 Fit(학습)·S(정지용)만 쓴다. 학습 규칙 확정용.

V(inner validation)·outer test 는 쓰지 않는다 (check_learning_rate 와 같은 S-only 분할 파일). 모델 선택에 쓰지 않는다.

[1] GMU G1(lr 1e-4) 장기 학습 — 제안 B: max 5000 / patience 400 / min epoch 30 / min_delta 1e-4.
    인코더는 공식 규칙(S C-index, patience 10)으로 학습한 check_learning_rate 와 같은 것. D 가 본 점검, B·C 는 보조.
    곡선 전체를 보려고 5000 epoch 끝까지 기록하고, 규칙은 기록된 S NLL 에 사후 적용한다.

[2] 인코더 조기 종료 검토 — tabular(clin+report) · image, OS·PFS.
    (a) 공식 규칙(patience 10)으로 학습
    (b) patience 없이 최대 epoch(60/30)까지 학습하면서 매 epoch train/S 의 Efron NLL(eval, 전체)도 기록
    (a)와 (b)는 겹치는 epoch 에서 학습 경로가 같아야 한다 (기록이 학습을 바꾸지 않는다는 확인).
    (b) 곡선에 규칙들을 사후 적용: S C-index patience 10(현행) / S NLL patience 10·20.

Run:  python experiments/실험16_GMU_nestedCV/diagnostics/check_early_stopping.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))                 # diagnostics/
EXP_DIR = os.path.dirname(HERE)                                   # 실험16 폴더 (nested_fusion 패키지)
PROJECT_ROOT = os.path.dirname(os.path.dirname(EXP_DIR))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, HERE)

import json

import numpy as np
import pandas as pd
import torch

from check_learning_rate import INNER, OUTER, curve, stop_only_split

from nested_fusion.config import CANDIDATES, ENCODERS, SEED
from nested_fusion.dataset import feats, labels_tensor
from nested_fusion.model import build
from nested_fusion.train import EarlyStopper, run_encoders
from sclc import cohort, features, paths, plotstyle
from sclc import foldsafe_clinical as fsc
from sclc.encoders import build_encoder
from sclc.model import MODALITY_CONFIGS, make_model_factory
from sclc.train import ImageOnlyEvaluator, TrimodalEvaluator, cox_efron_loss, seed_everything

OUT = paths.outputs("gmu_nested", "es_check")
G1_RULE = dict(max_epochs=5000, patience=400, min_epoch=30, min_delta=1e-4)
MODELS = ("D_gmu", "B_concat", "C_fixed_gate")
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#2b2b2a", "#8a8980"


# ── [1] GMU G1 장기 ─────────────────────────────────────────────────────
def rule_on(values, patience, min_epoch, min_delta, max_epochs, higher_is_better=False):
    st = EarlyStopper(patience, min_epoch, min_delta)
    for e, v in enumerate(values[:max_epochs], start=1):
        _, halt = st.update(e, -float(v) if higher_is_better else float(v))
        if halt:
            return {"stop_epoch": e, "halted_by_patience": True, "best_epoch": st.best_epoch}
    return {"stop_epoch": min(len(values), max_epochs), "halted_by_patience": False, "best_epoch": st.best_epoch}


def gmu_summary(df, null_tr, null_st):
    tr, sp = df.train_nll.to_numpy(), df.stop_nll.to_numpy()
    r = rule_on(sp, G1_RULE["patience"], G1_RULE["min_epoch"], G1_RULE["min_delta"], G1_RULE["max_epochs"])
    after = np.arange(1, len(sp) + 1) >= G1_RULE["min_epoch"]
    smin = int(np.arange(1, len(sp) + 1)[after][np.argmin(sp[after])])
    pts = (30, 100, 500, 1000, 1500, 2000, 3000, 4000, 5000)
    return {
        "rule_B": r, "stop_nll_min_epoch_ge30": smin,
        "gain_vs_null": {e: {"train": round(null_tr - tr[e - 1], 4), "stop": round(null_st - sp[e - 1], 4),
                             "gap": round((null_tr - tr[e - 1]) - (null_st - sp[e - 1]), 4)} for e in pts},
        "gain_at_best": {"train": round(null_tr - tr[r["best_epoch"] - 1], 4),
                         "stop": round(null_st - sp[r["best_epoch"] - 1], 4)},
        "train_nll_drop_per_500": {f"{a}->{a + 500}": round(tr[a - 1] - tr[a + 499], 4)
                                   for a in (1, 500, 1000, 2000, 3000, 4000) if a + 499 < len(tr)},
    }


# ── [2] 인코더 ─────────────────────────────────────────────────────────
def encoder_run(kind, target, split_csv, ctx, patience, diag, save_dir):
    common = dict(target=target, epochs=ENCODERS[kind]["max_epochs"], seed=SEED + 100 * OUTER,
                  split_csv=split_csv, save_dir=save_dir, loss_fn=cox_efron_loss, patience=patience,
                  num_workers=4, val_loss_fn=cox_efron_loss if diag else None,
                  batch_size=ENCODERS[kind]["batch_size"], lr=ENCODERS[kind]["lr"],
                  weight_decay=ENCODERS[kind]["weight_decay"], fold_end_hook=lambda **_: None)
    if kind == "tabular":
        return TrimodalEvaluator(model_factory=make_model_factory(MODALITY_CONFIGS["clin_report"]),
                                 clinical_columns_fn=fsc.baseline_clinical_columns,
                                 extra_numeric_fn=fsc.make_extra_numeric_fn(ctx["raw"], []),
                                 recode_lung_disease=True, **ctx["text_kwargs"], **common).run()
    return ImageOnlyEvaluator(ckpt_tag="image_simplecnn", **common).run()


def comparable_pairs(ids, labels, target):
    sub = labels.loc[ids]
    t, e = sub[f"{target}_days"].to_numpy(), sub[f"{target}_event"].to_numpy()
    return int(((t[:, None] < t[None, :]) & (e[:, None] == 1)).sum())


def encoder_summary(h: pd.DataFrame, official: pd.DataFrame, null_tr, null_st) -> dict:
    n = len(official)
    same = all(np.array_equal(h[c].to_numpy()[:n], official[c].to_numpy())
               for c in ("train_loss", "train_cindex", "val_cindex"))
    ci, nll = h.val_cindex.to_numpy(), h.val_loss.to_numpy()
    cur = rule_on(ci, 10, 1, 0.0, len(ci), higher_is_better=True)
    best_c = cur["best_epoch"]
    return {
        "trajectory_identical_to_official": bool(same), "official_epochs_run": n,
        "official_best_epoch": int(official.epoch[official.saved].iloc[-1]),
        "rule_cindex_p10": cur,
        "cindex_max_all_epochs": {"epoch": int(np.argmax(ci) + 1), "value": round(float(ci.max()), 4)},
        "cindex_at_best": round(float(ci[best_c - 1]), 4),
        "later_epochs_exceeding_best_cindex": [int(e) for e in np.where(ci > ci[best_c - 1])[0] + 1],
        "nll_min_all_epochs": {"epoch": int(np.argmin(nll) + 1), "value": round(float(nll.min()), 4),
                               "gain_vs_null": round(null_st - float(nll.min()), 4)},
        "rule_nll_p10": rule_on(nll, 10, 1, 0.0, len(nll)),
        "rule_nll_p20": rule_on(nll, 20, 1, 0.0, len(nll)),
        "stop_gain_vs_null_at": {int(e): round(null_st - float(nll[e - 1]), 4)
                                 for e in sorted({1, best_c, 5, 10, 15, 20, 30, len(nll)}) if e <= len(nll)},
        "train_gain_vs_null_at": {int(e): round(null_tr - float(h.train_loss_eval[e - 1]), 4)
                                  for e in sorted({1, best_c, 5, 10, 15, 20, 30, len(nll)}) if e <= len(nll)},
    }


def plot_gmu(curves, summaries, nulls, path):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    targets = list(curves)
    fig, axes = plt.subplots(len(targets), len(MODELS), figsize=(13, 6.6), squeeze=False, facecolor="#fcfcfb")
    for r, t in enumerate(targets):
        for c, m in enumerate(MODELS):
            ax, df, s = axes[r][c], curves[t][m], summaries[t][m]
            ax.set_facecolor("#fcfcfb")
            ax.plot(df.epoch, nulls[t]["train"] - df.train_nll, color=BLUE, lw=2)
            ax.plot(df.epoch, nulls[t]["stop"] - df.stop_nll, color=BLUE, lw=2, ls="--")
            ax.axvline(s["rule_B"]["best_epoch"], color=INK, lw=1, ls=":")
            if s["rule_B"]["halted_by_patience"]:
                ax.axvline(s["rule_B"]["stop_epoch"], color=MUTED, lw=1, ls="-.")
            ax.axhline(0, color=MUTED, lw=0.8)
            ax.set_title(f"{t.upper()} · {m} · G1" + ("  (본 점검)" if m == "D_gmu" else "  (보조)"),
                         loc="left", fontsize=10, color=INK)
            ax.set_xlabel("epoch (full-batch step)", fontsize=8, color=MUTED)
            if c == 0:
                ax.set_ylabel("null 대비 NLL 개선 (클수록 좋음)", fontsize=8, color=MUTED)
            _style(ax)
    fig.legend(handles=[Line2D([], [], color=BLUE, lw=2, label="Train (Fit, eval 모드)"),
                        Line2D([], [], color=BLUE, lw=2, ls="--", label="정지용 S"),
                        Line2D([], [], color=INK, lw=1, ls=":", label="최적 epoch (B 규칙)"),
                        Line2D([], [], color=MUTED, lw=1, ls="-.", label="B 규칙 종료")],
               loc="upper center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_encoders(hist, summ, nulls, path):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    keys = list(hist)
    fig, axes = plt.subplots(len(keys), 2, figsize=(11, 2.7 * len(keys)), squeeze=False, facecolor="#fcfcfb")
    for r, key in enumerate(keys):
        h, s, t = hist[key], summ[key], key.split("/")[0]
        a1, a2 = axes[r]
        a1.plot(h.epoch, nulls[t]["train"] - h.train_loss_eval, color=BLUE, lw=2)
        a1.plot(h.epoch, nulls[t]["stop"] - h.val_loss, color=ORANGE, lw=2)
        a1.axhline(0, color=MUTED, lw=0.8)
        a2.plot(h.epoch, h.train_cindex, color=BLUE, lw=2)
        a2.plot(h.epoch, h.val_cindex, color=ORANGE, lw=2)
        a2.axhline(0.5, color=MUTED, lw=0.8)
        for ax, title in ((a1, "Efron NLL: null 대비 개선"), (a2, "Harrell C-index")):
            ax.set_facecolor("#fcfcfb")
            ax.axvline(s["rule_cindex_p10"]["best_epoch"], color=INK, lw=1, ls=":")
            ax.axvline(s["rule_cindex_p10"]["stop_epoch"], color=INK, lw=1, ls="-.")
            ax.axvline(s["nll_min_all_epochs"]["epoch"], color=MUTED, lw=1.5, ls="--")
            ax.set_title(f"{key} · {title}", loc="left", fontsize=9, color=INK)
            ax.set_xlabel("epoch", fontsize=8, color=MUTED)
            _style(ax)
    fig.legend(handles=[Line2D([], [], color=BLUE, lw=2, label="Train (Fit)"),
                        Line2D([], [], color=ORANGE, lw=2, label="정지용 S"),
                        Line2D([], [], color=INK, lw=1, ls=":", label="현행 최적 (S C-index)"),
                        Line2D([], [], color=INK, lw=1, ls="-.", label="현행 종료 (patience 10)"),
                        Line2D([], [], color=MUTED, lw=1.5, ls="--", label="S NLL 최저 epoch")],
               loc="upper center", ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _style(ax):
    ax.tick_params(colors=MUTED, labelsize=7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d4d3cd")
    ax.grid(axis="y", color="#ecebe6", lw=0.6)


def main() -> None:
    if os.path.exists(os.path.join(OUT, "es_check_summary.json")):
        raise SystemExit(f"{OUT} 에 이미 결과가 있다 — 덮어쓰지 않는다.")
    os.makedirs(OUT, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    plotstyle.apply()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corpus, _ = features.load_text_corpus(cohort.DEFAULT_MERGED_CSV)
    ctx = {"raw": fsc.load_raw_values(), "text_kwargs": build_encoder("radbert").evaluator_kwargs(corpus)}
    split_csv = os.path.join(OUT, "outer1_inner1_stop_only.csv")
    ids = stop_only_split(split_csv)
    labels = cohort.load_trimodal_cohort().drop_duplicates("research_id").set_index("research_id")

    out = {"status": "정지 규칙 진단 전용 — V·outer test 미사용, 모델 선택·평가에 쓰지 않음",
           "n_fit": len(ids["fit"]), "n_stop": len(ids["stop"]), "g1_rule": G1_RULE,
           "gmu": {}, "encoders": {}, "null_nll": {}, "stop_comparable_pairs": {}}
    gmu_curves, gmu_summ, enc_hist, enc_summ = {}, {}, {}, {}
    for target in ("os", "pfs"):
        seed_everything(SEED)
        # [1] 공식 규칙 인코더 -> 특징 -> G1 장기
        stores = {kind: run_encoders(kind, target, split_csv, ENCODERS[kind]["max_epochs"],
                                     ENCODERS[kind]["patience"], SEED + 100 * OUTER,
                                     os.path.join(OUT, "ckpt", target, f"{kind}_official"), None, ctx, [])
                  for kind in ENCODERS}
        official_hist = {kind: pd.DataFrame(ev.training_history) for kind, (_, ev) in stores.items()}
        T, I = stores["tabular"][0][INNER]["splits"], stores["image"][0][INNER]["splits"]
        tr = (feats(T["train"], I["train"], dev), *labels_tensor(ids["fit"], labels, target, dev))
        st = (feats(T["val"], I["val"], dev), *labels_tensor(ids["stop"], labels, target, dev))
        zero = lambda d: float(cox_efron_loss(torch.zeros(len(d[1]), device=dev), d[1], d[2]))
        nulls = {"train": zero(tr), "stop": zero(st)}
        out["null_nll"][target] = nulls
        out["stop_comparable_pairs"][target] = comparable_pairs(ids["stop"], labels, target)
        gmu_curves[target], gmu_summ[target] = {}, {}
        for m in MODELS:
            torch.manual_seed(SEED * 1000 + 100 * OUTER + 10 * INNER + 0)      # G1 = 후보 순번 0
            df = curve(build(m, CANDIDATES["G1"]).to(dev), tr, st, CANDIDATES["G1"],
                       G1_RULE["max_epochs"])
            df.to_csv(os.path.join(OUT, f"gmu_G1_long_{target}_{m}.csv"), index=False)
            gmu_curves[target][m] = df
            gmu_summ[target][m] = gmu_summary(df, nulls["train"], nulls["stop"])
            r = gmu_summ[target][m]["rule_B"]
            print(f"[gmu] {target} {m}: stop {r['stop_epoch']} best {r['best_epoch']} "
                  f"S-min {gmu_summ[target][m]['stop_nll_min_epoch_ge30']}")
        # [2] 인코더: patience 없이 끝까지 + NLL 기록
        for kind in ENCODERS:
            ev = encoder_run(kind, target, split_csv, ctx, None, True,
                             os.path.join(OUT, "ckpt", target, f"{kind}_full"))
            h = pd.DataFrame(ev.training_history)
            key = f"{target}/{kind}"
            h.to_csv(os.path.join(OUT, f"encoder_curve_{target}_{kind}.csv"), index=False)
            enc_hist[key] = h
            # null NLL 은 GMU 와 같은 집합(Fit·S)의 위험점수 0 기준
            enc_summ[key] = encoder_summary(h, official_hist[kind], nulls["train"], nulls["stop"])
            s = enc_summ[key]
            print(f"[enc] {key}: identical={s['trajectory_identical_to_official']} "
                  f"Cbest {s['rule_cindex_p10']['best_epoch']} stop {s['rule_cindex_p10']['stop_epoch']} "
                  f"Cmax@{s['cindex_max_all_epochs']['epoch']} NLLmin@{s['nll_min_all_epochs']['epoch']}")
    out["gmu"] = {t: v for t, v in gmu_summ.items()}
    out["encoders"] = enc_summ
    plot_gmu(gmu_curves, gmu_summ, out["null_nll"], os.path.join(OUT, "gmu_G1_long.png"))
    plot_encoders(enc_hist, enc_summ, out["null_nll"], os.path.join(OUT, "encoder_curves.png"))
    with open(os.path.join(OUT, "es_check_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
    print(f"[check_early_stopping] wrote {OUT}")


if __name__ == "__main__":
    main()
