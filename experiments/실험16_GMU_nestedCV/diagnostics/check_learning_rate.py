# -*- coding: utf-8 -*-
"""[실험16-7] GMU 학습률·최대 epoch 점검 — outer 1 / inner 1 의 Fit(학습)과 S(정지용)만 쓴다.

최적화 상태만 본다. **Inner Validation(V)·Outer Test 는 쓰지 않는다** — 인코더 평가기는 test
분할의 C-index 를 자동으로 계산하므로, 임시 분할 파일에서 test 자리에도 S 를 넣어 V 가 아예
들어오지 않게 한다. 모델 선택에 쓰지 않는다.

  1. 인코더(tabular·image)를 본 실행과 같은 설정·seed 로 Fit 학습, S C-index 로 최적 epoch
  2. 고정 특징으로 D(GMU) [+ 보조: B·C] x {G1 lr 1e-4, G4 lr 1e-3} 를 early stopping 없이 800 epoch
     매 epoch: Train NLL(학습 모드 = 실제 최적화 대상), Train NLL(eval 모드), S NLL(eval 모드)
  3. 기록된 S 곡선에 기존 규칙(max 500 / patience 40 / min epoch 30 / min_delta 1e-4)을 사후 적용
  NLL 은 위험집합 크기에 따라 크기가 달라서(Fit 120명 vs S 22명) 둘을 직접 빼면 안 된다.
  각 집합의 null 모델(위험점수 0) NLL 을 빼서 "null 대비 개선량"으로 맞춘 뒤 비교한다.

Run:  python experiments/실험16_GMU_nestedCV/diagnostics/check_learning_rate.py
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

from nested_fusion.config import CANDIDATES, ENCODERS, SEED
from nested_fusion.dataset import feats, labels_tensor, load_manifest
from nested_fusion.model import build
from nested_fusion.train import EarlyStopper, run_encoders
from sclc import cohort, features, paths, plotstyle
from sclc import foldsafe_clinical as fsc
from sclc.encoders import build_encoder
from sclc.train import cox_efron_loss, seed_everything

OUT = paths.outputs("gmu_nested", "lr_check")
OUTER, INNER = 1, 1
MAX_EPOCHS = 800
CANDS = ("G1", "G4")
MODELS = ("D_gmu", "B_concat", "C_fixed_gate")          # D 가 본 점검, B·C 는 보조
RULE = dict(patience=40, min_epoch=30, min_delta=1e-4)
CHECKPOINTS = (30, 50, 100, 200, 300, 400, 500, 600, 700, 800)


def stop_only_split(path: str) -> dict:
    man = load_manifest()
    ids = next(o for o in man["outer"] if o["k"] == OUTER)["inner"][INNER - 1]["ids"]
    rows = ([(i, "train", INNER) for i in ids["fit"]] + [(i, "val", INNER) for i in ids["stop"]]
            + [(i, "test", INNER) for i in ids["stop"]])               # V 대신 S
    pd.DataFrame(rows, columns=["research_id", "split", "fold"]).to_csv(path, index=False)
    return ids


def apply_rule(stop_nll: np.ndarray, max_epochs: int) -> dict:
    st = EarlyStopper(**RULE)
    halted = None
    for e, v in enumerate(stop_nll[:max_epochs], start=1):
        _, halt = st.update(e, float(v))
        if halt:
            halted = e
            break
    return {"stop_epoch": halted or max_epochs, "halted_by_patience": halted is not None,
            "best_epoch": st.best_epoch, "best_stop_nll": st.best}


def curve(model, tr, st, cfg, n_epochs: int = MAX_EPOCHS) -> pd.DataFrame:
    """early stopping 없이 n_epochs 학습하며 매 epoch Train(학습/eval 모드)·S 의 Efron NLL 을 기록한다."""
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    rows = []
    for epoch in range(1, n_epochs + 1):
        model.train()
        opt.zero_grad()
        loss = cox_efron_loss(model(tr[0]), tr[1], tr[2])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            rows.append({"epoch": epoch, "train_nll_trainmode": float(loss),
                         "train_nll": float(cox_efron_loss(model(tr[0]), tr[1], tr[2])),
                         "stop_nll": float(cox_efron_loss(model(st[0]), st[1], st[2]))})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, null_train: float, null_stop: float) -> dict:
    tr, sp = df.train_nll.to_numpy(), df.stop_nll.to_numpy()
    after = df.epoch.to_numpy() >= RULE["min_epoch"]
    gmin = int(df.epoch[after].iloc[int(np.argmin(sp[after]))])
    at = lambda e: {"train_nll": round(tr[e - 1], 5), "stop_nll": round(sp[e - 1], 5),
                    "train_gain_vs_null": round(null_train - tr[e - 1], 5),
                    "stop_gain_vs_null": round(null_stop - sp[e - 1], 5)}
    return {
        "rule_max500": apply_rule(sp, 500), "rule_max800": apply_rule(sp, 800),
        "stop_nll_min_epoch_ge30_within800": {"epoch": gmin, "stop_nll": round(float(sp[gmin - 1]), 5)},
        "at_epoch": {e: at(e) for e in CHECKPOINTS},
        "train_nll_drop": {"1->500": round(tr[0] - tr[499], 5), "400->500": round(tr[399] - tr[499], 5),
                           "500->800": round(tr[499] - tr[799], 5), "700->800": round(tr[699] - tr[799], 5)},
        "stop_nll_drop": {"1->500": round(sp[0] - sp[499], 5), "400->500": round(sp[399] - sp[499], 5),
                          "500->800": round(sp[499] - sp[799], 5), "700->800": round(sp[699] - sp[799], 5)},
    }


def plot(curves: dict, summaries: dict, nulls: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    color = {"G1": "#2a78d6", "G4": "#eb6834"}            # 검증된 범주 1·2 번 (dataviz 기본 팔레트)
    ink, muted = "#2b2b2a", "#8a8980"
    targets = list(curves)
    fig, axes = plt.subplots(len(targets), len(MODELS), figsize=(4.2 * len(MODELS), 3.4 * len(targets)),
                             squeeze=False, facecolor="#fcfcfb")
    for r, t in enumerate(targets):
        for c, m in enumerate(MODELS):
            ax = axes[r][c]
            ax.set_facecolor("#fcfcfb")
            for cand in CANDS:
                df = curves[t][(m, cand)]
                ax.plot(df.epoch, nulls[t]["train"] - df.train_nll, color=color[cand], lw=2, ls="-")
                ax.plot(df.epoch, nulls[t]["stop"] - df.stop_nll, color=color[cand], lw=2, ls="--")
                s = summaries[t][(m, cand)]["rule_max500"]
                ax.axvline(s["stop_epoch"], color=color[cand], lw=1, ls=":")
            ax.axvline(500, color=muted, lw=1)
            ax.axhline(0, color=muted, lw=0.8)
            ax.set_title(f"{t.upper()} · {m}" + ("  (본 점검)" if m == "D_gmu" else "  (보조)"),
                         color=ink, fontsize=10, loc="left")
            ax.set_xlabel("epoch (full-batch step)", color=muted, fontsize=8)
            if c == 0:
                ax.set_ylabel("null 대비 NLL 개선 (클수록 좋음)", color=muted, fontsize=8)
            ax.tick_params(colors=muted, labelsize=7)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color("#d4d3cd")
            ax.grid(axis="y", color="#ecebe6", lw=0.6)
    from matplotlib.lines import Line2D
    handles = ([Line2D([], [], color=color[c], lw=2, label=f"{c} (lr {CANDIDATES[c]['lr']:g})") for c in CANDS]
               + [Line2D([], [], color=ink, lw=2, ls="-", label="Train (Fit, eval 모드)"),
                  Line2D([], [], color=ink, lw=2, ls="--", label="정지용 S"),
                  Line2D([], [], color=ink, lw=1, ls=":", label="규칙상 종료 (max 500/pat 40)"),
                  Line2D([], [], color=muted, lw=1, label="max epoch 500")])
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False, fontsize=8,
               labelcolor=ink, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    global OUT
    if len(sys.argv) > 2 and sys.argv[1] == "--out_dir":       # 재현성 점검용 두 번째 실행
        OUT = os.path.abspath(sys.argv[2])
    os.makedirs(OUT, exist_ok=True)
    plotstyle.apply()                                    # 한글 폰트
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corpus, _ = features.load_text_corpus(cohort.DEFAULT_MERGED_CSV)
    ctx = {"raw": fsc.load_raw_values(), "text_kwargs": build_encoder("radbert").evaluator_kwargs(corpus)}
    split_csv = os.path.join(OUT, "outer1_inner1_stop_only.csv")
    ids = stop_only_split(split_csv)
    curves, summaries, nulls, enc_info = {}, {}, {}, {}
    for target in ("os", "pfs"):
        seed_everything(SEED)
        labels = cohort.load_trimodal_cohort().drop_duplicates("research_id").set_index("research_id")
        stores = {}
        for kind in ENCODERS:
            store, ev = run_encoders(kind, target, split_csv, ENCODERS[kind]["max_epochs"],
                                     ENCODERS[kind]["patience"], SEED + 100 * OUTER,
                                     os.path.join(OUT, "ckpt", target, kind), None, ctx, [])
            stores[kind] = store[INNER]["splits"]
            enc_info[f"{target}/{kind}"] = {"best_epoch": ev.fold_records[0]["best_epoch"],
                                            "epochs_run": len(ev.training_history)}
        T, I = stores["tabular"], stores["image"]
        tr = (feats(T["train"], I["train"], dev), *labels_tensor(ids["fit"], labels, target, dev))
        st = (feats(T["val"], I["val"], dev), *labels_tensor(ids["stop"], labels, target, dev))
        zero = lambda d: float(cox_efron_loss(torch.zeros(len(d[1]), device=dev), d[1], d[2]))
        nulls[target] = {"train": zero(tr), "stop": zero(st)}
        curves[target], summaries[target] = {}, {}
        for m in MODELS:
            for cand in CANDS:
                ci = list(CANDIDATES).index(cand)
                torch.manual_seed(SEED * 1000 + 100 * OUTER + 10 * INNER + ci)   # run_nested_cv 와 같은 seed
                df = curve(build(m, CANDIDATES[cand]).to(dev), tr, st, CANDIDATES[cand])
                df.to_csv(os.path.join(OUT, f"curve_{target}_{m}_{cand}.csv"), index=False)
                curves[target][(m, cand)] = df
                summaries[target][(m, cand)] = summarize(df, nulls[target]["train"], nulls[target]["stop"])
                s = summaries[target][(m, cand)]
                print(f"{target} {m:13s} {cand}: rule500 stop={s['rule_max500']['stop_epoch']} "
                      f"best={s['rule_max500']['best_epoch']} | min S NLL@{s['stop_nll_min_epoch_ge30_within800']['epoch']}")
    plot(curves, summaries, nulls, os.path.join(OUT, "learning_curves.png"))
    out = {"status": "최적화 점검 전용 — V·outer test 미사용, 모델 선택에 쓰지 않음",
           "outer": OUTER, "inner": INNER, "n_fit": len(ids["fit"]), "n_stop": len(ids["stop"]),
           "rule": RULE, "encoders": enc_info, "null_nll": nulls,
           "summaries": {t: {f"{m}/{c}": v for (m, c), v in s.items()} for t, s in summaries.items()}}
    with open(os.path.join(OUT, "lr_check_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
    print(f"[check_learning_rate] wrote {OUT}")


if __name__ == "__main__":
    main()
