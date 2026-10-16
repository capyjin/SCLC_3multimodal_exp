# -*- coding: utf-8 -*-
"""③ 결과 요약. run_v2_{os,pfs} 결과로 fold 별 C-index 표·그림, 선택된 후보, gate 분포를 만든다.

Run:  python experiments/실험16_GMU_nestedCV/summarize.py"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json

import numpy as np
import pandas as pd

from nested_fusion.config import MODALITIES
from nested_fusion.model import FUSION_MODELS
from sclc import paths, plotstyle

OUT = paths.outputs("gmu_nested", "report_v2")
REFERENCE = {"os": 0.7224, "pfs": 0.6470}
MAIN = ("A_primary", "B_concat", "C_fixed_gate", "D_gmu")
EXTRA = ("A_ensemble", "A_raw", "tabular_only", "image_only")
PAIRS = (("D_gmu", "A_primary"), ("D_gmu", "B_concat"), ("D_gmu", "C_fixed_gate"),
         ("C_fixed_gate", "B_concat"), ("B_concat", "A_primary"))
COLORS = {"A_primary": "#2a78d6", "B_concat": "#eb6834", "C_fixed_gate": "#1baf7a", "D_gmu": "#eda100"}
INK, MUTED, SURF = "#2b2b2a", "#8a8980", "#fcfcfb"


def load(target):
    d = paths.outputs("gmu_nested", f"run_v2_{target}")
    rd = lambda f: pd.read_csv(os.path.join(d, f))
    rj = lambda f: json.load(open(os.path.join(d, f), encoding="utf-8"))
    enc_hist = pd.concat([pd.read_csv(os.path.join(d, f"outer{k}", "encoder_history.csv")).assign(outer=k)
                          for k in sorted(rd("fold_metrics.csv").outer.unique())], ignore_index=True)
    return {"metrics": rd("fold_metrics.csv"), "inner": rd("inner_results.csv"), "oof": rd("oof_predictions.csv"),
            "enc_hist": enc_hist, "selection": rj("selection.json"),
            "A": rj("late_fusion_A.json"), "gate": rj("gate_analysis.json"), "dir": d}


def cindex_table(m):
    wide = m.pivot(index="model", columns="outer", values="cindex")
    wide["mean"], wide["sd"] = wide.mean(axis=1), wide.std(axis=1, ddof=1)
    return wide.loc[[x for x in MAIN + EXTRA if x in wide.index]]


def deltas(wide):
    folds = [c for c in wide.columns if isinstance(c, (int, np.integer))]
    out = {}
    for a, b in PAIRS:
        d = (wide.loc[a, folds] - wide.loc[b, folds]).astype(float)
        out[f"{a} - {b}"] = {"per_fold": [round(x, 4) for x in d], "mean": round(d.mean(), 4),
                             "sd": round(d.std(ddof=1), 4), "n_positive": int((d > 0).sum()), "n": len(d)}
    return out


def selection_table(sel):
    rows = []
    for k, s in sel.items():
        for m in FUSION_MODELS:
            rows.append({"outer": int(k), "model": m, "selected": s[m]["selected"],
                         "inner_mean_cindex_selected": s[m]["inner_mean_cindex"][s[m]["selected"]],
                         "inner_best_epochs": s[m]["inner_best_epochs"], "refit_epochs": s[m]["refit_epochs"]})
        rows.append({"outer": int(k), "model": "encoders", "selected": "-",
                     "refit_epochs": s["encoders_refit_epochs"]})
    return pd.DataFrame(rows)


def overfit_gmu(inner, sel):
    rows = []
    for k, s in sel.items():
        for m in FUSION_MODELS:
            sub = inner[(inner.outer == int(k)) & (inner.model == m) & (inner.candidate == s[m]["selected"])]
            rows.append({"outer": int(k), "model": m, "candidate": s[m]["selected"],
                         "train_gain_at_best": sub.train_gain_at_best.mean(),
                         "stop_gain_at_best": sub.stop_gain_at_best.mean(),
                         "gap": (sub.train_gain_at_best - sub.stop_gain_at_best).mean(),
                         "best_epoch_median": float(sub.best_epoch.median())})
    return pd.DataFrame(rows)


def overfit_encoders(enc_hist, inner):
    nulls = inner.groupby(["outer", "inner"])[["null_train", "null_stop"]].first()
    rows = []
    for (k, kind, fold), g in enc_hist[enc_hist.phase == "inner"].groupby(["outer", "encoder", "fold"]):
        b = g[g.saved].iloc[-1]
        last = g.iloc[-1]
        nt, ns = nulls.loc[(k, fold)]
        rows.append({"outer": k, "encoder": kind, "inner": fold, "best_epoch": int(b.epoch), "epochs_run": len(g),
                     "train_gain_at_best": nt - b.train_loss_eval, "stop_gain_at_best": ns - b.val_loss,
                     "gap_at_best": (nt - b.train_loss_eval) - (ns - b.val_loss),
                     "train_gain_at_end": nt - last.train_loss_eval, "stop_gain_at_end": ns - last.val_loss})
    return pd.DataFrame(rows)


def gate_summary(oof):
    rows = []
    for k, g in list(oof.groupby("outer")) + [("all", oof)]:
        for mod in MODALITIES:
            x = g[f"gate_D_gmu_{mod}"]
            rows.append({"outer": k, "modality": mod, "mean": x.mean(), "sd": x.std(ddof=1),
                         "q05": x.quantile(.05), "q50": x.median(), "q95": x.quantile(.95),
                         "C_fixed_gate": g[f"gate_C_fixed_gate_{mod}"].iloc[0] if k != "all" else np.nan})
    return pd.DataFrame(rows)


def a_diag(A):
    return pd.DataFrame([{"outer": int(k), **{f"coef_{c}": v for c, v in a["coef_z"].items()},
                          **{f"sd_ratio_{arm}": v for arm, v in a["diagnostics"]["sd_ratio_outer_over_inner"].items()},
                          "eff_img_w_inner": a["diagnostics"]["effective_image_weight"]["inner_oof"],
                          "eff_img_w_outer": a["diagnostics"]["effective_image_weight"]["outer_test"]}
                         for k, a in A.items()])


def plot_cindex(tables, path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(tables), figsize=(6.2 * len(tables), 3.8), facecolor=SURF, squeeze=False)
    for ax, (t, w) in zip(axes[0], tables.items()):
        ax.set_facecolor(SURF)
        folds = [c for c in w.columns if isinstance(c, (int, np.integer))]
        offs = np.linspace(-0.18, 0.18, len(folds))
        for i, m in enumerate(MAIN):
            ys = w.loc[m, folds].to_numpy(float)
            ax.scatter(i + offs, ys, s=64, color=COLORS[m], edgecolor=SURF, linewidth=2, zorder=3)
            ax.plot([i - 0.26, i + 0.26], [w.loc[m, "mean"]] * 2, color=INK, lw=2, zorder=4)
            ax.text(i + 0.29, w.loc[m, "mean"], f"{w.loc[m, 'mean']:.3f}", ha="left", va="center",
                    fontsize=8, color=INK)
        for f, off in enumerate(offs):
            ax.plot(np.arange(len(MAIN)) + off, [w.loc[m, folds[f]] for m in MAIN], color="#d4d3cd", lw=1, zorder=1)
        ax.axhline(REFERENCE[t], color=MUTED, lw=1, ls="--")
        ax.text(len(MAIN) - 0.5, REFERENCE[t], f"기존 late fusion {REFERENCE[t]} (참고)", fontsize=7,
                color=MUTED, ha="right", va="bottom")
        ax.set_xticks(range(len(MAIN)), ["A\nlate fusion", "B\nconcat", "C\nfixed gate", "D\nGMU"], fontsize=8,
                      color=INK)
        ax.set_title(f"{t.upper()} — outer test C-index (점 = fold, 선 = 평균, 회색선 = 같은 fold)",
                     loc="left", fontsize=9, color=INK)
        ax.tick_params(colors=MUTED, labelsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#d4d3cd")
        ax.grid(axis="y", color="#ecebe6", lw=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_gates(oofs, path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(oofs), figsize=(5.6 * len(oofs), 3.6), facecolor=SURF, squeeze=False)
    names = {"clinical": "Clinical", "report": "Report", "image": "Image"}
    for ax, (t, oof) in zip(axes[0], oofs.items()):
        ax.set_facecolor(SURF)
        data = [oof[f"gate_D_gmu_{m}"].to_numpy() for m in MODALITIES]
        bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=True,
                        medianprops=dict(color=INK, lw=2), whiskerprops=dict(color=MUTED),
                        capprops=dict(color=MUTED), flierprops=dict(marker="o", markersize=3,
                                                                    markerfacecolor=MUTED, markeredgecolor=SURF))
        for patch in bp["boxes"]:
            patch.set(facecolor=COLORS["D_gmu"], edgecolor=SURF, alpha=0.85)
        cvals = oof.groupby("outer")[[f"gate_C_fixed_gate_{m}" for m in MODALITIES]].first()
        for i, m in enumerate(MODALITIES, start=1):
            ax.scatter([i + 0.38] * len(cvals), cvals[f"gate_C_fixed_gate_{m}"], marker="D", s=22,
                       color=COLORS["C_fixed_gate"], edgecolor=SURF, linewidth=1, zorder=3)
        ax.set_xticks([1, 2, 3], [names[m] for m in MODALITIES], fontsize=8, color=INK)
        ax.set_ylim(0, 1)
        ax.set_title(f"{t.upper()} — D(GMU) 환자별 gate (outer test 전체, 상자)\n◆ = C 고정 gate (outer fold 별)",
                     loc="left", fontsize=9, color=INK)
        ax.tick_params(colors=MUTED, labelsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#d4d3cd")
        ax.grid(axis="y", color="#ecebe6", lw=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    import matplotlib
    matplotlib.use("Agg")
    plotstyle.apply()
    os.makedirs(OUT, exist_ok=True)
    report, tables, oofs = {"reference_only": REFERENCE}, {}, {}
    for t in ("os", "pfs"):
        r = load(t)
        w = cindex_table(r["metrics"])
        tables[t], oofs[t] = w, r["oof"]
        w.round(4).to_csv(os.path.join(OUT, f"cindex_{t}.csv"))
        sel = selection_table(r["selection"])
        sel.to_csv(os.path.join(OUT, f"selection_{t}.csv"), index=False)
        og, oe = overfit_gmu(r["inner"], r["selection"]), overfit_encoders(r["enc_hist"], r["inner"])
        og.round(4).to_csv(os.path.join(OUT, f"overfit_gmu_{t}.csv"), index=False)
        oe.round(4).to_csv(os.path.join(OUT, f"overfit_encoders_{t}.csv"), index=False)
        gs = gate_summary(r["oof"])
        gs.round(4).to_csv(os.path.join(OUT, f"gates_{t}.csv"), index=False)
        ad = a_diag(r["A"])
        ad.round(4).to_csv(os.path.join(OUT, f"late_fusion_A_diag_{t}.csv"), index=False)
        inner = r["inner"]
        report[t] = {
            "cindex": {m: {"mean": round(w.loc[m, "mean"], 4), "sd": round(w.loc[m, "sd"], 4),
                           "folds": [round(x, 4) for x in w.loc[m, [c for c in w.columns if isinstance(c, (int, np.integer))]]]}
                       for m in w.index},
            "delta": deltas(w),
            "D_minus_reference": round(w.loc["D_gmu", "mean"] - REFERENCE[t], 4),
            "selection": sel.to_dict(orient="records"),
            "gmu_hit_max_epochs": inner[inner.epochs_run == inner.rule_max_epochs][
                ["outer", "inner", "model", "candidate"]].values.tolist(),
            "gates_all": gs[gs.outer == "all"].round(4).to_dict(orient="records"),
        }
    plot_cindex(tables, os.path.join(OUT, "cindex_by_fold.png"))
    plot_gates(oofs, os.path.join(OUT, "gates.png"))
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1, default=str)
    print(f"[summarize] wrote {OUT}")


if __name__ == "__main__":
    main()
