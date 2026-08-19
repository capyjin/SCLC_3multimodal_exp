# -*- coding: utf-8 -*-
"""순열 p값 · 부트스트랩 CI(F팔) · 표 · 그림 — 학습 없이 산출물만 만든다.

읽는 것: ``outputs/image_permutation/{runs.jsonl, oof_*_r*.json, trivial_stats.json}``
쓰는 것: 같은 폴더의 ``results.json``, ``plots/null_distribution.png``

검정통계량은 5-fold 평균 C-index, 순열 p 는

    p = (1 + #{귀무 ≥ 관측}) / (N + 1)      (Phipson-Smyth)

이다. +1 은 "관측 자신도 하나의 순열"이라는 관례이며, 없으면 N 이 작을 때 p=0 이
나와 과대해석된다. **따라서 N=100 에서 얻을 수 있는 최소 p 는 1/101 ≈ 0.0099 이고,
그보다 강한 주장은 할 수 없다.**

귀무는 순열 범위별로 따로 모은다. 두 범위(stratified/global)는 서로 다른 귀무
구성이므로 한 p값에 섞으면 안 된다.

Run:  python 실험10_영상단독_난수대조검정/analyze.py
"""
import argparse
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core import plotstyle as ps  # noqa: E402
from core.metrics import cindex  # noqa: E402

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "image_permutation")
TARGETS = ("os", "pfs")
# 앞이 주력. global 만 귀무 중심이 0.50 이다 — stratified 는 환자의 event 비트를
# 고정하므로 "영상 -> 중도절단 여부" 연관(AUC 0.63)이 귀무에 남아 중심이 0.53 으로
# 밀린다. 그래서 stratified 는 "중도절단으로 설명되는 몫을 뺀 뒤에도 남는가"를 묻는
# 더 엄격한 조건부 검정으로만 쓴다.
SCOPES = ("global", "stratified")
N_BOOT, BOOT_SEED = 2000, 42


def load_runs(path: str, epochs: int) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        runs = [json.loads(line) for line in fh if line.strip()]
    missing = [r for r in runs if "epochs" not in r]
    if missing:
        raise SystemExit(f"runs.jsonl 에 epochs 가 없는 레코드 {len(missing)}개 — "
                         "학습 조건이 다른 기록이 섞였을 수 있다. backfill 후 다시 실행하라.")
    return [r for r in runs if r["epochs"] == epochs]


def select(runs, arm: str, target: str, scope=None) -> list[dict]:
    hits = [r for r in runs if r["arm"] == arm and r["target"] == target]
    return [r for r in hits if r["perm_scope"] == scope] if arm == "C" else hits


def means(runs, arm: str, target: str, scope=None) -> np.ndarray:
    return np.array([r["mean"] for r in select(runs, arm, target, scope)])


def permutation_p(observed: float, null: np.ndarray) -> float:
    return float((1 + int((null >= observed).sum())) / (len(null) + 1))


def describe(vals: np.ndarray) -> dict:
    return {"n": int(len(vals)), "mean": float(vals.mean()), "std": float(vals.std()),
            "min": float(vals.min()), "max": float(vals.max()),
            "q025": float(np.percentile(vals, 2.5)), "q95": float(np.percentile(vals, 95)),
            "q975": float(np.percentile(vals, 97.5))}


def bootstrap_ci(oof: list[dict], n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """fold 내 환자 재표집으로 관측값의 표본변동을 잰다.

    ⚠️ 이 구간은 **학습된 모델을 고정한 채** test 환자만 다시 뽑은 것이라
    학습/시드 변동을 포함하지 않는다. 0.6570 전체의 불확실성이 아니라
    "이 모델들이 이 환자들에서 낸 값"의 구간이다. 학습 변동은 B팔이 따로 잰다.

    fold 마다 위험점수 척도가 다르므로 238명을 이어붙이지 않고 fold 안에서만
    재표집한 뒤 fold 평균을 낸다(core.metrics.fold_mean_cindex 관례).
    """
    by_fold: dict[int, list[dict]] = {}
    for row in oof:
        by_fold.setdefault(int(row["fold"]), []).append(row)
    packed = [(np.array([r["duration"] for r in rows], float),
               np.array([r["risk_score"] for r in rows], float),
               np.array([r["event"] for r in rows], float)) for rows in by_fold.values()]

    rng = np.random.default_rng(seed)
    draws, dropped = [], 0
    for _ in range(n_boot):
        fold_cis = []
        for dur, risk, evt in packed:
            idx = rng.integers(0, len(dur), len(dur))
            if evt[idx].sum() == 0:      # C-index 정의 불가 -> fold 만 빼면 추정량이 바뀌므로 draw 전체를 버린다
                fold_cis = None
                break
            fold_cis.append(cindex(dur[idx], risk[idx], evt[idx]))
        if fold_cis is None:
            dropped += 1
            continue
        draws.append(float(np.mean(fold_cis)))
    draws = np.array(draws)
    return {"n_boot": int(len(draws)), "n_dropped": dropped, "mean": float(draws.mean()),
            "ci_lo": float(np.percentile(draws, 2.5)), "ci_hi": float(np.percentile(draws, 97.5)),
            "boot_frac_below_0.5": float((draws <= 0.5).mean()),
            "note": "학습된 모델 고정, test 환자만 재표집 — 학습/시드 변동 미포함"}


def summarize(runs, target: str) -> dict:
    baseline = select(runs, "A", target)
    if not baseline:
        return {"target": target}
    obs = min(baseline, key=lambda r: r["replicate"])
    out = {"target": target, "observed": obs["mean"], "observed_replicate": obs["replicate"],
           "observed_folds": obs["folds"], "nulls": {}, "permutation_p": {}}

    for scope in SCOPES:
        null = means(runs, "C", target, scope)
        if len(null):
            out["nulls"][scope] = describe(null)
            out["permutation_p"][scope] = permutation_p(out["observed"], null)

    for arm, key in (("B", "real_seeds"), ("D", "noise_image")):
        vals = means(runs, arm, target)
        if len(vals):
            out[key] = describe(vals)

    # B팔 전체가 귀무 95분위 위에 있는가 — 중앙값으로 가짜 p값을 만들지 않고 이렇게만 말한다
    primary = out["nulls"].get(SCOPES[0])
    if primary and "real_seeds" in out:
        out["all_seeds_above_null_q95"] = bool(out["real_seeds"]["min"] > primary["q95"])

    oof_path = os.path.join(OUT_DIR, f"oof_{target}_r{obs['replicate']}.json")
    if os.path.exists(oof_path):
        with open(oof_path, encoding="utf-8") as fh:
            out["bootstrap"] = bootstrap_ci(json.load(fh))
    return out


def plot(summary: dict, runs, path: str) -> None:
    import matplotlib.pyplot as plt
    ps.apply()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    for ax, target in zip(axes, TARGETS):
        s = summary[target]
        null = means(runs, "C", target, SCOPES[0])
        if len(null):
            ax.hist(null, bins=24, color=ps.GRID, edgecolor=ps.BASE,
                    label=f"귀무분포 (라벨순열 {len(null)}회)")
        ax.axvline(0.5, color=ps.MUTED, ls=":", lw=1.4, label="동전던지기 0.50")
        if "noise_image" in s:
            ax.axvline(s["noise_image"]["mean"], color=ps.MAGENTA, ls="--", lw=1.6,
                       label=f"잡음 이미지 {s['noise_image']['mean']:.3f}")
        if "trivial" in s:
            ax.axvline(s["trivial"], color=ps.GREEN, ls="-.", lw=1.8,
                       label=f"전역통계 6개 {s['trivial']:.3f}")
        if s.get("observed") is not None:
            ax.axvline(s["observed"], color=ps.TARGET_COLOR[target], lw=2.8,
                       label=f"관측 영상단독 {s['observed']:.4f}")
        title = target.upper()
        if SCOPES[0] in s.get("permutation_p", {}):
            title += f"   순열 p = {s['permutation_p'][SCOPES[0]]:.4f}"
        ax.set_title(title)
        ax.set_xlabel("5-fold 평균 C-index")
        ax.legend(fontsize=8, frameon=False)
    axes[0].set_ylabel("순열 횟수")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"-> {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30, help="이 학습 조건의 기록만 모은다")
    args = ap.parse_args()

    runs = load_runs(os.path.join(OUT_DIR, "runs.jsonl"), args.epochs)
    summary = {t: summarize(runs, t) for t in TARGETS}

    trivial_path = os.path.join(OUT_DIR, "trivial_stats.json")
    if os.path.exists(trivial_path):
        with open(trivial_path, encoding="utf-8") as fh:
            trivial = json.load(fh)
        summary["trivial_stats"] = trivial
        for t in TARGETS:
            summary[t]["trivial"] = trivial["results"]["ALL6"][t]["mean"]

    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    def row(label, fn):
        print(f"{label:<30}{fn('os'):>24}{fn('pfs'):>24}")
    fmt = lambda v, d=4: "-" if v is None else f"{v:.{d}f}"

    print(f"\n{'':<30}{'OS':>24}{'PFS':>24}")
    row("관측 영상단독 (A)", lambda t: fmt(summary[t].get("observed")))
    for scope in SCOPES:
        row(f"귀무 평균 [{scope}]",
            lambda t, s=scope: (f"{summary[t]['nulls'][s]['mean']:.4f} ± {summary[t]['nulls'][s]['std']:.4f}"
                                f" (n={summary[t]['nulls'][s]['n']})" if s in summary[t].get("nulls", {}) else "-"))
        row(f"귀무 95분위 [{scope}]",
            lambda t, s=scope: fmt(summary[t]["nulls"][s]["q95"]) if s in summary[t].get("nulls", {}) else "-")
        row(f"순열 p [{scope}]",
            lambda t, s=scope: fmt(summary[t]["permutation_p"][s]) if s in summary[t].get("permutation_p", {}) else "-")
    row("시드 재현 (B)", lambda t: (f"{summary[t]['real_seeds']['mean']:.4f} "
                                    f"[{summary[t]['real_seeds']['min']:.3f}, {summary[t]['real_seeds']['max']:.3f}]"
                                    if "real_seeds" in summary[t] else "-"))
    row("잡음 이미지 (D)", lambda t: fmt(summary[t].get("noise_image", {}).get("mean")))
    row("전역통계 6개, CNN없음 (E)", lambda t: fmt(summary[t].get("trivial")))
    row("부트스트랩 95%* (F)", lambda t: (f"[{summary[t]['bootstrap']['ci_lo']:.4f}, "
                                          f"{summary[t]['bootstrap']['ci_hi']:.4f}]"
                                          if "bootstrap" in summary[t] else "-"))
    print("  * F는 학습된 모델을 고정한 채 test 환자만 재표집한 구간 — 학습/시드 변동은 B가 잰다.")

    plot(summary, runs, os.path.join(OUT_DIR, "plots", "null_distribution.png"))


if __name__ == "__main__":
    main()
