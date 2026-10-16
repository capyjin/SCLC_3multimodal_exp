# -*- coding: utf-8 -*-
"""① nested CV 분할 생성. 기존 outer 5-fold 안에 inner 4-fold 와 조기 종료용 분할을 만든다.
확정본: --out_dir data/splits/gmu_nested_v1 --status LOCKED

Run:  python experiments/실험16_GMU_nestedCV/make_splits.py"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import StratifiedKFold, train_test_split

from sclc import cohort, paths
from sclc.train import fold_plan

DEFAULT_OUT = paths.outputs("gmu_nested", "splits_draft")


def seed_for(base: int, outer: int, inner: int = 0) -> int:
    return base + 100 * outer + inner


def strata(labels: pd.DataFrame, ids, rule: str) -> np.ndarray:
    sub = labels.loc[ids]
    if rule == "os_event":
        return sub["os_event"].to_numpy()
    if rule == "os_event_x_os_time_tertile":
        tert = pd.qcut(sub["os_days"].rank(method="first"), 3, labels=False).to_numpy()
        return sub["os_event"].to_numpy() * 3 + tert
    raise ValueError(rule)


def describe(labels: pd.DataFrame, ids) -> dict:
    sub = labels.loc[list(ids)]
    return {"n": len(sub),
            "os_event_rate": round(float(sub.os_event.mean()), 4),
            "pfs_event_rate": round(float(sub.pfs_event.mean()), 4),
            "os_days_median": float(sub.os_days.median()),
            "pfs_days_median": float(sub.pfs_days.median())}


def check_balance(labels, v_ids, rest_ids, max_gap, min_p) -> dict:
    """V 와 나머지의 PFS 사건률 차이와 관찰시간 분포(KS 검정) 점검."""
    v, r = labels.loc[v_ids], labels.loc[rest_ids]
    out = {"pfs_event_gap": round(abs(float(v.pfs_event.mean() - r.pfs_event.mean())), 4),
           "os_event_gap": round(abs(float(v.os_event.mean() - r.os_event.mean())), 4),
           "ks_p_os_days": round(float(ks_2samp(v.os_days, r.os_days).pvalue), 4),
           "ks_p_pfs_days": round(float(ks_2samp(v.pfs_days, r.pfs_days).pvalue), 4)}
    out["pass"] = bool(out["pfs_event_gap"] <= max_gap and out["ks_p_os_days"] >= min_p
                       and out["ks_p_pfs_days"] >= min_p)
    return out


def split_outer(labels, t_ids, k, args, rule):
    t_ids = np.array(sorted(t_ids))
    skf = StratifiedKFold(n_splits=args.n_inner, shuffle=True, random_state=seed_for(args.seed, k))
    inner = []
    for j, (tr_pos, v_pos) in enumerate(skf.split(t_ids, strata(labels, t_ids, rule)), start=1):
        tr, v = t_ids[tr_pos], t_ids[v_pos]
        fit, stop = train_test_split(tr, test_size=args.stop_frac, shuffle=True,
                                     random_state=seed_for(args.seed, k, j),
                                     stratify=labels.loc[tr, "os_event"].to_numpy())
        inner.append({"j": j, "fit": sorted(map(int, fit)), "stop": sorted(map(int, stop)),
                      "val": sorted(map(int, v)),
                      "balance": check_balance(labels, v, tr, args.max_pfs_event_gap, args.min_ks_p)})
    return inner


def invariants(t_ids, o_ids, inner) -> dict:
    t, o = set(t_ids), set(o_ids)
    vals = [set(f["val"]) for f in inner]
    ok = {"outer_train_test_disjoint": not (t & o),
          "inner_val_partition_of_outer_train": set().union(*vals) == t
          and sum(len(v) for v in vals) == len(t)}
    ok["fit_stop_val_disjoint_and_cover"] = all(
        not (set(f["fit"]) & set(f["stop"])) and not ((set(f["fit"]) | set(f["stop"])) & set(f["val"]))
        and (set(f["fit"]) | set(f["stop"]) | set(f["val"])) == t for f in inner)
    ok["outer_test_never_inside"] = all(
        not ((set(f["fit"]) | set(f["stop"]) | set(f["val"])) & o) for f in inner)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n_inner", type=int, default=4)
    ap.add_argument("--stop_frac", type=float, default=0.15, help="N2 정지용 분할 비율")
    ap.add_argument("--seed", type=int, default=42, help="N7 seed 기준값")
    ap.add_argument("--max_pfs_event_gap", type=float, default=0.10, help="N1")
    ap.add_argument("--min_ks_p", type=float, default=0.05, help="N1")
    ap.add_argument("--out_dir", default=DEFAULT_OUT)
    ap.add_argument("--status", default="DRAFT — N1~N7 확정 전",
                    help="fold_manifest 에 남길 상태 문구")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cdf = cohort.load_trimodal_cohort()
    labels = cdf.drop_duplicates("research_id").set_index("research_id")[
        ["os_days", "os_event", "pfs_days", "pfs_event"]]
    manifest = {"config": {**vars(args), "out_dir": os.path.relpath(args.out_dir, PROJECT_ROOT),
                           "outer_source": os.path.relpath(cohort.DEFAULT_SPLIT_CSV, PROJECT_ROOT),
                           "inner_primary_rule": "os_event",
                           "inner_fallback_rule": "os_event_x_os_time_tertile",
                           "status": args.status},
                "cohort": describe(labels, labels.index), "outer": []}
    seen_test = []
    for k, ids in fold_plan(cdf):
        o_ids = ids["test"]
        t_ids = ids["train"] + ids["val"]
        seen_test += o_ids
        rule = "os_event"
        inner = split_outer(labels, t_ids, k, args, rule)
        if not all(f["balance"]["pass"] for f in inner):
            rule = "os_event_x_os_time_tertile"
            inner = split_outer(labels, t_ids, k, args, rule)
        rows = []
        for f in inner:
            rows += [(i, "train", f["j"]) for i in f["fit"]]
            rows += [(i, "val", f["j"]) for i in f["stop"]]
            rows += [(i, "test", f["j"]) for i in f["val"]]
        pd.DataFrame(rows, columns=["research_id", "split", "fold"]).to_csv(
            os.path.join(args.out_dir, f"outer{k}_inner.csv"), index=False)
        pd.DataFrame([(i, "train", k) for i in sorted(t_ids)] + [(i, "test", k) for i in sorted(o_ids)],
                     columns=["research_id", "split", "fold"]).to_csv(
            os.path.join(args.out_dir, f"outer{k}_refit.csv"), index=False)
        manifest["outer"].append({
            "k": k, "rule_used": rule,
            "outer_train": describe(labels, t_ids), "outer_test": describe(labels, o_ids),
            "outer_test_ids": sorted(map(int, o_ids)),
            "inner": [{"j": f["j"], "fit": describe(labels, f["fit"]), "stop": describe(labels, f["stop"]),
                       "val": describe(labels, f["val"]), "balance": f["balance"],
                       "ids": {"fit": f["fit"], "stop": f["stop"], "val": f["val"]}} for f in inner],
            "invariants": invariants(t_ids, o_ids, inner),
        })
    manifest["each_patient_outer_test_once"] = (sorted(seen_test) == sorted(labels.index)
                                                and len(seen_test) == len(set(seen_test)))
    with open(os.path.join(args.out_dir, "fold_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)

    print(f"cohort {manifest['cohort']}  each_patient_outer_test_once="
          f"{manifest['each_patient_outer_test_once']}")
    for o in manifest["outer"]:
        print(f"\nouter {o['k']}  rule={o['rule_used']}  T={o['outer_train']['n']} O={o['outer_test']['n']}  "
              f"invariants={all(o['invariants'].values())}")
        for f in o["inner"]:
            b = f["balance"]
            print(f"  j{f['j']} fit {f['fit']['n']:3d} stop {f['stop']['n']:2d} val {f['val']['n']:2d} | "
                  f"V os_ev {f['val']['os_event_rate']:.3f} pfs_ev {f['val']['pfs_event_rate']:.3f} "
                  f"| gap pfs {b['pfs_event_gap']:.3f} ks_os {b['ks_p_os_days']:.3f} "
                  f"ks_pfs {b['ks_p_pfs_days']:.3f} {'PASS' if b['pass'] else 'FAIL'}")
    print(f"\n[splits] wrote {args.out_dir}")


if __name__ == "__main__":
    main()
