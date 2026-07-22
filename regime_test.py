# -*- coding: utf-8 -*-
"""Step-matched regime test: enlarge the Cox risk set (bigger batch) while
holding optimizer steps ~constant by scaling epochs. Isolates the risk-set
effect from the undertraining confound seen in the fixed-epoch sweep.

Pairs are (config, batch_size, epochs). Train fold ~= 190 patients, so
steps/epoch = ceil(190/bs); total steps = steps/epoch * epochs.
"""
import argparse
import math
import numpy as np
from train import TrimodalEvaluator
from ablation import make_factory, CONFIGS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="os")
    # "config:bs:ep,config:bs:ep,..."
    ap.add_argument("--runs", default="clin_report:32:60,clin_report:64:120,clin_report:32:90")
    args = ap.parse_args()

    results = []
    for spec in args.runs.split(","):
        cfg, bs, ep = spec.split(":")
        bs, ep = int(bs), int(ep)
        steps = math.ceil(190 / bs) * ep
        print(f"\n########## {cfg} bs={bs} ep={ep} (~{steps} steps) target={args.target} ##########")
        ev = TrimodalEvaluator(
            target=args.target, epochs=ep, batch_size=bs,
            save_dir=f"outputs/regime/{cfg}_bs{bs}_ep{ep}_{args.target}",
            model_factory=make_factory(CONFIGS[cfg]),
        ).run()
        m, s = float(np.mean(ev.c_indices)), float(np.std(ev.c_indices))
        results.append((cfg, bs, ep, steps, m, s, [round(c, 4) for c in ev.c_indices]))
        print(f"[REGIME] {cfg} bs={bs} ep={ep}: {m:.4f} +/- {s:.4f}")

    print(f"\n======== REGIME TEST (target={args.target}) ========")
    print(f"{'config':<12}{'bs':>4}{'ep':>5}{'steps':>7}{'mean':>9}{'std':>8}")
    for cfg, bs, ep, steps, m, s, folds in results:
        print(f"{cfg:<12}{bs:>4}{ep:>5}{steps:>7}{m:>9.4f}{s:>8.4f}   {folds}")


if __name__ == "__main__":
    main()
