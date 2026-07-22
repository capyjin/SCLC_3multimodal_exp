# -*- coding: utf-8 -*-
"""Test the batch=16 minibatch-Cox hypothesis: sweep batch size for the
tabular-only clin+report config (fast, no CNN). If C-index climbs toward the
original 0.7083 as batch grows, the small Cox risk set was the handicap."""
import argparse
import numpy as np
from train import TrimodalEvaluator
from ablation import make_factory, CONFIGS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="clin_report")
    ap.add_argument("--target", default="os")
    ap.add_argument("--batches", default="16,32,64,128,256")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    results = {}
    for bs in [int(b) for b in args.batches.split(",")]:
        print(f"\n########## {args.config} batch_size={bs} target={args.target} ##########")
        ev = TrimodalEvaluator(
            target=args.target, epochs=args.epochs, batch_size=bs,
            save_dir=f"outputs/batch_sweep/{args.config}_bs{bs}_{args.target}",
            model_factory=make_factory(CONFIGS[args.config]),
        ).run()
        m, s = float(np.mean(ev.c_indices)), float(np.std(ev.c_indices))
        results[bs] = (m, s, [round(c, 4) for c in ev.c_indices])
        print(f"[SWEEP] {args.config} bs={bs}: {m:.4f} +/- {s:.4f}")

    print(f"\n======== BATCH SWEEP ({args.config}, target={args.target}) ========")
    print(f"{'batch':>6}{'mean':>9}{'std':>8}")
    for bs, (m, s, folds) in results.items():
        print(f"{bs:>6}{m:>9.4f}{s:>8.4f}   {folds}")


if __name__ == "__main__":
    main()
