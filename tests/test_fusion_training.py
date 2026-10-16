# -*- coding: utf-8 -*-
"""실험16 융합 모델 B·C·D 와 학습 규칙(EarlyStopper·후보 선택·재학습 epoch) 단위 테스트.

Run:  python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "실험16_GMU_nestedCV"))

import torch

from nested_fusion.config import CANDIDATES, FEATURE_DIMS
from nested_fusion.model import FUSION_MODELS, build
from nested_fusion.train import (EarlyStopper, effective_candidates, refit_epochs, rule_for, select_candidate,
                                 train_fusion)
from sclc.train import cox_efron_loss


def run_stopper(values, **kw):
    st = EarlyStopper(**kw)
    for e, v in enumerate(values, start=1):
        _, halt = st.update(e, v)
        if halt:
            return st.best_epoch, e
    return st.best_epoch, len(values)


class TestEarlyStopper(unittest.TestCase):
    def test_patience_counts_from_last_improvement(self):
        vals = [5, 4, 3, 2, 2.5, 2.4, 2.3, 2.2, 2.1, 2.05]
        self.assertEqual(run_stopper(vals, patience=3), (4, 7))

    def test_min_epoch_excludes_earlier_epochs(self):
        vals = [0.1, 0.2] + [1.0, 0.9, 0.95, 0.96, 0.97]
        best, stop = run_stopper(vals, patience=3, min_epoch=3)
        self.assertEqual(best, 4)                      # epoch 1 의 0.1 은 후보가 아니다
        self.assertEqual(stop, 7)

    def test_never_stops_before_min_epoch_plus_patience(self):
        best, stop = run_stopper([1.0] * 100, patience=40, min_epoch=30)
        self.assertEqual((best, stop), (30, 70))

    def test_min_delta_and_ties_keep_earlier(self):
        vals = [1.0, 0.99995, 1.0, 0.9998]           # 5e-5 개선은 min_delta 1e-4 미만
        st = EarlyStopper(patience=10, min_delta=1e-4)
        flags = [st.update(e, v)[0] for e, v in enumerate(vals, start=1)]
        self.assertEqual(flags, [True, False, False, True])
        self.assertEqual(st.best_epoch, 4)


class TestRules(unittest.TestCase):
    def test_select_candidate(self):
        self.assertEqual(select_candidate({"G1": 0.700, "G3": 0.704, "G2": 0.703}), "G1")
        self.assertEqual(select_candidate({"G1": 0.690, "G3": 0.704, "G2": 0.700}), "G2")
        self.assertEqual(select_candidate({"G1": 0.690, "G4": 0.6995, "G3": 0.704}), "G4")

    def test_rules_by_learning_rate(self):
        self.assertEqual(rule_for(CANDIDATES["G1"]),
                         dict(max_epochs=5000, patience=400, min_epoch=30, min_delta=1e-4))
        self.assertEqual(rule_for(CANDIDATES["G4"]),
                         dict(max_epochs=500, patience=40, min_epoch=30, min_delta=1e-4))
        with self.assertRaises(ValueError):
            rule_for({"lr": 3e-4})

    def test_effective_candidates_drop_duplicate_configs(self):
        self.assertEqual(list(effective_candidates("D_gmu")), list(CANDIDATES))
        for m in ("B_concat", "C_fixed_gate"):
            self.assertEqual(list(effective_candidates(m)), ["G1", "G3", "G4", "G5", "G6"])

    def test_refit_epochs(self):
        self.assertEqual(refit_epochs([40, 43, 41, 42]), 42)   # 41.5 -> 42
        self.assertEqual(refit_epochs([30, 31, 35, 90]), 33)
        with self.assertRaises(ValueError):
            refit_epochs([None])


def toy(n=60, seed=0):
    g = torch.Generator().manual_seed(seed)
    feats = {m: torch.randn(n, d, generator=g) for m, d in FEATURE_DIMS.items()}
    dur = torch.randint(1, 20, (n,), generator=g).float()
    evt = (torch.rand(n, generator=g) < 0.8).float()
    return feats, dur, evt


class TestModels(unittest.TestCase):
    def test_shapes_and_gates(self):
        f, _, _ = toy()
        for name in FUSION_MODELS:
            for cand in ("G1", "G3"):
                m = build(name, CANDIDATES[cand])
                self.assertEqual(tuple(m(f).shape), (60, 1))
        c = build("C_fixed_gate", CANDIDATES["G1"])
        self.assertTrue(torch.equal(c.gates(f), c.gates(f)[0].expand(60, -1)))
        d = build("D_gmu", CANDIDATES["G1"])
        self.assertGreater(float(d.gates(f).std(dim=0).sum()), 0)     # 환자마다 다르다

    def test_train_fusion_restores_best_and_fixed_mode(self):
        torch.manual_seed(0)
        tr, st = toy(seed=1), toy(seed=2)
        m = build("D_gmu", CANDIDATES["G4"])
        res = train_fusion(m, tr, st, CANDIDATES["G4"], cox_efron_loss,
                              max_epochs=80, patience=10, min_epoch=5)
        saved = [r for r in res["history"] if r.get("saved")]
        self.assertEqual(res["best_epoch"], saved[-1]["epoch"])
        self.assertGreaterEqual(res["best_epoch"], 5)
        with torch.no_grad():
            now = float(cox_efron_loss(m(st[0]), st[1], st[2]))
        self.assertAlmostEqual(now, saved[-1]["stop_loss"], places=5)  # 최적 가중치로 되돌아왔다
        m2 = build("B_concat", CANDIDATES["G1"])
        res2 = train_fusion(m2, tr, None, CANDIDATES["G1"], cox_efron_loss, max_epochs=7)
        self.assertEqual((res2["epochs_run"], res2["best_epoch"]), (7, 7))

    def test_feature_stats_keys(self):
        f, _, _ = toy()
        s = build("D_gmu", CANDIDATES["G1"]).feature_stats(f)
        self.assertEqual(set(s), {"pre_projection_norm", "pre_projection_elem_sd",
                                  "post_projection_norm", "gates"})
        self.assertNotIn("gates", build("B_concat", CANDIDATES["G1"]).feature_stats(f))


if __name__ == "__main__":
    unittest.main()
