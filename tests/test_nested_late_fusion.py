# -*- coding: utf-8 -*-
"""실험16 비교 모델 A (nested late fusion) 단위 테스트 — 합성 데이터만 쓴다.

Run:  python -m unittest discover -s tests -v
"""
import inspect
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "실험16_GMU_nestedCV"))

import numpy as np
import pandas as pd

from nested_fusion.config import ARMS
from nested_fusion.model import effective_image_weight, fit_stacker, nested_late_fusion
from nested_fusion.processing import pool_inner_oof, standardize, standardizer
from sclc.evaluation import cindex


def synth(n_t=400, n_o=100, seed=0, b_tab=1.0, b_img=0.5):
    """잠재 위험 a(tabular), b(image) 로 생존시간을 만든다. 모델 점수 = 잠재값 x 모델별 배율 + 이동."""
    rng = np.random.default_rng(seed)
    n = n_t + n_o
    a, b = rng.normal(size=n), rng.normal(size=n)
    t = rng.exponential(np.exp(-(b_tab * a + b_img * b)))
    c = rng.exponential(np.median(t) * 3, size=n)
    labels = pd.DataFrame({"os_days": np.minimum(t, c), "os_event": (t <= c).astype(int)},
                          index=np.arange(n))
    t_ids, o_ids = np.arange(n_t), np.arange(n_t, n)
    return rng, a, b, labels, t_ids, o_ids


def build_records(rng, a, b, t_ids, o_ids, n_inner=4, scale_fn=None, noise=0.3):
    """inner 4개 + outer 1개 모델의 점수. scale_fn(model_index, arm) -> (배율, 이동)."""
    scale_fn = scale_fn or (lambda m, arm: (1.0, 0.0))
    latent = {"tabular": a, "image": b}
    folds = np.array_split(rng.permutation(t_ids), n_inner)
    inner = []
    for j, v in enumerate(folds, start=1):
        fit = np.setdiff1d(t_ids, v)
        arms = {}
        for arm in ARMS:
            s, sh = scale_fn(j, arm)
            score = lambda ids: s * (latent[arm][ids] + noise * rng.normal(size=len(ids))) + sh
            arms[arm] = {"fit_risk": score(fit), "val_risk": score(v), "extra_risk": score(o_ids)}
        inner.append({"j": j, "val_ids": list(map(int, v)), "arms": arms})
    arms = {}
    for arm in ARMS:
        s, sh = scale_fn(0, arm)
        arms[arm] = {"train_risk": s * (latent[arm][t_ids] + noise * rng.normal(size=len(t_ids))) + sh,
                     "test_risk": s * (latent[arm][o_ids] + noise * rng.normal(size=len(o_ids))) + sh}
    return inner, {"test_ids": list(map(int, o_ids)), "arms": arms}


class TestPoolingAndStandardization(unittest.TestCase):
    def setUp(self):
        self.rng, self.a, self.b, self.labels, self.t, self.o = synth()
        self.inner, self.outer = build_records(self.rng, self.a, self.b, self.t, self.o)

    def test_oof_is_exact_partition(self):
        oof = pool_inner_oof(self.inner, self.t)
        self.assertEqual(sorted(oof.index), sorted(self.t.tolist()))
        bad = [dict(r) for r in self.inner]
        bad[1] = {**bad[1], "val_ids": bad[0]["val_ids"]}          # 중복 + 누락
        with self.assertRaises(ValueError):
            pool_inner_oof(bad, self.t)
        with self.assertRaises(ValueError):
            pool_inner_oof(self.inner[:3], self.t)                # 누락

    def test_standardized_by_own_training_scores(self):
        oof = pool_inner_oof(self.inner, self.t)
        rec = self.inner[2]
        a = rec["arms"]["image"]
        expect = (np.asarray(a["val_risk"]) - np.mean(a["fit_risk"])) / np.std(a["fit_risk"], ddof=1)
        np.testing.assert_allclose(oof.loc[rec["val_ids"], "z_image"].to_numpy(), expect)


class TestStacker(unittest.TestCase):
    def test_recovers_generating_weights(self):
        rng, a, b, labels, t, o = synth(n_t=3000, n_o=10, seed=1)
        x = pd.DataFrame({"z_tabular": a[t], "z_image": b[t]}, index=t)
        coef = fit_stacker(x, labels.loc[t], "os")
        self.assertAlmostEqual(coef["z_tabular"], 1.0, delta=0.1)
        self.assertAlmostEqual(coef["z_image"], 0.5, delta=0.1)

    def test_no_outer_labels_enter(self):
        params = inspect.signature(nested_late_fusion).parameters
        self.assertNotIn("o_labels", params)
        rng, a, b, labels, t, o = synth(seed=2)
        inner, outer = build_records(rng, a, b, t, o)
        r1 = nested_late_fusion(inner, outer, t, labels.loc[t], "os")
        outer2 = {**outer, "arms": {arm: {**outer["arms"][arm],
                                          "test_risk": outer["arms"][arm]["test_risk"][::-1]}
                                    for arm in ARMS}}
        r2 = nested_late_fusion(inner, outer2, t, labels.loc[t], "os")
        self.assertEqual(r1["coef_z"], r2["coef_z"])                # 계수는 inner OOF 로만 정해진다
        lab_bad = labels.copy()
        lab_bad.loc[o, "os_days"] = 1.0                              # O_k 레이블을 망가뜨려도
        r3 = nested_late_fusion(inner, outer, t, lab_bad.loc[t], "os")
        self.assertEqual(r1["coef_z"], r3["coef_z"])
        with self.assertRaises(KeyError):                            # T_k 레이블이 없으면 실패
            nested_late_fusion(inner, outer, t, labels.loc[o], "os")


class TestScaleConsistency(unittest.TestCase):
    def test_primary_invariant_to_per_model_scale_and_shift(self):
        rng, a, b, labels, t, o = synth(seed=3)
        state = rng.bit_generator.state
        inner1, outer1 = build_records(rng, a, b, t, o)
        rng.bit_generator.state = state                              # 같은 잡음, 배율만 다르게
        scales = {(m, arm): (float(np.exp(rng_s)), float(sh)) for (m, arm), rng_s, sh in zip(
            [(m, arm) for m in range(5) for arm in ARMS],
            np.random.default_rng(9).normal(size=10), np.random.default_rng(10).normal(size=10) * 5)}
        inner2, outer2 = build_records(rng, a, b, t, o, scale_fn=lambda m, arm: scales[(m, arm)])
        r1 = nested_late_fusion(inner1, outer1, t, labels.loc[t], "os")
        r2 = nested_late_fusion(inner2, outer2, t, labels.loc[t], "os")
        for v in ("primary", "ensemble"):
            np.testing.assert_allclose(r1["pred"][v], r2["pred"][v], rtol=1e-5, atol=1e-6,
                                       err_msg=f"{v} should not depend on per-model scale")
        # raw 는 모델별 배율이 다르면 순위가 바뀐다 (표준화가 필요한 이유)
        c1 = cindex(labels.loc[o, "os_days"], r1["pred"]["raw"], labels.loc[o, "os_event"])
        c2 = cindex(labels.loc[o, "os_days"], r2["pred"]["raw"], labels.loc[o, "os_event"])
        self.assertNotAlmostEqual(c1, c2, places=6)

    def test_sd_ratio_detects_outer_spread_change(self):
        rng, a, b, labels, t, o = synth(seed=4)
        inner, outer = build_records(rng, a, b, t, o)
        r = nested_late_fusion(inner, outer, t, labels.loc[t], "os")
        for arm in ARMS:
            self.assertAlmostEqual(r["diagnostics"]["sd_ratio_outer_over_inner"][arm], 1.0, delta=0.25)
        outer_wide = {**outer, "arms": {**outer["arms"], "image": {
            **outer["arms"]["image"], "test_risk": outer["arms"]["image"]["test_risk"] * 3}}}
        r2 = nested_late_fusion(inner, outer_wide, t, labels.loc[t], "os")
        self.assertGreater(r2["diagnostics"]["sd_ratio_outer_over_inner"]["image"], 2.0)
        self.assertGreater(r2["diagnostics"]["effective_image_weight"]["outer_test"],
                           r2["diagnostics"]["effective_image_weight"]["inner_oof"])

    def test_ensemble_is_mean_of_inner_standardized(self):
        rng, a, b, labels, t, o = synth(seed=5)
        inner, outer = build_records(rng, a, b, t, o)
        r = nested_late_fusion(inner, outer, t, labels.loc[t], "os")
        z = {arm: np.mean([standardize(rec["arms"][arm]["extra_risk"],
                                           standardizer(rec["arms"][arm]["fit_risk"]))
                           for rec in inner], axis=0) for arm in ARMS}
        expect = r["coef_z"]["z_tabular"] * z["tabular"] + r["coef_z"]["z_image"] * z["image"]
        np.testing.assert_allclose(r["pred"]["ensemble"], expect)

    def test_effective_weight_and_slopes(self):
        self.assertAlmostEqual(effective_image_weight({"z_tabular": 2.0, "z_image": -1.0},
                                                          {"tabular": 1.0, "image": 2.0}), 0.5)
        rng, a, b, labels, t, o = synth(seed=6)
        inner, outer = build_records(rng, a, b, t, o)
        slopes = nested_late_fusion(inner, outer, t, labels.loc[t], "os")["diagnostics"]["calibration_slopes"]
        self.assertEqual(sorted(slopes), [1, 2, 3, 4])
        self.assertTrue(all(s["tabular"] > 0 and s["image"] > 0 for s in slopes.values()))

    def test_risk_direction(self):
        """선형 예측자는 클수록 위험 — 저장소 C-index 관례와 같은 방향."""
        rng, a, b, labels, t, o = synth(n_t=1500, n_o=500, seed=7, b_img=1.0)
        inner, outer = build_records(rng, a, b, t, o, noise=0.05)
        r = nested_late_fusion(inner, outer, t, labels.loc[t], "os")
        self.assertGreater(cindex(labels.loc[o, "os_days"], r["pred"]["primary"],
                                  labels.loc[o, "os_event"]), 0.7)


if __name__ == "__main__":
    unittest.main()
