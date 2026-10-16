# -*- coding: utf-8 -*-
"""``sclc.train.cox_partial_nll`` (Efron/Breslow) 단위 테스트 — 합성 데이터만 쓴다.

외부 기준 두 개(lifelines CoxPHFitter = Efron, scikit-survival CoxPHSurvivalAnalysis
= Efron/Breslow)와 독립 구현(순수 파이썬 루프), 손으로 푼 예제로 값을 맞추고,
불변성·그래디언트·경계 조건을 확인한다.

Run:  python -m unittest discover -s tests -v
"""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np
import pandas as pd
import torch

from sclc.train import cox_efron_loss, cox_partial_nll, cox_ph_loss


def naive_nll(eta, time, event, ties):
    """정의 그대로의 이중 루프 구현 (사건 1건당 평균)."""
    eta, time, event = map(np.asarray, (eta, time, event))
    total, n_event = 0.0, int(event.sum())
    for t in np.unique(time[event == 1]):
        dset = (time == t) & (event == 1)
        rset = time >= t
        a, b, d = np.exp(eta[rset]).sum(), np.exp(eta[dset]).sum(), int(dset.sum())
        total += eta[dset].sum()
        for l in range(d):
            c = l / d if ties == "efron" else 0.0
            total -= math.log(a - c * b)
    return -total / n_event


def tied_data(n=120, p=2, seed=0, n_times=12):
    """동점이 많은 합성 생존자료 (정수 시간, 약 75% 사건)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, p))
    beta = np.array([0.8, -0.5, 0.3][:p])
    raw = rng.exponential(scale=np.exp(-x @ beta))
    time = np.ceil(raw / raw.max() * n_times).astype(float)
    event = (rng.uniform(size=n) < 0.75).astype(float)
    return x, time, event


def t64(*arrays):
    return [torch.as_tensor(np.asarray(a), dtype=torch.float64) for a in arrays]


def fit_beta(x, time, event, ties):
    """우리 손실을 LBFGS 로 최소화한 선형 Cox 계수."""
    xt, tt, et = t64(x, time, event)
    beta = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([beta], lr=1.0, max_iter=500, tolerance_grad=1e-12,
                            tolerance_change=1e-14, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = cox_partial_nll(xt @ beta, tt, et, ties=ties)
        loss.backward()
        return loss

    opt.step(closure)
    return beta.detach().numpy()


class TestAgainstReferences(unittest.TestCase):
    def test_matches_naive_loop(self):
        for seed in range(5):
            x, time, event = tied_data(n=60, seed=seed, n_times=6)
            eta = np.random.default_rng(seed + 10).normal(size=len(time))
            for ties in ("efron", "breslow"):
                ours = cox_partial_nll(*t64(eta, time, event), ties=ties).item()
                self.assertAlmostEqual(ours, naive_nll(eta, time, event, ties), places=10)

    def test_hand_computed_example(self):
        # 시간 [1,1,2,3], 사건 [1,1,1,0]: t=1 에 동점 사건 2건
        eta = np.array([0.1, -0.2, 0.3, 0.0])
        time, event = np.array([1, 1, 2, 3.0]), np.array([1, 1, 1, 0.0])
        e = np.exp(eta)
        a1, b1 = e.sum(), e[0] + e[1]
        a2 = e[2] + e[3]
        efron = -((eta[0] + eta[1] - math.log(a1) - math.log(a1 - 0.5 * b1))
                  + (eta[2] - math.log(a2))) / 3
        breslow = -((eta[0] + eta[1] - 2 * math.log(a1)) + (eta[2] - math.log(a2))) / 3
        self.assertAlmostEqual(cox_partial_nll(*t64(eta, time, event), ties="efron").item(), efron, places=12)
        self.assertAlmostEqual(cox_partial_nll(*t64(eta, time, event), ties="breslow").item(), breslow, places=12)
        self.assertLess(efron, breslow)   # 동점이 있으면 Efron 분모가 작다 -> NLL 이 작다

    def test_loglik_matches_lifelines_efron(self):
        from lifelines import CoxPHFitter
        x, time, event = tied_data()
        df = pd.DataFrame(x, columns=["x0", "x1"]).assign(T=time, E=event)
        cph = CoxPHFitter().fit(df, duration_col="T", event_col="E")
        eta = x @ cph.params_.to_numpy()
        ours_total = cox_partial_nll(*t64(eta, time, event), ties="efron").item() * event.sum()
        self.assertAlmostEqual(ours_total, -cph.log_likelihood_, places=6)

    def test_coefficients_match_lifelines_and_sksurv(self):
        from lifelines import CoxPHFitter
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        from sksurv.util import Surv
        x, time, event = tied_data(seed=3)
        y = Surv.from_arrays(event=event.astype(bool), time=time)
        df = pd.DataFrame(x, columns=["x0", "x1"]).assign(T=time, E=event)
        lif = CoxPHFitter().fit(df, duration_col="T", event_col="E").params_.to_numpy()
        for ties in ("efron", "breslow"):
            ours = fit_beta(x, time, event, ties)
            sk = CoxPHSurvivalAnalysis(alpha=0, ties=ties, n_iter=200, tol=1e-12).fit(x, y).coef_
            np.testing.assert_allclose(ours, sk, atol=1e-5, err_msg=f"sksurv {ties}")
            if ties == "efron":
                np.testing.assert_allclose(ours, lif, atol=1e-4, err_msg="lifelines efron")


class TestProperties(unittest.TestCase):
    def test_no_ties_all_three_agree(self):
        rng = np.random.default_rng(1)
        n = 80
        time = rng.exponential(size=n) + np.arange(n) * 1e-6        # 동점 없음
        event = (rng.uniform(size=n) < 0.7).astype(float)
        eta = rng.normal(size=n)
        args = t64(eta, time, event)
        efron = cox_partial_nll(*args, ties="efron").item()
        self.assertAlmostEqual(efron, cox_partial_nll(*args, ties="breslow").item(), places=10)
        self.assertAlmostEqual(efron, cox_ph_loss(*args).item(), places=10)

    def test_permutation_invariant(self):
        x, time, event = tied_data(seed=2)
        eta = np.random.default_rng(5).normal(size=len(time))
        base = cox_efron_loss(*t64(eta, time, event)).item()
        for seed in range(5):
            p = np.random.default_rng(seed).permutation(len(time))
            self.assertAlmostEqual(cox_efron_loss(*t64(eta[p], time[p], event[p])).item(), base, places=10)

    def test_old_loss_depends_on_order_with_ties(self):
        """기존 cox_ph_loss 를 바꾸지 않고 새 함수를 만든 이유의 기록."""
        x, time, event = tied_data(seed=2)
        eta = np.random.default_rng(5).normal(size=len(time))
        vals = {round(cox_ph_loss(*t64(eta[p], time[p], event[p])).item(), 8)
                for p in (np.random.default_rng(s).permutation(len(time)) for s in range(5))}
        self.assertGreater(len(vals), 1)

    def test_shift_invariant(self):
        x, time, event = tied_data(seed=4)
        eta = np.random.default_rng(6).normal(size=len(time))
        for ties in ("efron", "breslow"):
            a = cox_partial_nll(*t64(eta, time, event), ties=ties).item()
            b = cox_partial_nll(*t64(eta + 7.3, time, event), ties=ties).item()
            self.assertAlmostEqual(a, b, places=9)

    def test_large_scores_stable(self):
        x, time, event = tied_data(seed=7)
        eta = np.random.default_rng(8).normal(size=len(time)) * 50 + 500
        val = cox_efron_loss(*t64(eta, time, event))
        self.assertTrue(torch.isfinite(val))
        self.assertAlmostEqual(val.item(), cox_efron_loss(*t64(eta - 500, time, event)).item(), places=8)

    def test_gradcheck(self):
        x, time, event = tied_data(n=25, seed=9, n_times=5)
        _, tt, et = t64(x, time, event)
        for ties in ("efron", "breslow"):
            eta = torch.randn(len(time), dtype=torch.float64, requires_grad=True)
            self.assertTrue(torch.autograd.gradcheck(
                lambda r: cox_partial_nll(r, tt, et, ties=ties), (eta,)))

    def test_float32_close_to_float64(self):
        x, time, event = tied_data(seed=11)
        eta = np.random.default_rng(12).normal(size=len(time))
        a = cox_efron_loss(*t64(eta, time, event)).item()
        b = cox_efron_loss(*[torch.as_tensor(v, dtype=torch.float32) for v in (eta, time, event)]).item()
        self.assertAlmostEqual(a, b, places=5)


class TestEdgeCases(unittest.TestCase):
    def test_zero_events_returns_graph_zero(self):
        eta = torch.randn(6, dtype=torch.float64, requires_grad=True)
        loss = cox_efron_loss(eta, torch.arange(6.0), torch.zeros(6))
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertTrue(torch.equal(eta.grad, torch.zeros(6, dtype=torch.float64)))

    def test_single_event(self):
        eta = np.array([0.5, -1.0, 0.2])
        time, event = np.array([2.0, 1.0, 3.0]), np.array([0, 1, 0.0])
        expect = -(eta[1] - math.log(np.exp(eta).sum()))
        self.assertAlmostEqual(cox_efron_loss(*t64(eta, time, event)).item(), expect, places=12)

    def test_all_tied_all_events(self):
        # 전원 같은 시각에 사건: A = B, Efron 분모 = A(1 - l/d)
        eta = np.array([0.3, -0.4, 1.1, 0.0])
        a, d = np.exp(eta).sum(), 4
        expect = -(eta.sum() - sum(math.log(a * (1 - l / d)) for l in range(d))) / d
        val = cox_efron_loss(*t64(eta, np.ones(4), np.ones(4))).item()
        self.assertAlmostEqual(val, expect, places=12)

    def test_column_vector_and_validation(self):
        x, time, event = tied_data(n=30, seed=13)
        eta = np.random.default_rng(14).normal(size=len(time))
        a = cox_efron_loss(*t64(eta, time, event)).item()
        b = cox_efron_loss(*t64(eta[:, None], time, event)).item()
        self.assertEqual(a, b)
        with self.assertRaises(ValueError):
            cox_efron_loss(*t64(eta[:-1], time, event))
        with self.assertRaises(ValueError):
            cox_partial_nll(*t64(eta, time, event), ties="exact")


if __name__ == "__main__":
    unittest.main()
