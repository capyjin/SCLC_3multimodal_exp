# -*- coding: utf-8 -*-
"""``sclc.train.fit`` 의 정지 지표(monitor) 테스트 — 장난감 데이터.

Run:  python -m unittest discover -s tests -v
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch
from torch.utils.data import DataLoader, TensorDataset

from sclc.train import best_epoch, cox_efron_loss, evaluate_loss, fit


def toy(n, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 5, generator=g)
    t = torch.exp(-(x[:, 0] - 0.5 * x[:, 1])) * torch.rand(n, generator=g) * 10 + 0.1
    e = (torch.rand(n, generator=g) < 0.8).float()
    return TensorDataset(x, t, e)


class Lin(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.f = torch.nn.Linear(5, 1)

    def forward(self, x):
        return self.f(x)


def run(monitor, patience=None, min_delta=0.0, epochs=40, lr=0.05):
    torch.manual_seed(0)
    model = Lin()
    tr = DataLoader(toy(120, 1), batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(0))
    va = DataLoader(toy(40, 2), batch_size=32)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.pt")
        hist = fit(model, tr, va, torch.optim.SGD(model.parameters(), lr=lr), torch.device("cpu"), epochs,
                   path, loss_fn=cox_efron_loss, patience=patience, val_loss_fn=cox_efron_loss,
                   monitor=monitor, min_delta=min_delta)
        model.load_state_dict(torch.load(path))
    return hist, model, va


class TestMonitor(unittest.TestCase):
    def test_loss_monitor_saves_on_val_loss_improvement(self):
        hist, model, va = run("loss", min_delta=1e-4)
        best = float("inf")
        for row in hist:
            expect = row["val_loss"] < best - 1e-4
            self.assertEqual(row["saved"], expect, row)
            best = row["val_loss"] if expect else best
        be = best_epoch(hist)
        restored = evaluate_loss(model, va.dataset, torch.device("cpu"), cox_efron_loss)
        self.assertAlmostEqual(restored, hist[be - 1]["val_loss"], places=5)   # 최적 가중치로 복원

    def test_patience_on_loss(self):
        hist, _, _ = run("loss", patience=3, epochs=200, lr=0.5)
        be = best_epoch(hist)
        self.assertEqual(len(hist), be + 3)                     # 마지막 개선 뒤 3 epoch 에서 멈춤
        self.assertTrue(all(not r["saved"] for r in hist[be:]))

    def test_min_epoch(self):
        """min_epoch 전에는 저장도 patience 집계도 없다; 첫 추적 epoch 은 무조건 저장."""
        torch.manual_seed(0)
        model = Lin()
        tr = DataLoader(toy(120, 1), batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(0))
        va = DataLoader(toy(40, 2), batch_size=32)
        with tempfile.TemporaryDirectory() as d:
            hist = fit(model, tr, va, torch.optim.SGD(model.parameters(), lr=0.5), torch.device("cpu"), 200,
                       os.path.join(d, "m.pt"), loss_fn=cox_efron_loss, patience=3, val_loss_fn=cox_efron_loss,
                       monitor="loss", min_delta=1e-4, min_epoch=8)
            with self.assertRaises(ValueError):
                fit(Lin(), tr, va, None, torch.device("cpu"), 5, os.path.join(d, "x.pt"),
                    val_loss_fn=cox_efron_loss, monitor="loss", min_epoch=6)
        self.assertTrue(all(not r["saved"] for r in hist[:7]))
        self.assertTrue(hist[7]["saved"])
        be = best_epoch(hist)
        self.assertGreaterEqual(be, 8)
        self.assertEqual(len(hist), be + 3)                     # 마지막 개선 뒤 patience 3

    def test_cindex_monitor_unchanged(self):
        hist, _, _ = run("cindex")
        best = -1.0
        for row in hist:
            self.assertEqual(row["saved"], row["val_cindex"] > best)
            best = max(best, row["val_cindex"])

    def test_recording_does_not_change_trajectory(self):
        """val_loss_fn 기록(난수 상태 복원) 유무와 관계없이 학습 경로가 같다 (dropout 있는 모델)."""
        def go(record):
            torch.manual_seed(0)
            model = torch.nn.Sequential(torch.nn.Linear(5, 8), torch.nn.Dropout(0.5), torch.nn.Linear(8, 1))
            tr = DataLoader(toy(120, 1), batch_size=16, shuffle=True, generator=torch.Generator().manual_seed(0))
            va = DataLoader(toy(40, 2), batch_size=32)
            with tempfile.TemporaryDirectory() as d:
                return fit(model, tr, va, torch.optim.SGD(model.parameters(), lr=0.05), torch.device("cpu"),
                           8, os.path.join(d, "m.pt"), loss_fn=cox_efron_loss,
                           val_loss_fn=cox_efron_loss if record else None)
        a, b = go(False), go(True)
        self.assertEqual([r["train_loss"] for r in a], [r["train_loss"] for r in b])

    def test_bad_monitor(self):
        with self.assertRaises(ValueError):
            fit(Lin(), None, None, None, torch.device("cpu"), 1, "x.pt", monitor="auc")
        with self.assertRaises(ValueError):
            fit(Lin(), None, None, None, torch.device("cpu"), 1, "x.pt", monitor="loss")


if __name__ == "__main__":
    unittest.main()
