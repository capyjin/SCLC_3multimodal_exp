# -*- coding: utf-8 -*-
"""융합 모델. ConcatFusion · FixedGateFusion · JointGMU(GMU) 는 인코더 특징을,
nested_late_fusion(Late Fusion) · GatedLateFusion 은 인코더 위험점수를 결합한다."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from lifelines import CoxPHFitter

from sclc.evaluation import cindex
from sclc.train import cox_efron_loss

from .config import ARMS, FEATURE_DIMS, GATED_LATE_LAMBDAS, MODALITIES
from .processing import pool_inner_oof, standardize, standardizer

F64 = torch.float64


# ── Concat fusion / FixedGate fusion / JointGMU fusion 특징 수준 결합 ──
class FeatureFusion(nn.Module):
    """특징 수준 융합 모델의 공통 부모. 모달리티별 특징을 Linear 로 d 차원에 맞추고 L2 정규화한다."""

    def __init__(self, d: int, dropout: float, feature_dims: dict | None = None):
        super().__init__()
        dims = feature_dims or FEATURE_DIMS
        self.d = int(d)
        self.proj = nn.ModuleDict({m: nn.Linear(dims[m], d) for m in MODALITIES})
        self.dropout = nn.Dropout(dropout)

    @classmethod
    def from_config(cls, cfg: dict) -> "FeatureFusion":
        return cls(d=cfg["d"], bottleneck=cfg["bottleneck"], dropout=cfg["dropout"])

    def projected(self, feats: dict) -> list[torch.Tensor]:
        return [F.normalize(self.proj[m](feats[m]), dim=1) for m in MODALITIES]

    def gates(self, feats: dict) -> torch.Tensor | None:
        return None

    @torch.no_grad()
    def feature_stats(self, feats: dict) -> dict:
        """projection 전후 특징 크기와 gate 분포 요약 (기록용)."""
        self.eval()
        out = {"pre_projection_norm": {m: _summ(feats[m].norm(dim=1)) for m in MODALITIES},
               "pre_projection_elem_sd": {m: round(float(feats[m].std(dim=0).mean()), 5) for m in MODALITIES},
               "post_projection_norm": {m: _summ(self.proj[m](feats[m]).norm(dim=1)) for m in MODALITIES}}
        g = self.gates(feats)
        if g is not None:
            out["gates"] = {m: _summ(g[:, i]) for i, m in enumerate(MODALITIES)}
        return out


class ConcatFusion(FeatureFusion):
    """Concat: 세 특징을 이어 붙여 Linear 로 위험점수를 낸다."""

    def __init__(self, d: int = 16, dropout: float = 0.3, **_):
        super().__init__(d, dropout)
        self.head = nn.Linear(3 * d, 1, bias=False)

    def forward(self, feats: dict) -> torch.Tensor:
        return self.head(self.dropout(torch.cat(self.projected(feats), dim=1)))


class GatedFusion(FeatureFusion):
    """FixedGateFusion · JointGMU 공통: gate 로 가중합한 특징을 Linear 로 위험점수로 바꾼다. 하위 클래스는 gate 만 정한다."""

    def __init__(self, d: int = 16, dropout: float = 0.3, **gate_kwargs):
        super().__init__(d, dropout)
        # 생성 순서(projection → gate → head)가 초기화 난수 순서를 정하므로 바꾸지 않는다
        self._build_gate(d, **gate_kwargs)
        self.head = nn.Linear(d, 1, bias=False)

    def _build_gate(self, d: int, **kwargs) -> None:
        raise NotImplementedError

    def _gate_values(self, hs: list[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def gates(self, feats: dict) -> torch.Tensor:
        return self._gate_values(self.projected(feats))

    def forward(self, feats: dict) -> torch.Tensor:
        hs = self.projected(feats)
        g = self._gate_values(hs)
        fused = (g.unsqueeze(-1) * torch.stack(hs, dim=1)).sum(dim=1)
        return self.head(self.dropout(fused))


class FixedGateFusion(GatedFusion):
    """Constant Gate: 모든 환자에게 같은 gate 3개 (학습 파라미터)."""

    def _build_gate(self, d: int, **_) -> None:
        self.gate_logit = nn.Parameter(torch.zeros(len(MODALITIES)))

    def _gate_values(self, hs: list[torch.Tensor]) -> torch.Tensor:
        return torch.sigmoid(self.gate_logit).expand(hs[0].size(0), -1)


class JointGMU(GatedFusion):
    """GMU: 세 특징을 함께 보고 환자마다 gate 3개를 계산한다."""

    def _build_gate(self, d: int, bottleneck: int = 8, **_) -> None:
        self.gate_net = nn.Sequential(nn.Linear(3 * d, bottleneck), nn.ReLU(),
                                      nn.Linear(bottleneck, len(MODALITIES)))

    def _gate_values(self, hs: list[torch.Tensor]) -> torch.Tensor:
        return torch.sigmoid(self.gate_net(torch.cat(hs, dim=1)))


# 키는 결과 파일에 기록되는 모델 이름이다
FUSION_MODELS = {"B_concat": ConcatFusion, "C_fixed_gate": FixedGateFusion, "D_gmu": JointGMU}


def build(name: str, cfg: dict) -> FeatureFusion:
    return FUSION_MODELS[name].from_config(cfg)


def _summ(x: torch.Tensor) -> dict:
    x = x.detach().float().cpu()
    q = torch.quantile(x, torch.tensor([0.05, 0.5, 0.95]))
    return {"mean": round(float(x.mean()), 5), "sd": round(float(x.std()), 5),
            "q05": round(float(q[0]), 5), "q50": round(float(q[1]), 5), "q95": round(float(q[2]), 5)}


# ── Late fusion 위험점수 수준 결합 ──
def fit_stacker(x: pd.DataFrame, t_labels: pd.DataFrame, target: str) -> dict:
    """CoxPH(Efron) 계수. 학습 환자 레이블만 받는다."""
    frame = x.join(t_labels[[f"{target}_days", f"{target}_event"]], how="inner")
    if len(frame) != len(x):
        raise ValueError("레이블이 없는 T_k 환자가 있다")
    cph = CoxPHFitter().fit(frame, duration_col=f"{target}_days", event_col=f"{target}_event")
    return {c: float(cph.params_[c]) for c in x.columns}


def linear_predictor(coefs: dict, x: dict) -> np.ndarray:
    return sum(coefs[c] * np.asarray(x[c], dtype=float) for c in coefs)


def calibration_slopes(oof: pd.DataFrame, t_labels: pd.DataFrame, target: str) -> dict:
    out = {}
    for j, part in oof.groupby("inner_fold"):
        out[int(j)] = {arm: fit_stacker(part[[f"z_{arm}"]], t_labels.loc[part.index], target)[f"z_{arm}"]
                       for arm in ARMS}
    return out


def effective_image_weight(coefs: dict, sd: dict) -> float:
    """영상 arm 의 실효 가중치 |b_img|·SD_img / (|b_tab|·SD_tab + |b_img|·SD_img)."""
    w_tab = abs(coefs["z_tabular"]) * sd["tabular"]
    w_img = abs(coefs["z_image"]) * sd["image"]
    return float(w_img / (w_tab + w_img))


def nested_late_fusion(inner: list[dict], outer: dict, t_ids, t_labels: pd.DataFrame,
                       target: str) -> dict:
    """Late Fusion: inner OOF 위험점수로 CoxPH 계수를 정하고 outer 시험 환자 점수에 적용한다.
    시험 환자 레이블은 받지 않는다. 예측은 primary(z-score) · raw · ensemble 세 가지."""
    oof = pool_inner_oof(inner, t_ids)
    t_lab = t_labels.loc[oof.index]
    coef_z = fit_stacker(oof[[f"z_{a}" for a in ARMS]], t_lab, target)
    coef_raw = fit_stacker(oof[[f"raw_{a}" for a in ARMS]], t_lab, target)

    z_out = {f"z_{a}": standardize(outer["arms"][a]["test_risk"],
                                   standardizer(outer["arms"][a]["train_risk"])) for a in ARMS}
    raw_out = {f"raw_{a}": np.asarray(outer["arms"][a]["test_risk"], dtype=float) for a in ARMS}
    pred = {"primary": linear_predictor(coef_z, z_out), "raw": linear_predictor(coef_raw, raw_out)}

    if all("extra_risk" in rec["arms"][a] for rec in inner for a in ARMS):
        z_ens = {f"z_{a}": np.mean([standardize(rec["arms"][a]["extra_risk"],
                                                standardizer(rec["arms"][a]["fit_risk"]))
                                    for rec in inner], axis=0) for a in ARMS}
        pred["ensemble"] = linear_predictor(coef_z, z_ens)

    sd_oof = {a: float(oof[f"z_{a}"].std(ddof=1)) for a in ARMS}
    sd_out = {a: float(np.std(z_out[f"z_{a}"], ddof=1)) for a in ARMS}
    diagnostics = {
        "calibration_slopes": calibration_slopes(oof, t_lab, target),
        "sd_inner_oof_z": sd_oof, "sd_outer_test_z": sd_out,
        "sd_ratio_outer_over_inner": {a: sd_out[a] / sd_oof[a] for a in ARMS},
        "effective_image_weight": {"inner_oof": effective_image_weight(coef_z, sd_oof),
                                   "outer_test": effective_image_weight(coef_z, sd_out)},
    }
    return {"coef_z": coef_z, "coef_raw": coef_raw, "pred": pred,
            "test_ids": list(outer["test_ids"]), "oof": oof, "diagnostics": diagnostics}


# ── Gated Late fusion 위험점수 수준 결합 ──
class GatedLateFusion:
    """Gated Late Fusion: 두 위험점수 z 를 환자별 gate 로 섞는다. r = Σ b_m · 2σ(c_m·z) · z_m
    c = 0 이면 Late Fusion(nested_late_fusion)과 같은 식이다. b·c 는 Efron 손실 + λ‖c‖² 로 학습한다."""

    LAMBDAS = GATED_LATE_LAMBDAS

    def __init__(self, lam: float, gated: bool = True):
        self.lam, self.gated = lam, gated
        self.beta, self.C = None, None

    @staticmethod
    def forward(beta, C, Z):
        g = 2.0 * torch.sigmoid(Z @ C.T)
        return (g * beta * Z).sum(dim=1), g

    def predict(self, Z):
        """-> (위험점수, gate)"""
        return self.forward(self.beta, self.C, Z)

    def fit(self, Z, t, e) -> "GatedLateFusion":
        gated, lam = self.gated, self.lam
        beta = torch.zeros(2, dtype=F64, requires_grad=True)
        C = torch.zeros(2, 2, dtype=F64, requires_grad=gated)
        params = [p for p in (beta, C) if p.requires_grad]
        opt = torch.optim.LBFGS(params, lr=1.0, max_iter=500, tolerance_grad=1e-10, tolerance_change=1e-12,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            r, _ = self.forward(beta, C, Z)
            loss = cox_efron_loss(r, t, e) + (lam * (C ** 2).sum() if gated else 0.0)
            loss.backward()
            return loss

        opt.step(closure)
        self.beta, self.C = beta.detach(), C.detach()
        return self

    @classmethod
    def select_lambda(cls, Z, t, e, folds):
        """inner fold 검증 C-index 평균이 가장 높은 λ (동점이면 큰 λ)."""
        scores = {}
        for lam in cls.LAMBDAS:
            cis = []
            for j in sorted(np.unique(folds)):
                tr, va = folds != j, folds == j
                r, _ = cls(lam).fit(Z[tr], t[tr], e[tr]).predict(Z[va])
                cis.append(cindex(t[va].numpy(), r.numpy(), e[va].numpy()))
            scores[lam] = float(np.mean(cis))
        best = max(scores.values())
        return max(lam for lam, s in scores.items() if s >= best - 1e-12), scores

    @staticmethod
    def gate_summary(g: torch.Tensor) -> dict:
        g = g.numpy()
        return {arm: {"mean": float(g[:, i].mean()), "sd": float(g[:, i].std(ddof=1)),
                      "min": float(g[:, i].min()), "max": float(g[:, i].max())} for i, arm in enumerate(ARMS)}
