# -*- coding: utf-8 -*-
"""학습. 인코더 학습, 특징 수준 융합 모델(Concat · Constant Gate · GMU)의 full-batch 학습과 조기 종료, 하이퍼파라미터 후보 선택 규칙."""

import copy
import statistics

import torch
import torch.nn as nn

from sclc import foldsafe_clinical as fsc
from sclc.model import MODALITY_CONFIGS, make_model_factory
from sclc.train import ImageOnlyEvaluator, TrimodalEvaluator, cox_efron_loss

from .config import (BASELINES, CANDIDATES, COMMON_RULE, ENCODER_STOP, ENCODERS, LR_RULES, SELECTION_TOLERANCE,
                     SIMPLICITY_ORDER, TRAIN_RULE, USED_KEYS, guard)
from .processing import ImageFeatureHook, StateHook, TabularFeatureHook


# ── 인코더 학습 ──
def run_encoders(kind, target, split_csv, epochs, patience, seed, save_dir, extra_ids, ctx, audit,
                 min_epoch=None):
    """tabular 또는 image 인코더를 학습하면서 fold 마다 특징·위험점수를 뽑아 store 로 돌려준다."""
    store = {}
    common = dict(target=target, epochs=epochs, seed=seed, split_csv=split_csv, save_dir=save_dir,
                  loss_fn=cox_efron_loss, patience=patience, extra_eval_ids=extra_ids, num_workers=4,
                  val_loss_fn=cox_efron_loss, monitor=ENCODER_STOP["monitor"],
                  min_delta=ENCODER_STOP["min_delta"],
                  min_epoch=ENCODERS[kind]["min_epoch"] if min_epoch is None else min_epoch)
    if kind == "tabular":
        ev = TrimodalEvaluator(
            batch_size=ENCODERS["tabular"]["batch_size"], lr=ENCODERS["tabular"]["lr"],
            weight_decay=ENCODERS["tabular"]["weight_decay"],
            model_factory=make_model_factory(MODALITY_CONFIGS["clin_report"]),
            clinical_columns_fn=fsc.baseline_clinical_columns,
            extra_numeric_fn=fsc.make_extra_numeric_fn(ctx["raw"], audit), recode_lung_disease=True,
            fold_end_hook=TabularFeatureHook(store, keep_model=False), **ctx["text_kwargs"], **common).run()
    else:
        ev = ImageOnlyEvaluator(
            batch_size=ENCODERS["image"]["batch_size"], lr=ENCODERS["image"]["lr"],
            weight_decay=ENCODERS["image"]["weight_decay"], ckpt_tag="image_simplecnn",
            fold_end_hook=ImageFeatureHook(store, keep_model=False), **common).run()
    for fold, rec in store.items():
        for split, d in rec["splits"].items():
            for name, tensor in d.items():
                if name != "tabular":
                    guard(bool(torch.isfinite(tensor).all()), f"{kind} fold {fold} {split}/{name} 에 비유한값")
            guard(torch.allclose(d["risk"], d["head_risk"], atol=1e-4),
                  f"{kind} fold {fold} {split}: 추출 특징으로 위험점수가 재현되지 않는다")
    return store, ev


# ── ⑥ 비교 모델 학습 ──
def image_model_factory(spec: dict):
    """영상 단독 비교 모델의 (model_factory, optimizer_factory). SimpleCNN 은 (None, None) 으로 evaluator 기본 모델을 쓴다."""
    import torch.optim as optim

    if spec["backbone"] == "simplecnn":
        return None, None
    from sclc.model import RadImageNetDeepSurv, ResNet18DeepSurv
    if spec["backbone"] == "resnet18":
        make = lambda: ResNet18DeepSurv(gray_scale=True, pretrained=True, dropout=spec["dropout"])
        backbone = lambda m: m.base
    elif spec["backbone"] == "radimagenet":
        make = lambda: RadImageNetDeepSurv(pretrained=True, dropout=spec["dropout"])
        backbone = lambda m: m.backbone
    else:
        raise ValueError(f"unknown backbone {spec['backbone']!r}")

    def optimizer_factory(model):
        return optim.Adam([{"params": backbone(model).parameters(), "lr": spec["backbone_lr"]},
                           {"params": model.head.parameters(), "lr": spec["head_lr"]}],
                          weight_decay=spec["weight_decay"])

    return make, optimizer_factory


def run_baseline(name, target, split_csv, epochs, patience, seed, save_dir, extra_ids, ctx, audit, min_epoch=None):
    """비교 모델 하나를 분할 파일의 fold 마다 학습한다. evaluator 와 학습 인자는 run_encoders 와 같고 모델만 다르다.
    돌려주는 store 에는 fold 별 분할 ID 와 가중치 해시만 있다 (특징 추출 없음)."""
    spec = BASELINES[name]
    rule = ENCODERS[spec["trainer"]]
    store = {}
    common = dict(target=target, epochs=epochs, seed=seed, split_csv=split_csv, save_dir=save_dir,
                  loss_fn=cox_efron_loss, patience=patience, extra_eval_ids=extra_ids, num_workers=4,
                  val_loss_fn=cox_efron_loss, monitor=ENCODER_STOP["monitor"],
                  min_delta=ENCODER_STOP["min_delta"],
                  min_epoch=rule["min_epoch"] if min_epoch is None else min_epoch,
                  batch_size=rule["batch_size"], fold_end_hook=StateHook(store, keep_model=False))
    if spec["trainer"] == "tabular":
        ev = TrimodalEvaluator(
            lr=rule["lr"], weight_decay=rule["weight_decay"],
            model_factory=make_model_factory(MODALITY_CONFIGS[spec["model_config"]]),
            clinical_columns_fn=fsc.baseline_clinical_columns,
            extra_numeric_fn=fsc.make_extra_numeric_fn(ctx["raw"], audit), recode_lung_disease=True,
            **ctx["report_kwargs"][spec["report"]], **common).run()
    else:
        model_factory, optimizer_factory = image_model_factory(spec)
        extra = ({} if spec["backbone"] == "simplecnn" else
                 dict(resize=spec["resize"], augment=spec["augment"], model_factory=model_factory,
                      optimizer_factory=optimizer_factory))
        ev = ImageOnlyEvaluator(lr=rule["lr"], weight_decay=rule["weight_decay"],
                                ckpt_tag=f"image_{spec['backbone']}", **extra, **common).run()
    guard(sorted(store) == sorted(r["fold"] for r in ev.fold_records), f"{name}: hook 기록과 fold 기록이 다르다")
    return store, ev


# ── 융합 모델 학습 ──
class EarlyStopper:
    """검증 손실이 min_delta 넘게 줄면 최적 epoch 을 갱신하고, patience 동안 개선이 없으면 멈춘다.
    min_epoch 전에는 판단하지 않는다."""

    def __init__(self, patience: int, min_epoch: int = 1, min_delta: float = 0.0):
        self.patience, self.min_epoch, self.min_delta = patience, min_epoch, min_delta
        self.best, self.best_epoch, self.wait = float("inf"), None, 0

    def update(self, epoch: int, value: float) -> tuple[bool, bool]:
        if epoch < self.min_epoch:
            return False, False
        if value < self.best - self.min_delta:
            self.best, self.best_epoch, self.wait = value, epoch, 0
            return True, False
        self.wait += 1
        return False, self.patience is not None and self.wait >= self.patience


def train_fusion(model: nn.Module, train: tuple, stop: tuple | None, cfg: dict, loss_fn,
                 max_epochs: int = TRAIN_RULE["max_epochs"], patience: int | None = TRAIN_RULE["patience"],
                 min_epoch: int = TRAIN_RULE["min_epoch"], min_delta: float = TRAIN_RULE["min_delta"]) -> dict:
    """full-batch AdamW (1 epoch = 1 step). stop 분할이 있으면 조기 종료 후 최적 가중치로 되돌리고,
    없으면 max_epochs 만큼 고정 학습한다."""
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    feats, dur, evt = train
    stopper = EarlyStopper(patience, min_epoch, min_delta) if stop is not None else None
    history, best_state = [], None
    for epoch in range(1, max_epochs + 1):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(feats), dur, evt)
        loss.backward()
        opt.step()
        row = {"epoch": epoch, "train_loss": float(loss.item())}
        if not torch.isfinite(loss):
            raise FloatingPointError(f"epoch {epoch}: 학습 손실이 유한하지 않다")
        if stopper is not None:
            model.eval()
            with torch.no_grad():
                row["stop_loss"] = float(loss_fn(model(stop[0]), stop[1], stop[2]).item())
                row["train_loss_eval"] = float(loss_fn(model(feats), dur, evt).item())
            if not (torch.isfinite(torch.tensor(row["stop_loss"])) and torch.isfinite(torch.tensor(row["train_loss_eval"]))):
                raise FloatingPointError(f"epoch {epoch}: 정지용/학습 손실이 유한하지 않다")
            improved, halt = stopper.update(epoch, row["stop_loss"])
            if improved:
                best_state = copy.deepcopy(model.state_dict())
                row["saved"] = True
            history.append(row)
            if halt:
                break
        else:
            history.append(row)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    best = stopper.best_epoch if stopper is not None else len(history)
    return {"history": history, "best_epoch": best, "epochs_run": len(history)}


# ── 후보 선택 규칙 ──
def rule_for(cfg: dict) -> dict:
    """후보의 학습률에 맞는 정지 규칙."""
    if cfg["lr"] not in LR_RULES:
        raise ValueError(f"no training rule for lr {cfg['lr']}")
    return {**LR_RULES[cfg["lr"]], **COMMON_RULE}


def effective_candidates(name: str) -> dict:
    """모델이 실제로 쓰는 값이 같은 후보는 하나만 남긴다 (ConcatFusion · FixedGateFusion 은 bottleneck 을 안 써서 G2 = G1)."""
    seen, out = set(), {}
    for cand, cfg in CANDIDATES.items():
        key = tuple(cfg[k] for k in USED_KEYS[name])
        if key not in seen:
            seen.add(key)
            out[cand] = cfg
    return out


def select_candidate(mean_cindex: dict[str, float], tolerance: float = SELECTION_TOLERANCE,
                     order=SIMPLICITY_ORDER) -> str:
    """최고 평균 C-index 와 0.005 이내인 후보 중 가장 단순한 것."""
    top = max(mean_cindex.values())
    for name in order:
        if name in mean_cindex and mean_cindex[name] >= top - tolerance:
            return name
    raise ValueError("no candidate")


def refit_epochs(best_epochs) -> int:
    """inner 최적 epoch 의 중앙값 (.5 는 올림)."""
    vals = [int(e) for e in best_epochs if e is not None]
    if not vals:
        raise ValueError("no best epochs")
    return int(statistics.median(vals) + 0.5)
