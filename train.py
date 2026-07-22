# -*- coding: utf-8 -*-
"""Cox loss, training loop, and K-fold evaluators for the image-involving
models (early/trimodal fusion, and the image-only late-fusion arm).

``cox_ph_loss``/``train_one_epoch``/``fit``/seeding helpers are copied from
``clinical+image/SCLC_simple_CNN-main/train.py`` (image training has no
pycox equivalent that supports on-the-fly PNG loading, so both source
projects use this custom loop for anything involving images -- unchanged
here). The clinical-only/report-only unimodal arms used by late fusion
instead reuse pycox's ``CoxPH`` + torchtuples (see ``late_fusion.py``),
matching ``clinical+report``'s ``TabularSplitEvaluator`` pattern.
"""
import os
import random
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from lifelines.utils import concordance_index
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import cohort
import dataset as ds
import features
from model import ImageOnlyDeepSurv, TrimodalConcatDeepSurv


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cox_ph_loss(risk_scores: torch.Tensor, durations: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    """Cox negative partial log-likelihood. Higher risk_scores = higher risk
    = shorter expected survival. See clinical+image/train.py for the full
    derivation comment -- logic is unchanged here."""
    if risk_scores.ndim == 2 and risk_scores.size(1) == 1:
        risk_scores = risk_scores.squeeze(1)

    order = torch.argsort(durations, descending=True)
    risk_scores = risk_scores[order]
    events = events[order]

    log_cumsum_exp = torch.logcumsumexp(risk_scores, dim=0)
    diff = risk_scores - log_cumsum_exp
    event_mask = events == 1

    if event_mask.sum() == 0:
        return torch.tensor(0.0, device=risk_scores.device, requires_grad=True)

    return -diff[event_mask].mean()


@torch.no_grad()
def evaluate_c_index(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    durations, events, risks = [], [], []
    for batch in loader:
        if len(batch) == 4:
            images, tabular, dur, evt = batch
            risk = model(images.to(device), tabular.to(device)).squeeze(1).cpu().numpy()
        else:
            images, dur, evt = batch
            risk = model(images.to(device)).squeeze(1).cpu().numpy()
        risks.extend(risk.tolist())
        durations.extend(dur.numpy().tolist())
        events.extend(evt.numpy().tolist())
    return concordance_index(
        event_times=np.array(durations, dtype=np.float32),
        predicted_scores=-np.array(risks, dtype=np.float32),
        event_observed=np.array(events, dtype=np.float32),
    )


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss, total_count = 0.0, 0
    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        if len(batch) == 4:
            images, tabular, durations, events = batch
            tabular = tabular.to(device)
        else:
            images, durations, events = batch
            tabular = None

        images = images.to(device)
        durations = durations.to(device)
        events = events.to(device)

        optimizer.zero_grad()
        risk_scores = model(images, tabular).squeeze(1) if tabular is not None else model(images).squeeze(1)
        loss = cox_ph_loss(risk_scores, durations, events)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size
        pbar.set_postfix({"batch_loss": f"{loss.item():.4f}", "avg_loss": f"{total_loss / total_count:.4f}"})

    return total_loss / total_count


def fit(model, train_loader, val_loader, optimizer, device, epochs, save_path):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    best_val_cindex = -np.inf
    history: list[dict] = []

    for epoch in tqdm(range(1, epochs + 1), desc="Epochs"):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        train_cindex = evaluate_c_index(model, train_loader, device)
        val_cindex = evaluate_c_index(model, val_loader, device)
        print(f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} train_cindex={train_cindex:.4f} val_cindex={val_cindex:.4f}")
        history.append({"epoch": int(epoch), "train_loss": float(train_loss), "train_cindex": float(train_cindex), "val_cindex": float(val_cindex)})

        if val_cindex > best_val_cindex:
            best_val_cindex = val_cindex
            torch.save(model.state_dict(), save_path)
            print(f"  -> best model saved (val_cindex={best_val_cindex:.4f})")

    print(f"\nBest validation c-index: {best_val_cindex:.4f}")
    return history


def _fold_plan(cohort_df: pd.DataFrame, max_folds: int | None):
    folds = sorted(int(f) for f in cohort_df["fold"].unique())
    if max_folds is not None:
        folds = folds[:max_folds]
    plan = []
    for fold in folds:
        fold_df = cohort_df[cohort_df["fold"] == fold]
        ids = {name: fold_df.loc[fold_df["split"] == name, "research_id"].astype(int).tolist() for name in ("train", "val", "test")}
        plan.append((fold, ids))
    return plan


class TrimodalEvaluator:
    """K-fold trainer/evaluator for the early (concat) fusion model:
    image + clinical + report -> one Cox head, trained end-to-end.
    """

    def __init__(
        self, target: str, merged_csv: str = cohort.DEFAULT_MERGED_CSV,
        image_dir: str = cohort.DEFAULT_IMAGE_DIR, split_csv: str = cohort.DEFAULT_SPLIT_CSV,
        resize: int = 512, gray_scale: bool = True,
        tfidf_max_features: int = 400, tfidf_ngram_range=(2, 4),
        image_proj_dim: int = 128, image_dropout: float = 0.2,
        clinical_hidden=(128, 128, 128, 128), clinical_dropout: float = 0.5,
        report_hidden=(32, 16), report_dropout: float = 0.3, fusion_dropout: float = 0.3,
        lr: float = 1e-4, weight_decay: float = 1e-4, epochs: int = 30, batch_size: int = 16,
        save_dir: str = "checkpoints", max_folds: int | None = None,
        device: torch.device | None = None, num_workers: int = 4, seed: int = 42,
        model_factory=None,
    ):
        self.target = target
        self.model_factory = model_factory
        self.merged_csv = merged_csv
        self.image_dir = image_dir
        self.split_csv = split_csv
        self.resize = resize
        self.gray_scale = gray_scale
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_ngram_range = tuple(tfidf_ngram_range)
        self.image_proj_dim = image_proj_dim
        self.image_dropout = image_dropout
        self.clinical_hidden = tuple(clinical_hidden)
        self.clinical_dropout = clinical_dropout
        self.report_hidden = tuple(report_hidden)
        self.report_dropout = report_dropout
        self.fusion_dropout = fusion_dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.max_folds = max_folds
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_workers = num_workers
        self.seed = seed
        os.makedirs(self.save_dir, exist_ok=True)

        self.c_indices: list[float] = []
        self.fold_records: list[dict] = []
        self.oof_predictions: list[dict] = []
        self.training_history: list[dict] = []

    def run(self) -> "TrimodalEvaluator":
        cohort_df = cohort.load_trimodal_cohort(self.merged_csv, self.split_csv)
        clinical_frame = cohort_df.drop_duplicates("research_id").set_index("research_id")
        standardize_cols, categorical_cols = features.resolve_clinical_columns(clinical_frame)
        corpus, _ = features.load_text_corpus(self.merged_csv)

        self.c_indices, self.fold_records, self.oof_predictions, self.training_history = [], [], [], []

        plan = _fold_plan(cohort_df, max_folds=self.max_folds)
        for plan_index, (fold, ids) in enumerate(plan):
            print(f"\n=== [early_fusion/{self.target}] Fold {fold} ===")
            _seed_everything(self.seed + fold)

            train_samples = ds.preprocess_data(self.image_dir, clinical_frame.loc[ids["train"]].reset_index(), self.target, self.gray_scale)
            val_samples = ds.preprocess_data(self.image_dir, clinical_frame.loc[ids["val"]].reset_index(), self.target, self.gray_scale)
            test_samples = ds.preprocess_data(self.image_dir, clinical_frame.loc[ids["test"]].reset_index(), self.target, self.gray_scale)
            print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
            if not test_samples:
                print(f"  [skip] fold {fold}: no test samples found")
                continue

            tabular, clinical_dim, report_dim = features.build_fold_multimodal_tabular(
                clinical_frame.loc[ids["train"]], clinical_frame.loc[ids["val"]], clinical_frame.loc[ids["test"]],
                corpus, standardize_cols, categorical_cols,
                tfidf_max_features=self.tfidf_max_features, tfidf_ngram_range=self.tfidf_ngram_range,
            )

            train_ds, val_ds = ds.create_dataset(
                train_samples, val_samples, self.resize, False, self.gray_scale, False,
                train_tabular=tabular["train"], test_tabular=tabular["val"],
            )
            _, test_ds = ds.create_dataset(
                train_samples, test_samples, self.resize, False, self.gray_scale, False,
                train_tabular=tabular["train"], test_tabular=tabular["test"],
            )

            loader_kwargs = {"num_workers": self.num_workers, "pin_memory": True, "worker_init_fn": _seed_worker}
            generator = torch.Generator()
            generator.manual_seed(self.seed + fold)
            train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, generator=generator, **loader_kwargs)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, **loader_kwargs)
            test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False, **loader_kwargs)

            if self.model_factory is not None:
                model = self.model_factory(clinical_dim, report_dim).to(self.device)
            else:
                model = TrimodalConcatDeepSurv(
                    clinical_dim=clinical_dim, report_dim=report_dim, gray_scale=self.gray_scale,
                    image_proj_dim=self.image_proj_dim, image_dropout=self.image_dropout,
                    clinical_hidden=self.clinical_hidden, clinical_dropout=self.clinical_dropout,
                    report_hidden=self.report_hidden, report_dropout=self.report_dropout,
                    fusion_dropout=self.fusion_dropout,
                ).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

            if plan_index == 0:
                sample_batch = next(iter(train_loader))
                s_img, s_tab, _, _ = sample_batch
                print(f"[Fusion] image batch shape: {s_img.shape}, tabular batch shape: {s_tab.shape}")
                model.eval()
                with torch.no_grad():
                    s_out = model(s_img.to(self.device), s_tab.to(self.device))
                print(f"[Fusion] model output shape: {s_out.shape}")
                model.train()

            save_path = os.path.join(self.save_dir, f"fold{fold}_early_fusion_{self.target}.pt")
            fold_history = fit(model, train_loader, val_loader, optimizer, self.device, self.epochs, save_path)
            self.training_history.extend([{"target": self.target, "fold": fold, **row} for row in fold_history])

            model.load_state_dict(torch.load(save_path, map_location=self.device))
            model.eval()

            risks, durations, events = [], [], []
            with torch.no_grad():
                for images, tabular_batch, dur, evt in test_loader:
                    risk = model(images.to(self.device), tabular_batch.to(self.device)).squeeze(1).cpu().numpy()
                    risks.extend(risk.tolist())
                    durations.extend(dur.numpy().tolist())
                    events.extend(evt.numpy().tolist())

            risks = np.array(risks, dtype=np.float32)
            durations = np.array(durations, dtype=np.float32)
            events = np.array(events, dtype=np.float32)
            ci = concordance_index(durations, -risks, events)
            self.c_indices.append(ci)
            self.fold_records.append({
                "target": self.target, "modality": "early_fusion_image_clinical_report", "fold": fold,
                "c_index": round(float(ci), 4),
                "n": len(risks), "event_count": int((events == 1).sum()), "censored_count": int((events == 0).sum()),
            })
            self.oof_predictions.extend([
                {"research_id": rid, "target": self.target, "modality": "early_fusion_image_clinical_report",
                 "fold": fold, "duration": float(d), "event": float(e), "risk_score": float(r)}
                for rid, d, e, r in zip(ids["test"], durations, events, risks)
            ])
            print(f"Fold {fold}: C-index = {ci:.4f} (n={len(risks)})")

        if not self.c_indices:
            raise RuntimeError("No folds were completed.")
        best_fold = self.fold_records[int(np.argmax(self.c_indices))]["fold"]
        best_src = os.path.join(self.save_dir, f"fold{best_fold}_early_fusion_{self.target}.pt")
        best_dst = os.path.join(self.save_dir, f"best_early_fusion_{self.target}.pt")
        shutil.copy2(best_src, best_dst)
        print(f"\nCompleted {len(self.c_indices)} fold(s) C-index: {np.mean(self.c_indices):.4f} +/- {np.std(self.c_indices):.4f}")
        return self


class ImageOnlyEvaluator:
    """K-fold trainer/evaluator for the pure-image DeepSurv arm used as one
    of the three unimodal inputs to late fusion."""

    def __init__(
        self, target: str, merged_csv: str = cohort.DEFAULT_MERGED_CSV,
        image_dir: str = cohort.DEFAULT_IMAGE_DIR, split_csv: str = cohort.DEFAULT_SPLIT_CSV,
        resize: int = 512, gray_scale: bool = True,
        lr: float = 1e-4, weight_decay: float = 1e-4, epochs: int = 30, batch_size: int = 16,
        save_dir: str = "checkpoints", max_folds: int | None = None,
        device: torch.device | None = None, num_workers: int = 4, seed: int = 42,
        model_factory=None, optimizer_factory=None, augment: bool = False,
        ckpt_tag: str = "image_only",
    ):
        self.target = target
        self.merged_csv = merged_csv
        self.image_dir = image_dir
        self.split_csv = split_csv
        self.resize = resize
        self.gray_scale = gray_scale
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.save_dir = save_dir
        # Backward-compatible hooks (default None reproduces the SimpleCNN arm
        # exactly). ``model_factory() -> nn.Module`` swaps the image backbone;
        # ``optimizer_factory(model) -> Optimizer`` allows differential LR
        # (e.g. low-LR pretrained backbone + higher-LR head); ``augment``
        # toggles the train-fold flip augmentation in create_dataset.
        self.model_factory = model_factory
        self.optimizer_factory = optimizer_factory
        self.augment = augment
        self.ckpt_tag = ckpt_tag
        self.max_folds = max_folds
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_workers = num_workers
        self.seed = seed
        os.makedirs(self.save_dir, exist_ok=True)

        self.c_indices: list[float] = []
        self.fold_records: list[dict] = []
        self.oof_predictions: list[dict] = []
        self.training_history: list[dict] = []

    def run(self) -> "ImageOnlyEvaluator":
        cohort_df = cohort.load_trimodal_cohort(self.merged_csv, self.split_csv)
        clinical_frame = cohort_df.drop_duplicates("research_id").set_index("research_id")

        self.c_indices, self.fold_records, self.oof_predictions, self.training_history = [], [], [], []

        for fold, ids in _fold_plan(cohort_df, max_folds=self.max_folds):
            print(f"\n=== [late_fusion/image_only/{self.target}] Fold {fold} ===")
            _seed_everything(self.seed + fold)

            train_samples = ds.preprocess_data(self.image_dir, clinical_frame.loc[ids["train"]].reset_index(), self.target, self.gray_scale)
            val_samples = ds.preprocess_data(self.image_dir, clinical_frame.loc[ids["val"]].reset_index(), self.target, self.gray_scale)
            test_samples = ds.preprocess_data(self.image_dir, clinical_frame.loc[ids["test"]].reset_index(), self.target, self.gray_scale)
            if not test_samples:
                print(f"  [skip] fold {fold}: no test samples found")
                continue

            train_ds, val_ds = ds.create_dataset(train_samples, val_samples, self.resize, self.augment, self.gray_scale, False)
            _, test_ds = ds.create_dataset(train_samples, test_samples, self.resize, False, self.gray_scale, False)

            loader_kwargs = {"num_workers": self.num_workers, "pin_memory": True, "worker_init_fn": _seed_worker}
            generator = torch.Generator()
            generator.manual_seed(self.seed + fold)
            train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, generator=generator, **loader_kwargs)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, **loader_kwargs)
            test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False, **loader_kwargs)

            if self.model_factory is not None:
                model = self.model_factory().to(self.device)
            else:
                model = ImageOnlyDeepSurv(gray_scale=self.gray_scale).to(self.device)
            if self.optimizer_factory is not None:
                optimizer = self.optimizer_factory(model)
            else:
                optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

            save_path = os.path.join(self.save_dir, f"fold{fold}_{self.ckpt_tag}_{self.target}.pt")
            fold_history = fit(model, train_loader, val_loader, optimizer, self.device, self.epochs, save_path)
            self.training_history.extend([{"target": self.target, "fold": fold, **row} for row in fold_history])

            model.load_state_dict(torch.load(save_path, map_location=self.device))
            model.eval()

            risks, durations, events = [], [], []
            with torch.no_grad():
                for images, dur, evt in test_loader:
                    risk = model(images.to(self.device)).squeeze(1).cpu().numpy()
                    risks.extend(risk.tolist())
                    durations.extend(dur.numpy().tolist())
                    events.extend(evt.numpy().tolist())

            risks = np.array(risks, dtype=np.float32)
            durations = np.array(durations, dtype=np.float32)
            events = np.array(events, dtype=np.float32)
            ci = concordance_index(durations, -risks, events)
            self.c_indices.append(ci)
            self.fold_records.append({
                "target": self.target, "modality": "image_only", "fold": fold,
                "c_index": round(float(ci), 4),
                "n": len(risks), "event_count": int((events == 1).sum()), "censored_count": int((events == 0).sum()),
            })
            self.oof_predictions.extend([
                {"research_id": rid, "target": self.target, "modality": "image_only",
                 "fold": fold, "duration": float(d), "event": float(e), "risk_score": float(r)}
                for rid, d, e, r in zip(ids["test"], durations, events, risks)
            ])
            print(f"Fold {fold}: C-index = {ci:.4f} (n={len(risks)})")

        if not self.c_indices:
            raise RuntimeError("No folds were completed.")
        print(f"\nCompleted {len(self.c_indices)} fold(s) C-index: {np.mean(self.c_indices):.4f} +/- {np.std(self.c_indices):.4f}")
        return self
