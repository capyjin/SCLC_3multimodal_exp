# -*- coding: utf-8 -*-
"""인코더 출력 처리. 학습된 인코더에서 특징·위험점수를 뽑고(FeatureHook), 저장된 체크포인트로 같은 값을
다시 만들며(CheckpointRebuilder), 위험점수를 late fusion 입력(z-score, inner OOF)으로 정리한다."""

import hashlib
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sclc import cohort, features
from sclc import dataset as ds
from sclc import foldsafe_clinical as fsc
from sclc.model import MODALITY_CONFIGS, ImageOnlyDeepSurv, make_model_factory
from sclc.train import fold_plan

from .config import ARMS


# ── 특징 추출 ──
def state_hash(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class FeatureHook:
    """fold 학습이 끝나면 호출되어 split 별 특징·위험점수를 store[fold] 에 저장한다. 하위 클래스가 extract 를 정한다."""

    def __init__(self, store: dict, keep_model: bool = True):
        self.store, self.keep_model = store, keep_model

    def __call__(self, fold, ids, model, datasets, device):
        out = self._record(fold, ids, model, datasets)
        for split, ds_ in datasets.items():
            if len(ds_) == 0:
                continue
            out[split] = self.extract(model, ds_, device)

    def extract(self, model, dataset, device) -> dict:
        raise NotImplementedError

    def _record(self, fold, ids, model, datasets) -> dict:
        self.store[fold] = {"ids": ids, "hash": state_hash(model), "training_flag": model.training,
                            "image_norm": self._image_norm(datasets), "splits": {},
                            "model": model if self.keep_model else None,
                            "datasets": datasets if self.keep_model else None}
        return self.store[fold]["splits"]

    @staticmethod
    def _image_norm(datasets) -> dict | None:
        try:
            norm = datasets["train"].transform.transforms[2]
            return {"mean": [float(x) for x in norm.mean], "std": [float(x) for x in norm.std]}
        except (AttributeError, IndexError, KeyError):
            return None


class TabularFeatureHook(FeatureHook):
    """임상·판독지 인코더: clinical [n,128], report [n,16] 특징."""

    def extract(self, model, dataset, device) -> dict:
        ec, et, risk, head_risk = [], [], [], []
        with torch.no_grad():
            for img, tab, _, _ in DataLoader(dataset, batch_size=64, shuffle=False):
                img, tab = img.to(device), tab.to(device)
                f = model.branch_features(img, tab)
                ec.append(f["clinical"].cpu())
                et.append(f["report"].cpu())
                risk.append(model(img, tab).cpu())
                head_risk.append(model.head(torch.cat([f["clinical"], f["report"]], 1)).cpu())
        return {"clinical": torch.cat(ec), "report": torch.cat(et),
                "risk": torch.cat(risk).squeeze(1), "head_risk": torch.cat(head_risk).squeeze(1),
                "tabular": torch.as_tensor(dataset.tabular)}


class ImageFeatureHook(FeatureHook):
    """PET MIP 인코더(SimpleCNN): image [n,512] 특징."""

    def extract(self, model, dataset, device) -> dict:
        ei, risk, head_risk = [], [], []
        with torch.no_grad():
            for img, _, _ in DataLoader(dataset, batch_size=32, shuffle=False):
                img = img.to(device)
                e = model.backbone(img)
                ei.append(e.cpu())
                risk.append(model(img).cpu())
                head_risk.append(model.head(e).cpu())
        return {"image": torch.cat(ei), "risk": torch.cat(risk).squeeze(1),
                "head_risk": torch.cat(head_risk).squeeze(1)}


# ── 체크포인트 재구성 ──
class CheckpointRebuilder:
    """저장된 인코더 체크포인트를 다시 불러, 학습 때와 같은 전처리로 특징·위험점수를 재계산한다 (재학습 없음)."""

    def __init__(self, target, ctx):
        self.target, self.ctx = target, ctx
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.full = cohort.load_trimodal_cohort(recode_lung_disease=True).drop_duplicates(
            "research_id").set_index("research_id")

    def _frames(self, split_csv):
        cdf = cohort.load_trimodal_cohort(split_csv=split_csv, recode_lung_disease=True)
        return cdf, cdf.drop_duplicates("research_id").set_index("research_id")

    def _samples(self, frame):
        return ds.preprocess_data(cohort.DEFAULT_IMAGE_DIR, frame.reset_index(), self.target, True)

    def tabular(self, split_csv, ckpt_dir, extra_ids):
        cdf, clin = self._frames(split_csv)
        std_cols, cat_cols = fsc.baseline_clinical_columns(clin)
        extra_fn = fsc.make_extra_numeric_fn(self.ctx["raw"], [])
        cols = [c for c in cdf.columns if c != "research_id"]
        extra_frame = self.full.loc[extra_ids, cols] if extra_ids else None
        store = {}
        for fold, ids in fold_plan(cdf):
            build = lambda test: features.build_fold_multimodal_tabular(
                clin.loc[ids["train"]], clin.loc[ids["val"]], test, self.ctx["corpus"], std_cols, cat_cols,
                extra_numeric_fn=extra_fn, text_encoder_fn=self.ctx["text_fn"])
            tab, cdim, rdim = build(clin.loc[ids["test"]])
            tr_s = self._samples(clin.loc[ids["train"]])
            train_ds, val_ds = ds.create_dataset(tr_s, self._samples(clin.loc[ids["val"]]), 512, False, True, False,
                                                 train_tabular=tab["train"], test_tabular=tab["val"])
            _, test_ds = ds.create_dataset(tr_s, self._samples(clin.loc[ids["test"]]), 512, False, True, False,
                                           train_tabular=tab["train"], test_tabular=tab["test"])
            dsets = {"train": train_ds, "val": val_ds, "test": test_ds}
            if extra_frame is not None:
                etab, _, _ = build(extra_frame)
                assert np.array_equal(etab["train"], tab["train"])
                _, dsets["extra"] = ds.create_dataset(tr_s, self._samples(extra_frame), 512, False, True, False,
                                                      train_tabular=tab["train"], test_tabular=etab["test"])
            model = make_model_factory(MODALITY_CONFIGS["clin_report"])(cdim, rdim).to(self.dev)
            model.load_state_dict(torch.load(os.path.join(ckpt_dir, f"fold{fold}_early_fusion_{self.target}.pt"),
                                             map_location=self.dev), strict=True)
            model.eval()
            TabularFeatureHook(store, keep_model=False)(fold=fold, ids=ids, model=model, datasets=dsets, device=self.dev)
        return store

    def image(self, split_csv, ckpt_dir, extra_ids):
        cdf, clin = self._frames(split_csv)
        cols = [c for c in cdf.columns if c != "research_id"]
        extra_frame = self.full.loc[extra_ids, cols] if extra_ids else None
        store = {}
        for fold, ids in fold_plan(cdf):
            tr_s = self._samples(clin.loc[ids["train"]])
            train_ds, val_ds = ds.create_dataset(tr_s, self._samples(clin.loc[ids["val"]]), 512, False, True, False)
            _, test_ds = ds.create_dataset(tr_s, self._samples(clin.loc[ids["test"]]), 512, False, True, False)
            dsets = {"train": train_ds, "val": val_ds, "test": test_ds}
            if extra_frame is not None:
                _, dsets["extra"] = ds.create_dataset(tr_s, self._samples(extra_frame), 512, False, True, False)
            model = ImageOnlyDeepSurv(gray_scale=True).to(self.dev)
            model.load_state_dict(torch.load(os.path.join(ckpt_dir, f"fold{fold}_image_simplecnn_{self.target}.pt"),
                                             map_location=self.dev), strict=True)
            model.eval()
            ImageFeatureHook(store, keep_model=False)(fold=fold, ids=ids, model=model, datasets=dsets, device=self.dev)
        return store


# ── late fusion 입력 ──
def standardizer(train_scores) -> tuple[float, float]:
    """학습 환자 점수의 (평균, 표준편차)."""
    x = np.asarray(train_scores, dtype=float)
    sd = float(x.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("학습 환자 점수의 SD 가 0 이거나 계산되지 않는다")
    return float(x.mean()), sd


def standardize(x, stats: tuple[float, float]) -> np.ndarray:
    return (np.asarray(x, dtype=float) - stats[0]) / stats[1]


def arm_records(tab_s: dict, img_s: dict, rt: dict, ri: dict, test_ids) -> tuple[list[dict], dict]:
    """인코더 hook 결과를 nested_late_fusion · GatedLateFusion 의 입력 형식(inner fold 별 점수, outer 점수)으로 바꾼다."""
    arms = (("tabular", tab_s), ("image", img_s))
    inner = [{"j": j, "val_ids": tab_s[j]["ids"]["test"],
              "arms": {arm: {"fit_risk": s[j]["splits"]["train"]["risk"].numpy(),
                             "val_risk": s[j]["splits"]["test"]["risk"].numpy(),
                             "extra_risk": s[j]["splits"]["extra"]["risk"].numpy()}
                       for arm, s in arms}} for j in sorted(tab_s)]
    outer = {"test_ids": test_ids,
             "arms": {"tabular": {"train_risk": rt["train"]["risk"].numpy(), "test_risk": rt["test"]["risk"].numpy()},
                      "image": {"train_risk": ri["train"]["risk"].numpy(), "test_risk": ri["test"]["risk"].numpy()}}}
    return inner, outer


def pool_inner_oof(inner: list[dict], t_ids) -> pd.DataFrame:
    """inner fold 별 검증 점수를 모아 환자당 한 줄의 OOF 표를 만든다 (각 점수는 자기 inner 모델 기준 z-score)."""
    rows = []
    for rec in inner:
        cols = {}
        for arm in ARMS:
            a = rec["arms"][arm]
            if len(a["val_risk"]) != len(rec["val_ids"]):
                raise ValueError(f"inner {rec['j']} {arm}: val 점수 수 != val 환자 수")
            cols[f"raw_{arm}"] = np.asarray(a["val_risk"], dtype=float)
            cols[f"z_{arm}"] = standardize(a["val_risk"], standardizer(a["fit_risk"]))
        rows.append(pd.DataFrame({"research_id": list(rec["val_ids"]), "inner_fold": rec["j"], **cols}))
    df = pd.concat(rows, ignore_index=True)
    if df["research_id"].duplicated().any():
        raise ValueError(f"OOF 에 중복 환자: {sorted(df.loc[df.research_id.duplicated(), 'research_id'])}")
    if set(df["research_id"]) != set(int(i) for i in t_ids):
        missing = sorted(set(int(i) for i in t_ids) - set(df["research_id"]))
        extra = sorted(set(df["research_id"]) - set(int(i) for i in t_ids))
        raise ValueError(f"OOF 가 T_k 와 다르다 (빠짐 {missing}, 남음 {extra})")
    return df.set_index("research_id").sort_index()
