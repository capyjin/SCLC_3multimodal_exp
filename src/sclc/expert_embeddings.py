# -*- coding: utf-8 -*-
"""이미 학습·고정된 expert 체크포인트에서 fold-safe (in-sample train / OOF test)
임베딩을 뽑는다 -- 실험11(MoE gating)의 입력을 만드는 공용 모듈.

[왜 src/sclc 에 있나] 이 추출 로직은 exp_gate_m0.py 하나가 아니라 향후 M1(attention)
실험도 같이 쓸 인프라라서 core 로 올린다(코드_구조.md 원칙 1).

[재학습을 하지 않는 이유] image 는 ``outputs/image_cph/emb_{target}_fold{k}.npz``
(실험8 산출물, fold-safe OOF 캐시)를 **읽기만** 한다. tabular(clinical+report)는
``outputs/late_fusion_B/tabular_{target}/fold{k}_early_fusion_{target}.pt``
(이미 시너지를 학습한 clin_report 체크포인트, OS 0.708 을 낸 그 모델)를 로드해
forward-pass 만 한다 -- 둘 다 가중치 업데이트가 없으므로 여기서 새로운 누수가
생길 자리가 없다. 실제로 학습되는 것은 이 임베딩 위에 얹는 gate 뿐이다
(exp_gate_m0.py).

[누수 방지 원칙 -- 실험8 과 동일]
  - outer fold 의 train 환자 임베딩은 그 fold 의 체크포인트로 뽑은 것이라
    in-sample(그 체크포인트가 본 데이터)이고, test 환자 임베딩은 같은
    체크포인트로 뽑은 **진짜 OOF**다. gate 는 반드시 train 임베딩에만 fit 하고
    test 임베딩에는 forward 만 해야 한다(exp_gate_m0.py 가 강제).
  - image/tabular 둘 다 같은 ``cohort.DEFAULT_SPLIT_CSV`` 기반 ``fold_plan`` 을
    쓰므로 fold 멤버십이 동일해야 한다 -- ``load_gate_inputs`` 가 rid 집합
    불일치를 assert 로 막는다.
"""
from __future__ import annotations

import functools
import os

import numpy as np
import torch

from sclc import cohort, features
from sclc.model import ConcatDeepSurv
from sclc.train import fold_plan

IMAGE_EMB_DIR = os.path.join(cohort.PROJECT_ROOT, "outputs", "image_cph")
TABULAR_CKPT_DIR = os.path.join(cohort.PROJECT_ROOT, "outputs", "late_fusion_B")
RADBERT_TABULAR_CKPT_DIR = os.path.join(cohort.PROJECT_ROOT, "outputs", "late_fusion_B_radbert")
REPORT_ENCODERS = ("tfidf", "radbert")

# TrimodalEvaluator/get_tabular_oof 가 clin_report 체크포인트를 만들 때 쓴 기본값과
# 정확히 일치해야 한다(다르면 tabular 텐서 shape/의미가 학습 때와 어긋난다).
_TEXT_SOURCE = "concl_find"
_TFIDF_MAX_FEATURES = 400
_TFIDF_NGRAM_RANGE = (2, 4)


def _tabular_ckpt_path(ckpt_dir: str, target: str, fold: int) -> str:
    return os.path.join(ckpt_dir, f"tabular_{target}", f"fold{fold}_early_fusion_{target}.pt")


@functools.lru_cache(maxsize=None)
def resolve_report_encoder(name: str = "tfidf") -> tuple[str, object]:
    """(tabular_ckpt_dir, tabular_text_encoder_fn) 를 **쌍으로** 돌려준다.

    둘은 절대 따로 움직이면 안 된다 -- 체크포인트의 report_branch 첫 Linear 폭
    (TF-IDF 400 / RadBERT 768)이 인코더가 만드는 블록 폭과 일치해야 하기 때문이다.
    지금은 400≠768 이라 어긋나면 ``load_tabular_oof``의 ``model.load_state_dict``가
    곧바로 실패하지만(RuntimeError), 언젠가 폭이 같은 인코더가 추가되면 **조용히**
    틀린 숫자가 나올 수 있다 -- 그래서 이 둘을 개별 인자로 노출하지 않고 이 함수를
    통해서만 쌍으로 얻도록 강제한다.

    ``lru_cache``: RadBERT 임베딩 구축(모델 forward, 248개 문서)은 프로세스당
    1회만 하면 된다 -- fold/seed 와 무관한 고정값이므로(모듈독스트링 참고).
    """
    if name == "tfidf":
        return TABULAR_CKPT_DIR, None
    if name == "radbert":
        from sclc import bert_features  # transformers/sklearn 지연 import(tfidf 경로는 안 필요)
        return RADBERT_TABULAR_CKPT_DIR, bert_features.build_radbert_report_encoder()
    raise ValueError(f"알 수 없는 report_encoder {name!r} -- {list(REPORT_ENCODERS)} 중 하나여야 함")


def load_image_oof(target: str, emb_dir: str = IMAGE_EMB_DIR) -> dict[int, dict]:
    """실험8 캐시에서 fold별 (train/test) SimpleCNN 512차원 임베딩 + 자체 risk 를 읽는다."""
    out = {}
    for fold in range(1, 6):
        path = os.path.join(emb_dir, f"emb_{target}_fold{fold}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} 없음 -- 실험8/exp_image_cph.py 산출물이 필요함")
        z = np.load(path)
        out[fold] = {
            "train": {"emb": z["train_emb"], "risk": z["train_risk"], "rid": z["train_rid"],
                     "duration": z["train_duration"], "event": z["train_event"]},
            "test": {"emb": z["test_emb"], "risk": z["test_risk"], "rid": z["test_rid"],
                    "duration": z["test_duration"], "event": z["test_event"]},
        }
    return out


@torch.no_grad()
def load_tabular_oof(target: str, ckpt_dir: str = TABULAR_CKPT_DIR,
                     max_folds: int | None = None, text_encoder_fn=None) -> dict[int, dict]:
    """fold별로 clin_report 체크포인트를 로드해 (clinical 128d, report 16d) 임베딩과
    그 모델 자신의 결합 risk 를 train/test 각각에 대해 forward 로 뽑는다.

    ``text_encoder_fn`` (기본 None=TF-IDF, 기존 동작과 완전히 동일)은 판독지 블록을
    다른 인코더(RadBERT 등)로 대체한다 -- ``sclc.late_fusion.get_tabular_oof``가
    체크포인트를 **학습할 때** 쓴 것과 반드시 같아야 한다(다르면 report_branch 입력
    분포가 학습 때와 달라져 forward 결과가 무의미해진다). ``ckpt_dir``도 그 학습이
    저장한 경로와 맞춰야 한다(예: RadBERT 체크포인트는
    ``실험11_MoE/prepare_tabular_radbert_ckpt.py``가 ``outputs/late_fusion_B_radbert/``
    에 저장)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cohort_df = cohort.load_trimodal_cohort()
    clinical_frame = cohort_df.drop_duplicates("research_id").set_index("research_id")
    standardize_cols, categorical_cols = features.resolve_clinical_columns(clinical_frame)
    corpus, _ = features.load_text_corpus(cohort.DEFAULT_MERGED_CSV, source=_TEXT_SOURCE)

    out = {}
    for fold, ids in fold_plan(cohort_df, max_folds=max_folds):
        tabular, clinical_dim, report_dim = features.build_fold_multimodal_tabular(
            clinical_frame.loc[ids["train"]], clinical_frame.loc[ids["val"]], clinical_frame.loc[ids["test"]],
            corpus, standardize_cols, categorical_cols,
            tfidf_max_features=_TFIDF_MAX_FEATURES, tfidf_ngram_range=_TFIDF_NGRAM_RANGE,
            text_encoder_fn=text_encoder_fn,
        )
        model = ConcatDeepSurv(
            clinical_dim=clinical_dim, report_dim=report_dim,
            use_image=False, use_clinical=True, use_report=True,
        ).to(device)
        ckpt_path = _tabular_ckpt_path(ckpt_dir, target, fold)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)  # strict=True 기본값 -- 키가 하나라도 안 맞으면 바로 에러
        model.eval()

        split_out = {}
        for split, rid_list in (("train", ids["train"]), ("test", ids["test"])):
            tab = torch.from_numpy(tabular[split]).float().to(device)
            feats = model.branch_features(None, tab)
            risk = model.head(model.fusion_dropout(torch.cat([feats["clinical"], feats["report"]], dim=1)))
            labels = clinical_frame.loc[rid_list]
            split_out[split] = {
                "clinical_emb": feats["clinical"].cpu().numpy(),
                "report_emb": feats["report"].cpu().numpy(),
                "risk": risk.squeeze(1).cpu().numpy(),
                "rid": np.asarray(rid_list, dtype=np.int64),
                "duration": labels[f"{target}_days"].to_numpy(dtype=np.float64),
                "event": labels[f"{target}_event"].to_numpy(dtype=np.float64),
            }
        out[fold] = split_out
    return out


def _align(image_split: dict, tabular_split: dict) -> dict:
    """image/tabular 의 같은 split(rid 집합은 같아야 함)을 rid 기준으로 정렬해 합친다."""
    img_rid = list(map(int, image_split["rid"]))
    tab_rid = list(map(int, tabular_split["rid"]))
    if set(img_rid) != set(tab_rid):
        raise AssertionError(
            f"image/tabular rid 집합 불일치 (image={len(img_rid)}, tabular={len(tab_rid)}) -- "
            "split_csv 가 다르거나 fold_plan 호출 조건이 다름"
        )
    img_pos = {r: i for i, r in enumerate(img_rid)}
    tab_pos = {r: i for i, r in enumerate(tab_rid)}
    order = sorted(img_rid)
    ii = [img_pos[r] for r in order]
    ti = [tab_pos[r] for r in order]
    return {
        "rid": np.array(order, dtype=np.int64),
        "image_emb": image_split["emb"][ii],
        "image_risk": image_split["risk"][ii],
        "clinical_emb": tabular_split["clinical_emb"][ti],
        "report_emb": tabular_split["report_emb"][ti],
        "tabular_risk": tabular_split["risk"][ti],
        "duration": image_split["duration"][ii],
        "event": image_split["event"][ii],
    }


def load_gate_inputs(target: str, max_folds: int | None = None,
                     report_encoder: str = "tfidf") -> dict[int, dict]:
    """image + tabular 를 fold별로 합쳐 gate 학습에 바로 쓸 수 있는 형태로 반환.

    반환: {fold: {"train": {...}, "test": {...}}}. 각 split 딕셔너리는
    rid/image_emb(512)/image_risk/clinical_emb(128)/report_emb(16)/tabular_risk/
    duration/event 를 rid 오름차순으로 정렬해 담는다.

    ``report_encoder``: "tfidf"(기본, published 체크포인트) 또는 "radbert"
    (``outputs/late_fusion_B_radbert``). ckpt_dir 과 text_encoder_fn 은
    ``resolve_report_encoder``가 항상 쌍으로 내주므로 둘이 어긋날 일이 없다
    (개별 인자로는 노출하지 않음 -- report_dim 이 다르면 즉시 RuntimeError 로
    막히지만, 우연히 같아지는 인코더가 생기면 조용히 틀릴 수 있어서다).
    """
    ckpt_dir, text_encoder_fn = resolve_report_encoder(report_encoder)
    image = load_image_oof(target)
    tabular = load_tabular_oof(target, ckpt_dir=ckpt_dir, max_folds=max_folds,
                               text_encoder_fn=text_encoder_fn)
    out = {}
    for fold in tabular:
        out[fold] = {
            "train": _align(image[fold]["train"], tabular[fold]["train"]),
            "test": _align(image[fold]["test"], tabular[fold]["test"]),
        }
    return out
