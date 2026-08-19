# -*- coding: utf-8 -*-
"""[실험] 영상 유니모달 = SimpleCNN(우리 구조) 특징 추출 + 고전적 CoxPH

동기:
  지금까지 영상 단독 성능(OS 0.6570 / PFS 0.6154)은 전부 **DeepSurv**로만 쟀다.
  즉 SimpleCNN 백본(512차원) 뒤에 Dropout→Linear(512,1) 헤드를 붙여
  Cox 부분우도로 **end-to-end** 학습한 값이다.
  아직 해보지 않은 것: 백본은 그대로 얼리고 **위험 예측만 고전 CoxPH(lifelines)** 로 바꾸기.
  = 라디오믹스 논문들이 흔히 쓰는 "CNN feature → PCA → CoxPH" 파이프라인.

무엇이 고정되고 무엇이 달라지나:
  고정 = 코호트(238명), split(trimodal_common_5fold_seed42_v1), 이미지 전처리(512 resize,
         train fold 픽셀 mean/std), 백본 구조(model.SimpleCNNBackbone 그대로),
         백본 학습 조건(bs16/ep30/lr1e-4/wd1e-4, val C-index 최고 체크포인트).
  달라짐 = **마지막 위험 점수를 내는 방법만**.
         deepsurv_head       : 신경망 자신의 Linear(512,1)   ← 기존 0.6570/0.6154 기준선
         cph_selected        : ★대표값★ CoxPH 계열 후보를 **train fold 내부 교차검증**으로
                               고른 뒤 171명 전체로 재적합 → test 적용 (test는 선택에 관여 안 함)
         cph_pca{k}          : 512차원 → 표준화 → PCA(k) → CoxPH        (탐색용)
         cph_ridge_full_p{L} : 512차원 전부 → 표준화 → 릿지 CoxPH       (탐색용)
         cph_unicox_top{k}   : train 단변량 Cox |z| 상위 k개 → CoxPH    (탐색용)
         ctrl_random_cnn_pca16 : **학습 안 한** 백본 특징 → PCA16 → CoxPH (대조군)

누수 방지 (설계 감사 반영):
  - 백본 체크포인트는 fold의 train으로 학습하고 val로 고른다. val/test는 서로 배타적이라
    (split CSV 검증 완료) test로 새지 않는다.
  - StandardScaler / PCA / 단변량 스크리닝 / CoxPH는 **전부 train fold 171명만**으로 fit.
  - **하이퍼파라미터(변형·k·penalizer) 선택도 test를 보지 않는다.** 대표값 `cph_selected`는
    outer train 171명 안에서 inner 5-fold CV로 후보를 고른다(= nested selection).
    `cph_pca*` 등 개별 변형 표는 **탐색용(exploratory)** 이며 대표값으로 쓰면 안 된다.
  - 임베딩 npz 캐시는 체크포인트 sha1·resize·seed 등을 함께 저장하고, 하나라도 다르면
    자동으로 다시 뽑는다(오래된 캐시로 잘못된 숫자가 나오는 것을 막음).
  - 기본 체크포인트 디렉터리(outputs/late_fusion_B/...)는 **읽기 전용**으로만 쓴다.
    (RESULTS.md의 0.6570/0.6154를 낸 파일이라 절대 덮어쓰지 않는다.)

알려진 한계(누수는 아니지만 논문에 반드시 명시):
  CoxPH는 CNN이 이미 학습한 그 171명의 임베딩 위에서 beta를 추정한다. test 정보는 전혀 안
  들어가므로 누수가 아니라 **편향**이며, 방향은 CoxPH에게 불리하다(in-sample 임베딩은
  실제보다 잘 분리돼 있어 beta가 과대 추정됨). 완전히 없애려면 outer train 안에서 백본을
  inner-CV로 다시 학습해 out-of-fold 임베딩을 만들어야 한다(백본 25회 재학습/타깃).

Run:
  python exp_image_cph.py --targets os,pfs
  python exp_image_cph.py --smoke            # 1 fold, 후보 축소 (경로 점검용)
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import hashlib
import json
import warnings

import numpy as np
import pandas as pd
import sklearn
import torch
from lifelines import CoxPHFitter
from lifelines import __version__ as lifelines_version
from lifelines.exceptions import ConvergenceError, ConvergenceWarning
from lifelines.utils import concordance_index
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import transforms

from core import cohort
from core import dataset as ds
from core.model import ImageOnlyDeepSurv
from core.train import ImageOnlyEvaluator, fold_plan, seed_everything

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "image_cph")
os.makedirs(OUT_DIR, exist_ok=True)

CKPT_TAG = "image_simplecnn"
# 기준 학습 조건 = RESULTS.md 3절의 영상 arm. 새로 학습할 때 이 조건이 아니면 막는다.
REFERENCE_TRAIN_CONF = {"epochs": 30, "batch_size": 16, "resize": 512}
CACHE_VERSION = 2  # 캐시 형식이 바뀌면 올린다


def published_ckpt_dir(target: str) -> str:
    """late fusion 영상 arm이 저장해 둔 체크포인트(bs16/ep30/seed42).
    RESULTS.md의 0.6570/0.6154를 낸 바로 그 파일이라 **읽기 전용**으로만 쓴다."""
    return os.path.join(PROJECT_ROOT, "outputs", "late_fusion_B", f"image_simplecnn_{target}")


def _ckpt_path(ckpt_dir: str, fold: int, target: str) -> str:
    return os.path.join(ckpt_dir, f"fold{fold}_{CKPT_TAG}_{target}.pt")


def _sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 1) 백본 체크포인트 준비
# ---------------------------------------------------------------------------
def resolve_ckpt_dir(target: str, args):
    """(체크포인트 디렉터리, 학습해도 되는가) 를 돌려준다.
    감사 지적 S1: 기본 디렉터리에 학습을 흘려보내면 RESULTS.md 기준 체크포인트가
    덮어써진다(--smoke의 2 epoch 모델로도!). 그래서 기본 경로는 읽기 전용으로 고정."""
    if args.retrain:
        return os.path.join(OUT_DIR, f"ckpt_{target}"), True
    if args.ckpt_dir:
        return args.ckpt_dir, True
    return published_ckpt_dir(target), False


def ensure_checkpoints(target, ckpt_dir, may_train, folds, args):
    missing = [f for f in folds if not os.path.exists(_ckpt_path(ckpt_dir, f, target))]
    if not missing and not args.retrain:
        print(f"[image_cph/{target}] 체크포인트 재사용: {ckpt_dir}")
        return
    if not may_train:
        raise RuntimeError(
            f"[image_cph/{target}] 체크포인트 없음: fold {missing} in {ckpt_dir}\n"
            f"  이 디렉터리는 RESULTS.md 기준 산출물이라 여기에 학습하지 않는다.\n"
            f"  먼저 `python late_fusion_tab_image.py --targets {target}` 로 영상 arm을 만들거나,\n"
            f"  `--retrain` 을 주어 outputs/image_cph/ckpt_{target} 에 새로 학습하라."
        )
    bad = {k: getattr(args, k) for k, v in REFERENCE_TRAIN_CONF.items() if getattr(args, k) != v}
    if bad and not args.allow_nonreference_train:
        raise RuntimeError(
            f"[image_cph/{target}] 기준 학습 조건과 다름 {bad} (기준 {REFERENCE_TRAIN_CONF}).\n"
            f"  의도한 것이면 --allow_nonreference_train 를 붙여라."
        )
    print(f"[image_cph/{target}] 체크포인트 학습 -> {ckpt_dir} (missing={missing}, retrain={args.retrain})")
    ImageOnlyEvaluator(
        target=target, epochs=args.epochs, batch_size=args.batch_size, resize=args.resize,
        save_dir=ckpt_dir, ckpt_tag=CKPT_TAG, max_folds=args.max_folds, seed=args.seed,
    ).run()


# ---------------------------------------------------------------------------
# 2) 특징 추출: 백본 512차원 임베딩 + 신경망 자신의 위험 점수
# ---------------------------------------------------------------------------
def _eval_transform(train_samples, resize, gray_scale):
    """dataset.create_dataset 의 test_transforms 와 **동일**하되, 정규화 통계를
    fold당 한 번만 계산한다(감사 NIT: 원래는 split마다 171장을 다시 훑었음).
    통계는 train fold 이미지에서만 나온다 = 학습 때와 같은 조건."""
    mean, std = ds._compute_mean_std_from_samples(train_samples, gray_scale=gray_scale)
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((resize, resize)),
        transforms.Normalize(mean, std),
    ]), mean, std


@torch.no_grad()
def _forward_split(model, samples, split_ids, transform, device, batch_size, num_workers):
    """한 split의 512차원 임베딩과 DeepSurv 위험 점수를 뽑는다."""
    dset = ds.PetSurvivalDataset(samples, transform)
    loader = DataLoader(dset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    model.eval()
    embs, risks, durs, evts = [], [], [], []
    for images, dur, evt in loader:
        emb = model.backbone(images.to(device))   # [B, 512]
        risk = model.head(emb).squeeze(1)         # eval 모드라 Dropout은 항등함수
        embs.append(emb.cpu().numpy())
        risks.append(risk.cpu().numpy())
        durs.extend(dur.numpy().tolist())
        evts.extend(evt.numpy().tolist())
    rids = np.array([split_ids[s["source_position"]] for s in samples], dtype=np.int64)
    return {
        "emb": np.concatenate(embs).astype(np.float32),
        "risk": np.concatenate(risks).astype(np.float32),
        "duration": np.array(durs, dtype=np.float64),
        "event": np.array(evts, dtype=np.float64),
        "rid": rids,
    }


@torch.no_grad()
def _warmup_batchnorm(model, samples, transform, device, batch_size, num_workers, passes=2):
    """감사 지적 S9: 학습 안 한 대조군 백본을 그냥 eval()로 쓰면 BatchNorm이
    running_mean=0/var=1 이라 **항등함수**가 되어 '같은 구조'가 아니게 된다.
    train fold 이미지만으로 BN 통계를 채워 구조적으로 공정한 대조군을 만든다(누수 없음)."""
    dset = ds.PetSurvivalDataset(samples, transform)
    loader = DataLoader(dset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    model.train()
    for _ in range(passes):
        for images, _dur, _evt in loader:
            model.backbone(images.to(device))
    model.eval()


def _cache_meta(target, fold, ckpt_path, args, include_random):
    return {
        "cache_version": CACHE_VERSION,
        "target": target,
        "fold": int(fold),
        "ckpt_path": os.path.abspath(ckpt_path),
        "ckpt_sha1": _sha1(ckpt_path),
        "resize": int(args.resize),
        "gray_scale": True,
        "seed": int(args.seed),
        "has_random": bool(include_random),
        "split_csv_sha1": _sha1(args.split_csv),
        "bn_warmup_passes": int(args.bn_warmup_passes),
    }


def build_fold_features(target, ckpt_dir, folds_plan, clinical_frame, args, include_random):
    """fold마다 train/test 임베딩을 뽑아 npz로 캐시한다.
    감사 지적 B2: 캐시에 provenance(체크포인트 sha1 등)를 같이 적고, 하나라도 다르면
    캐시를 무시하고 다시 뽑는다."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds_data = {}

    for fold, ids in folds_plan:
        cache = os.path.join(OUT_DIR, f"emb_{target}_fold{fold}.npz")
        meta = _cache_meta(target, fold, _ckpt_path(ckpt_dir, fold, target), args, include_random)

        if os.path.exists(cache) and not args.refresh_cache:
            z = np.load(cache, allow_pickle=False)
            cached_meta = json.loads(str(z["meta_json"])) if "meta_json" in z.files else None
            if cached_meta == meta:
                folds_data[fold] = {k: z[k] for k in z.files if k != "meta_json"}
                print(f"[image_cph/{target}] fold {fold}: 캐시 사용 {os.path.basename(cache)}")
                continue
            diff = [k for k in meta if cached_meta is None or cached_meta.get(k) != meta[k]]
            print(f"[image_cph/{target}] fold {fold}: 캐시 무효({diff}) -> 재추출")

        print(f"\n=== [image_cph/{target}] fold {fold} 특징 추출 ===")
        seed_everything(args.seed + fold)
        train_samples = ds.preprocess_data(args.image_dir, clinical_frame.loc[ids["train"]].reset_index(), target, True)
        test_samples = ds.preprocess_data(args.image_dir, clinical_frame.loc[ids["test"]].reset_index(), target, True)
        transform, mean, std = _eval_transform(train_samples, args.resize, True)

        model = ImageOnlyDeepSurv(gray_scale=True).to(device)
        model.load_state_dict(torch.load(_ckpt_path(ckpt_dir, fold, target), map_location=device))

        tr = _forward_split(model, train_samples, ids["train"], transform, device, args.batch_size, args.num_workers)
        te = _forward_split(model, test_samples, ids["test"], transform, device, args.batch_size, args.num_workers)

        rec = {
            "train_emb": tr["emb"], "train_duration": tr["duration"], "train_event": tr["event"],
            "train_risk": tr["risk"], "train_rid": tr["rid"],
            "test_emb": te["emb"], "test_duration": te["duration"], "test_event": te["event"],
            "test_risk": te["risk"], "test_rid": te["rid"],
        }

        if include_random:
            seed_everything(1000 + args.seed + fold)
            rnd = ImageOnlyDeepSurv(gray_scale=True).to(device)
            if args.bn_warmup_passes > 0:
                _warmup_batchnorm(rnd, train_samples, transform, device, args.batch_size,
                                  args.num_workers, passes=args.bn_warmup_passes)
            rtr = _forward_split(rnd, train_samples, ids["train"], transform, device, args.batch_size, args.num_workers)
            rte = _forward_split(rnd, test_samples, ids["test"], transform, device, args.batch_size, args.num_workers)
            rec["train_emb_random"] = rtr["emb"]
            rec["test_emb_random"] = rte["emb"]

        np.savez_compressed(cache, meta_json=np.array(json.dumps(meta)), **rec)
        folds_data[fold] = rec
        print(f"[image_cph/{target}] fold {fold}: train {rec['train_emb'].shape} / "
              f"test {rec['test_emb'].shape} (pixel mean={mean[0]:.4f} std={std[0]:.4f}) "
              f"-> {os.path.basename(cache)}")

    return folds_data


# ---------------------------------------------------------------------------
# 3) 고전 CoxPH 적합 (전부 train 행만으로 fit)
# ---------------------------------------------------------------------------
PENALIZER_LADDER = [0.0, 0.01, 0.1, 1.0, 10.0]
_SAFE_FIT_ERRORS = (ConvergenceError, ValueError, np.linalg.LinAlgError, ZeroDivisionError)


def _fit_cph(X_tr, dur_tr, evt_tr, penalizer):
    """수렴 실패하면 penalizer를 올려가며 재시도. 실제 사용된 값을 함께 반환.
    감사 지적 S2: 요청값이 사다리에 없어도 **요청값부터** 시도해야 한다."""
    cols = [f"f{i}" for i in range(X_tr.shape[1])]
    df = pd.DataFrame(X_tr, columns=cols)
    df["duration"] = dur_tr
    df["event"] = evt_tr
    ladder = [penalizer] + [p for p in PENALIZER_LADDER if p > penalizer]
    last_err = None
    for pen in ladder:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                cph = CoxPHFitter(penalizer=pen, l1_ratio=0.0)
                cph.fit(df, duration_col="duration", event_col="event")
            return cph, cols, pen
        except _SAFE_FIT_ERRORS as err:
            last_err = err
            continue
    raise RuntimeError(f"CoxPH 적합 실패 (penalizer ladder {ladder}): {last_err}")


def _cph_risks(X_tr, X_te, dur_tr, evt_tr, penalizer):
    cph, cols, pen = _fit_cph(X_tr, dur_tr, evt_tr, penalizer)
    risk_te = cph.predict_partial_hazard(pd.DataFrame(X_te, columns=cols)).to_numpy()
    risk_tr = cph.predict_partial_hazard(pd.DataFrame(X_tr, columns=cols)).to_numpy()
    return risk_te, risk_tr, pen


def _cindex(dur, risk, evt):
    return float(concordance_index(dur, -np.asarray(risk, dtype=float), evt))


def _scaled(rec, key_tr, key_te):
    sc = StandardScaler().fit(rec[key_tr])
    return sc.transform(rec[key_tr]), sc.transform(rec[key_te])


def _screen_univariate(rec, cache):
    """train 행만으로 512번 단변량 Cox -> |z|. 같은 train 행 집합이면 한 번만 계산해 캐시.
    감사 지적 S4: 실패를 조용히 0으로 만들면 '상위 k = 0..k-1번 차원'이 되어버린다.
    실패 수를 세어 보고하고 10%를 넘으면 멈춘다."""
    key = (int(rec["train_rid"][0]), int(rec["train_rid"][-1]), len(rec["train_rid"]),
           float(rec["train_duration"].sum()))
    if key in cache:
        return cache[key]

    Xtr, _ = _scaled(rec, "train_emb", "test_emb")
    zs = np.zeros(Xtr.shape[1])
    base = pd.DataFrame({"duration": rec["train_duration"], "event": rec["train_event"]})
    n_failed = 0
    for j in range(Xtr.shape[1]):
        d = base.copy()
        d["x"] = Xtr[:, j]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                f = CoxPHFitter(penalizer=0.01).fit(d, duration_col="duration", event_col="event")
            zs[j] = abs(float(f.summary.loc["x", "z"]))
        except _SAFE_FIT_ERRORS:
            zs[j] = np.nan
            n_failed += 1
    if n_failed > 0.1 * Xtr.shape[1]:
        raise RuntimeError(f"단변량 스크리닝 실패 과다: {n_failed}/{Xtr.shape[1]}")
    out = (zs, n_failed)
    cache[key] = out
    return out


# --- 후보 추정기(estimator): rec(train/test 행) -> (test 위험, train 위험, 부가정보) ----
def est_deepsurv(rec, ctx, floor):
    return rec["test_risk"].astype(float), rec["train_risk"].astype(float), {}


def make_est_pca(k, penalizer=0.0, random_backbone=False):
    keys = ("train_emb_random", "test_emb_random") if random_backbone else ("train_emb", "test_emb")

    def fn(rec, ctx, floor):
        Xtr, Xte = _scaled(rec, *keys)
        k_eff = int(min(k, Xtr.shape[0] - 1, Xtr.shape[1]))
        pca = PCA(n_components=k_eff, svd_solver="full").fit(Xtr)
        risk_te, risk_tr, pen = _cph_risks(pca.transform(Xtr), pca.transform(Xte),
                                           rec["train_duration"], rec["train_event"],
                                           max(penalizer, floor))
        return risk_te, risk_tr, {"n_components": k_eff,
                                  "explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
                                  "penalizer_used": pen}
    return fn


def make_est_ridge(penalizer):
    def fn(rec, ctx, floor):
        Xtr, Xte = _scaled(rec, "train_emb", "test_emb")
        risk_te, risk_tr, pen = _cph_risks(Xtr, Xte, rec["train_duration"], rec["train_event"],
                                           max(penalizer, floor))
        return risk_te, risk_tr, {"n_features": int(Xtr.shape[1]), "penalizer_used": pen}
    return fn


def make_est_unicox(k, penalizer=0.1):
    def fn(rec, ctx, floor):
        Xtr, Xte = _scaled(rec, "train_emb", "test_emb")
        zs, n_failed = _screen_univariate(rec, ctx["screen"])
        order = np.argsort(-np.nan_to_num(zs, nan=-np.inf))
        top = order[:k]
        risk_te, risk_tr, pen = _cph_risks(Xtr[:, top], Xte[:, top],
                                           rec["train_duration"], rec["train_event"],
                                           max(penalizer, floor))
        return risk_te, risk_tr, {"selected": top.tolist(), "n_screen_failed": int(n_failed),
                                  "penalizer_used": pen}
    return fn


def build_candidates(pca_ks, ridge_penalizers, uni_ks):
    """cph_selected 가 고를 수 있는 후보군 (전부 고전 CoxPH 계열)."""
    cands = [(f"cph_pca{k}", make_est_pca(k)) for k in pca_ks]
    cands += [(f"cph_ridge_full_p{p:g}", make_est_ridge(p)) for p in ridge_penalizers]
    cands += [(f"cph_unicox_top{k}", make_est_unicox(k)) for k in uni_ks]
    return cands


# ---------------------------------------------------------------------------
# 4) inner CV 로 후보 고르기 (test 미사용) — 감사 지적 B1
# ---------------------------------------------------------------------------
def _sub_rec(rec, tr_idx, va_idx):
    """outer train 171명을 inner-train / inner-val 로 쪼갠 가짜 rec."""
    out = {
        "train_emb": rec["train_emb"][tr_idx], "test_emb": rec["train_emb"][va_idx],
        "train_duration": rec["train_duration"][tr_idx], "test_duration": rec["train_duration"][va_idx],
        "train_event": rec["train_event"][tr_idx], "test_event": rec["train_event"][va_idx],
        "train_rid": rec["train_rid"][tr_idx], "test_rid": rec["train_rid"][va_idx],
        "train_risk": rec["train_risk"][tr_idx], "test_risk": rec["train_risk"][va_idx],
    }
    if "train_emb_random" in rec:
        out["train_emb_random"] = rec["train_emb_random"][tr_idx]
        out["test_emb_random"] = rec["train_emb_random"][va_idx]
    return out


def select_by_inner_cv(rec, candidates, ctx, n_splits, seed):
    """outer train 171명 안에서만 inner K-fold -> 후보별 평균 C-index -> 최고 후보 이름 반환."""
    n = len(rec["train_rid"])
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = {name: [] for name, _ in candidates}
    for tr_idx, va_idx in kf.split(np.arange(n)):
        sub = _sub_rec(rec, tr_idx, va_idx)
        for name, fn in candidates:
            try:
                risk_va, _, _ = fn(sub, ctx, 0.0)
                scores[name].append(_cindex(sub["test_duration"], risk_va, sub["test_event"]))
            except Exception as err:  # 후보 하나가 죽어도 선택 절차는 계속
                print(f"    [inner-cv] {name} 실패: {type(err).__name__}: {err}")
                scores[name].append(np.nan)
    mean_scores = {name: float(np.nanmean(v)) if not np.all(np.isnan(v)) else -np.inf
                   for name, v in scores.items()}
    best = max(mean_scores, key=mean_scores.get)
    return best, mean_scores


# ---------------------------------------------------------------------------
# 5) fold 단위 평가 + penalizer 통일 + 페어드 부트스트랩
# ---------------------------------------------------------------------------
def eval_estimator(name, fn, folds, folds_data, ctx, floor=0.0):
    per_fold = []
    for fold in folds:
        rec = folds_data[fold]
        risk_te, risk_tr, extra = fn(rec, ctx, floor)
        per_fold.append({
            "fold": fold,
            "c_index": _cindex(rec["test_duration"], risk_te, rec["test_event"]),
            "train_insample_c_index": _cindex(rec["train_duration"], risk_tr, rec["train_event"]),
            "risk_test": np.asarray(risk_te, dtype=float),
            "extra": extra,
        })
    return per_fold


def eval_with_uniform_penalizer(name, fn, folds, folds_data, ctx):
    """감사 지적 S3: penalizer 사다리가 fold마다 다르게 올라가면 5-fold 평균이
    서로 다른 추정기의 평균이 된다. 1차로 돌려 최대값을 찾고, 다르면 전 fold를 그 값으로 재적합."""
    per_fold = eval_estimator(name, fn, folds, folds_data, ctx, floor=0.0)
    used = [f["extra"].get("penalizer_used") for f in per_fold if "penalizer_used" in f["extra"]]
    if used and len(set(used)) > 1:
        floor = max(used)
        print(f"  [!] {name}: fold마다 penalizer가 달랐음 {sorted(set(used))} -> 전 fold를 {floor:g} 로 재적합")
        per_fold = eval_estimator(name, fn, folds, folds_data, ctx, floor=floor)
    return per_fold


def paired_bootstrap(per_fold_a, per_fold_b, folds_data, n_boot, seed):
    """같은 환자 리샘플에서 두 방법의 C-index를 동시에 계산해 차이(a-b)의 분포를 낸다.
    fold마다 beta·스케일러가 달라 위험 점수를 fold 간에 합칠 수 없으므로 fold별로 계산 후 평균."""
    rng = np.random.default_rng(seed)
    a_by_fold = {f["fold"]: f["risk_test"] for f in per_fold_a}
    b_by_fold = {f["fold"]: f["risk_test"] for f in per_fold_b}
    diffs = []
    for _ in range(n_boot):
        da, db = [], []
        for fold, rec in folds_data.items():
            n = len(rec["test_duration"])
            idx = rng.integers(0, n, size=n)
            dur, evt = rec["test_duration"][idx], rec["test_event"][idx]
            if evt.sum() < 2 or len(np.unique(dur)) < 2:
                continue
            try:
                da.append(_cindex(dur, a_by_fold[fold][idx], evt))
                db.append(_cindex(dur, b_by_fold[fold][idx], evt))
            except (ZeroDivisionError, ValueError):
                continue
        if da:
            diffs.append(float(np.mean(da) - np.mean(db)))
    diffs = np.asarray(diffs)
    if diffs.size == 0:
        return None
    return {
        "n_boot": int(diffs.size),
        "mean_diff": float(diffs.mean()),
        "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "p_gt_0": float((diffs > 0).mean()),
    }


def summarize(per_fold):
    ci = [f["c_index"] for f in per_fold]
    return {
        "fold_cindex": [round(float(c), 4) for c in ci],
        "mean": float(np.mean(ci)),
        "std": float(np.std(ci)),
        "train_insample_cindex_mean": float(np.mean([f["train_insample_c_index"] for f in per_fold])),
        "extras": [f["extra"] for f in per_fold],
    }


# ---------------------------------------------------------------------------
# 6) 실행
# ---------------------------------------------------------------------------
def run_target(target, args, cohort_df):
    clinical_frame = cohort_df.drop_duplicates("research_id").set_index("research_id")
    folds_plan = fold_plan(cohort_df, max_folds=args.max_folds)
    folds = [f for f, _ in folds_plan]
    include_random = not args.no_random

    ckpt_dir, may_train = resolve_ckpt_dir(target, args)
    ensure_checkpoints(target, ckpt_dir, may_train, folds, args)
    folds_data = build_fold_features(target, ckpt_dir, folds_plan, clinical_frame, args, include_random)

    ctx = {"screen": {}}
    candidates = build_candidates(args.pca_ks, args.ridge_penalizers, args.uni_ks)
    results, per_fold_store = {}, {}

    # (a) 기준선: 신경망 자신의 헤드
    per_fold_store["deepsurv_head"] = eval_estimator("deepsurv_head", est_deepsurv, folds, folds_data, ctx)
    results["deepsurv_head"] = summarize(per_fold_store["deepsurv_head"])
    print(f"[image_cph/{target}/deepsurv_head] MEAN {results['deepsurv_head']['mean']:.4f} "
          f"+/- {results['deepsurv_head']['std']:.4f}")

    # (b) ★대표값★ inner CV 로 고른 CoxPH (test는 선택에 관여하지 않음)
    print(f"\n--- [image_cph/{target}] cph_selected: outer train 안에서 inner {args.inner_folds}-fold 로 후보 선택 ---")
    sel_per_fold, sel_choice = [], []
    for fold in folds:
        rec = folds_data[fold]
        best, mean_scores = select_by_inner_cv(rec, candidates, ctx, args.inner_folds, args.seed + fold)
        fn = dict(candidates)[best]
        risk_te, risk_tr, extra = fn(rec, ctx, 0.0)
        sel_per_fold.append({
            "fold": fold,
            "c_index": _cindex(rec["test_duration"], risk_te, rec["test_event"]),
            "train_insample_c_index": _cindex(rec["train_duration"], risk_tr, rec["train_event"]),
            "risk_test": np.asarray(risk_te, dtype=float),
            "extra": {"selected_variant": best, "inner_cv_mean": round(mean_scores[best], 4), **extra},
        })
        sel_choice.append(best)
        print(f"[image_cph/{target}/cph_selected] fold {fold}: 선택={best} "
              f"(inner {mean_scores[best]:.4f}) -> test C-index={sel_per_fold[-1]['c_index']:.4f}")
    per_fold_store["cph_selected"] = sel_per_fold
    results["cph_selected"] = summarize(sel_per_fold)
    results["cph_selected"]["selected_per_fold"] = sel_choice
    print(f"[image_cph/{target}/cph_selected] MEAN {results['cph_selected']['mean']:.4f} "
          f"+/- {results['cph_selected']['std']:.4f}")

    # (c) 탐색용 개별 변형 (test로 고르면 안 됨 — 표에 exploratory 라고 표시)
    print(f"\n--- [image_cph/{target}] 탐색용 개별 변형 (대표값으로 쓰지 말 것) ---")
    exploratory = list(candidates)
    if include_random:
        exploratory.append(("ctrl_random_cnn_pca16", make_est_pca(16, random_backbone=True)))
    for name, fn in exploratory:
        per_fold = eval_with_uniform_penalizer(name, fn, folds, folds_data, ctx)
        per_fold_store[name] = per_fold
        results[name] = summarize(per_fold)
        results[name]["exploratory"] = True
        print(f"[image_cph/{target}/{name}] MEAN {results[name]['mean']:.4f} +/- {results[name]['std']:.4f} "
              f"(train in-sample {results[name]['train_insample_cindex_mean']:.4f})")

    # (d) 기준선 대비 페어드 부트스트랩 (감사 지적 S8)
    print(f"\n--- [image_cph/{target}] deepsurv_head 대비 페어드 부트스트랩 (B={args.n_boot}) ---")
    for name in results:
        if name == "deepsurv_head":
            continue
        bs = paired_bootstrap(per_fold_store[name], per_fold_store["deepsurv_head"],
                              folds_data, args.n_boot, args.seed)
        results[name]["vs_deepsurv_head"] = {
            "per_fold_diff": [round(a["c_index"] - b["c_index"], 4)
                              for a, b in zip(per_fold_store[name], per_fold_store["deepsurv_head"])],
            "mean_diff": round(results[name]["mean"] - results["deepsurv_head"]["mean"], 4),
            "paired_bootstrap": bs,
        }
        if bs:
            print(f"  {name:<26} d={bs['mean_diff']:+.4f}  95% CI [{bs['ci95'][0]:+.4f}, {bs['ci95'][1]:+.4f}]"
                  f"  P(d>0)={bs['p_gt_0']:.3f}")

    provenance = {
        "seed": args.seed, "resize": args.resize, "batch_size": args.batch_size,
        "epochs_if_trained": args.epochs, "gray_scale": True,
        "inner_folds": args.inner_folds, "n_boot": args.n_boot,
        "bn_warmup_passes": args.bn_warmup_passes,
        "ckpt_dir": ckpt_dir,
        "ckpt_sha1": {int(f): _sha1(_ckpt_path(ckpt_dir, f, target)) for f in folds},
        "split_csv": args.split_csv, "split_csv_sha1": _sha1(args.split_csv),
        "versions": {"torch": torch.__version__, "lifelines": lifelines_version,
                     "sklearn": sklearn.__version__, "numpy": np.__version__},
        "fold_sizes": {int(f): {"n_train": int(len(folds_data[f]["train_rid"])),
                                "n_test": int(len(folds_data[f]["test_rid"])),
                                "n_test_events": int(folds_data[f]["test_event"].sum())}
                       for f in folds},
    }
    return {
        "target": target,
        "n_folds": len(folds),
        "headline": {"reference": "deepsurv_head", "classical_cox": "cph_selected"},
        "note": ("cph_pca*/cph_ridge*/cph_unicox* 는 exploratory 이며 test fold에서 최대값을 고르면 "
                 "낙관 편향이 생긴다. 대표값은 inner-CV로 고른 cph_selected 를 쓸 것. "
                 "train_insample_cindex_mean 은 방법 간 비교 불가(정의상 CoxPH가 더 in-sample)."),
        "provenance": provenance,
        "variants": results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="os,pfs")
    ap.add_argument("--max_folds", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=30, help="체크포인트가 없을 때만 사용")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--resize", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--inner_folds", type=int, default=5, help="cph_selected 후보 선택용 inner CV")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--bn_warmup_passes", type=int, default=2, help="랜덤 대조군 BN 통계 채우기(train fold만)")
    ap.add_argument("--ckpt_dir", default=None, help="비우면 outputs/late_fusion_B/image_simplecnn_{target} (읽기 전용)")
    ap.add_argument("--retrain", action="store_true", help="outputs/image_cph/ckpt_{target} 에 백본을 새로 학습")
    ap.add_argument("--allow_nonreference_train", action="store_true",
                    help="기준(ep30/bs16/512)과 다른 조건으로 학습하는 것을 허용")
    ap.add_argument("--refresh_cache", action="store_true", help="임베딩 npz 캐시 무시하고 다시 추출")
    ap.add_argument("--no_random", action="store_true", help="랜덤 초기화 백본 대조군 생략")
    ap.add_argument("--pca_ks", default=None)
    ap.add_argument("--ridge_penalizers", default=None)
    ap.add_argument("--uni_ks", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "results.json"))
    args = ap.parse_args()

    # --smoke 는 명시적으로 준 후보 목록을 덮어쓰지 않는다 (감사 NIT)
    defaults = {"pca_ks": "2,4,8,16,32,64", "ridge_penalizers": "0.1,1.0,10.0", "uni_ks": "8,16"}
    smoke_defaults = {"pca_ks": "4,16", "ridge_penalizers": "1.0", "uni_ks": ""}
    for key, dflt in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, smoke_defaults[key] if args.smoke else dflt)
    if args.smoke:
        args.max_folds = args.max_folds or 1
        args.n_boot = min(args.n_boot, 200)

    args.pca_ks = sorted({int(x) for x in args.pca_ks.split(",") if x.strip()})
    args.ridge_penalizers = sorted({float(x) for x in args.ridge_penalizers.split(",") if x.strip()})
    args.uni_ks = sorted({int(x) for x in args.uni_ks.split(",") if x.strip()})
    args.merged_csv = cohort.DEFAULT_MERGED_CSV
    args.image_dir = cohort.DEFAULT_IMAGE_DIR
    args.split_csv = cohort.DEFAULT_SPLIT_CSV

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    if not targets:
        raise SystemExit("--targets 가 비었다")

    cohort_df = cohort.load_trimodal_cohort(args.merged_csv, args.split_csv)  # 타깃마다 다시 읽지 않는다
    all_results = {}
    for target in targets:
        all_results[target] = run_target(target, args, cohort_df)
        with open(os.path.join(OUT_DIR, f"results_{target}.json"), "w") as f:
            json.dump(all_results[target], f, indent=2, ensure_ascii=False)

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[image_cph] wrote {args.out}")

    print("\n================ 영상 유니모달: DeepSurv 헤드 vs 고전 CoxPH ================")
    names = list(all_results[targets[0]]["variants"].keys())
    width = 20
    header = f"{'variant':<26}" + "".join(f"{t.upper():>{width}}" for t in targets)
    print(header)
    print("-" * len(header))
    for name in names:
        row = f"{name:<26}"
        for t in targets:
            v = all_results[t]["variants"].get(name)
            row += f"{v['mean']:.4f} +/- {v['std']:.3f}".rjust(width) if v else " " * width
        tag = "  (exploratory)" if all_results[targets[0]]["variants"][name].get("exploratory") else ""
        print(row + tag)
    print("\n대표값 = deepsurv_head(기존 기준선) vs cph_selected(inner-CV로 고른 고전 CoxPH).")


if __name__ == "__main__":
    main()
