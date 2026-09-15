# -*- coding: utf-8 -*-
"""RadImageNet(방사선영상 사전학습) ResNet50 이 SimpleCNN 대비 새로운 정보를
주는지 검정 -- MoE 4/5-expert 후보에 넣을 만한지 판단하기 위한 실험.

동기:
  ResNet18(ImageNet, 자연영상)은 도메인 미스매치로 SimpleCNN 보다 못했다
  (OS 0.633 < 0.657, RESULTS.md 3절). RadImageNet 은 그 미스매치의 정확히 그
  부분(자연영상 통계)만 없앤 사전학습이다 -- CT/MRI/초음파 130만 장으로
  학습됐지만 PET 자체는 학습 데이터에 없다. "방사선영상 통계는 도움이 되지만
  PET 특이성은 없다"는 가설을 두 가지로 검정한다.

검정 1 -- 단독 성능이 SimpleCNN 천장을 넘는가 (실험8/10 식):
  ImageOnlyEvaluator 로 5-fold DeepSurv 를 pretrained=True/False 두 조건으로
  학습해 OOF C-index 를 비교한다. random 대조군을 반드시 같이 낸다 -- 사전학습
  자체의 효과와 "ResNet50 구조가 어차피 포착하는 것"을 구분하기 위해서다.

검정 2 -- SimpleCNN 과 중복되는가:
  outputs/image_cph 캐시(실험8, 읽기전용)의 SimpleCNN OOF DeepSurv 위험점수·
  512차원 임베딩과, 여기서 새로 뽑은 RadImageNet OOF 위험점수·2048차원
  임베딩 사이의 상관을 본다 (위험점수 Pearson/Spearman, PCA20 후 CCA/RV계수 --
  exp_cnn_radiomics_corr.py 와 같은 방법론).

Run:
  python 실험11_MoE/exp_radimagenet.py                       # 본실행(오래 걸림)
  python 실험11_MoE/exp_radimagenet.py --smoke                # 경로 점검용
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import argparse
import json

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import transforms

from sclc import cohort, late_fusion
from sclc import dataset as ds
from sclc.model import RadImageNetDeepSurv
from sclc.train import fold_plan

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "image_radimagenet")
EMB_DIR = os.path.join(PROJECT_ROOT, "outputs", "image_cph")  # 실험8 SimpleCNN 캐시 (읽기 전용)
PCA_K = 20
SIMPLECNN_BASELINE = {"os": 0.6570, "pfs": 0.6154}


def load_simplecnn_oof(target: str):
    """실험8 캐시에서 SimpleCNN 의 OOF 위험점수/임베딩/rid 를 모은다 (읽기 전용)."""
    risk, emb, rids = [], [], []
    for fold in range(1, 6):
        z = np.load(os.path.join(EMB_DIR, f"emb_{target}_fold{fold}.npz"))
        risk.append(z["test_risk"])
        emb.append(z["test_emb"])
        rids.append(z["test_rid"])
    risk, emb, rids = np.concatenate(risk), np.concatenate(emb), np.concatenate(rids)
    order = np.argsort(rids)
    return risk[order], emb[order], rids[order]


@torch.no_grad()
def extract_radimagenet_oof_embeddings(target: str, ckpt_dir: str, pretrained: bool, folds_plan,
                                       image_dir: str, resize: int, device,
                                       batch_size: int = 16, num_workers: int = 4):
    """저장된 fold 체크포인트에서 test(=OOF) 2048차원 임베딩을 뽑는다
    (실험8 ``_forward_split`` 과 같은 패턴 -- 재학습 없이 forward pass만)."""
    tag = "image_radimagenet" if pretrained else "image_radimagenet_random"
    cohort_df = cohort.load_trimodal_cohort()
    clinical_frame = cohort_df.drop_duplicates("research_id").set_index("research_id")

    embs, rids = [], []
    for fold, ids in folds_plan:
        train_samples = ds.preprocess_data(image_dir, clinical_frame.loc[ids["train"]].reset_index(), target, True)
        test_samples = ds.preprocess_data(image_dir, clinical_frame.loc[ids["test"]].reset_index(), target, True)
        mean, std = ds._compute_mean_std_from_samples(train_samples, gray_scale=True)
        transform = transforms.Compose([
            transforms.ToTensor(), transforms.Resize((resize, resize)), transforms.Normalize(mean, std),
        ])
        model = RadImageNetDeepSurv(pretrained=pretrained).to(device)
        ckpt_path = os.path.join(ckpt_dir, f"fold{fold}_{tag}_{target}.pt")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        loader = DataLoader(ds.PetSurvivalDataset(test_samples, transform), batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
        fold_emb = []
        for images, _dur, _evt in loader:
            fold_emb.append(model.backbone(images.to(device)).cpu().numpy())
        embs.append(np.concatenate(fold_emb))
        rids.append(np.array(ids["test"], dtype=np.int64))

    embs, rids = np.concatenate(embs), np.concatenate(rids)
    order = np.argsort(rids)
    return embs[order], rids[order]


def cv_rank1_cca(X: np.ndarray, Y: np.ndarray, n_splits: int = 5, seed: int = 42):
    """첫 정준상관을 held-out fold 에 투영해 계산 (in-sample CCA 는 차원이
    크면 인위적으로 1에 가까워지므로 CV 필수 -- exp_cnn_radiomics_corr.py 참고)."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    corrs = []
    for tr_idx, te_idx in kf.split(X):
        x_scaler, y_scaler = StandardScaler().fit(X[tr_idx]), StandardScaler().fit(Y[tr_idx])
        cca = CCA(n_components=1).fit(x_scaler.transform(X[tr_idx]), y_scaler.transform(Y[tr_idx]))
        u, v = cca.transform(x_scaler.transform(X[te_idx]), y_scaler.transform(Y[te_idx]))
        corrs.append(float(np.corrcoef(u[:, 0], v[:, 0])[0, 1]))
    return float(np.mean(corrs)), corrs


def rv_coefficient(X: np.ndarray, Y: np.ndarray) -> float:
    Xc, Yc = X - X.mean(axis=0), Y - Y.mean(axis=0)
    Sxy, Sxx, Syy = Xc.T @ Yc, Xc.T @ Xc, Yc.T @ Yc
    return float(np.trace(Sxy @ Sxy.T) / np.sqrt(np.trace(Sxx @ Sxx.T) * np.trace(Syy @ Syy.T)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="os,pfs")
    ap.add_argument("--skip_random", action="store_true", help="랜덤초기화 대조군 생략(시간 절약)")
    ap.add_argument("--max_folds", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--smoke", action="store_true", help="1 fold / 2 epoch 로 경로만 점검")
    args = ap.parse_args()
    if args.smoke:
        args.max_folds, args.epochs = 1, 2

    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    results = {}

    print(f"device = {device}")
    for target in targets:
        print(f"\n{'=' * 70}\n[radimagenet/{target}] 검정 1: 단독 DeepSurv OOF C-index (pretrained)\n{'=' * 70}")
        ev_pre = late_fusion.get_image_oof_radimagenet(
            target, pretrained=True, epochs=args.epochs, max_folds=args.max_folds)
        results.setdefault(target, {})["pretrained_cindex"] = {
            "folds": [round(c, 4) for c in ev_pre.c_indices],
            "mean": float(np.mean(ev_pre.c_indices)), "std": float(np.std(ev_pre.c_indices)),
        }
        print(f"[radimagenet/{target}] pretrained OOF C-index = {np.mean(ev_pre.c_indices):.4f} "
              f"+/- {np.std(ev_pre.c_indices):.4f}  (SimpleCNN 기준선 {SIMPLECNN_BASELINE[target]})")

        if not args.skip_random:
            print(f"\n[radimagenet/{target}] 검정 1b: 랜덤초기화 대조군")
            ev_rnd = late_fusion.get_image_oof_radimagenet(
                target, pretrained=False, epochs=args.epochs, max_folds=args.max_folds)
            results[target]["random_cindex"] = {
                "folds": [round(c, 4) for c in ev_rnd.c_indices],
                "mean": float(np.mean(ev_rnd.c_indices)), "std": float(np.std(ev_rnd.c_indices)),
            }
            print(f"[radimagenet/{target}] random OOF C-index = {np.mean(ev_rnd.c_indices):.4f} "
                  f"+/- {np.std(ev_rnd.c_indices):.4f}")

        print(f"\n[radimagenet/{target}] 검정 2: SimpleCNN 과 위험점수/임베딩 중복도")
        sc_risk, sc_emb, sc_rids = load_simplecnn_oof(target)
        ckpt_dir = os.path.join(late_fusion.DEFAULT_OUT_DIR, f"image_radimagenet_{target}")
        plan = fold_plan(cohort.load_trimodal_cohort(), max_folds=args.max_folds)
        ri_emb, ri_rids = extract_radimagenet_oof_embeddings(
            target, ckpt_dir, True, plan, cohort.DEFAULT_IMAGE_DIR, resize=224, device=device)

        # smoke(--max_folds 1) 에서는 두 rid 집합이 부분집합일 수 있으니 교집합으로 정렬
        common = sorted(set(map(int, ri_rids)) & set(map(int, sc_rids)))
        sc_pos = {int(r): i for i, r in enumerate(sc_rids)}
        ri_pos = {int(r): i for i, r in enumerate(ri_rids)}
        sc_idx = [sc_pos[r] for r in common]
        ri_idx = [ri_pos[r] for r in common]

        ri_risk_by_rid = late_fusion.oof_dict(ev_pre.oof_predictions)
        ri_risk = np.array([ri_risk_by_rid[r] for r in common])
        pearson = float(np.corrcoef(ri_risk, sc_risk[sc_idx])[0, 1])
        spearman = float(spearmanr(ri_risk, sc_risk[sc_idx])[0])

        k = min(PCA_K, len(common) - 1)
        ri_pca = PCA(n_components=k, random_state=42).fit_transform(ri_emb[ri_idx])
        sc_pca = PCA(n_components=k, random_state=42).fit_transform(sc_emb[sc_idx])
        n_splits = min(5, len(common))
        cca_r, cca_folds = cv_rank1_cca(ri_pca, sc_pca, n_splits=n_splits)
        rv = rv_coefficient(StandardScaler().fit_transform(ri_pca), StandardScaler().fit_transform(sc_pca))

        results[target]["redundancy_vs_simplecnn"] = {
            "n_common_patients": len(common),
            "risk_score_pearson": pearson, "risk_score_spearman": spearman,
            "embedding_cca_rank1_heldout": {"mean": cca_r, "per_fold": [round(c, 4) for c in cca_folds]},
            "embedding_rv_coefficient": rv,
        }
        print(f"  위험점수 상관: pearson={pearson:.4f} spearman={spearman:.4f}")
        print(f"  임베딩 CCA(held-out) = {cca_r:.4f}, RV = {rv:.4f}")

    out_path = os.path.join(OUT_DIR, "results.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(f"\n-> {out_path}")

    print(f"\n{'=' * 70}\n요약 (SimpleCNN 기준선: OS {SIMPLECNN_BASELINE['os']} / PFS {SIMPLECNN_BASELINE['pfs']})\n{'=' * 70}")
    for target in targets:
        r = results[target]
        line = f"{target.upper():<5} pretrained={r['pretrained_cindex']['mean']:.4f}"
        if "random_cindex" in r:
            line += f"  random={r['random_cindex']['mean']:.4f}"
        line += (f"  |  risk_corr(pearson)={r['redundancy_vs_simplecnn']['risk_score_pearson']:.3f}"
                f"  emb_RV={r['redundancy_vs_simplecnn']['embedding_rv_coefficient']:.3f}")
        print(line)


if __name__ == "__main__":
    main()
