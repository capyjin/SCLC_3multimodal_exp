"""단일 모달 모델들이 서로 얼마나 '같은 환자'를 위험하다고 보는지(상관) 측정.
상관이 높으면 = 같은 정보를 담고 있다 = 합쳐도 새 정보가 없다(중복)."""
import os, sys, json
import numpy as np, torch
import cohort, features, dataset as ds
from ablation import AblatableConcatDeepSurv, CONFIGS
from torch.utils.data import DataLoader
from scipy.stats import spearmanr

TARGET = "pfs"
SINGLES = ["image_only", "clin_only", "report_only"]

cohort_df = cohort.load_trimodal_cohort()
cf = cohort_df.drop_duplicates("research_id").set_index("research_id")
std_cols, cat_cols = features.resolve_clinical_columns(cf)
corpus, _ = features.load_text_corpus(cohort.DEFAULT_MERGED_CSV)

risks = {c: [] for c in SINGLES}     # 모든 fold의 test 위험점수를 이어붙임
for fold in [1, 2, 3, 4, 5]:
    f_df = cohort_df[cohort_df["fold"] == fold]
    ids = {s: f_df.loc[f_df["split"] == s, "research_id"].astype(int).tolist()
           for s in ("train", "val", "test")}
    tab, cdim, rdim = features.build_fold_multimodal_tabular(
        cf.loc[ids["train"]], cf.loc[ids["val"]], cf.loc[ids["test"]],
        corpus, std_cols, cat_cols, tfidf_max_features=400, tfidf_ngram_range=(2, 4))
    tr_s = ds.preprocess_data(cohort.DEFAULT_IMAGE_DIR, cf.loc[ids["train"]].reset_index(), TARGET, True)
    te_s = ds.preprocess_data(cohort.DEFAULT_IMAGE_DIR, cf.loc[ids["test"]].reset_index(), TARGET, True)
    _, te_ds = ds.create_dataset(tr_s, te_s, 512, False, True, False,
                                 train_tabular=tab["train"], test_tabular=tab["test"])
    for cfg in SINGLES:
        m = AblatableConcatDeepSurv(cdim, rdim, **CONFIGS[cfg])
        m.load_state_dict(torch.load(
            f"outputs/ablation_why_pfs/{cfg}_{TARGET}/fold{fold}_early_fusion_{TARGET}.pt",
            map_location="cpu"))
        m.eval()
        r = []
        with torch.no_grad():
            for img, tb, _, _ in DataLoader(te_ds, batch_size=16):
                r.append(m(img, tb).squeeze(1).numpy())
        risks[cfg].append(np.concatenate(r))

flat = {c: np.concatenate(v) for c, v in risks.items()}
print(f"\n=== 단일 모달 위험점수 간 상관 (PFS, 전체 {len(flat['image_only'])}명 OOF) ===")
out = {}
for a, b in [("image_only", "report_only"), ("image_only", "clin_only"), ("clin_only", "report_only")]:
    rho, p = spearmanr(flat[a], flat[b])
    out[f"{a}~{b}"] = {"spearman": float(rho), "p": float(p)}
    print(f"  {a:<12} ~ {b:<12}  spearman = {rho:+.3f}  (p={p:.2g})")
json.dump(out, open(f"outputs/ablation_why_pfs/redundancy_{TARGET}.json", "w"), indent=2)
