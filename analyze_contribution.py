"""각 모달리티가 최종 위험점수를 실제로 얼마나 흔드는지(기여도 표준편차) 측정."""
import glob, json, os, sys
import numpy as np, torch
import cohort, features, dataset as ds
from ablation import AblatableConcatDeepSurv, CONFIGS
from torch.utils.data import DataLoader

TARGET = os.environ.get("T","pfs")
DIMS = {"image": 128, "clinical": 128, "report": 16}
dev = torch.device("cpu")

cohort_df = cohort.load_trimodal_cohort()
clinical_frame = cohort_df.drop_duplicates("research_id").set_index("research_id")
std_cols, cat_cols = features.resolve_clinical_columns(clinical_frame)
corpus, _ = features.load_text_corpus(cohort.DEFAULT_MERGED_CSV)

out = {}
for cfg in ["clin_report", "clin_image", "all"]:
    flags = CONFIGS[cfg]
    names = [n for n in ("image", "clinical", "report") if flags[f"use_{n}"]]
    acc = {n: [] for n in names}
    for fold in [1, 2, 3]:                      # 3개 fold만 (CPU라 시간 절약)
        f_df = cohort_df[cohort_df["fold"] == fold]
        ids = {s: f_df.loc[f_df["split"] == s, "research_id"].astype(int).tolist()
               for s in ("train", "val", "test")}
        tab, cdim, rdim = features.build_fold_multimodal_tabular(
            clinical_frame.loc[ids["train"]], clinical_frame.loc[ids["val"]],
            clinical_frame.loc[ids["test"]], corpus, std_cols, cat_cols,
            tfidf_max_features=400, tfidf_ngram_range=(2, 4))
        tr_s = ds.preprocess_data(cohort.DEFAULT_IMAGE_DIR, clinical_frame.loc[ids["train"]].reset_index(), TARGET, True)
        te_s = ds.preprocess_data(cohort.DEFAULT_IMAGE_DIR, clinical_frame.loc[ids["test"]].reset_index(), TARGET, True)
        _, te_ds = ds.create_dataset(tr_s, te_s, 512, False, True, False,
                                     train_tabular=tab["train"], test_tabular=tab["test"])
        model = AblatableConcatDeepSurv(cdim, rdim, **flags)
        model.load_state_dict(torch.load(
            f"outputs/ablation_why_pfs/{cfg}_{TARGET}/fold{fold}_early_fusion_{TARGET}.pt",
            map_location="cpu"))
        model.eval()
        w = model.head.weight.squeeze(0)
        # 블록별 가중치 슬라이스
        sl, i = {}, 0
        for n in names:
            sl[n] = slice(i, i + DIMS[n]); i += DIMS[n]
        parts = {n: [] for n in names}
        with torch.no_grad():
            for img, tb, _, _ in DataLoader(te_ds, batch_size=16):
                feats = {}
                if flags["use_image"]:
                    feats["image"] = torch.nn.functional.normalize(model.img_proj(model.backbone(img)), dim=1)
                cx, rx = tb[:, :cdim], tb[:, cdim:]
                if flags["use_clinical"]:
                    feats["clinical"] = torch.nn.functional.normalize(model.clinical_branch(cx), dim=1)
                if flags["use_report"]:
                    feats["report"] = torch.nn.functional.normalize(model.report_branch(rx), dim=1)
                for n in names:
                    parts[n].append((feats[n] * w[sl[n]]).sum(1).numpy())
        for n in names:
            acc[n].append(float(np.concatenate(parts[n]).std()))
    out[cfg] = {n: float(np.mean(v)) for n, v in acc.items()}
    tot = sum(out[cfg].values())
    print(f"--- {cfg} ({TARGET}) : 위험점수 기여도 표준편차 ---")
    for n in names:
        print(f"   {n:<9} sd={out[cfg][n]:.4f}   share={out[cfg][n]/tot:.1%}")
json.dump(out, open(f"outputs/ablation_why_pfs/contribution_{TARGET}.json", "w"), indent=2)
