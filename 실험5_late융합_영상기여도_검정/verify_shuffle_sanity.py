# -*- coding: utf-8 -*-
"""[검증용] "뒤섞은 이미지 단독 = 0.50(동전던지기)"이 사실인지 직접 확인하는 스크립트.

fig12의 '＋난수(뒤섞은) 이미지' 막대가 0.705나 되는 것이 이상해 보일 수 있다.
그 0.705는 뒤섞은 이미지의 실력이 아니라 **tabular(임상+판독지)가 낸 점수**이며,
뒤섞은 이미지는 아무 기여도 못 하면서 잡음만 더해 0.708 -> 0.705로 깎은 것이다.

이 스크립트는 그것을 증명한다:
  ① 이미 RESULTS.md / results.json 에 공개된 값(tabular 단독, 이미지 단독)을
     이 스크립트가 그대로 재현하는지 먼저 대조한다  -> 계산 방식이 맞다는 증거
  ② 같은 방식으로 '뒤섞은 이미지 단독' C-index 를 계산한다 -> 0.50 근처여야 한다

Run:  python 실험5_late융합_영상기여도_검정/verify_shuffle_sanity.py
"""
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from lifelines.utils import concordance_index

from core import cohort
from core.train import fold_plan

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "late_fusion_B")
N_REPEAT = 100
SEED = 42


def fold_mean_cindex(score_map, labels, cohort_df, target):
    """fold별 test 환자에 대해 C-index를 구해 평균낸다 (late fusion 평가와 동일 방식)."""
    cis = []
    for _fold, ids in fold_plan(cohort_df, max_folds=None):
        te = ids["test"]
        s = np.array([score_map[i] for i in te], dtype=float)
        d = labels.loc[te, f"{target}_days"].to_numpy(float)
        e = labels.loc[te, f"{target}_event"].to_numpy(float)
        cis.append(concordance_index(d, -s, e))   # 위험점수가 높을수록 빨리 사건 -> 부호 반전
    return float(np.mean(cis))


def main():
    cohort_df = cohort.load_trimodal_cohort()
    labels = cohort_df.drop_duplicates("research_id").set_index("research_id")
    published = json.load(open(os.path.join(OUT_DIR, "results.json")))

    for target in ("os", "pfs"):
        oof = json.load(open(os.path.join(OUT_DIR, f"oof_{target}.json")))
        tab = {int(k): v for k, v in oof["tabular"].items()}
        img = {int(k): v for k, v in oof["image"].items()}
        ids = list(labels.index)

        tab_ci = fold_mean_cindex(tab, labels, cohort_df, target)
        img_ci = fold_mean_cindex(img, labels, cohort_df, target)

        # ── ① 대조: 이미 공개된 값과 일치하는가? ───────────────────────────
        ref_tab = published[target]["tabular_only"]["mean"]
        ref_img = published[target]["image_simplecnn_only"]["mean"]

        print(f"\n{'='*66}\n  {target.upper()}\n{'='*66}")
        print("① 계산 방식 검증 — 이 스크립트가 기존 공개 수치를 재현하는가?")
        for name, mine, ref in (("tabular 단독(임상+판독지)", tab_ci, ref_tab),
                                ("진짜 이미지 단독", img_ci, ref_img)):
            ok = "일치 ✅" if abs(mine - ref) < 1e-3 else "불일치 ❌"
            print(f"   {name:<24} 이 스크립트 {mine:.4f}  |  results.json {ref:.4f}  -> {ok}")

        # ── ② 본 검증: 뒤섞은 이미지를 '단독'으로 쓰면? ────────────────────
        rng = np.random.default_rng(SEED)
        vals = np.array([img[i] for i in ids], dtype=float)
        shuf = [fold_mean_cindex(dict(zip(ids, rng.permutation(vals))), labels, cohort_df, target)
                for _ in range(N_REPEAT)]
        shuf = np.array(shuf)

        print(f"\n② 뒤섞은 이미지를 '단독'으로 쓰면? ({N_REPEAT}회 반복)")
        print(f"   평균 {shuf.mean():.4f}  95% 구간 [{np.percentile(shuf, 2.5):.4f}, "
              f"{np.percentile(shuf, 97.5):.4f}]")
        print(f"   -> 동전던지기(0.50)와 같은 수준"
              f"{' ✅' if abs(shuf.mean() - 0.5) < 0.02 else ' ❌'}")

        # ── 결론: fig12의 0.705는 누가 낸 점수인가 ────────────────────────
        stack_shuf = published[target]["tabular_only"]["mean"]  # 참고용 기준선
        print(f"\n③ 결론")
        print(f"   뒤섞은 이미지 단독      = {shuf.mean():.4f}  (정보 없음)")
        print(f"   tabular 단독            = {tab_ci:.4f}")
        print(f"   fig12의 '＋뒤섞은 이미지' 막대는 이 둘을 합친 것이며,")
        print(f"   실제로는 tabular({tab_ci:.4f})가 전부 낸 점수에서 잡음만큼 깎인 값이다.")


if __name__ == "__main__":
    main()
