# -*- coding: utf-8 -*-
"""진단용(공식 결과 아님) -- M0-tuned의 inner-CV 후보 선택이 "이 표본을 하필 이렇게
쪼갠 우연" 때문인지, 진짜 신호인지 확인.

exp_gate_m0_tuned.py::inner_cv_select 는 KFold(random_state=SELECTION_SEED=42) 로
171명을 5조각(모의고사)으로 쪼개는 방식을 **딱 1가지**로만 고정해서 썼다. 1등과
2등의 모의고사 점수 차이가 fold마다 0.0001~0.0041 밖에 안 났다(사용자 확인 요청으로
outputs/moe_gate_m0_tuned/results.json 의 inner_cv_scores 를 직접 까본 결과) --
이 근소한 차이가 "진짜 이 후보가 더 나아서"인지 "모의고사를 하필 이렇게 쪼갠
우연"인지 구분이 안 된다.

그래서 쪼개는 방식(KFold random_state)과 게이트 학습 시드를 42/43/44/45 로 4가지
다르게 반복해서, **같은 fold에서 1등이 매번 같은 후보로 나오는지** 본다.
  - 매번 같으면 -> 진짜 신호 (선택이 강건함)
  - 계속 바뀌면 -> 처음 의심대로 노이즈 (탐색면이 평평함, 한 번 더 확인됨)

exp_gate_m0_tuned.py 를 건드리지 않는다(공식 결과 재현성 보호) -- CANDIDATES/
build_tensors_for_candidate/train_and_eval 만 그대로 가져다 쓰고, KFold의
random_state 를 42 고정 대신 인자로 받는 얇은 버전만 여기서 새로 만든다.

Run: python 실험11_MoE/diag_inner_cv_seed_stability.py --targets os,pfs
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json

import numpy as np
import torch
from sklearn.model_selection import KFold

from sclc import expert_embeddings as ee
from exp_gate_m0_tuned import CANDIDATES, INNER_FOLDS, build_tensors_for_candidate, train_and_eval

OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "diag_inner_cv_seed_stability")
SELECTION_SEEDS = (42, 43, 44, 45)  # 42 = 공식 결과와 동일 조건(재현 검증용), 43~45 = 신규


def inner_cv_select_seeded(fold_data: dict, device, selection_seed: int):
    """exp_gate_m0_tuned.inner_cv_select 와 완전히 같은 절차, KFold random_state와
    train_and_eval 의 seed만 인자로 받는다(그 파일은 42로 고정돼 있음)."""
    train = fold_data["train"]
    n = len(train["duration"])
    kf = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=selection_seed)

    scores = {cand: [] for cand in CANDIDATES}
    for tr_idx, va_idx in kf.split(np.arange(n)):
        inner_train = {k: v[tr_idx] for k, v in train.items() if k != "rid"}
        inner_val = {k: v[va_idx] for k, v in train.items() if k != "rid"}
        inner_fold_data = {"train": inner_train, "test": inner_val}
        for gate_input, wd in CANDIDATES:
            tensors = build_tensors_for_candidate(inner_fold_data, gate_input)
            ci = train_and_eval(tensors, wd, seed=selection_seed, device=device)
            scores[(gate_input, wd)].append(ci)

    mean_scores = {c: float(np.mean(v)) for c, v in scores.items()}
    best = max(mean_scores, key=mean_scores.get)
    return best, mean_scores


def run_target(target: str, max_folds=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = ee.load_gate_inputs(target, max_folds=max_folds)

    fold_results = []
    for fold, fold_data in data.items():
        print(f"\n[{target}] fold {fold}:")
        per_seed = {}
        for seed in SELECTION_SEEDS:
            best, scores = inner_cv_select_seeded(fold_data, device, seed)
            key = f"{best[0]}_wd{best[1]:g}"
            per_seed[seed] = {"selected": key, "score": scores[best]}
            print(f"    seed={seed}: 1등={key:16s} (모의고사 점수={scores[best]:.4f})")

        winners = {v["selected"] for v in per_seed.values()}
        agree = len(winners) == 1
        print(f"    -> {'일치 (강건)' if agree else f'불일치 ({len(winners)}가지 다른 1등)'}")
        fold_results.append({"fold": fold, "per_seed": per_seed, "agree": agree,
                             "n_distinct_winners": len(winners)})

    n_agree = sum(1 for f in fold_results if f["agree"])
    print(f"\n[{target}] {len(fold_results)}개 fold 중 {n_agree}개가 seed 4개 전부 같은 후보 선택")
    return {"target": target, "folds": fold_results, "n_agree": n_agree, "n_folds": len(fold_results)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="os,pfs")
    ap.add_argument("--max_folds", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    all_results = {}
    for target in [t.strip() for t in args.targets.split(",") if t.strip()]:
        all_results[target] = run_target(target, max_folds=args.max_folds)

    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 70}\n요약 (진단용, 공식 결과 아님)\n{'=' * 70}")
    for target, r in all_results.items():
        print(f"[{target.upper()}] {r['n_agree']}/{r['n_folds']} fold 에서 seed 4개 전부 동일 후보 선택")
    print(f"\n-> {os.path.join(OUT_DIR, 'results.json')}")


if __name__ == "__main__":
    main()
