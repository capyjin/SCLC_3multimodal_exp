# -*- coding: utf-8 -*-
"""[실험15] 채택 모델을 **fold-safe 결측 대치**로 다시 학습하면 성능이 바뀌는가?

채택 모델(late fusion 2-way, RadBERT)의 임상 입력 중 연속형 7개는 238명 전체
(정확히는 321명 임상 데이터셋 전체) 중앙값으로 미리 채워져 있다. 그 중앙값에
test fold 환자가 들어가 있으므로 논문 리뷰어가 누수로 지적할 수 있다.
이 실험은 **대치 방식 하나만** 바꿔 채택 모델을 다시 잰다.

    tab_orig      concat[임상+판독지(RadBERT)]  CSV 그대로 (전체 중앙값 대치)  <- 재현 대조
    tab_foldsafe  concat[임상+판독지(RadBERT)]  7열을 train fold 중앙값으로 재대치
    image         SimpleCNN 영상 축 — outputs/late_fusion_B/oof_{target}.json 재사용
                  (영상 축은 임상변수를 안 쓰므로 대치 방식과 무관, 재학습 불필요)

    late_orig     tab_orig     + image  -> 채택 모델 재현 (OS 0.7224 / PFS 0.6470)
    late_foldsafe tab_foldsafe + image  -> 이번 실험의 답

[고정] RadBERT(ko2en, raw 768) · brain_meta 수정 · seed · bs32/ep60 · 분할 파일.
[불가피한 차이] fold-safe 7열이 임상 블록 **뒤쪽**으로 옮겨 붙는다(폭 21 은 동일).
  열 순서가 바뀌면 가중치 초기값이 다른 열에 대응하므로, 대치와 무관한 작은 흔들림이
  섞일 수 있다. 그 크기는 ``--seed`` 를 바꿔 여러 번 돌려 가늠한다.

Run:  python experiments/실험15_결측대치_foldsafe_최종모델/exp_foldsafe_final.py --target os
      python experiments/실험15_결측대치_foldsafe_최종모델/exp_foldsafe_final.py --target pfs
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json

import numpy as np

import foldsafe_clinical as fsc
from sclc import paths
from sclc.encoders import build_encoder
from sclc.evaluation import fold_cindices
from sclc.experiments.fusion import Axis, Combination, LateFusionExperiment
from sclc.train import fold_plan

#: 채택 모델 수치 (MODEL_SUMMARY.md §2-1·§3-2, RadBERT). 재현 대조용.
KNOWN_TAB = {"os": 0.7152524737017621, "pfs": 0.6455753258335233}
KNOWN_TAB_FOLDS = {"os": [0.705499, 0.760668, 0.6875, 0.719733, 0.702863],
                   "pfs": [0.645307, 0.669065, 0.664557, 0.621368, 0.62758]}
KNOWN_LATE = {"os": 0.7224, "pfs": 0.6470}


class FoldSafeFinal(LateFusionExperiment):
    """대치 방식만 바꿔 채택 late fusion 모델을 다시 잰다."""

    name = "foldsafe_final"
    default_out_dir = "foldsafe_final"
    model_config = "clin_report + image_only"
    baseline = "tab_orig"
    known_baseline = KNOWN_TAB
    known_baseline_folds = KNOWN_TAB_FOLDS

    def __init__(self, args):
        super().__init__(args)
        self.impute_audit: list = []
        self._raw = None

    @property
    def raw(self):
        if self._raw is None:
            self._raw = fsc.load_raw_values()
        return self._raw

    def build_axes(self) -> list[Axis]:
        return [
            Axis("tab_orig", "concat[임상+판독지(RadBERT)] — CSV 전체 중앙값 대치 (채택 모델)",
                 model_config="clin_report", encoder=build_encoder("radbert")),
            Axis("tab_foldsafe", "concat[임상+판독지(RadBERT)] — 7열 train fold 중앙값 재대치",
                 model_config="clin_report", encoder=build_encoder("radbert")),
            Axis("image", "영상 단독 SimpleCNN (late_fusion_B OOF 재사용)"),
        ]

    def build_combinations(self) -> list[Combination]:
        return [
            Combination("late_orig", "tab_orig", "image", "채택 모델 재현"),
            Combination("late_foldsafe", "tab_foldsafe", "image", "fold-safe 대치"),
        ]

    def evaluator_kwargs(self, name: str) -> dict:
        kwargs = super().evaluator_kwargs(name)
        if name == "tab_foldsafe":
            kwargs["clinical_columns_fn"] = fsc.reduced_clinical_columns
            kwargs["extra_numeric_fn"] = fsc.make_extra_numeric_fn(self.raw, self.impute_audit)
        return kwargs

    def extra_record(self, name: str) -> dict:
        if name == "image":
            return {"source": self.image_cache_path()}
        rec = super().extra_record(name)
        if name == "tab_foldsafe":
            rec["impute_audit"] = list(self.impute_audit)
        return rec

    def image_cache_path(self) -> str:
        return paths.outputs("late_fusion_B", f"oof_{self.target}.json")

    def run_variant(self, name: str) -> dict:
        if name != "image":
            return super().run_variant(name)
        with open(self.image_cache_path(), encoding="utf-8") as fh:
            risk = {int(k): float(v) for k, v in json.load(fh)["image"].items()}
        self.oof[name] = risk
        labels = self.cohort_df.drop_duplicates("research_id").set_index("research_id")
        cis = fold_cindices(risk, labels, fold_plan(self.cohort_df), self.target)
        return {"name": name, "desc": self.describe(name),
                "mean": float(np.mean(cis)), "std": float(np.std(cis)),
                "folds": [round(float(c), 4) for c in cis], "train_val_gap_mean": None,
                **self.extra_record(name)}

    def reference_points(self) -> dict:
        return {"채택 late fusion (MODEL_SUMMARY)": KNOWN_LATE[self.target]}


if __name__ == "__main__":
    FoldSafeFinal.main()
