# -*- coding: utf-8 -*-
"""[실험6-c] 최고 모델의 tabular 축 인코더만 RadBERT 로 바꾸면 최고 기록을 넘는가?

현재 최고: late fusion( concat[임상+판독지(TF-IDF)] , 영상 SimpleCNN ) = OS 0.7221.
concat 단계에서는 이미 RadBERT 가 앞선다(0.7153 vs 0.7076) — 그럼 영상까지 얹으면
최고 기록을 넘는가? 이 조합은 미시험이었다.

측정:
  축   tab_tfidf         concat[임상+판독지(TF-IDF)]
       tab_radbert       concat[임상+판독지(RadBERT)]
       image_simplecnn   영상 단독 SimpleCNN (bs16/ep30, 영상 arm 의 표준 조건)
  결합 late_tab_tfidf+img / late_tab_radbert+img

  두 결합 결과의 차이가 곧 "인코더 교체의 순효과"다.

누수 방지: 검증된 ``sclc.fusion_stack.combine_two`` 를 그대로 쓴다
  (fold 마다 train 환자의 OOF 점수로만 CoxPH 적합).

Run:  python experiments/실험6_판독지_인코더_비교/exp_encoder_trimodal.py --target os
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from sclc.encoders import build_encoder
from sclc.experiments.fusion import Axis, Combination, LateFusionExperiment
from sclc.fusion_stack import get_image_oof_simplecnn

# 현재 최고 (late fusion + SimpleCNN). 결합 결과를 이 값과 대조한다.
KNOWN_BEST = {"os": 0.7221, "pfs": 0.6678}


class EncoderTrimodalFusion(LateFusionExperiment):
    """tabular 축(임상+판독지)을 인코더 2종으로 학습해 영상 축과 결합한다."""

    name = "encoder_trimodal"
    default_out_dir = "radbert_full"     # 기존 산출물 폴더를 그대로 이어 쓴다
    model_config = "clin_report + image_only"

    @classmethod
    def add_arguments(cls, ap) -> None:
        ap.add_argument("--image_epochs", type=int, default=30,
                        help="영상 축 에폭 (기본 %(default)s = 영상 arm 의 표준 조건). "
                             "tabular 축의 --epochs 와 다른 게 정상이다.")
        ap.add_argument("--image_batch_size", type=int, default=16,
                        help="영상 축 배치 크기 (기본 %(default)s = 영상 arm 의 표준 조건).")

    def build_axes(self) -> list[Axis]:
        return [
            Axis("tab_tfidf", "concat[임상+판독지(TF-IDF 400)]",
                 model_config="clin_report", encoder=build_encoder("tfidf")),
            Axis("tab_radbert", "concat[임상+판독지(RadBERT, 한글 ko2en, 축소 없음)]",
                 model_config="clin_report", encoder=build_encoder("radbert")),
            # 영상 축은 TrimodalEvaluator 가 아니라 ImageOnlyEvaluator 를 쓴다.
            # 기본 out_dir(outputs/late_fusion_B)에 체크포인트를 남기므로 기존
            # late fusion 실행과 같은 영상 모델을 그대로 재사용한다.
            Axis("image_simplecnn", "영상 단독 SimpleCNN",
                 runner=lambda: get_image_oof_simplecnn(
                     self.target, epochs=self.args.image_epochs,
                     batch_size=self.args.image_batch_size, seed=self.seed)),
        ]

    def build_combinations(self) -> list[Combination]:
        return [
            Combination("late_tab_tfidf+img", "tab_tfidf", "image_simplecnn",
                        "tabular(TF-IDF) + 영상"),
            Combination("late_tab_radbert+img", "tab_radbert", "image_simplecnn",
                        "tabular(RadBERT) + 영상"),
        ]

    def reference_points(self) -> dict[str, float]:
        return {"현재 최고 (late fusion + SimpleCNN)": KNOWN_BEST[self.target]}


if __name__ == "__main__":
    EncoderTrimodalFusion.main()
