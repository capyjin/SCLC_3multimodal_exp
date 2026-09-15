# -*- coding: utf-8 -*-
"""[실험6-b] RadBERT 에 맞는 융합 방식 찾기 — late fusion 에서 인코더만 갈아 끼운다.

동기 — 관측된 모순:
  판독지 **단독**으로는 RadBERT 가 TF-IDF 를 확실히 이긴다
      OS 0.6685 vs 0.6268, PFS 0.6354 vs 0.6094 (10 fold Δ=+0.034, p=0.027)
  그런데 임상변수와 **concat(조기융합)** 하면 우위가 사라진다 (Δ=-0.0006, p=0.951).
  RadBERT 실험은 그때까지 전부 concat 이었다. 그런데 이 프로젝트의 OS 최고 모델은
  late fusion 이다. RadBERT 는 "단독 모델로서" 더 좋으므로, 단독 성능을 그대로 쓰는
  late fusion 과 궁합이 맞을 가능성이 있는데 한 번도 시험된 적이 없었다.

이 스크립트가 재는 것:
  축   clin_only        임상 단독
       report_tfidf     판독지 단독 (TF-IDF)
       report_radbert   판독지 단독 (RadBERT)
  결합 late_clin+tfidf   / late_clin+radbert

  핵심 비교는 "같은 late fusion 틀에서 인코더만 바꿨을 때 RadBERT 가 이기는가"다.
  단독에서의 +0.034 가 여기서 살아남는다면, concat 이 RadBERT 를 못 살렸다는 뜻이 된다.

누수 방지:
  - 각 단독 모델의 OOF 위험점수는 그 환자가 test 였던 fold 의 모델이 낸 값이다.
  - CoxPH 결합은 fold 마다 train 환자의 OOF 점수로만 fit 하고 test 에 적용한다
    (``sclc.late_fusion.combine_two`` — 검증된 코드를 그대로 재사용).
  - ⚠️ nested CV 가 아니라는 알려진 한계는 sclc/experiments/fusion.py 참고.

Run:  python experiments/실험6_판독지_인코더_비교/exp_encoder_fusion.py --target os
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from sclc.encoders import build_encoder
from sclc.experiments.fusion import Axis, Combination, LateFusionExperiment

# concat 기준선 (clin_report, TF-IDF). 결합 결과를 이 값과 대조한다.
CONCAT_BASELINE = {"os": 0.7076, "pfs": 0.6678}


class EncoderLateFusion(LateFusionExperiment):
    """임상 단독 + 판독지 단독을 CoxPH 로 묶고, 판독지 인코더만 바꿔 비교한다."""

    name = "encoder_fusion"
    default_out_dir = "radbert_fusion"   # 기존 산출물 폴더를 그대로 이어 쓴다
    log_tag = "AXIS"
    # 축마다 모델 조합이 다르므로 실험 전체를 대표하는 단일 config 가 없다.
    model_config = "mixed(clin_only|report_only)"

    def build_axes(self) -> list[Axis]:
        return [
            Axis("clin_only", "임상 단독", model_config="clin_only"),
            Axis("report_tfidf", "판독지 단독 (TF-IDF 400)",
                 model_config="report_only", encoder=build_encoder("tfidf")),
            Axis("report_radbert", "판독지 단독 (RadBERT, 한글 ko2en, 축소 없음)",
                 model_config="report_only", encoder=build_encoder("radbert")),
        ]

    def build_combinations(self) -> list[Combination]:
        return [
            Combination("late_clin+tfidf", "clin_only", "report_tfidf",
                        "임상 + 판독지(TF-IDF)"),
            Combination("late_clin+radbert", "clin_only", "report_radbert",
                        "임상 + 판독지(RadBERT)"),
        ]

    def reference_points(self) -> dict[str, float]:
        return {"concat 기준선 (clin_report, TF-IDF)": CONCAT_BASELINE[self.target]}


if __name__ == "__main__":
    EncoderLateFusion.main()
