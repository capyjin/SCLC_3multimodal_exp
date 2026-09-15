# -*- coding: utf-8 -*-
"""[실험13] late fusion 2-way 에서 **어느 두 모달리티를 묶는가**가 성능을 가르는가?

채택 모델(2-way)은 세 모달리티를 **다 쓰되** 그중 둘만 한 모델로 묶어 학습하고
나머지 하나는 따로 학습해 CoxPH 로 결합한다. 묶는 조합은 셋뿐이다:

    G1  concat[임상+판독지]  (+)  영상 단독      <- 채택.  OS 0.7224 / PFS 0.6470
    G2  concat[임상+영상]    (+)  판독지 단독    <- 미측정
    G3  concat[판독지+영상]  (+)  임상 단독      <- 미측정

[왜 이 실험이 필요한가]
  논문 리뷰어의 예상 질문은 "왜 하필 임상과 판독지를 묶었나"이고, 그 답이 되는
  건 G1 vs G2 vs G3 다. 세 셀은 **모달리티 집합(3개)도 같고 late fusion 구조도
  같다** — 바뀌는 건 묶음 하나뿐이라서 성능 차이를 묶음 선택에 귀속시킬 수 있다.
  "영상을 뺐다"는 오해도 같이 차단된다.

  기존 표(RESULTS_TABLE_final.md 표3-1)의 (a) early concat = 셋 다 묶음,
  (b) late 3-way = 아무것도 안 묶음 이 이 축의 양 끝이고, G1~G3 가 그 사이를
  채운다. 이 스크립트는 미측정인 **G2·G3 만** 잰다.

[조건] 표3-1 과 같게 맞춘다 — RadBERT · brain_meta 수정 후 · seed 42 · bs32/ep60.
  캐시(``outputs/late_fusion_B/oof_*.json``)는 쓰지 않는다. G2·G3 는 영상이
  묶음 **안에서** 함께 학습되므로 재학습이 불가피하고, 단일 축도 이번 실행에서
  새로 학습한다.

  ⚠️ 영상의 학습 조건이 G1 과 G2·G3 사이에 다른 건 설계상 불가피하다.
  G1 에서 영상은 **독립 축**이라 영상 arm 표준 조건(bs16/ep30)으로 돌지만,
  G2·G3 에서 영상은 묶음 안에 있어 묶음 축의 조건(bs32/ep60)을 따른다.
  "따로 학습이냐 같이 학습이냐"가 곧 이 실험이 재는 것이므로, 이건 교란이
  아니라 처치 그 자체다. (실험3의 ``clin_image`` 도 같은 bs32/ep60 이다.)

[누수 방지] 새로 짠 곳이 없다 — 전부 검증된 경로를 상속해서 쓴다.
  · fold별 인코더 적합: ``sclc.encoders`` 의 ``ReportEncoder`` 계약
    (train fold 환자로만 fit). 학습 전에 ``preflight`` 가 fold 마다 실제
    행렬을 만들어 폭·분산을 확인한다.
  · 임상 스케일러/TF-IDF: ``features.build_fold_multimodal_tabular`` (train-only).
  · CoxPH 결합: ``sclc.late_fusion.combine_risk_scores`` — fold 마다 train
    환자의 OOF 점수로만 적합하고 test 에 적용.
  · ``brain_meta`` 기저시점 누수 수정은 기본값(``fix_brain_meta=True``)으로 적용.
  · 남아 있는 알려진 한계(nested CV 가 아님)는 G1 과 **동일**하므로 세 셀의
    비교는 공정하다. 자세한 설명은 ``sclc.experiments.fusion`` docstring.

[내장 재현 대조] ``clin_only`` 축은 실험12가 같은 조건으로 이미 측정했다
  (OS 0.6388 / PFS 0.6223). 이 축이 그 값을 재현하지 못하면 학습 경로가
  달라졌다는 뜻이므로 이번 실행의 결과를 믿으면 안 된다 — ``check_baseline``
  이 자동으로 대조하고 경고한다.

Run:  python experiments/실험13_융합묶음_조합비교/exp_grouping.py --target os
      python experiments/실험13_융합묶음_조합비교/exp_grouping.py --target pfs
      python experiments/실험13_융합묶음_조합비교/exp_grouping.py --preflight_only
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from sclc.encoders import build_encoder
from sclc.experiments.fusion import Axis, Combination, LateFusionExperiment
from sclc.utils import cli

#: G1 (채택 2-way) — RESULTS_TABLE_final.md 표3-1 (c). 같은 조건으로 측정된 값이라
#: 이번 실행의 G2·G3 와 직접 비교된다.
KNOWN_G1 = {"os": 0.7224, "pfs": 0.6470}

#: 표3-1 의 양 끝 — (a) 셋 다 묶음 / (b) 아무것도 안 묶음. 요약표에 같이 찍는다.
KNOWN_EARLY_CONCAT = {"os": 0.6879, "pfs": 0.6555}
KNOWN_LATE_3WAY = {"os": 0.7045, "pfs": 0.6545}

#: 실험12가 같은 조건(RadBERT/bs32/ep60/seed42)으로 측정한 단일 축 — 재현 대조용.
CLIN_ONLY_KNOWN = {"os": 0.6387690795698175, "pfs": 0.6222844400296995}
CLIN_ONLY_KNOWN_FOLDS = {"os": [0.6291, 0.6558, 0.6544, 0.5605, 0.694],
                         "pfs": [0.6146, 0.6115, 0.6429, 0.5661, 0.6764]}


class GroupingFusion(LateFusionExperiment):
    """묶음 조합만 바꿔 가며 2-way late fusion 을 측정한다."""

    name = "fusion_grouping"
    default_out_dir = "fusion_grouping"
    model_config = "grouping(clin_image|report_image + 남은 단일 축)"
    baseline = "clin_only"          # 실험12 재현 대조 + 요약표의 delta 기준

    def __init__(self, args):
        super().__init__(args)
        self._preflighted = False

    @property
    def encoder_spec(self) -> str:
        # 속성이 아니라 프로퍼티인 이유: 부모 ``__init__`` 이 ``settings()`` 를
        # 먼저 부르므로, 거기서 이미 읽을 수 있어야 한다.
        return self.args.encoder

    @classmethod
    def add_arguments(cls, ap) -> None:
        ap.add_argument("--encoder", default="radbert",
                        help="판독지 인코더 (기본 %(default)s = 표3-1 의 조건 통일 설정). "
                             "tfidf 로 바꾸면 내장 재현 대조는 건너뛴다.")
        ap.add_argument("--axes", type=cli.comma_list, default=None,
                        help="이번에 돌릴 축만 골라 지정 (기본: 전부). "
                             "중단된 실행을 이어 붙일 때 쓴다.")

    # ── 축 ───────────────────────────────────────────────────────────────
    def build_axes(self) -> list[Axis]:
        """네 축. 실험12와 같이 **모든 축에 인코더를 넘긴다**.

        판독지를 안 쓰는 축(``clin_only``/``clin_image``)에도 넘기는 건 낭비처럼
        보이지만, 실험12의 ``clin_only`` 를 비트 단위로 재현하려면 호출 경로가
        같아야 한다. 모델이 report 블록을 안 보므로(``use_report=False``) 수치에
        영향은 없다. 인코더는 축마다 새로 만든다 — ``leakage_audit`` 기록이 축별로
        섞이지 않게 하려는 것이고, RadBERT 임베딩은 디스크 캐시라 비용은 없다.
        """
        def enc():
            return build_encoder(self.encoder_spec)

        return [
            # 재현 대조용 축을 맨 앞에 둔다 — 학습 경로가 틀어졌다면 영상 축
            # 두 개(가장 비싼 부분)를 돌리기 전에 경고가 뜬다.
            Axis("clin_only", "임상 단독 (G3 의 남은 축)",
                 model_config="clin_only", encoder=enc()),
            Axis("report_only", f"판독지 단독 ({self.encoder_spec}) (G2 의 남은 축)",
                 model_config="report_only", encoder=enc()),
            Axis("clin_image", "concat[임상+영상] (G2 의 묶음 축)",
                 model_config="clin_image", encoder=enc()),
            Axis("report_image", f"concat[판독지({self.encoder_spec})+영상] (G3 의 묶음 축)",
                 model_config="report_image", encoder=enc()),
        ]

    def variants(self) -> list[str]:
        names = list(self.axes)
        if self.args.axes is None:
            return names
        return cli.check_names(self.args.axes, names)

    # ── 결합 ─────────────────────────────────────────────────────────────
    def build_combinations(self) -> list[Combination]:
        return [
            Combination("G2_clin+img|report", "clin_image", "report_only",
                        "concat[임상+영상] + 판독지 단독",
                        coef_names=("risk_clin_image", "risk_report")),
            Combination("G3_report+img|clin", "report_image", "clin_only",
                        "concat[판독지+영상] + 임상 단독",
                        coef_names=("risk_report_image", "risk_clinical")),
        ]

    # ── 학습 전 점검 ─────────────────────────────────────────────────────
    def preflight(self, name: str) -> None:
        """인코더의 fold-safe 점검을 실행당 1회 돌린다.

        축 4개가 **같은 설정의 인코더**를 쓰므로 점검 결과도 같다. 그래도
        건너뛰지 않고 반드시 한 번은 돌린다 — 몇 초에 끝나고, 몇 시간짜리
        실행이 잘못된 feature 위에서 도는 걸 막는 유일한 지점이다.
        """
        if self._preflighted:
            return
        self._preflighted = True
        self.axes[name].encoder.preflight(self.corpus, self.log)

    # ── 재현 대조 / 참고값 ───────────────────────────────────────────────
    @property
    def known_baseline(self) -> dict:
        # 알려진 값은 RadBERT 조건의 것이다. 인코더를 바꾸면 대조가 무의미하다.
        return CLIN_ONLY_KNOWN if self.encoder_spec == "radbert" else {}

    @property
    def known_baseline_folds(self) -> dict:
        return CLIN_ONLY_KNOWN_FOLDS if self.encoder_spec == "radbert" else {}

    def settings(self) -> dict:
        return {**super().settings(), "encoder": self.encoder_spec}

    def reference_points(self) -> dict[str, float]:
        t = self.target
        return {
            "G1 채택 2-way  concat[임상+판독지] + 영상": KNOWN_G1[t],
            "(a) early concat  셋 다 묶음":              KNOWN_EARLY_CONCAT[t],
            "(b) late 3-way    아무것도 안 묶음":         KNOWN_LATE_3WAY[t],
        }


if __name__ == "__main__":
    GroupingFusion.main()
