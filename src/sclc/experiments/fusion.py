# -*- coding: utf-8 -*-
"""late fusion 형태 실험의 부모 클래스 — "축을 각각 학습하고 CoxPH 로 묶는다".

[골격]
  1. **축(axis)** 을 하나씩 5-fold 로 학습해 OOF 위험점수를 얻는다
     (축 = 임상 단독 / 판독지 단독 / 영상 단독 / concat[임상+판독지] ...).
  2. 축 두 개를 골라 fold 마다 **train 환자의 OOF 점수로만** CoxPH 를 적합하고
     test 에 적용해 결합 성능을 잰다 (``fusion_stack.combine_two``).

  1번은 ``BaseExperiment`` 의 항목 루프가 그대로 처리한다 — 축이 곧 항목이다.
  이 클래스가 더하는 건 2번(결합 단계)과 축별로 다른 평가기를 쓰는 길뿐이다.

⚠️ 알려진 한계 (기존 late fusion 결과와 동일 조건이라 그대로 유지한다):
  이 방식은 nested CV 가 아니다. 메타학습기(CoxPH)의 입력이 test fold 를 본
  모델에서 나온다. 계수가 2개뿐이라 편향은 작지만 0은 아니다. 기존 결과
  (OS 0.7221)와 비교 가능하게 하려고 같은 조건을 유지한다.
"""
import os
import sys
from abc import abstractmethod
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sclc import cohort
from sclc.encoders.base import ReportEncoder
from sclc.experiments.base import BaseExperiment, ReportCorpusMixin
from sclc.fusion_stack import combine_two
from sclc.model import MODALITY_CONFIGS, make_model_factory
from sclc.utils.logging import banner, section
from sclc.utils.summary import Table


@dataclass
class Axis:
    """결합에 들어갈 축 하나.

    ``runner`` 가 있으면 그걸로 학습한다 (영상 축처럼 ``TrimodalEvaluator`` 가
    아닌 평가기를 쓰는 경우). 없으면 ``model_config`` + ``encoder`` 로
    ``TrimodalEvaluator`` 를 만든다.
    """
    tag: str
    desc: str
    model_config: str | None = None
    encoder: ReportEncoder | None = None
    runner: object | None = field(default=None, repr=False)


@dataclass
class Combination:
    """CoxPH 로 묶을 축 두 개."""
    name: str
    left: str
    right: str
    desc: str = ""


class LateFusionExperiment(ReportCorpusMixin, BaseExperiment):
    item_key = "axes"
    log_tag = "AXIS"
    #: 결과 JSON 에서 결합 결과가 들어갈 키
    combination_key = "combinations"

    def __init__(self, args):
        super().__init__(args)
        self._axes: dict[str, Axis] | None = None
        self.combined: dict[str, dict] = {}
        self._cohort_df = None

    # ── 서브클래스가 채우는 부분 ──────────────────────────────────────────
    @abstractmethod
    def build_axes(self) -> list[Axis]:
        """이 실험이 학습할 축들."""

    @abstractmethod
    def build_combinations(self) -> list[Combination]:
        """결합할 축 쌍들."""

    # ── 공통 구현 ────────────────────────────────────────────────────────
    @property
    def axes(self) -> dict[str, Axis]:
        if self._axes is None:
            self._axes = {a.tag: a for a in self.build_axes()}
        return self._axes

    @property
    def flags(self) -> dict:
        """실험 전체를 대표하는 모달리티 조합 — late fusion 에는 **없다**.

        축마다 모델 조합이 다르므로(``clin_only`` + ``report_only`` 처럼)
        서브클래스는 ``model_config`` 에 ``"mixed(clin_only|report_only)"`` 같은
        **설명용 라벨**을 넣는다. 그 라벨은 ``MODALITY_CONFIGS`` 의 키가 아니라서
        부모의 ``flags`` 가 KeyError 로 죽었다 (실험6-b/6-c 가 실행 즉시 죽는
        원인이었다). 이 값은 로그 헤더와 결과 JSON 의 기록용일 뿐이고 실제
        모델은 축별 ``model_factory`` 가 만들므로, 레지스트리에 없으면 빈 dict 를
        돌려준다. 축별 조합은 ``extra_record`` 가 축마다 따로 남긴다.
        """
        return MODALITY_CONFIGS.get(self.model_config, {})

    @property
    def cohort_df(self):
        if self._cohort_df is None:
            self._cohort_df = cohort.load_trimodal_cohort()
        return self._cohort_df

    def variants(self) -> list[str]:
        return list(self.axes)

    def describe(self, name: str) -> str:
        return self.axes[name].desc

    def model_factory(self, name: str):
        axis = self.axes[name]
        return make_model_factory(MODALITY_CONFIGS[axis.model_config])

    def evaluator_kwargs(self, name: str) -> dict:
        axis = self.axes[name]
        if axis.encoder is None:
            return {}
        return axis.encoder.evaluator_kwargs(self.corpus)

    def extra_record(self, name: str) -> dict:
        axis = self.axes[name]
        rec = {"model_config": axis.model_config}
        if axis.encoder is not None:
            rec["encoder"] = axis.encoder.config()
            if axis.encoder.audit:
                rec["leakage_audit"] = list(axis.encoder.audit)
        return rec

    def run_variant(self, name: str) -> dict:
        axis = self.axes[name]
        if axis.runner is not None:
            return self.record_from(name, axis.runner())
        return super().run_variant(name)

    # ── 결합 단계 ────────────────────────────────────────────────────────
    def after_variants(self) -> None:
        for combo in self.build_combinations():
            missing = [t for t in (combo.left, combo.right) if t not in self.oof]
            if missing:
                self.log.warning(f"[COMBINE] {combo.name}: 축 {missing} 의 OOF 가 없어 건너뛴다 "
                                 "(그 축을 이번에 안 돌렸다면 정상).")
                continue
            banner(self.log, f"COMBINE {combo.name}  target={self.target}")
            out = combine_two(self.cohort_df, self.target,
                              self.oof[combo.left], self.oof[combo.right])
            self.combined[combo.name] = {
                "name": combo.name, "desc": combo.desc,
                "axes": [combo.left, combo.right],
                "mean": out["mean"], "std": out["std"],
                "folds": [round(float(c), 4) for c in out["fold_cindex"]],
                "coefs_per_fold": out.get("coefs_per_fold"),
                "mean_coef": out.get("mean_coef"),
            }
            self.log.info(f"[COMBINE] {combo.name} {self.target}: {out['mean']:.4f} "
                          f"+/- {out['std']:.4f}  coef({combo.left},{combo.right})="
                          f"{out.get('mean_coef')}")
        self.store.update_header(**{self.combination_key: self.combined})
        self.store.save()

    # ── 요약표: 축 표 + 결합 표 ──────────────────────────────────────────
    def summarize(self) -> None:
        super().summarize()
        if not self.combined:
            return
        section(self.log, f"COMBINED (CoxPH late fusion, target={self.target})")
        table = Table([("combination", -26), ("mean", 9), ("std", 9),
                       ("mean_coef", -22), ("folds", -2)])
        for rec in self.combined.values():
            table.add(rec["name"], f"{rec['mean']:.4f}", f"{rec['std']:.4f}",
                      str(rec.get("mean_coef")), f"  {rec['folds']}")
        table.emit(self.log)
        for label, known in self.reference_points().items():
            self.log.info(f"  참고: {label} = {known:.4f}")

    def reference_points(self) -> dict[str, float]:
        """요약 끝에 같이 찍을 알려진 기준값 {설명: 값}."""
        return {}
