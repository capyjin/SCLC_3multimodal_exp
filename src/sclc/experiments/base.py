# -*- coding: utf-8 -*-
"""실험 파이프라인의 추상 부모 클래스.

[왜 이 클래스가 존재하는가]
  실험6의 세 스크립트(인코더 비교 · 텍스트 source · SUV 특징)를 나란히 놓으면
  **골격이 글자 단위로 같았다**:

      1. 인자 파싱 (--target/--epochs/--batch_size)  + 이름 검증
      2. 학습 전 통계 출력 (토큰 통계 / 텍스트 길이 / SUV 분포)
      3. 결과 JSON 열기 + "설정이 다르면 이어붙이지 않기" 20줄
      4. 항목마다: 구분선 -> preflight -> TrimodalEvaluator.run() ->
         mean/std/folds/과적합격차/fold_records/training_history 수집 -> 즉시 저장
      5. 기준선 항목이면 알려진 값과 재현 대조
      6. 요약표 (기준선 대비 delta)

  다른 건 4번의 "이 항목에서 유일하게 바뀌는 것" 한 줄뿐이었다
  (``text_encoder_fn=`` / ``text_source=`` / ``extra_numeric_fn=``).
  그 한 줄을 위해 200줄이 세 벌 있었다. 여기서는 골격을 한 번만 쓰고,
  서브클래스는 그 한 줄과 설명 문자열만 채운다.

[서브클래스가 채우는 것 — 최소 3개]
  ``variants()``           이번에 돌릴 항목 이름 목록
  ``describe(name)``       항목 한 줄 설명 (로그·결과 JSON 에 남는다)
  ``evaluator_kwargs(name)`` 이 항목에서만 달라지는 TrimodalEvaluator 인자

[선택적으로 덮어쓰는 것]
  ``add_arguments`` / ``prepare`` / ``preflight`` / ``extra_record`` /
  ``settings`` / ``summary_table`` / ``run_variant`` / ``after_variants``
"""
import os
import sys
from abc import ABC, abstractmethod

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse

import numpy as np

from sclc import paths
from sclc.metrics import train_val_gap
from sclc.model import MODALITY_CONFIGS, make_model_factory
from sclc.utils import cli
from sclc.utils.logging import banner, get_logger, section
from sclc.utils.results import ResultStore
from sclc.utils.summary import (FOLDS_COLUMN, N_FOLD_WILCOXON_NOTE, Table,
                                comparison_row)


class BaseExperiment(ABC):
    """"조건 하나만 바꿔 가며 5-fold 를 반복하고 기준선과 비교한다" 형태의 실험."""

    #: 로그/헤더에 찍히는 실험 이름
    name: str = "experiment"
    #: outputs/ 아래 기본 폴더 이름
    default_out_dir: str = "experiment"
    #: 결과 JSON 안에서 항목들이 들어갈 키 (arms / variants / steps ...)
    item_key: str = "items"
    #: 모델 조합 (sclc.model.MODALITY_CONFIGS). 실험 내내 고정한다.
    model_config: str = "clin_report"
    #: 재현 대조에 쓸 기준선 항목 이름 (없으면 None)
    baseline: str | None = None
    #: {target: 알려진 평균 C-index}. 기준선 항목이 이 값을 재현하는지 확인한다.
    known_baseline: dict = {}
    #: {target: 알려진 fold별 C-index}
    known_baseline_folds: dict = {}
    #: 요약표 접두어
    log_tag: str = "EXP"

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.target = args.target
        self.epochs = int(args.epochs)
        self.batch_size = int(args.batch_size)
        self.seed = int(getattr(args, "seed", cli.DEFAULT_SEED))
        self.out_dir = args.out_dir or paths.outputs(self.default_out_dir)
        os.makedirs(self.out_dir, exist_ok=True)
        self.log = get_logger(self.name, self.out_dir)
        self.results_path = os.path.join(self.out_dir, f"results_{self.target}.json")
        self.store = ResultStore(self.results_path, self.settings(),
                                 item_key=self.item_key, logger=self.log)
        #: 항목별 OOF 위험점수 {항목: {research_id: risk}} — late fusion 실험이 쓴다
        self.oof: dict[str, dict] = {}

    # ── 서브클래스가 채우는 부분 ──────────────────────────────────────────
    @classmethod
    def add_arguments(cls, ap: argparse.ArgumentParser) -> None:
        """실험 고유 인자. 공통 인자(--target 등)는 이미 붙어 있다."""

    @abstractmethod
    def variants(self) -> list[str]:
        """이번 실행에서 돌릴 항목 이름들 (순서가 곧 요약표 순서)."""

    @abstractmethod
    def describe(self, name: str) -> str:
        """항목 한 줄 설명."""

    @abstractmethod
    def evaluator_kwargs(self, name: str) -> dict:
        """이 항목에서만 달라지는 ``TrimodalEvaluator`` 인자.

        여기 안 적힌 건 전부 실험 내내 고정이다 — 그게 이 실험 설계의 핵심이다.
        """

    def prepare(self) -> None:
        """항목 루프 전 1회. 학습 전 통계 출력·전역 계산에 쓴다."""

    def preflight(self, name: str) -> None:
        """학습 없이 feature 를 실제로 만들어 보는 사전 점검 (기본은 없음)."""

    def extra_record(self, name: str) -> dict:
        """결과 JSON 의 항목 레코드에 더 넣을 것 (감사 기록, 통계 등)."""
        return {}

    def settings(self) -> dict:
        """"이 값들이 같아야 기존 결과 파일에 이어붙인다"는 판정 기준."""
        return {"target": self.target, "epochs": self.epochs,
                "batch_size": self.batch_size, "model_config": self.model_config}

    # ── 공통 구현 ────────────────────────────────────────────────────────
    @property
    def flags(self) -> dict:
        return MODALITY_CONFIGS[self.model_config]

    def model_factory(self, name: str):
        return make_model_factory(self.flags)

    def save_dir(self, name: str) -> str:
        """체크포인트 폴더. ``@`` 는 경로에 쓰기 껄끄러우므로 ``_`` 로 바꾼다."""
        return os.path.join(self.out_dir, f"{name.replace('@', '_')}_{self.target}")

    def build_evaluator(self, name: str):
        from sclc.train import TrimodalEvaluator   # torch 로딩이 느려서 여기서 import
        return TrimodalEvaluator(
            target=self.target, epochs=self.epochs, batch_size=self.batch_size,
            seed=self.seed, save_dir=self.save_dir(name),
            model_factory=self.model_factory(name),
            **self.evaluator_kwargs(name),
        )

    def run_variant(self, name: str) -> dict:
        """항목 하나를 학습·평가하고 결과 레코드를 만든다.

        축마다 평가기가 다른 실험(영상 축은 ``ImageOnlyEvaluator``)은 이 메서드를
        덮어쓴다 — 그래도 아래 ``record_from`` 은 그대로 재사용한다.
        """
        ev = self.build_evaluator(name).run()
        return self.record_from(name, ev)

    def record_from(self, name: str, ev) -> dict:
        """평가기 -> 결과 레코드. 지표의 정의를 여기 한 곳에 묶어 둔다
        (과적합 격차 계산이 4개 파일에 복사돼 있던 게 정리 전 상태다)."""
        cis = ev.c_indices
        gaps = train_val_gap(ev.training_history)
        if getattr(ev, "oof_predictions", None):
            from sclc.fusion_stack import oof_dict
            self.oof[name] = oof_dict(ev.oof_predictions)
        return {
            "name": name,
            "desc": self.describe(name),
            "mean": float(np.mean(cis)),
            "std": float(np.std(cis)),
            "folds": [round(float(c), 4) for c in cis],
            "train_val_gap_mean": float(np.mean(gaps)) if gaps else None,
            "fold_records": ev.fold_records,
            "training_history": ev.training_history,
            **self.extra_record(name),
        }

    def check_baseline(self, name: str, record: dict) -> None:
        """기준선 항목이 알려진 값을 재현했는지 확인한다.

        재현되지 않으면 **결과를 신뢰하면 안 된다** — 리팩터가 기본 경로 동작을
        바꿨다는 뜻이기 때문이다. 죽이지는 않고 크게 경고만 한다 (다른 항목의
        결과는 여전히 저장할 가치가 있으므로).
        """
        if name != self.baseline:
            return
        if (self.epochs, self.batch_size) != (cli.DEFAULT_EPOCHS, cli.DEFAULT_BATCH_SIZE):
            # 알려진 기준선은 bs32/ep60 에서 나온 값이다. smoke test(ep2)를 그 값과
            # 대조하면 항상 MISMATCH 가 떠서 진짜 경보가 묻힌다.
            self.log.info(f"[{self.log_tag}] baseline reproduction check: 설정이 확립된 "
                          f"bs{cli.DEFAULT_BATCH_SIZE}/ep{cli.DEFAULT_EPOCHS} 가 아니라 "
                          f"bs{self.batch_size}/ep{self.epochs} 이므로 건너뛴다.")
            return
        known = self.known_baseline.get(self.target)
        known_folds = self.known_baseline_folds.get(self.target)
        if known is None:
            self.log.info(f"[{self.log_tag}] baseline reproduction check: "
                          f"{self.target} 의 알려진 기준선이 없어 건너뛴다.")
            return
        ok_mean = abs(record["mean"] - known) < 1e-3
        self.log.info(f"[{self.log_tag}] baseline reproduction check: mean {record['mean']:.4f} "
                      f"vs known {known} -> {'MATCH' if ok_mean else '*** MISMATCH ***'}")
        ok_folds = True
        if known_folds:
            ok_folds = all(abs(a - b) < 1e-3 for a, b in zip(record["folds"], known_folds))
            self.log.info(f"       folds {record['folds']}")
            self.log.info(f"       known {known_folds} -> {'MATCH' if ok_folds else '*** MISMATCH ***'}")
        if not (ok_mean and ok_folds):
            self.log.warning(
                f"*** WARNING: {name} 항목이 기존 결과를 재현하지 못했다. 기본 경로 동작이 "
                "바뀌었다는 뜻이므로 이 실행의 결과를 신뢰하면 안 된다. ***")

    def after_variants(self) -> None:
        """항목 루프가 끝난 뒤 1회 (late fusion 결합 단계 등)."""

    # ── 요약표 ───────────────────────────────────────────────────────────
    def baseline_reference(self) -> tuple[float | None, list | None, str]:
        """(기준선 평균, 기준선 fold값, 출처). 이번에 돌린 기준선을 우선 쓰고,
        없으면 알려진 값으로 대체한다."""
        rec = self.store.get(self.baseline) if self.baseline else None
        if rec:
            return rec["mean"], rec["folds"], "this run"
        return (self.known_baseline.get(self.target),
                self.known_baseline_folds.get(self.target), "known (RESULTS.md)")

    def summary_columns(self) -> list[tuple[str, int]]:
        return [("name", -20), ("mean", 8), ("std", 8), ("delta", 9),
                ("impr", 7), ("t_p", 9), ("wilcox_p", 10), ("gap", 8), FOLDS_COLUMN]

    def summary_values(self, name: str, record: dict, cmp: dict) -> list:
        gap = record.get("train_val_gap_mean")
        return [name, f"{record['mean']:.4f}", f"{record['std']:.4f}",
                cmp["delta"], cmp["improved"], cmp["ttest_p"], cmp["wilcoxon_p"],
                f"{gap:.4f}" if gap is not None else None, f"  {record['folds']}"]

    def summarize(self) -> None:
        base_mean, base_folds, base_src = self.baseline_reference()
        section(self.log, f"{self.log_tag} SUMMARY (target={self.target}, "
                          f"model={self.model_config}, bs{self.batch_size}/ep{self.epochs})")
        if base_mean is None:
            self.log.info("baseline = (없음) -> delta/검정 칸은 '-' 로 둔다")
        else:
            self.log.info(f"baseline = {self.baseline} {base_mean:.4f} [{base_src}]  {base_folds}")

        table = Table(self.summary_columns())
        for name in self.summary_order():
            record = self.store.get(name)
            if not record:
                continue
            cmp = ({"delta": None, "improved": None, "ttest_p": None, "wilcoxon_p": None}
                   if name == self.baseline
                   else comparison_row(record, base_folds, base_mean))
            table.add(*self.summary_values(name, record, cmp))
        table.emit(self.log)
        self.log.info(N_FOLD_WILCOXON_NOTE)
        self.log.info(f"\nwrote {self.results_path}")

    def summary_order(self) -> list[str]:
        """요약표 행 순서. 이번에 안 돌렸지만 파일에 남아 있는 항목도 같이 보여 준다."""
        order = list(self.variants())
        order += [n for n in self.store.items if n not in order]
        return order

    # ── 진입점 ───────────────────────────────────────────────────────────
    def run(self) -> None:
        self.log.info(f"=== {self.name} === target={self.target} model={self.model_config}"
                      f"{self.flags} bs={self.batch_size} ep={self.epochs} seed={self.seed}")
        self.store.update_header(flags=self.flags, seed=self.seed)
        self.prepare()

        for name in self.variants():
            banner(self.log, f"{self.log_tag}: {name}  ({self.describe(name)})  "
                             f"target={self.target}")
            self.preflight(name)
            if self.args.preflight_only:
                continue
            record = self.run_variant(name)
            self.store.put(name, record)       # 항목마다 즉시 저장 (crash-safe)
            self.log.info(f"[{self.log_tag}] {name} {self.target}: {record['mean']:.4f} "
                          f"+/- {record['std']:.4f}  folds={record['folds']}")
            self.check_baseline(name, record)

        if self.args.preflight_only:
            self.log.info("\npreflight_only: 학습은 건너뛰었다.")
            return
        self.after_variants()
        self.summarize()

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        ap = argparse.ArgumentParser(description=cls.__doc__)
        cli.add_common_args(ap)
        cls.add_arguments(ap)
        return ap

    @classmethod
    def main(cls, argv=None) -> None:
        cls(cls.build_parser().parse_args(argv)).run()


class ReportCorpusMixin:
    """판독지 텍스트를 쓰는 실험에 ``self.corpus`` 를 붙인다.

    실험마다 ``features.load_text_corpus(cohort.DEFAULT_MERGED_CSV)`` 를 각자
    부르고 있었다. 한 번만 읽으면 되는 데다, ``text_source`` 를 바꾸는 실험에서는
    "어느 source 로 읽었나"가 결과 해석의 전제라서 한 곳에 두는 게 맞다.

    ⚠️ ``corpus`` 의 값은 판독지 **원문**이다. 로그·결과 파일에 절대 넣지 않는다.
    """

    #: 판독지의 어느 부분을 쓸지 (features.load_text_corpus 참고)
    text_source: str = "concl_find"

    @property
    def corpus(self) -> dict[int, str]:
        if getattr(self, "_corpus", None) is None:
            from sclc import cohort, features
            self._corpus, self._corpus_stats = features.load_text_corpus(
                cohort.DEFAULT_MERGED_CSV, source=self.text_source)
        return self._corpus

    @property
    def corpus_stats(self) -> dict:
        self.corpus       # 로드 보장
        return self._corpus_stats
