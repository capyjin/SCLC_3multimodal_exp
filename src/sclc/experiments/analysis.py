# -*- coding: utf-8 -*-
"""**재학습 없는 분석**의 추상 부모 클래스.

[BaseExperiment 와 무엇이 다른가]
  ``BaseExperiment`` 는 "조건을 바꿔 가며 5-fold 를 학습한다" — GPU 몇 시간짜리다.
  이 클래스는 그 학습이 남긴 산출물(OOF 위험점수 캐시, runs.jsonl, results.json)
  만 읽어 통계를 내는 쪽이다 — 수 초짜리이고, 몇 번을 다시 돌려도 체크포인트를
  건드리지 않는다. 두 성질이 다르므로 골격도 따로 둔다.

[이 클래스가 존재하는 이유]
  정리 전 late fusion 분석 스크립트 5개(실험5 두 개 · 실험1 두 개 · 실험7 하나)를
  나란히 놓으면 앞뒤 20~30줄이 같았다:

      PROJECT_ROOT 앵커 2줄 -> argparse -> cohort.load_trimodal_cohort()
      -> labels = df.drop_duplicates("research_id").set_index("research_id")
      -> plan = fold_plan(df) -> 타깃 루프 -> print 구분선
      -> json.dump(..., indent=2) -> print(f"wrote {path}")

  달라지는 건 가운데 "이 분석이 실제로 재는 것" 한 덩어리뿐이었다. 게다가
  ``labels`` 를 만드는 그 한 줄은 실수하기 쉬운 곳이다 — ``drop_duplicates``
  를 빠뜨리면 한 환자가 여러 행으로 들어와 ``.loc`` 이 조용히 더 긴 배열을
  돌려주고, C-index 가 틀린 채로 계산된다. 그래서 여기 한 곳에 둔다.

[두 단계로 나뉜다]
  ``BaseAnalysis``        데이터 로딩 · 로깅 · 저장 · CLI 만 담당. 서브클래스는
                          ``compute()`` 하나만 채운다. 결과 모양이 타깃 루프가
                          아닌 분석(시드 sweep 요약의 meta/crosscheck 등)이 여기.
  ``TargetLoopAnalysis``  "타깃마다 같은 계산을 반복한다"는 가장 흔한 모양.
                          ``compute_target`` / ``report_target`` 두 개만 채운다.
"""
import argparse
import json
import os
from abc import ABC, abstractmethod

from sclc import paths
from sclc.utils.cli import TARGETS, comma_list
from sclc.utils.logging import get_logger, section


class BaseAnalysis(ABC):
    """저장된 산출물만 읽어 통계를 내는 분석 하나."""

    #: 로그에 찍히는 이름 (서브커맨드 이름과 같게 두면 찾기 쉽다)
    name: str = "analysis"
    #: ``--help`` 에 나오는 한 줄 설명
    description: str = ""
    #: 읽고 쓰는 기본 폴더 (outputs/ 아래 이름). 산출물 폴더 이름은 문서·그림
    #: 스크립트와의 계약이므로 바꾸지 않는다.
    default_out_dir: str = ""
    #: 결과 JSON 파일 이름. ``None`` 이면 파일을 쓰지 않고 출력만 한다.
    out_name: str | None = None
    #: 기본 타깃 목록
    targets: tuple = TARGETS
    #: 코호트를 brain_meta 누수 수정본으로 읽을지 (기존 분석 전부 True)
    fix_brain_meta: bool = True

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.targets = tuple(getattr(args, "targets", None) or self.targets)
        self.out_dir = getattr(args, "out_dir", None) or paths.outputs(self.default_out_dir)
        self.log = get_logger(self.name, self.out_dir)
        self._cohort_df = None
        self._labels = None
        self._plan = None

    # ── 데이터 (필요할 때 한 번만 로드) ───────────────────────────────────
    @property
    def cohort_df(self):
        if self._cohort_df is None:
            from sclc import cohort
            self._cohort_df = cohort.load_trimodal_cohort(fix_brain_meta=self.fix_brain_meta)
        return self._cohort_df

    @property
    def labels(self):
        """research_id 인덱스의 코호트 프레임 (``labels.loc[ids, "os_days"]`` 용).

        ``drop_duplicates`` 가 여기 한 번만 있다 — 이 한 줄이 5개 파일에
        복사돼 있었고, 빠뜨리면 조용히 틀린 C-index 가 나온다.
        """
        if self._labels is None:
            self._labels = self.cohort_df.drop_duplicates("research_id").set_index("research_id")
        return self._labels

    @property
    def plan(self):
        """``[(fold, {"train": [...], "val": [...], "test": [...]}), ...]`` (고정 분할)."""
        if self._plan is None:
            from sclc.train import fold_plan
            self._plan = fold_plan(self.cohort_df)
        return self._plan

    def patient_ids(self) -> list:
        return list(self.labels.index)

    # ── 서브클래스가 채우는 부분 ──────────────────────────────────────────
    @abstractmethod
    def compute(self) -> dict:
        """이 분석이 재는 것 전부. 반환값이 그대로 결과 JSON 이 된다."""

    def report(self, result: dict) -> None:
        """계산이 끝난 뒤 한 번 — 종합 표·결론 출력 (기본은 없음)."""

    @classmethod
    def add_arguments(cls, ap: argparse.ArgumentParser) -> None:
        """분석 고유 인자. 공통 인자(--targets/--out_dir)는 이미 붙어 있다."""

    # ── 공통 구현 ────────────────────────────────────────────────────────
    @property
    def out_path(self) -> str | None:
        return os.path.join(self.out_dir, self.out_name) if self.out_name else None

    def save(self, result: dict) -> None:
        path = self.out_path
        if path is None:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, path)      # 쓰다가 죽어도 반쪽짜리 JSON 이 남지 않는다
        self.log.info(f"\nwrote {path}")

    def run(self) -> dict:
        result = self.compute()
        self.report(result)
        self.save(result)
        return result

    # ── 진입점 ───────────────────────────────────────────────────────────
    @classmethod
    def build_parser(cls, ap: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
        ap = ap or argparse.ArgumentParser(description=cls.description or cls.__doc__)
        ap.add_argument("--targets", type=comma_list, default=None,
                        help=f"분석할 타깃 (기본 {','.join(cls.targets)})")
        ap.add_argument("--out_dir", default=None,
                        help=f"읽고 쓸 폴더 (기본 outputs/{cls.default_out_dir})")
        cls.add_arguments(ap)
        return ap

    @classmethod
    def main(cls, argv=None):
        return cls(cls.build_parser().parse_args(argv)).run()


class TargetLoopAnalysis(BaseAnalysis):
    """타깃(os/pfs)마다 같은 계산을 반복하는 분석 — 이 저장소에서 가장 흔한 모양.

    결과는 ``{타깃: compute_target(타깃)}`` 이고, 타깃 하나가 끝날 때마다 곧바로
    출력한다(오래 걸리는 분석에서 앞 타깃 결과를 먼저 볼 수 있게).
    """

    @abstractmethod
    def compute_target(self, target: str) -> dict:
        """이 타깃에서 재는 것."""

    @abstractmethod
    def report_target(self, target: str, result: dict) -> None:
        """``compute_target`` 의 결과를 사람이 읽을 형태로 출력."""

    def compute(self) -> dict:
        out = {}
        for target in self.targets:
            section(self.log, f"target = {target.upper()}")
            out[target] = self.compute_target(target)
            self.report_target(target, out[target])
        return out


def dispatch(analyses: dict, argv=None, description: str = ""):
    """분석 여러 개를 서브커맨드 하나로 묶는다.

        dispatch({"contribution": ImageContribution, "shuffle": ShuffleSanity})
        -> python analyze_late_fusion.py contribution --targets os

    한 주제의 분석을 파일 하나에 모으되 각 분석은 자기 인자를 그대로 갖게 하는
    장치다. 스크립트가 흩어져 있으면 "이 주제에 어떤 검정이 이미 있는가"를
    ``ls`` 로 알 수 없다.
    """
    ap = argparse.ArgumentParser(description=description)
    sub = ap.add_subparsers(dest="command", required=True, metavar="COMMAND")
    for key, cls in analyses.items():
        cls.build_parser(sub.add_parser(key, help=cls.description,
                                        description=cls.description or cls.__doc__))
    args = ap.parse_args(argv)
    return analyses[args.command](args).run()
