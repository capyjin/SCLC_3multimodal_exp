# -*- coding: utf-8 -*-
"""요약표 출력 — 실험 4개가 각자 갖고 있던 f-string 정렬 코드를 한 곳으로.

표 자체는 사소하지만, **기준선 대비 delta 와 쌍대검정을 어떻게 붙이느냐**는
사소하지 않다. 정리 전에는 exp_bert_text 만 쌍대검정을 붙였고 나머지는 delta 만
찍었다. 같은 저장소의 두 표가 서로 다른 기준으로 "좋아졌다"를 판정하고 있었다.
여기서는 기준선이 있으면 항상 delta + 개선 fold 수 + p 값을 같이 낸다.
"""
from sclc.metrics import paired_pvalues


class Table:
    """고정폭 텍스트 표. ``columns`` 는 ``(헤더, 폭)`` 목록이고 폭이 음수면 좌측정렬."""

    def __init__(self, columns: list[tuple[str, int]]):
        self.columns = columns
        self.rows: list[list[str]] = []

    def _fmt(self, value, width: int) -> str:
        text = "-" if value is None else str(value)
        return f"{text:<{abs(width)}}" if width < 0 else f"{text:>{width}}"

    def add(self, *values) -> None:
        self.rows.append([self._fmt(v, w) for v, (_, w) in zip(values, self.columns)])

    def render(self) -> list[str]:
        head = "".join(self._fmt(name, width) for name, width in self.columns)
        return [head] + ["".join(r) for r in self.rows]

    def emit(self, logger) -> None:
        for line in self.render():
            logger.info(line)


def comparison_row(record: dict, baseline_folds=None, baseline_mean=None) -> dict:
    """한 항목을 기준선과 비교한 값들을 만든다.

    기준선이 없으면 (delta 등이) 전부 None 이라 표에 ``-`` 로 찍힌다 — "기준선이
    없어서 비교를 안 한 것"과 "비교했는데 차이가 0"이 구분된다.
    """
    out = {"delta": None, "improved": None, "ttest_p": None, "wilcoxon_p": None}
    if baseline_mean is not None:
        out["delta"] = f"{record['mean'] - baseline_mean:+.4f}"
    if baseline_folds and record.get("folds"):
        pv = paired_pvalues(record["folds"], baseline_folds)
        out["improved"] = f"{pv['n_improved']}/{pv['n_folds']}"
        out["ttest_p"] = f"{pv['ttest_p']:.4f}" if pv["ttest_p"] is not None else None
        out["wilcoxon_p"] = f"{pv['wilcoxon_p']:.4f}" if pv["wilcoxon_p"] is not None else None
    return out


#: 요약표 마지막 열 — fold별 C-index 리스트는 길이가 들쭉날쭉해서 폭을 주지 않고
#: 좌측정렬한다. 앞 열과 붙지 않도록 헤더·값 모두 공백 두 칸으로 시작한다.
FOLDS_COLUMN = ("  folds", -2)


N_FOLD_WILCOXON_NOTE = ("  (n=5 fold 라 wilcoxon 양측 p 는 최소 0.0625 — 0.05 를 원리적으로 "
                        "못 넘는다. 참고용.)")
