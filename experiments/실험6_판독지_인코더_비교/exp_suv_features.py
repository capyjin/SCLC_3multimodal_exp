# -*- coding: utf-8 -*-
"""[실험6-e] 판독지에서 뽑은 SUVmax 수치가 현재 최고 모델을 더 좋게 하는가?

동기:
  판독지는 TF-IDF(char n-gram)로만 쓰이고 있어서, 판독지 안의 **수치**(SUVmax)는
  사실상 문자 조각으로 흩어져 버린다. SUVmax 는 SCLC 예후 인자로 알려져 있으므로
  이걸 명시적인 숫자 feature 로 뽑아 주면 이득이 있는지 3단계로 쌓아 올리며 본다.

단계 (누적):
  none                 SUV 없음 = clin_report 그대로 (재현 기준선, 0.7076/0.6678)
  suv_max              + 문서 내 최대 SUV
  suv_max+count        + 값 개수(≈수치 기재된 병변 수, 결측 지시자 역할 겸함)
  suv_max+count+mean   + 평균 SUV

고정되는 것(= 오직 SUV 열만 달라진다):
  모델 조합(clin_report), fold split, seed, epochs, batch_size, TF-IDF(400차원,
  패딩 고정), 텍스트 source(concl_find). SUV 열은 **임상 블록 뒤**에 붙어서
  clinical 브랜치로 들어간다 -> report 브랜치 차원(400)은 변하지 않고
  모델 구조 변경도 없다.

누수 방지:
  정규식 파싱은 환자별 결정론적 연산이라 전역 1회 수행해도 무방하다.
  환자들을 가로질러 계산되는 통계(결측 대치 median, StandardScaler)만 fold 별로
  **train fold 환자만** 써서 fit 한다 (suv_features.make_extra_numeric_fn).
  실제 사용된 fold별 median/표본수는 로그와 결과 JSON(leakage_audit)에 남긴다.

Run:  python experiments/실험6_판독지_인코더_비교/exp_suv_features.py --target os
      python .../exp_suv_features.py --target os --steps none,suv_max
      python .../exp_suv_features.py --target os --preflight_only
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import numpy as np

import suv_features   # 같은 폴더 형제 모듈 (이 실험 전용 — core 승격 기준 미달)
from sclc import cohort, features
from sclc.experiments.base import BaseExperiment
from sclc.utils import cli

STEP_ORDER = ["none", "suv_max", "suv_max+count", "suv_max+count+mean"]
STEP_DESC = {
    "none":               "SUV 없음 (clin_report 재현 기준선)",
    "suv_max":            "문서 내 최대 SUV",
    "suv_max+count":      "최대 SUV + 값 개수",
    "suv_max+count+mean": "최대 SUV + 개수 + 평균",
}
KNOWN_BASELINE = {"os": 0.7076, "pfs": 0.6678}
TFIDF_MAX_FEATURES = 400


class SuvFeatureLadder(BaseExperiment):
    """SUV 숫자 열을 한 개씩 쌓아 올리며 같은 모델을 5-fold 로 반복 학습한다."""

    name = "suv_features"
    default_out_dir = "suv_features"
    item_key = "steps"
    log_tag = "SUV"
    model_config = "clin_report"
    baseline = "none"
    known_baseline = KNOWN_BASELINE

    @classmethod
    def add_arguments(cls, ap) -> None:
        ap.add_argument("--steps", default=",".join(STEP_ORDER[1:]),
                        help="쉼표 목록. 'none'(기준선) 포함 가능. 기본값은 SUV 3단계.")

    def __init__(self, args):
        super().__init__(args)
        self.names = cli.check_names(cli.comma_list(args.steps), suv_features.STEPS)
        self.table = None
        self.audits: dict[str, list] = {}

    # ── BaseExperiment 계약 ──────────────────────────────────────────────
    def variants(self) -> list[str]:
        return self.names

    def describe(self, name: str) -> str:
        return STEP_DESC[name]

    def columns(self, name: str):
        return suv_features.STEPS[name]

    def make_extra_fn(self, name: str, audit: list):
        cols = self.columns(name)
        return suv_features.make_extra_numeric_fn(self.table, cols, audit=audit) if cols else None

    def evaluator_kwargs(self, name: str) -> dict:
        audit = self.audits.setdefault(name, [])
        audit.clear()      # preflight 에서 쌓인 기록은 버리고 본 실행 것만 남긴다
        # <- 이 실험에서 유일하게 바뀌는 것
        return {"extra_numeric_fn": self.make_extra_fn(name, audit)}

    def extra_record(self, name: str) -> dict:
        return {"step": name, "suv_cols": list(self.columns(name)),
                "leakage_audit": list(self.audits.get(name, []))}

    # ── 추출 self-check (전역 1회) ───────────────────────────────────────
    def prepare(self) -> None:
        self.table = suv_features.extract_suv_table()
        vs = self.table.attrs["value_stats"]
        n_reports, n_legend = self.table.attrs["n_reports"], self.table.attrs["n_legend"]
        self.log.info("\n=== SUV extraction self-check ===")
        self.log.info(f"reports={n_reports}  legend_sentence={n_legend} "
                      f"({n_legend / n_reports * 100:.1f}%)  "
                      f"docs_with_value={int(self.table['suv_available'].sum())} "
                      f"({self.table['suv_available'].mean() * 100:.1f}%)  n_values={vs['n']}")
        self.log.info(f"values: min={vs['min']} median={vs['median']} mean={vs['mean']:.2f} "
                      f"p95={vs['p95']} max={vs['max']}  "
                      f"outside_plausible_range={vs['n_outside_plausible_range']}")
        csum = suv_features.cohort_summary(self.table)
        self.log.info(f"cohort(238): suv_available={csum['n_available']} "
                      f"missing={csum['n_missing']}  "
                      f"suv_max median={csum['suv_max']['median']} "
                      f"unique={csum['suv_max']['n_unique']}  count hist={csum['suv_count_hist']}")
        self.store.update_header(extraction={
            "n_reports": n_reports, "n_legend": n_legend,
            "n_docs_with_value": int(self.table["suv_available"].sum()),
            "value_stats": vs, "cohort": csum})

    # ── 사전 점검 ────────────────────────────────────────────────────────
    def preflight(self, name: str) -> None:
        """학습 전에 fold별 feature 행렬을 실제로 만들어 보고 (1) 텐서 폭이 기대대로인지,
        (2) SUV 열이 환자마다 실제로 변하는지(상수/전부결측이 아닌지),
        (3) 결측 대치 median 이 정말 train fold 환자만으로 계산됐는지를 확인한다."""
        cols = self.columns(name)
        cohort_df = cohort.load_trimodal_cohort()
        clinical_frame = cohort_df.drop_duplicates("research_id").set_index("research_id")
        std_cols, cat_cols = features.resolve_clinical_columns(clinical_frame)
        corpus, _ = features.load_text_corpus(cohort.DEFAULT_MERGED_CSV)
        audit = self.audits.setdefault(name, [])
        audit.clear()
        fn = self.make_extra_fn(name, audit)

        dims, checks = None, []
        for fold in sorted(int(f) for f in cohort_df["fold"].unique()):
            fdf = cohort_df[cohort_df["fold"] == fold]
            ids = {s: fdf.loc[fdf["split"] == s, "research_id"].astype(int).tolist()
                   for s in ("train", "val", "test")}
            tab, clinical_dim, report_dim = features.build_fold_multimodal_tabular(
                clinical_frame.loc[ids["train"]], clinical_frame.loc[ids["val"]],
                clinical_frame.loc[ids["test"]], corpus, std_cols, cat_cols,
                tfidf_max_features=TFIDF_MAX_FEATURES, extra_numeric_fn=fn,
            )
            dims = (clinical_dim, report_dim, tab["train"].shape[1])
            if cols:
                # SUV 블록은 [len(std)+len(cat) : clinical_dim) 구간에 있다.
                base = len(std_cols) + len(cat_cols)
                blk = tab["train"][:, base:clinical_dim]
                checks.append({
                    "fold": fold,
                    "n_unique": [int(len(np.unique(blk[:, j]))) for j in range(blk.shape[1])],
                    "std": [round(float(blk[:, j].std()), 4) for j in range(blk.shape[1])],
                })

        clinical_dim, report_dim, width = dims
        base = len(std_cols) + len(cat_cols)
        n_extra = clinical_dim - base
        self.log.info(f"[preflight/{name}] clinical={len(std_cols)}+{len(cat_cols)}={base} "
                      f"+ suv={n_extra} -> clinical_dim={clinical_dim}; "
                      f"report(tfidf)={report_dim}; tabular width={width} "
                      f"(expected {base} + {len(cols)} + {TFIDF_MAX_FEATURES} "
                      f"= {base + len(cols) + TFIDF_MAX_FEATURES})")
        if n_extra != len(cols) or width != clinical_dim + report_dim:
            raise AssertionError("feature width mismatch")
        for c in checks:
            self.log.info(f"[preflight/{name}] fold {c['fold']} SUV columns {list(cols)}: "
                          f"n_unique={c['n_unique']} std={c['std']} (상수면 1/0.0 이 찍힌다)")
        for a in audit:
            self.log.info(f"[leakage-audit] fold n_train={a['n_train']} "
                          f"(SUV 있는 train 환자 {a['n_train_with_suv']}명) "
                          f"-> train-only median={a['train_fold_medians']} | "
                          f"결측 대치된 환자 수 {a['n_imputed']} | "
                          f"n_val={a['n_val']} n_test={a['n_test']}")

    def summary_order(self) -> list[str]:
        order = [n for n in STEP_ORDER if n in self.names or self.store.has(n)]
        return order + [n for n in self.store.items if n not in order]


if __name__ == "__main__":
    SuvFeatureLadder.main()
