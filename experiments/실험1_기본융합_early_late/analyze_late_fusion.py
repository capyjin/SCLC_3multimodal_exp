# -*- coding: utf-8 -*-
"""late fusion **분석** — 저장된 OOF 위험점수만 읽는다. 재학습 없음(수 초).

    contribution     영상이 실제로 정보를 더하는가 (β_img · 우도비 · 난수 대조)
    shuffle-sanity   "뒤섞은 영상 단독 = 0.50" 확인 (contribution 결과 읽는 법)
    seed-sweep       시드 sweep(runs.jsonl) 강건성 통계
    seed-ensemble    멀티시드 딥앙상블 (재학습 없이 시드 평균)
    variant-followup 임상 결측처리 변이별 tabular 를 영상과 결합했을 때 (실험7 후속)

정리 전에는 이 다섯이 실험 폴더 세 곳에 흩어진 다섯 파일이었다(896줄). 다섯 다
앞뒤 20~30줄이 같았고(코호트 로딩 · labels 인덱싱 · fold_plan · JSON 저장),
``fold별 C-index`` 와 ``CoxPH stack`` 을 각자 다시 구현하고 있었다. 지금은
그 배관이 ``sclc.experiments.analysis.BaseAnalysis`` 에, 계산 원자가
``sclc.late_fusion_tests`` 에 있고, 여기 남은 것은 **각 분석이 무엇을 재는가**뿐이다.

[모든 분석이 공유하는 규율]
  · fold 안에서만 비교한다 (fold 마다 위험점수 척도가 다르다 — 실험9 함정).
  · 메타학습기(CoxPH)는 fold 의 train 환자 OOF 로만 적합한다.
  · 결과를 보고 조건을 고르지 않는다 (시드는 전부 쓰고, 항목은 전부 낸다).

Run:  python experiments/실험1_기본융합_early_late/analyze_late_fusion.py <COMMAND>
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import json

import numpy as np
from scipy import stats

from sclc import late_fusion_tests, paths
from sclc.experiments.analysis import BaseAnalysis, TargetLoopAnalysis, dispatch
from sclc.late_fusion import combine_two, load_oof_cache, oof_cache_path, save_oof_cache
from sclc.evaluation import cindex, fold_cindices, fold_mean_cindex, paired_pvalues
from sclc.utils.cli import comma_list
from sclc.utils.summary import Table

#: 두 축의 이름 = CoxPH 설계행렬의 컬럼명 (결합기와 같은 이름을 써야 계수를 대조할 수 있다)
TAB, IMG = "risk_tabular", "risk_image"


class LateFusionOofMixin:
    """``outputs/late_fusion_B/oof_<target>.json`` 을 읽어 오는 공통 경로.

    실험5의 분석 두 개, 실험1의 3-way 재실행, 실험7의 결합 후속이 각자
    이 파일을 열고 있었다. 캐시가 없을 때 어떻게 할지가 파일마다 달랐던 것이
    문제였다 — 어떤 건 죽고 어떤 건 조용히 재학습을 시작했다.
    """

    def oof_scores(self, target: str) -> dict:
        """``{"tabular": {...}, "image": {...}}``. 캐시가 없으면 저장된 체크포인트로
        예측만 다시 해서(epochs=0, 재학습 아님) 만든 뒤 캐시에 남긴다."""
        if os.path.exists(oof_cache_path(target, self.out_dir)):
            return load_oof_cache(target, self.out_dir)

        from sclc.late_fusion import (get_image_oof_simplecnn, get_tabular_oof, oof_dict)
        self.log.info(f"[oof] 캐시 없음 -> 체크포인트로 예측만 재생성한다 (target={target}).")
        tab = get_tabular_oof(target, epochs=0, batch_size=32, seed=42, out_dir=self.out_dir)
        img = get_image_oof_simplecnn(target, epochs=0, batch_size=16, seed=42, out_dir=self.out_dir)
        scores = {"tabular": oof_dict(tab.oof_predictions), "image": oof_dict(img.oof_predictions)}
        save_oof_cache(target, scores, self.out_dir)
        return scores

    def risks(self, target: str) -> dict:
        """CoxPH 설계행렬용 ``{컬럼명: {research_id: 위험점수}}``."""
        s = self.oof_scores(target)
        return {TAB: s["tabular"], IMG: s["image"]}


# ===========================================================================
# 1) contribution — 영상이 실제로 정보를 더하는가
# ===========================================================================
class ImageContribution(LateFusionOofMixin, TargetLoopAnalysis):
    """★교수님 질문 — late fusion 에서 왜 OS만 오르고 PFS는 떨어지는가?

    최종 결합은 CoxPH 2변수 회귀 ``log h = β_tab·(tabular) + β_img·(image)`` 이므로,
    "영상이 도움이 되는가"는 **β_img 가 0과 유의하게 다른가**로 직접 검정된다.

      ① fold별 β_img + 95%CI + p      신호가 안정적인가?
      ② 우도비 검정 (tab vs tab+img)   영상이 정보를 더하는가?
      ③ tabular↔image 위험점수 상관    중복인가?
      ④ tabular 가 틀린 쌍에서의 구제율 보완적인가?
      ⑤ 영상 점수를 난수로 바꿔 200회   난수와 구분되는가?

    결론(RESULTS.md §10): OS 는 β=+0.322 (p=0.008, 5/5 fold 양수)로 확실한 신호,
    PFS 는 β=+0.008 (p=0.948)로 0과 구분 불가. 난수 대조에서도 OS 는 난수가
    0/200 승, PFS 는 난수가 146/200(73%) 승 — **영상에 PFS 고유 정보가 없다.**
    """

    name = "contribution"
    description = "영상 기여도 검정 (β_img · 우도비 · 상관 · 구제율 · 난수 대조)"
    default_out_dir = "late_fusion_B"
    out_name = "pfs_diagnosis.json"

    @classmethod
    def add_arguments(cls, ap) -> None:
        ap.add_argument("--n_permutation", type=int, default=200,
                        help="난수 대조 반복 수 (기본 %(default)s — 기존 산출물과 같은 값)")
        ap.add_argument("--permutation_seed", type=int, default=42)

    def compute_target(self, target: str) -> dict:
        risks = self.risks(target)
        labels, plan = self.labels, self.plan
        ids_all = self.patient_ids()
        res = {"target": target}

        # ── ① fold별 β_img ────────────────────────────────────────────────
        rows = late_fusion_tests.coef_per_fold(risks, labels, plan, target, covariate=IMG)
        coefs = np.array([r["coef"] for r in rows])
        res["beta_img_per_fold"] = rows
        res["beta_img_mean"] = float(coefs.mean())
        res["beta_img_positive_folds"] = int((coefs > 0).sum())
        res["beta_img_significant_folds"] = int(sum(1 for r in rows if r["p"] < 0.05))

        # ── ② 우도비 검정 (전체 238명 OOF, pooled) ────────────────────────
        full = late_fusion_tests.risk_frame(risks, ids_all, labels, target)
        lrt = late_fusion_tests.likelihood_ratio_test(full, [TAB], [TAB, IMG])
        res["lrt"] = {"stat": lrt["stat"], "df": lrt["df"], "p": lrt["p"],
                      "ll_tabular": lrt["ll_reduced"], "ll_both": lrt["ll_full"]}
        pooled = late_fusion_tests.coef_summary(full, IMG)
        res["beta_img_pooled"] = {k: pooled[k] for k in ("coef", "lo", "hi", "p")}

        # ── ③ 두 위험점수의 상관 ──────────────────────────────────────────
        # (a) pooled: fold 마다 모델이 달라 척도가 다르므로 fold 간 이질성이
        #     만드는 가짜 성분이 섞인다. 참고용으로만 남긴다.
        rho, p = stats.spearmanr(full[TAB], full[IMG])
        res["risk_correlation"] = {"spearman": float(rho), "p": float(p)}
        # (b) within-fold: 척도 문제가 없다. **문서가 인용하는 값은 이쪽이다.**
        per_fold = [{"fold": int(fold),
                     "spearman": float(stats.spearmanr(
                         [risks[TAB][i] for i in ids["test"]],
                         [risks[IMG][i] for i in ids["test"]])[0])}
                    for fold, ids in plan]
        res["risk_correlation_within_fold"] = {
            "per_fold": per_fold,
            "mean": float(np.mean([r["spearman"] for r in per_fold]))}

        # ── ④ 단독 C-index & 보완성 ───────────────────────────────────────
        dur, evt = full["duration"].to_numpy(), full["event"].to_numpy()
        rt, ri = full[TAB].to_numpy(), full[IMG].to_numpy()
        res["cindex_tabular"] = cindex(dur, rt, evt)
        res["cindex_image"] = cindex(dur, ri, evt)
        res["image_rescue_rate"] = late_fusion_tests.rescue_rate(dur, evt, rt, ri)

        # ── ⑤ 난수 대조군 ─────────────────────────────────────────────────
        # late fusion 파이프라인을 그대로 돌리되 영상 위험점수만 무작위로 섞는다.
        # 진짜 영상이 난수와 성능이 같다면, 영상은 정보가 아니라 잡음만 준다.
        def stack_with(image_map) -> float:
            return late_fusion_tests.stack_mean_cindex({TAB: risks[TAB], IMG: image_map},
                                                 labels, plan, target)

        res["stack_real_image"] = stack_with(risks[IMG])
        res["stack_tabular_only"] = late_fusion_tests.stack_mean_cindex(risks, labels, plan,
                                                                  target, names=[TAB])
        draws = late_fusion_tests.permutation_draws(ids_all, risks[IMG], stack_with,
                                              n_repeat=self.args.n_permutation,
                                              seed=self.args.permutation_seed)
        res["stack_shuffled_image"] = late_fusion_tests.null_block(draws, res["stack_real_image"])
        return res

    def report_target(self, target: str, r: dict) -> None:
        log = self.log
        log.info("① beta_img (fold별)")
        for row in r["beta_img_per_fold"]:
            mark = "*" if row["p"] < 0.05 else " "
            log.info(f"   fold {row['fold']}: {row['coef']:+.3f}  "
                     f"[{row['lo']:+.3f}, {row['hi']:+.3f}]  p={row['p']:.3f} {mark}")
        log.info(f"   평균 {r['beta_img_mean']:+.3f} | 양수 {r['beta_img_positive_folds']}/5 "
                 f"| 유의(p<.05) {r['beta_img_significant_folds']}/5")
        b = r["beta_img_pooled"]
        log.info(f"   전체 pooled: {b['coef']:+.3f} [{b['lo']:+.3f}, {b['hi']:+.3f}] p={b['p']:.4f}")
        log.info(f"② 우도비 검정 (영상을 더하면 설명력이 느는가): "
                 f"chi2={r['lrt']['stat']:.2f}, p={r['lrt']['p']:.4f}")
        wf = r["risk_correlation_within_fold"]
        log.info("③ tabular↔image 위험점수 상관:")
        log.info(f"     pooled       rho={r['risk_correlation']['spearman']:+.3f} "
                 f"(p={r['risk_correlation']['p']:.3g})  ← fold 척도차가 섞여 과대평가됨")
        log.info(f"     within-fold  rho={wf['mean']:+.3f}  "
                 f"(fold별 {[round(x['spearman'], 3) for x in wf['per_fold']]})  ← 이 값을 문서에 인용")
        log.info(f"④ 단독 C-index  tabular={r['cindex_tabular']:.4f}  image={r['cindex_image']:.4f}")
        rr = r["image_rescue_rate"]
        log.info(f"   tabular가 틀린 {rr['total']}쌍 중 영상이 맞힌 비율 = {rr['rate']:.3f}")
        sh = r["stack_shuffled_image"]
        log.info(f"⑤ late fusion C-index:  tabular단독={r['stack_tabular_only']:.4f}  "
                 f"진짜영상={r['stack_real_image']:.4f}  "
                 f"난수영상={sh['mean']:.4f} [{sh['p2_5']:.4f}, {sh['p97_5']:.4f}]")
        log.info(f"   난수가 진짜를 이긴 비율 = {sh['frac_random_beats_real']:.3f} "
                 f"({sh['n_repeat']}회 반복)")


# ===========================================================================
# 2) shuffle-sanity — "뒤섞은 영상 단독 = 0.50" 확인
# ===========================================================================
class ShuffleSanity(LateFusionOofMixin, TargetLoopAnalysis):
    """fig12 의 '＋난수(뒤섞은) 영상' 막대가 0.705 나 되는 것을 어떻게 읽어야 하나.

    그 0.705 는 뒤섞은 영상의 실력이 아니라 **tabular(임상+판독지)가 낸 점수**이며,
    뒤섞은 영상은 아무 기여도 못 하면서 잡음만 더해 0.708 -> 0.705 로 깎은 것이다.

      ① 이미 공개된 값(tabular 단독 · 영상 단독)을 이 계산이 재현하는지 대조
         -> 계산 방식이 맞다는 증거
      ② 같은 방식으로 '뒤섞은 영상 **단독**' C-index 를 계산 -> 0.50 근처여야 한다
    """

    name = "shuffle_sanity"
    description = "뒤섞은 영상 단독이 0.50(동전던지기)인지 확인"
    default_out_dir = "late_fusion_B"
    out_name = None            # 출력만 한다 (기존 스크립트와 동일)

    @classmethod
    def add_arguments(cls, ap) -> None:
        ap.add_argument("--n_repeat", type=int, default=100)
        ap.add_argument("--seed", type=int, default=42)
        ap.add_argument("--tol", type=float, default=1e-3, help="공개 수치 재현 허용오차")

    def compute_target(self, target: str) -> dict:
        scores = self.oof_scores(target)
        labels, plan = self.labels, self.plan
        with open(os.path.join(self.out_dir, "results.json"), encoding="utf-8") as fh:
            published = json.load(fh)[target]

        def score_fn(m):
            return fold_mean_cindex(m, labels, plan, target)

        draws = late_fusion_tests.permutation_draws(self.patient_ids(), scores["image"], score_fn,
                                              n_repeat=self.args.n_repeat, seed=self.args.seed)
        return {
            "tabular_only": score_fn(scores["tabular"]),
            "image_only": score_fn(scores["image"]),
            "published_tabular_only": published["tabular_only"]["mean"],
            "published_image_only": published["image_simplecnn_only"]["mean"],
            "shuffled_image_only": {"mean": float(draws.mean()),
                                    "p2_5": float(np.percentile(draws, 2.5)),
                                    "p97_5": float(np.percentile(draws, 97.5)),
                                    "n_repeat": len(draws)},
        }

    def report_target(self, target: str, r: dict) -> None:
        log, tol = self.log, self.args.tol
        log.info("① 계산 방식 검증 — 이 분석이 기존 공개 수치를 재현하는가?")
        for label, mine, ref in (("tabular 단독(임상+판독지)", r["tabular_only"], r["published_tabular_only"]),
                                 ("진짜 영상 단독", r["image_only"], r["published_image_only"])):
            ok = "일치 ✅" if abs(mine - ref) < tol else "불일치 ❌"
            log.info(f"   {label:<24} 이 분석 {mine:.4f}  |  results.json {ref:.4f}  -> {ok}")

        sh = r["shuffled_image_only"]
        log.info(f"\n② 뒤섞은 영상을 '단독'으로 쓰면? ({sh['n_repeat']}회 반복)")
        log.info(f"   평균 {sh['mean']:.4f}  95% 구간 [{sh['p2_5']:.4f}, {sh['p97_5']:.4f}]")
        log.info(f"   -> 동전던지기(0.50)와 같은 수준"
                 f"{' ✅' if abs(sh['mean'] - 0.5) < 0.02 else ' ❌'}")
        log.info("\n③ 결론")
        log.info(f"   뒤섞은 영상 단독      = {sh['mean']:.4f}  (정보 없음)")
        log.info(f"   tabular 단독            = {r['tabular_only']:.4f}")
        log.info("   fig12의 '＋뒤섞은 영상' 막대는 이 둘을 합친 것이며,")
        log.info(f"   실제로는 tabular({r['tabular_only']:.4f})가 전부 낸 점수에서 잡음만큼 깎인 값이다.")


# ===========================================================================
# 3) seed-sweep — 시드 강건성 통계
# ===========================================================================
PRACTICAL_DELTA = 0.016     # 실용적 유의미성 참고선 (실험6 REPORT_ENCODER_FINAL.md)
REFERENCE_SEED = 42

# 부수적 교차검증 — 이미 이 저장소에 있는 독립 실행과 대조한다.
# 실험10(영상단독, seed=42+100*r) / 실험7(tabular, DEFAULT_SEEDS=(42,142,242))
CROSSCHECK_IMAGE = {"os": {142: 0.6519, 242: 0.6531, 342: 0.6527, 442: 0.6573, 542: 0.6525}}
CROSSCHECK_TAB = {"os": {142: 0.6551, 242: 0.6795}, "pfs": {142: 0.6342, 242: 0.6345}}
CROSSCHECK_TOL = 1e-3


class SeedSweepSummary(BaseAnalysis):
    """``runs.jsonl`` 을 읽어 "late > tabular 가 seed 운인가"를 판정한다.

    핵심 통계는 **paired per-seed delta** Δ_s = late(s) − tabular(s) 다. late 는
    그 seed 의 tabular OOF 위에서 CoxPH 로 재적합되므로 두 팔이 강하게 공기(共起)
    한다 — unpaired(평균들의 평균)로 비교하면 공통 잡음에 검정력을 다 뺏긴다.

    ⚠️ 판정선 Δ0.016 은 "단일 5-fold 실행의 fold간 쌍대비교 해상도"라서 seed 간
    비교(분산이 훨씬 작다)에는 원리적으로 안 맞는다. 그래서 통계적 판정은
    부호검정/paired t 로 하고 Δ0.016 은 실용적 참고선으로만 표기한다.
    """

    name = "seed_sweep"
    description = "시드 sweep(runs.jsonl) 강건성 통계"
    default_out_dir = "late_fusion_seed_sweep"
    out_name = "summary.json"

    @classmethod
    def add_arguments(cls, ap) -> None:
        ap.add_argument("--report_encoder", choices=("tfidf", "radbert"), default="tfidf",
                        help="읽을 sweep 폴더를 고른다 (radbert -> ..._radbert)")
        ap.add_argument("--jsonl", default=None, help="기본 <out_dir>/runs.jsonl")
        # 조건이 다른 기록은 섞으면 안 된다 — 아래 값과 일치하는 실행만 통계에 넣는다
        ap.add_argument("--tab_epochs", type=int, default=60)
        ap.add_argument("--img_epochs", type=int, default=30)
        ap.add_argument("--max_folds", type=int, default=None)

    def __init__(self, args):
        if not getattr(args, "out_dir", None) and args.report_encoder != "tfidf":
            args.out_dir = f"{paths.outputs(self.default_out_dir)}_{args.report_encoder}"
        super().__init__(args)
        self.jsonl = args.jsonl or os.path.join(self.out_dir, "runs.jsonl")

    def load_runs(self) -> list[dict]:
        """조건이 일치하는 기록만 남긴다 — smoke 실행이 통계에 섞이면 결론이 바뀐다."""
        if not os.path.exists(self.jsonl):
            raise SystemExit(f"[analyze] {self.jsonl} 이 없다. 먼저 exp_late_fusion.py seed-sweep 을 돌려라.")
        want = {"fix_brain_meta": True, "tab_epochs": self.args.tab_epochs,
                "img_epochs": self.args.img_epochs, "max_folds": self.args.max_folds}
        kept, dropped = [], 0
        with open(self.jsonl, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if any(r.get(k) != v for k, v in want.items()):
                    dropped += 1
                    continue
                kept.append(r)
        if dropped:
            self.log.info(f"[analyze] 조건 불일치로 {dropped}개 기록 제외 ({want})")
        return kept

    # ── 통계 ─────────────────────────────────────────────────────────────
    @staticmethod
    def _paired_block(rows: list[dict]) -> dict:
        deltas = np.array([r["delta_late_minus_tab"] for r in rows], dtype=float)
        n = len(deltas)
        if n < 2:
            return {"n": n, "mean": float(deltas[0]) if n else None}
        mean, sd = float(deltas.mean()), float(deltas.std(ddof=1))
        se = sd / np.sqrt(n)
        tcrit = float(stats.t.ppf(0.975, df=n - 1))
        pp = paired_pvalues([r["late_simplecnn"]["mean"] for r in rows],
                            [r["tabular_only"]["mean"] for r in rows])
        k_pos = int((deltas > 0).sum())
        return {"n": n, "mean": mean, "sd": sd, "se": se,
                "ci95": [mean - tcrit * se, mean + tcrit * se],
                "k_positive": k_pos,
                # 편측(late>tabular 방향) 정확 이항검정. H0: P(Δ>0)=0.5
                "sign_test_p_one_sided": float(
                    stats.binomtest(k_pos, n, p=0.5, alternative="greater").pvalue),
                "ttest_p": pp["ttest_p"], "wilcoxon_p": pp["wilcoxon_p"]}

    def summarize_target(self, runs: list[dict], target: str) -> dict:
        rows = sorted([r for r in runs if r["target"] == target], key=lambda r: r["seed"])
        if not rows:
            return {}

        def arm_stats(key):
            vals = [r[key]["mean"] for r in rows]
            return {"mean": float(np.mean(vals)),
                    "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "min": float(np.min(vals)), "max": float(np.max(vals)), "values": vals}

        arms = ("tabular_only", "image_simplecnn_only", "late_simplecnn")
        ref_row = next((r for r in rows if r["seed"] == REFERENCE_SEED), None)
        out = {
            "per_seed": [{"seed": r["seed"], **{a: r[a]["mean"] for a in arms},
                          "delta": r["delta_late_minus_tab"]} for r in rows],
            "arms": {a: arm_stats(a) for a in arms},
            # primary = 새 seed 만. seed42 는 공개 수치의 출처라 "또 하나의 표본"이
            # 아니므로 기본 통계에서 뺀다 (pooled 는 따로 낸다).
            "paired_delta_primary": self._paired_block([r for r in rows if r["seed"] != REFERENCE_SEED]),
            "paired_delta_pooled": self._paired_block(rows) if ref_row else None,
        }
        if ref_row:
            out["reference_seed42"] = {**{a: ref_row[a]["mean"] for a in arms},
                                       "delta": ref_row["delta_late_minus_tab"]}
            combined = sorted(r["late_simplecnn"]["mean"] for r in rows)
            out["seed42_percentile_late"] = float(
                np.searchsorted(combined, ref_row["late_simplecnn"]["mean"], side="right") / len(combined))

        prim = out["paired_delta_primary"]
        if prim.get("n", 0) >= 2:
            direction = ("강건(전 seed 동일방향)" if prim["k_positive"] == prim["n"]
                         else "일관되게 반대방향" if prim["k_positive"] == 0 else "혼재")
            magnitude = ("판정선(0.016) 이상" if prim["mean"] >= PRACTICAL_DELTA
                         else "판정선(0.016) 미달")
            out["verdict"] = (f"{prim['k_positive']}/{prim['n']} seed 에서 late>tabular ({direction}); "
                              f"paired Δ={prim['mean']:+.4f}±{prim['sd']:.4f}, "
                              f"sign-test p={prim['sign_test_p_one_sided']:.4f}; 크기는 {magnitude}")
        else:
            out["verdict"] = "(신규 seed 2개 미만 -- 통계 불가, --seeds 로 더 채운 뒤 재분석)"
        out["practical_line"] = PRACTICAL_DELTA
        return out

    @staticmethod
    def crosscheck(runs: list[dict]) -> list[dict]:
        """기존 실험7/실험10 의 독립 실행과 같은 seed 에서 같은 값이 나오는지."""
        results = []
        for r in runs:
            tgt, seed = r["target"], r["seed"]
            expected = [("실험10_armB", "image_simplecnn_only", CROSSCHECK_IMAGE.get(tgt, {}).get(seed))]
            # tabular 대조값은 TF-IDF 시절 기록이라 radbert 실행에는 안 맞는다(다른 게 정상).
            if r.get("report_encoder", "tfidf") == "tfidf":
                expected.append(("실험7_original", "tabular_only", CROSSCHECK_TAB.get(tgt, {}).get(seed)))
            for source, key, exp in expected:
                if exp is None:
                    continue
                obs = r[key]["mean"]
                results.append({"source": source, "target": tgt, "seed": seed,
                                "expected": exp, "observed": obs,
                                "ok": abs(obs - exp) <= CROSSCHECK_TOL})
        return results

    def compute(self) -> dict:
        runs = self.load_runs()
        targets = sorted({r["target"] for r in runs})
        summary = {"meta": {
            "n_records": len(runs), "seeds": sorted({r["seed"] for r in runs}),
            "reference_seed": REFERENCE_SEED, "practical_delta": PRACTICAL_DELTA,
            "note": "fold 분할은 고정 -- 이 sweep은 학습(초기화/배치순서) 잡음만 잰다, "
                    "split 강건성은 별개"}}
        for target in targets:
            summary[target] = self.summarize_target(runs, target)
        summary["crosscheck"] = self.crosscheck(runs)
        return summary

    def report(self, summary: dict) -> None:
        log = self.log
        log.info(f"\n{'=' * 70}\nSeed 강건성 sweep 요약\n{'=' * 70}")
        for target, s in summary.items():
            if target in ("meta", "crosscheck") or not s:
                continue
            log.info(f"\n[{target.upper()}]")
            table = Table([("seed", 6), ("tabular", 9), ("image", 9), ("late", 9), ("Δ", 9)])
            for row in s["per_seed"]:
                table.add(row["seed"], f"{row['tabular_only']:.4f}",
                          f"{row['image_simplecnn_only']:.4f}",
                          f"{row['late_simplecnn']:.4f}", f"{row['delta']:+.4f}")
            table.emit(log)
            a = s["arms"]
            log.info(f"  arm 평균±sd (n={len(s['per_seed'])}): " + "  ".join(
                f"{k.split('_')[0]}={a[k]['mean']:.4f}±{a[k]['sd']:.4f}" for k in a))
            for label, block in (("primary(새 seed만", s["paired_delta_primary"]),
                                 ("pooled(42 포함", s.get("paired_delta_pooled"))):
                if block and block.get("n", 0) >= 2:
                    ci = block["ci95"]
                    log.info(f"  {label}, n={block['n']}): Δ={block['mean']:+.4f}±{block['sd']:.4f}  "
                             f"95%CI=[{ci[0]:+.4f},{ci[1]:+.4f}]  "
                             f"{block['k_positive']}/{block['n']} 양수  "
                             f"sign p={block['sign_test_p_one_sided']:.4f}")
            if "reference_seed42" in s:
                r42 = s["reference_seed42"]
                log.info(f"  seed42(참고): tabular={r42['tabular_only']:.4f} "
                         f"late={r42['late_simplecnn']:.4f} Δ={r42['delta']:+.4f}  "
                         f"(전체 중 late 백분위={s.get('seed42_percentile_late', float('nan')):.2f})")
            log.info(f"  판정: {s.get('verdict', '(seed 부족)')}")

        xc = summary["crosscheck"]
        if xc:
            log.info(f"\n교차검증(기존 실험7/10과 대조): {sum(1 for x in xc if x['ok'])}/{len(xc)} 일치")
            for x in xc:
                if not x["ok"]:
                    log.warning(f"  !! 불일치: {x['source']} seed={x['seed']} target={x['target']} "
                                f"기대={x['expected']:.4f} 관측={x['observed']:.4f}")


# ===========================================================================
# 4) seed-ensemble — 멀티시드 딥앙상블 (재학습 없음)
# ===========================================================================
class SeedEnsemble(TargetLoopAnalysis):
    """시드 sweep 이 캐시해 둔 OOF 만으로 "시드 평균이 실력을 올리는가"를 잰다.

    [동기] 채택 수치는 전부 seed=42 단일 실행인데, sweep 에서 tabular 축의 학습
    시드 잡음이 sd 0.0114(OS)/0.0147(PFS) 로 판정 참고선 0.015 에 육박한다. 즉
    지금까지 비교해 온 효과들과 시드 운이 같은 규모다. 한편 실험11(MoE)이 막힌
    이유는 "fold 당 171명에 학습 파라미터를 더 얹을 수 없다"였는데, 시드 평균은
    **학습 파라미터를 하나도 추가하지 않으므로** 그 논리에 걸리지 않는다.

    [누수 규율] 시드를 성능 보고 고르면 test 선택편향이므로 지정한 시드를
    **무조건 전부** 쓴다. 정규화는 (시드, fold) 그룹 안에서만 한다.

    [검증 게이트] 캐시에서 재계산한 per-seed C-index 가 sweep 의 summary.json 과
    일치하지 않으면 즉시 중단한다 — 정규화/정렬 코드가 조용히 틀리는 것을 막는
    유일한 방어선이다.
    """

    name = "seed_ensemble"
    description = "멀티시드 딥앙상블 (재학습 없이 시드 평균)"
    default_out_dir = "seed_ensemble"
    DEFAULT_SEEDS = (42, 142, 242, 342, 442, 542)

    @classmethod
    def add_arguments(cls, ap) -> None:
        ap.add_argument("--report_encoder", choices=("tfidf", "radbert"), default="radbert")
        ap.add_argument("--seeds", type=lambda v: [int(x) for x in comma_list(v)],
                        default=list(cls.DEFAULT_SEEDS))
        ap.add_argument("--sweep_dir", default=None, help="기본 outputs/late_fusion_seed_sweep[_<enc>]")
        ap.add_argument("--tol", type=float, default=1e-6, help="검증 게이트 허용오차")

    def __init__(self, args):
        super().__init__(args)
        self.seeds = list(args.seeds)
        suffix = "" if args.report_encoder == "tfidf" else f"_{args.report_encoder}"
        self.sweep_dir = args.sweep_dir or f"{paths.outputs('late_fusion_seed_sweep')}{suffix}"
        self.out_name = f"results_{args.report_encoder}.json"

    def load_seed_oof(self, seed: int, target: str) -> tuple[dict, dict]:
        with open(os.path.join(self.sweep_dir, "oof", f"seed{seed}_{target}.json"),
                  encoding="utf-8") as fh:
            d = json.load(fh)
        return ({int(k): float(v) for k, v in d["tabular"].items()},
                {int(k): float(v) for k, v in d["image"].items()})

    def verify_against_sweep(self, target, tab_folds, img_folds) -> None:
        """sweep summary.json 과 대조 — 어긋나면 조인/정렬이 틀렸다는 뜻이므로 중단."""
        with open(os.path.join(self.sweep_dir, "summary.json"), encoding="utf-8") as fh:
            rows = json.load(fh)[target]["per_seed"]
        for seed, tf, imf in zip(self.seeds, tab_folds, img_folds):
            row = next(r for r in rows if r["seed"] == seed)
            for label, mine, ref in (("tabular", np.mean(tf), row["tabular_only"]),
                                     ("image", np.mean(imf), row["image_simplecnn_only"])):
                if abs(float(mine) - ref) > self.args.tol:
                    raise SystemExit(
                        f"[{target}/seed{seed}/{label}] 캐시 재계산 {mine:.6f} != "
                        f"summary.json {ref:.6f} -- fold 정렬이나 라벨 조인이 어긋났다는 "
                        "뜻이므로 중단한다")

    def compute_target(self, target: str) -> dict:
        labels, plan = self.labels, self.plan
        tab_seeds, img_seeds = zip(*(self.load_seed_oof(s, target) for s in self.seeds))
        tab_folds = [fold_cindices(t, labels, plan, target) for t in tab_seeds]
        img_folds = [fold_cindices(i, labels, plan, target) for i in img_seeds]
        self.verify_against_sweep(target, tab_folds, img_folds)
        late = [combine_two(self.cohort_df, target, t, i) for t, i in zip(tab_seeds, img_seeds)]

        ind = {
            "tabular": {"per_seed_folds": tab_folds},
            "image": {"per_seed_folds": img_folds},
            "late": {"per_seed_folds": [c["fold_cindex"] for c in late]},
        }
        for arm in ind.values():
            arm["per_seed"] = [float(np.mean(f)) for f in arm["per_seed_folds"]]
            arm["mean"] = float(np.mean(arm["per_seed"]))
            arm["sd"] = float(np.std(arm["per_seed"], ddof=1))
            # fold 별로 "임의의 한 시드에서 기대되는 성적" = 시드 평균
            arm["expected_folds"] = np.mean(arm["per_seed_folds"], axis=0).tolist()

        res = {"individual": ind, "ensemble": {}}
        for mode in late_fusion_tests.NORMALIZE_MODES:
            ens_t = late_fusion_tests.ensemble_risks(tab_seeds, plan, mode)
            ens_i = late_fusion_tests.ensemble_risks(img_seeds, plan, mode)
            t_folds = fold_cindices(ens_t, labels, plan, target)
            i_folds = fold_cindices(ens_i, labels, plan, target)
            comb = combine_two(self.cohort_df, target, ens_t, ens_i)
            res["ensemble"][mode] = {
                "tabular": {"mean": float(np.mean(t_folds)), "folds": t_folds},
                "image": {"mean": float(np.mean(i_folds)), "folds": i_folds},
                "late": {"mean": comb["mean"], "folds": comb["fold_cindex"],
                         "mean_coef": comb["mean_coef"]},
                # 앙상블이 "임의의 한 시드에서 기대되는 성적"보다 나은가 (fold 쌍대)
                "paired_vs_expected": {
                    arm: paired_pvalues(folds, ind[arm]["expected_folds"])
                    for arm, folds in (("tabular", t_folds), ("late", comb["fold_cindex"]))},
                # 앙상블한 뒤에도 영상이 보태는가 (fold 쌍대)
                "paired_fusion_gain": paired_pvalues(comb["fold_cindex"], t_folds),
            }
        return res

    def compute(self) -> dict:
        return {"report_encoder": self.args.report_encoder, "seeds": self.seeds,
                "sweep_dir": self.sweep_dir, **super().compute()}

    def report_target(self, target: str, d: dict) -> None:
        log, ind = self.log, d["individual"]
        for arm in ("tabular", "late"):
            a = ind[arm]
            log.info(f"  {arm:<8} seed{self.seeds[0]}={a['per_seed'][0]:.4f}  "
                     f"개별평균={a['mean']:.4f}±{a['sd']:.4f}")
        for mode in late_fusion_tests.NORMALIZE_MODES:
            e = d["ensemble"][mode]
            pt, pl = e["paired_vs_expected"]["tabular"], e["paired_vs_expected"]["late"]
            log.info(f"   [{mode:>4}] tabular={e['tabular']['mean']:.4f} "
                     f"(vs기대 {e['tabular']['mean'] - ind['tabular']['mean']:+.4f}, "
                     f"{pt['n_improved']}/{pt['n_folds']} fold, t-p={pt['ttest_p']:.3f})   "
                     f"late={e['late']['mean']:.4f} "
                     f"(vs기대 {e['late']['mean'] - ind['late']['mean']:+.4f}, "
                     f"{pl['n_improved']}/{pl['n_folds']} fold, t-p={pl['ttest_p']:.3f})   "
                     f"융합이득={e['late']['mean'] - e['tabular']['mean']:+.4f}")


# ===========================================================================
# 5) variant-followup — 임상 결측처리 변이가 영상과 결합해도 살아남는가 (실험7 후속)
# ===========================================================================
class VariantFollowup(BaseAnalysis):
    """tabular 축에서 잰 이득이 **융합 후에도** 남는지 확인한다 (재학습 없음).

    tabular 단독에서 좋아졌다고 융합에서도 좋아지는 건 아니다: late fusion 은
    fold 마다 두 위험점수 위에 CoxPH 를 다시 적합하므로, 영상 점수가 이미
    그 개선분을 담고 있으면 결합 C-index 는 그대로일 수 있다.

    입력은 둘 다 이미 존재한다 — 재학습하지 않는다:
      tabular : 실험7 ``exp_missing_handling.py`` 가 저장한 (변이, seed)별 OOF
                (clin_report, bs32/ep60, 고정 분할)
      image   : ``outputs/late_fusion_B/oof_<target>.json`` 의 SimpleCNN 축
                (bs16/ep30, seed42). 영상 arm 은 임상 컬럼을 읽지 않으므로 이
                실험의 어떤 변경도 영상 축에 영향을 줄 수 없다.
    """

    name = "variant_followup"
    description = "임상 결측처리 변이별 tabular 를 영상과 결합 (실험7 후속)"
    default_out_dir = "EXP_20260805_clinical_missing_handling"
    out_name = "late_fusion_followup.json"

    @classmethod
    def add_arguments(cls, ap) -> None:
        ap.add_argument("--results", default=None, help="기본 <out_dir>/results.json")
        ap.add_argument("--variants", type=comma_list, default=["original", "A", "B", "C"])
        ap.add_argument("--seeds", type=lambda v: [int(x) for x in comma_list(v)],
                        default=[42, 142, 242])
        ap.add_argument("--image_oof_dir", default=None, help="기본 outputs/late_fusion_B")

    def load_tabular_oof(self, variant: str, target: str, seed: int) -> dict:
        path = self.args.results or os.path.join(self.out_dir, "results.json")
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        for run in payload["runs"]:
            if (run["config"] == "clin_report" and run["variant"] == variant
                    and run["target"] == target and run["seed"] == seed):
                return {int(r["research_id"]): float(r["risk_score"]) for r in run["oof_predictions"]}
        raise SystemExit(f"no clin_report/{variant}/{target}/seed{seed} run found in {path}")

    def compute(self) -> dict:
        image_dir = self.args.image_oof_dir or paths.outputs("late_fusion_B")
        runs = []
        for target in self.targets:
            image_risk = load_oof_cache(target, image_dir, keys=("image",))["image"]
            for variant in self.args.variants:
                for seed in self.args.seeds:
                    tab_risk = self.load_tabular_oof(variant, target, seed)
                    mismatch = set(image_risk) ^ set(tab_risk)
                    if mismatch:
                        raise SystemExit("tabular/image OOF 가 서로 다른 환자를 덮고 있다: "
                                         f"{sorted(mismatch)[:10]}")
                    combo = combine_two(self.cohort_df, target, tab_risk, image_risk)
                    runs.append({"target": target, "variant": variant, "seed": seed,
                                 "fused_mean": combo["mean"], "fused_std": combo["std"],
                                 "fused_folds": [round(c, 6) for c in combo["fold_cindex"]],
                                 "mean_coef": combo["mean_coef"]})
                    c = combo["mean_coef"]
                    self.log.info(f"[latefusion] {target}/{variant}/seed{seed}: "
                                  f"fused={combo['mean']:.4f} +/- {combo['std']:.4f}  "
                                  f"coef(tab,img)=({c['risk_tabular']:.3f},{c['risk_image']:.3f})")
        return {"runs": runs}

    def report(self, result: dict) -> None:
        self.log.info(f"\n{'=' * 80}\nLATE FUSION SUMMARY (tabular=clin_report + image=SimpleCNN)"
                      f"\n{'=' * 80}")
        table = Table([("target", -8), ("variant", -10), ("fused mean", 12),
                       ("sd across seeds", 18), ("  per-seed", -2)])
        for target in self.targets:
            for variant in self.args.variants:
                means = [r["fused_mean"] for r in result["runs"]
                         if r["target"] == target and r["variant"] == variant]
                if means:
                    table.add(target, variant, f"{np.mean(means):.4f}", f"{np.std(means):.4f}",
                              f"  {[round(m, 4) for m in means]}")
        table.emit(self.log)


ANALYSES = {
    "contribution": ImageContribution,
    "shuffle-sanity": ShuffleSanity,
    "seed-sweep": SeedSweepSummary,
    "seed-ensemble": SeedEnsemble,
    "variant-followup": VariantFollowup,
}

if __name__ == "__main__":
    dispatch(ANALYSES, description=__doc__)
