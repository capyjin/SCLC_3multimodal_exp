# -*- coding: utf-8 -*-
"""[실험6-a] 판독지 인코더 비교 — TF-IDF vs RadBERT.

동기:
  판독지는 오랫동안 char n-gram TF-IDF 로만 쓰였다. 글자 조각 빈도라서 의미를
  모르고, 특히 부정문("no evidence of metastasis")과 긍정문을 거의 구분하지
  못한다. 의학 코퍼스로 사전학습된 BERT 는 그 차이를 안다. 그래서 **텍스트
  인코더만** 바꿔서 이득이 있는지 본다.

arm (= sclc.encoders 의 후보 전부):
  tfidf          char n-gram TF-IDF 400 — 최종 채택 인코더 / 재현 기준선
  radbert        frozen RadBERT 768, 한글 ko2en 치환 (채택 레시피)
  radbert@strip  RadBERT + 한국어 삭제   ([UNK] 진단용 대조군, §5.2)
  radbert@raw    RadBERT + 원문 그대로   ([UNK] 진단용 대조군, §5.2)

  KM-BERT · mBERT · TF-IDF+RadBERT 결합 · SVD 랭크 통제군은 후보에서 제외됐다.
  근거와 당시 수치는 REPORT_ENCODER_FINAL.md §1·§3·§5, 코드 삭제 사유는
  src/sclc/encoders/registry.py docstring 참고.

고정되는 것(= 오직 텍스트 블록만 달라진다):
  모델 조합(--model_config, 기본 clin_report), fold split, seed, epochs,
  batch_size, 임상 블록, 텍스트 source(concl_find), 그리고 **report 브랜치 폭**.
  폭을 고정하는 게 중요하다 — 폭이 달라지면 파라미터 수와 추정 부담이 같이
  달라져서 "인코더가 좋아서"인지 "폭이 달라서"인지 구분이 안 된다.
  ``ReportEncoder.preflight`` 가 학습 전에 실제 폭을 재서 다르면 즉시 죽인다.

누수 방지:
  RadBERT forward 는 문서 하나만 보는 frozen 연산이라 전역 1회 계산해도 무방하다.
  환자를 가로질러 계산되는 통계(SVD 기저·StandardScaler·TF-IDF vocabulary)는
  전부 fold 별로 train fold 환자만 써서 fit 한다. 실제 쓰인 값은 결과 JSON 의
  ``leakage_audit`` 에 남는다.

환자 정보 보호: 판독지 원문은 stdout/로그/결과파일 어디에도 찍지 않는다.
  텍스트 통계는 집계값(평균 토큰 수, [UNK] 비율 등)만 낸다.

Run:  python experiments/실험6_판독지_인코더_비교/exp_encoder_compare.py --target os
      python .../exp_encoder_compare.py --target os --encoders tfidf,radbert
      python .../exp_encoder_compare.py --target os --model_config report_only  # 판독지 단독
      python .../exp_encoder_compare.py --target os --preflight_only            # 수 초
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from sclc.encoders import DEFAULT_ARMS, build_encoder
from sclc.experiments.base import BaseExperiment, ReportCorpusMixin
from sclc.model import MODALITY_CONFIGS
from sclc.utils import cli
from sclc.utils.summary import FOLDS_COLUMN

# 기존 기준선 (RESULTS.md / outputs/text_source/concl_find, report_only 는 ablation).
# 재현 확인용. 모델 조합마다 기준선이 다르므로 config -> target 으로 건다.
# 여기 없는 config 는 기준선 없음으로 보고 재현 검사를 건너뛴다 (죽지 않는다).
KNOWN_BASELINE = {
    "clin_report": {"os": 0.7076, "pfs": 0.6678},
    "report_only": {"os": 0.6268, "pfs": 0.6094},
}
KNOWN_BASELINE_FOLDS = {
    "clin_report": {
        "os":  [0.726, 0.7894, 0.6765, 0.654, 0.692],
        "pfs": [0.6444, 0.732, 0.6501, 0.641, 0.6717],
    },
    "report_only": {
        "os":  [0.5993, 0.6846, 0.5662, 0.6806, 0.6032],
        "pfs": [0.5921, 0.6322, 0.5832, 0.6729, 0.5666],
    },
}


class EncoderComparison(ReportCorpusMixin, BaseExperiment):
    """인코더만 갈아 끼우며 같은 모델을 5-fold 로 반복 학습한다."""

    name = "encoder_compare"
    default_out_dir = "bert_text"     # 기존 결과 폴더를 그대로 이어 쓴다
    item_key = "arms"
    log_tag = "ENC"
    baseline = "tfidf"

    @classmethod
    def add_arguments(cls, ap) -> None:
        ap.add_argument("--encoders", default=",".join(DEFAULT_ARMS),
                        help="쉼표 목록. 예: tfidf,radbert,radbert@strip "
                             "(기본 %(default)s). '@' 뒤는 RadBERT 의 한글 처리 변형.")
        ap.add_argument("--model_config", default="clin_report", choices=sorted(MODALITY_CONFIGS),
                        help="모델 조합. 기본 clin_report(임상+판독지). report_only 면 "
                             "판독지 브랜치만 남아 인코더 자체 실력을 잰다.")
        ap.add_argument("--out_dim", type=int, default=400,
                        help="텍스트 블록 목표 폭 (기본 %(default)s = TF-IDF 폭, 비교 공정성). "
                             "RadBERT 는 축소를 끄는 게 채택 설정이라 실제로는 768이 된다.")
        ap.add_argument("--svd", action="store_true",
                        help="RadBERT 블록에 train-only SVD 축소를 켠다. 기본은 꺼짐 "
                             "— §5.1 의 2×2 격자에서 축소가 성능을 깎는 것으로 판명됐다.")
        ap.add_argument("--scale", action="store_true",
                        help="RadBERT 블록에 train-only StandardScaler 를 켠다. 기본 꺼짐 (§5.1).")
        ap.add_argument("--embed_batch_size", type=int, default=16)
        ap.add_argument("--max_length", type=int, default=512)

    def __init__(self, args):
        self.model_config = args.model_config      # BaseExperiment.settings() 가 읽는다
        super().__init__(args)
        self.known_baseline = KNOWN_BASELINE.get(self.model_config, {})
        self.known_baseline_folds = KNOWN_BASELINE_FOLDS.get(self.model_config, {})
        self.encoders = {}
        for spec in cli.comma_list(args.encoders):
            kwargs = {"out_dim": args.out_dim}
            if spec.split("@")[0] == "radbert":
                kwargs.update(do_svd=args.svd, do_scale=args.scale,
                              max_length=args.max_length,
                              embed_batch_size=args.embed_batch_size)
            enc = build_encoder(spec, **kwargs)
            # 키는 항상 tfidf / radbert@ko2en 형태로 정규화된다 (표기 흔들림 방지).
            self.encoders[enc.key] = enc
        self._stats: dict[str, dict] = {}

    # ── BaseExperiment 계약 ──────────────────────────────────────────────
    def variants(self) -> list[str]:
        return list(self.encoders)

    def describe(self, name: str) -> str:
        return self.encoders[name].description

    def evaluator_kwargs(self, name: str) -> dict:
        return self.encoders[name].evaluator_kwargs(self.corpus)

    def settings(self) -> dict:
        return {**super().settings(), "out_dim": self.args.out_dim}

    def prepare(self) -> None:
        st = self.corpus_stats
        self.log.info(f"\n=== corpus ===  n_reports={st['n']}  total_chars={st['total_chars']}  "
                      "(원문은 출력하지 않는다)")
        self.store.update_header(corpus=st, model_config=self.model_config,
                                 known_baseline=self.known_baseline.get(self.target))

    def preflight(self, name: str) -> None:
        enc = self.encoders[name]
        stats = enc.corpus_stats(self.corpus)
        if stats:
            self._stats[name] = stats
            tok = stats.get("tokenization")
            if tok:
                self.log.info(f"[tokens/{name}] mean={tok['mean_tokens']:.1f} "
                              f"median={tok['median_tokens']:.0f} "
                              f"UNK={tok['unk_frac_mean'] * 100:.2f}% "
                              f"docs_with_UNK={tok['pct_docs_with_unk']:.1f}% "
                              f"truncated={tok['pct_docs_truncated']:.1f}%")
            cov = stats.get("ko2en_coverage")
            if cov:
                self.log.info(f"[ko2en] dict_entries={cov['dict_entries']} "
                              f"char_coverage={cov['char_coverage'] * 100:.1f}% "
                              f"unique_chunks={cov['unique_chunks']}")
        enc.preflight(self.corpus, self.log)
        enc.audit.clear()   # preflight 에서 쌓인 감사 기록은 버리고 본 실행 것만 남긴다

    def extra_record(self, name: str) -> dict:
        enc = self.encoders[name]
        return {"encoder": enc.config(),
                **self._stats.get(name, {}),
                "leakage_audit": list(enc.audit)}

    def summary_columns(self):
        return super().summary_columns()[:-1] + [("UNK%", 8), FOLDS_COLUMN]

    def summary_values(self, name, record, cmp):
        tok = (record.get("tokenization") or {})
        unk = f"{tok['unk_frac_mean'] * 100:.1f}" if tok else None
        return super().summary_values(name, record, cmp)[:-1] + [unk, f"  {record['folds']}"]


if __name__ == "__main__":
    EncoderComparison.main()
