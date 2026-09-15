# -*- coding: utf-8 -*-
"""late fusion **학습** 드라이버 — 축을 각각 학습해 CoxPH 로 묶는다 (서브커맨드 3개).

    method-b    2축 결합: [임상+판독지] + [영상]  ★프로젝트 최종 채택 모델(OS)
    three-way   3축 결합: 임상 · 판독지 · 영상을 각각 독립 학습한 뒤 가중합
    seed-sweep  method-b 레시피를 seed 만 바꿔 반복 (시드 강건성)

정리 전에는 이 셋이 각각 다른 파일이었다(``late_fusion_tab_image.py`` 142줄,
``exp_late_fusion_3modal_rerun.py`` 102줄, ``late_fusion_seed_sweep.py`` 272줄).
그런데 method-b 와 seed-sweep 은 **같은 두 축을 같은 조건으로 학습**하고 결과를
기록하는 방식만 달랐고, 그 축 실행 코드가 두 벌로 복사돼 있었다. 한쪽만 고치면
"seed 42 재현치"와 "본 실행"이 다른 코드로 계산되는데 아무도 알 수 없다.
여기서는 ``TabImageRun`` 하나가 그 축들을 돌리고, 두 서브커맨드는 그걸 상속해서
**기록 형식만** 다르게 한다.

    TabImageRun          tabular 축 + 영상 축(들) -> ArmResult -> CoxPH 결합
      +- MethodBRun      영상 축 2개(SimpleCNN, ResNet18), 소수 4자리
      +- SeedRun         영상 축 1개(SimpleCNN), 소수 6자리 + seed 메타

[누수 방지]
  각 환자의 위험점수는 그 환자가 test 였던 fold 의 모델이 낸 OOF 값이고,
  CoxPH 결합기는 fold 마다 train 환자의 OOF 로만 적합한다
  (``sclc.fusion_stack.combine_two`` — 검증된 코드를 그대로 쓴다).
  ⚠️ nested CV 가 아니라는 알려진 한계는 ``sclc/experiments/fusion.py`` 참고.

[출력이 print 인 이유]
  seed-sweep 은 학습 로그를 ``contextlib.redirect_stdout`` 으로 타깃별 로그
  파일에 몰아 넣는다. 로거 핸들러는 생성 시점의 스트림을 붙들고 있어 그 리디렉션을
  받지 않으므로, 이 파일은 print 를 유지한다. (분석 쪽 ``analyze_late_fusion.py``
  는 리디렉션이 없어서 ``BaseAnalysis`` 의 로거 + run.log 를 쓴다.)

Run:
  python experiments/실험1_기본융합_early_late/exp_late_fusion.py method-b --targets os,pfs
  python experiments/실험1_기본융합_early_late/exp_late_fusion.py three-way --target os
  python experiments/실험1_기본융합_early_late/exp_late_fusion.py seed-sweep --targets os
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import argparse
import contextlib
import json
import time

from sclc import cohort, paths
from sclc.encoders import build_encoder
from sclc.fusion_stack import (ArmResult, combine_two, combine_weighted_sum,
                               get_image_oof_resnet18, get_image_oof_simplecnn,
                               get_tabular_oof, load_oof_cache, oof_dict)
from sclc.utils.cli import comma_list

# 영상 축 목록: 축 키 -> (arm 결과 키, 결합 결과 키, OOF 추출기).
# 세 이름은 전부 기존 results.json 스키마와 문서 표기라서 바꾸지 않는다.
IMAGE_AXES = {
    "simplecnn": ("image_simplecnn_only", "late_simplecnn", get_image_oof_simplecnn),
    "resnet18": ("image_resnet18_only", "late_resnet18", get_image_oof_resnet18),
}


# ===========================================================================
# 공통 부모 — tabular 축 + 영상 축(들)을 학습하고 CoxPH 로 묶는 실행 하나
# ===========================================================================
class TabImageRun:
    """한 타깃 · 한 시드에 대해 축들을 학습하고 2축 결합까지 끝내는 실행 단위.

    서브클래스가 바꾸는 것은 두 가지뿐이다:
      ``image_axes``          어떤 영상 축을 돌릴지
      ``combination_record``  결합 결과를 결과 JSON 에 어떤 모양으로 넣을지
    """

    #: 돌릴 영상 축 (IMAGE_AXES 의 키)
    image_axes: tuple = ("simplecnn",)
    #: fold별 C-index 를 몇 자리로 반올림해 저장할지 (기존 파일과 맞춘다)
    fold_digits: int = 4
    #: 로그 접두어
    tag: str = "lateB"

    def __init__(self, target: str, *, out_dir: str, seed: int = 42,
                 tab_epochs: int = 60, tab_batch: int = 32,
                 img_epochs: int = 30, img_batch: int = 16, resnet_epochs: int = 30,
                 max_folds=None, fix_brain_meta: bool = True, text_encoder_fn=None,
                 cohort_df=None):
        self.target = target
        self.out_dir = out_dir
        self.seed = seed
        self.tab_epochs, self.tab_batch = tab_epochs, tab_batch
        self.img_epochs, self.img_batch = img_epochs, img_batch
        self.resnet_epochs = resnet_epochs
        self.max_folds = max_folds
        self.fix_brain_meta = fix_brain_meta
        self.text_encoder_fn = text_encoder_fn
        self._cohort_df = cohort_df
        #: 축 키 -> ArmResult (결합 뒤에도 OOF 를 꺼내 쓸 수 있게 남긴다)
        self.arms: dict[str, ArmResult] = {}

    @property
    def cohort_df(self):
        if self._cohort_df is None:
            self._cohort_df = cohort.load_trimodal_cohort(fix_brain_meta=self.fix_brain_meta)
        return self._cohort_df

    def _epochs_for(self, key: str) -> int:
        return self.resnet_epochs if key == "resnet18" else self.img_epochs

    def _banner(self, text: str) -> None:
        print(f"\n########## [{self.tag}] {text} target={self.target} ##########")

    def _log_arm(self, label: str, arm: ArmResult) -> None:
        print(f"[{self.tag}] {label} {self.target}: {arm.mean:.4f} +/- {arm.std:.4f}  "
              f"folds={arm.folds(4)}")

    # ── 축 학습 ──────────────────────────────────────────────────────────
    def run_tabular(self) -> ArmResult:
        """임상+판독지 결합 모델 (영상 제외, bs32/ep60) — 가장 강한 tabular 축."""
        self._banner(f"TABULAR (clin+report joint, bs{self.tab_batch}/ep{self.tab_epochs})")
        ev = get_tabular_oof(self.target, epochs=self.tab_epochs, batch_size=self.tab_batch,
                             max_folds=self.max_folds, seed=self.seed,
                             fix_brain_meta=self.fix_brain_meta, out_dir=self.out_dir,
                             text_encoder_fn=self.text_encoder_fn)
        arm = ArmResult.from_evaluator("tabular_only", ev)
        self._log_arm("tabular-only", arm)
        return arm

    def run_image(self, key: str) -> ArmResult:
        arm_key, _combo_key, runner = IMAGE_AXES[key]
        epochs = self._epochs_for(key)
        self._banner(f"IMAGE {key} (bs{self.img_batch}/ep{epochs})")
        ev = runner(self.target, epochs=epochs, batch_size=self.img_batch,
                    max_folds=self.max_folds, seed=self.seed, out_dir=self.out_dir)
        arm = ArmResult.from_evaluator(arm_key, ev)
        self._log_arm(f"image-{key}-only", arm)
        return arm

    # ── 결합 ─────────────────────────────────────────────────────────────
    def combination_record(self, combo: dict) -> dict:
        """결합 결과 중 결과 JSON 에 남길 부분.

        기본값은 기존 ``outputs/late_fusion_B/results.json`` 에 실제로 들어 있는
        다섯 키다. ``combine_two`` 는 ``fold_records``/``oof_predictions`` 도
        돌려주지만, 그 둘은 ``oof_<target>.json`` 에 이미 있고 결과 파일만
        수십 배로 불린다.
        """
        return {k: combo[k] for k in ("fold_cindex", "mean", "std",
                                      "coefs_per_fold", "mean_coef")}

    def run(self) -> dict:
        """축을 전부 학습하고 각 영상 축마다 2축 결합을 잰다."""
        record = {"target": self.target}
        tab = self.arms["tabular_only"] = self.run_tabular()
        record["tabular_only"] = tab.as_dict(self.fold_digits)

        combos = []
        for key in self.image_axes:
            arm_key, combo_key, _ = IMAGE_AXES[key]
            arm = self.arms[arm_key] = self.run_image(key)
            record[arm_key] = arm.as_dict(self.fold_digits)
            combos.append((key, arm_key, combo_key))

        for key, arm_key, combo_key in combos:
            self._banner(f"COMBINE tabular+{key}")
            combo = combine_two(self.cohort_df, self.target,
                                tab.risk, self.arms[arm_key].risk, max_folds=self.max_folds)
            record[combo_key] = self.combination_record(combo)
            print(f"[{self.tag}] late-fusion tabular+{key} {self.target}: "
                  f"{combo['mean']:.4f} +/- {combo['std']:.4f}  mean_coef={combo['mean_coef']}")
        return record


class MethodBRun(TabImageRun):
    """★채택 모델 — SimpleCNN 과 ResNet18 두 영상 축을 모두 잰다 (RESULTS.md §8)."""
    image_axes = ("simplecnn", "resnet18")
    fold_digits = 4


class SeedRun(TabImageRun):
    """시드 sweep 의 한 칸 — 채택 레시피(SimpleCNN)만 돌리고 seed 메타를 붙인다.

    ResNet18 축은 채택 모델이 아니므로 뺀다 (시간 절약 + 무관한 변동요인 제거).
    """
    image_axes = ("simplecnn",)
    fold_digits = 6
    tag = "seed_sweep"

    def combination_record(self, combo: dict) -> dict:
        return {k: combo[k] for k in ("mean", "std", "fold_cindex",
                                      "coefs_per_fold", "mean_coef")}

    def run(self, report_encoder: str = "tfidf") -> tuple[dict, dict]:
        started = time.time()
        record = super().run()
        tab_mean = record["tabular_only"]["mean"]
        record.update({
            "seed": self.seed, "report_encoder": report_encoder,
            "tab_epochs": self.tab_epochs, "tab_batch": self.tab_batch,
            "img_epochs": self.img_epochs, "img_batch": self.img_batch,
            "fix_brain_meta": self.fix_brain_meta, "max_folds": self.max_folds,
            # seed_everything(seed+fold) 이므로 실제로 쓰인 시드는 이 다섯 개다
            "effective_fold_seeds": [self.seed + f for f in range(1, 6)],
            "delta_late_minus_tab": record["late_simplecnn"]["mean"] - tab_mean,
            "delta_late_minus_tab_per_fold": [
                round(a - b, 6) for a, b in zip(record["late_simplecnn"]["fold_cindex"],
                                                self.arms["tabular_only"].c_indices)],
            "elapsed_s": round(time.time() - started, 1),
        })
        oof = {"tabular": self.arms["tabular_only"].risk,
               "image": self.arms["image_simplecnn_only"].risk}
        return record, oof


# ===========================================================================
# 서브커맨드 1 — method-b (2축 late fusion)
# ===========================================================================
def cmd_method_b(args) -> None:
    """방식 B: [임상+판독지 tabular] + [영상] 2축 결합.

    초기 융합(concat)으로 영상을 통째로 섞으면 임상+판독지만 썼을 때(0.708)보다
    셋 다 썼을 때(0.678)가 오히려 낮았다. 그래서 각 축을 **자기에게 맞는
    에포크/배치로 독립 학습**한 뒤 위험점수만 최적 비율로 합친다.
    """
    out_dir = args.out_dir or paths.outputs("late_fusion_B")
    os.makedirs(out_dir, exist_ok=True)
    if args.smoke:
        args.max_folds, args.tab_epochs, args.img_epochs, args.resnet_epochs = 1, 2, 2, 2

    results = {}
    for target in args.targets:
        results[target] = MethodBRun(
            target, out_dir=out_dir, seed=args.seed,
            tab_epochs=args.tab_epochs, img_epochs=args.img_epochs,
            resnet_epochs=args.resnet_epochs, max_folds=args.max_folds,
            fix_brain_meta=args.fix_brain_meta,
        ).run()

    out_path = args.out or os.path.join(out_dir, "results.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n[lateB] wrote {out_path}")

    print("\n================ METHOD B SUMMARY ================")
    for target, r in results.items():
        print(f"\n--- target={target} ---")
        for key, label in (("tabular_only", "tabular-only          "),
                           ("image_simplecnn_only", "image SimpleCNN-only  "),
                           ("image_resnet18_only", "image ResNet18-only   ")):
            print(f"  {label}: {r[key]['mean']:.4f} +/- {r[key]['std']:.4f}")
        for key, label in (("late_simplecnn", "late fusion +SimpleCNN"),
                           ("late_resnet18", "late fusion +ResNet18 ")):
            c = r[key]["mean_coef"]
            print(f"  {label}: {r[key]['mean']:.4f} +/- {r[key]['std']:.4f}"
                  f"   coef(tab,img)=({c['risk_tabular']:.3f},{c['risk_image']:.3f})")


# ===========================================================================
# 서브커맨드 2 — three-way (3축 가중합)
# ===========================================================================
#: batch16/ep30 조건으로 2026-07-22 에 나왔던 값 (참고용 대조)
KNOWN_LEGACY_3WAY = {"os": 0.6703, "pfs": 0.6288}


def cmd_three_way(args) -> None:
    """임상·판독지·영상을 각각 독립 학습한 뒤 CoxPH 로 가중합한다.

    임상/판독지 축은 ``sclc.fusion_arms`` 의 **pycox 경로**를 개선된 학습조건
    (bs32/ep60)으로 새로 학습한다. 영상 축은 **재학습하지 않고**
    ``outputs/late_fusion_B/oof_<target>.json`` 의 값을 재사용한다 — 영상 arm 의
    표준 조건(bs16/ep30)에서 이미 학습된 것이고, 임상 컬럼을 읽지 않으므로
    이 실험의 변경이 영상 축에 영향을 줄 수 없다.
    """
    from sclc import fusion_arms       # pycox/torchtuples 로딩이 느려서 여기서

    out_dir = args.out_dir or paths.outputs("late_fusion_3modal_rerun")
    os.makedirs(out_dir, exist_ok=True)
    cohort_df = cohort.load_trimodal_cohort()
    target = args.target

    print(f"\n########## clinical_only  target={target}  bs={args.batch_size} ep={args.epochs} ##########")
    clin = fusion_arms.run_clinical_only(cohort_df, target, batch_size=args.batch_size,
                                         epochs=args.epochs, seed=args.seed)
    print(f"\n########## report_only  target={target}  bs={args.batch_size} ep={args.epochs} ##########")
    rep = fusion_arms.run_report_only(cohort_df, target, batch_size=args.batch_size,
                                      epochs=args.epochs, seed=args.seed)

    def _folds(res):
        cis = [r["c_index"] for r in res["fold_records"]]
        return cis, sum(cis) / len(cis), [round(float(c), 4) for c in cis]

    clin_ci, clin_mean, clin_folds = _folds(clin)
    rep_ci, rep_mean, rep_folds = _folds(rep)
    print(f"[3WAY] clinical_only {target}: {clin_mean:.4f}  folds={clin_folds}")
    print(f"[3WAY] report_only {target}: {rep_mean:.4f}  folds={rep_folds}")

    print(f"\n########## image_only  target={target}  (재사용, 재학습 없음) ##########")
    img_risk = load_oof_cache(target, keys=("image",))["image"]
    print(f"[3WAY] image_only {target}: (재사용) n={len(img_risk)}")

    print(f"\n########## COMBINE (3-way weighted sum)  target={target} ##########")
    combined = combine_weighted_sum(cohort_df, target, img_risk,
                                    oof_dict(clin["oof_predictions"]),
                                    oof_dict(rep["oof_predictions"]))
    mean_ci = combined["mean"]
    folds = [round(float(c), 4) for c in combined["fold_cindex"]]
    print(f"[3WAY] late_fusion_weighted_sum {target}: {mean_ci:.4f}  folds={folds}")

    result = {
        "target": target, "batch_size": args.batch_size, "epochs": args.epochs,
        "clinical_only": {"mean": clin_mean, "folds": clin_folds},
        "report_only": {"mean": rep_mean, "folds": rep_folds},
        "image_only": {"note": "reused from outputs/late_fusion_B (no retrain)"},
        "late_fusion_weighted_sum": {
            "mean": mean_ci, "folds": folds,
            "coefficients_per_fold": [r["coefficients"] for r in combined["fold_records"]]},
    }
    path = os.path.join(out_dir, f"results_{target}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    print(f"\n================ SUMMARY ({target}) ================")
    print(f"clinical_only            {clin_mean:.4f}")
    print(f"report_only              {rep_mean:.4f}")
    print(f"late_fusion_weighted_sum {mean_ci:.4f}   "
          f"(batch16/ep30 조건 참고값: {KNOWN_LEGACY_3WAY[target]:.4f})")
    print(f"wrote {path}")


# ===========================================================================
# 서브커맨드 3 — seed-sweep (시드 강건성)
# ===========================================================================
SWEEP_OUT_DIR = paths.outputs("late_fusion_seed_sweep")
DEFAULT_SEEDS = (142, 242, 342, 442, 542)
REFERENCE_SEED = 42
MIN_SEED_GAP = 6           # seed_everything(seed+fold), fold∈1..5 -- 5 이하면 겹친다
ANCHOR_TOL = 1e-3

# fix_brain_meta=True 기준 seed=42 재현치 (MODEL_SUMMARY.md,
# outputs/brainfix/latefusion_os_fixed.json). 판독지 인코더 버전별로 나뉜다.
SEED42_ANCHOR = {
    "tfidf": {
        "os": {"tabular_only": 0.7057, "image_simplecnn_only": 0.6570, "late_simplecnn": 0.7143},
        "pfs": {"tabular_only": 0.6696, "image_simplecnn_only": 0.6154, "late_simplecnn": 0.6621},
    },
    # 영상 축은 판독지 인코더와 무관하므로 image 값이 동일한 게 정상이다.
    "radbert": {
        "os": {"tabular_only": 0.7153, "image_simplecnn_only": 0.6570, "late_simplecnn": 0.7224},
        "pfs": {"tabular_only": 0.6456, "image_simplecnn_only": 0.6154, "late_simplecnn": 0.6470},
    },
}

#: 이 폴더들에는 절대 쓰지 않는다 — RESULTS.md/MODEL_SUMMARY.md 기준 산출물이고,
#: 실험8이 image_simplecnn_* 체크포인트를 읽기전용으로 쓴다.
PROTECTED_DIRS = (paths.outputs("late_fusion_B"), paths.outputs("image_cph"))
REFERENCE_TRAIN_CONF = {"tab_epochs": 60, "tab_batch": 32, "img_epochs": 30, "img_batch": 16}


def assert_seeds_independent(seeds: list[int]) -> None:
    """base seed 가 서로 MIN_SEED_GAP 이상 떨어져 있는지 확인 (42 도 포함해서)."""
    all_seeds = sorted(set(seeds) | {REFERENCE_SEED})
    for i, a in enumerate(all_seeds):
        for b in all_seeds[i + 1:]:
            if b - a < MIN_SEED_GAP:
                raise SystemExit(
                    f"[seed_sweep] seed {a}와 {b}는 {b - a}밖에 안 떨어져 있다 "
                    f"(seed_everything(seed+fold), fold 1..5 -- {MIN_SEED_GAP} 이상 필요). "
                    "두 '독립' 시드 조건이 실제로는 같은 난수를 공유하게 된다.")


def assert_safe_out_dir(path: str) -> str:
    real = os.path.realpath(path)
    for p in PROTECTED_DIRS:
        preal = os.path.realpath(p)
        if real == preal or real.startswith(preal + os.sep):
            raise SystemExit(f"[seed_sweep] 보호된 경로에 쓰려 한다: {real} (금지: {p})")
    return path


def run_key(record: dict) -> tuple:
    """"이 조건의 실행이 이미 있는가"를 판정하는 열쇠 (runs.jsonl 이어달리기)."""
    return (record["seed"], record["target"], record.get("report_encoder", "tfidf"),
            record["tab_epochs"], record["img_epochs"],
            record["fix_brain_meta"], record["max_folds"])


def done_keys(jsonl_path: str) -> set:
    if not os.path.exists(jsonl_path):
        return set()
    with open(jsonl_path, encoding="utf-8") as fh:
        return {run_key(json.loads(line)) for line in fh if line.strip()}


def anchor_check(seed: int, target: str, record: dict) -> None:
    """seed=42 실행이 공개된 수치를 재현하는지 확인 (sweep 해석 전 사전점검)."""
    anchor = SEED42_ANCHOR.get(record.get("report_encoder", "tfidf"), {}).get(target)
    if seed != REFERENCE_SEED or not anchor:
        return
    for key, ref in anchor.items():
        obs = record[key]["mean"]
        dev = abs(obs - ref)
        status = "OK" if dev <= ANCHOR_TOL else "MISMATCH"
        print(f"  [anchor-check/{status}] {key}: 관측={obs:.4f} 기준={ref:.4f} 차이={dev:.4f}")
        if status == "MISMATCH":
            print(f"  !! seed=42 재현이 기준({ref})과 어긋난다 -- 코드/설정이 바뀌었을 수 있다. "
                  "sweep 전체를 해석하기 전에 원인을 확인하라.")


def cmd_seed_sweep(args) -> None:
    """채택 모델(late fusion: tabular+SimpleCNN)의 시드 강건성 검정.

    공개 수치(OS 0.7143 / PFS 0.6621)는 전부 seed=42 단일 실행이다. seed 만
    바꿔 같은 레시피를 재실행해 "late fusion > tabular_only" 가 seed 운인지 본다.
    간격 100 인 142/242/... 를 쓰는 이유는 실험7/실험10 의 시드와 직접 대조
    가능하게 하려는 것이다 (``analyze_late_fusion.py seed-sweep`` 의 교차검증).
    """
    if args.smoke:
        # max_folds=1 은 쓰지 않는다: combine_two 는 "모든 환자가 어느 fold 에선가
        # test 였다"를 요구하는데(각 환자의 OOF 는 그 fold 에서만 생긴다), fold 를
        # 하나만 돌리면 나머지 환자의 위험점수가 없어 KeyError 가 난다.
        args.tab_epochs, args.img_epochs = 2, 2

    if not args.smoke:
        assert_seeds_independent(args.seeds)
        bad = {k: getattr(args, k) for k in REFERENCE_TRAIN_CONF
               if getattr(args, k) != REFERENCE_TRAIN_CONF[k]}
        if bad and not args.allow_nonreference_train:
            raise SystemExit(
                f"[seed_sweep] 기준 학습 조건과 다름 {bad} (기준 {REFERENCE_TRAIN_CONF}). "
                "seed 만 바꾸는 절제실험이므로 학습 조건이 흔들리면 비교 자체가 무의미해진다. "
                "의도한 것이면 --allow_nonreference_train 를 붙여라.")

    out_root = args.out_dir or (SWEEP_OUT_DIR if args.report_encoder == "tfidf"
                                else f"{SWEEP_OUT_DIR}_{args.report_encoder}")
    assert_safe_out_dir(out_root)
    for sub in ("oof", "logs"):
        os.makedirs(os.path.join(out_root, sub), exist_ok=True)
    jsonl_path = os.path.join(out_root, "runs.jsonl")

    # 판독지 인코더: RadBERT 는 frozen + no_grad 라 fold/seed 와 무관하므로
    # 임베딩을 한 번만 계산해 모든 실행이 공유한다 (fold별 통계는 애초에 없다).
    text_encoder_fn = None
    if args.report_encoder != "tfidf":
        from sclc import features
        print(f"[seed_sweep] {args.report_encoder} 임베딩 계산 중 (1회, 모든 seed/fold 공용)...")
        corpus, _ = features.load_text_corpus(cohort.DEFAULT_MERGED_CSV)
        text_encoder_fn = build_encoder(args.report_encoder).build_encoder_fn(corpus)

    cohort_df = cohort.load_trimodal_cohort(fix_brain_meta=args.fix_brain_meta)

    for target in args.targets:
        log_path = os.path.join(out_root, "logs", f"{target}.log")
        for seed in args.seeds:
            planned = {"seed": seed, "target": target, "report_encoder": args.report_encoder,
                       "tab_epochs": args.tab_epochs, "img_epochs": args.img_epochs,
                       "fix_brain_meta": args.fix_brain_meta, "max_folds": args.max_folds}
            if run_key(planned) in done_keys(jsonl_path):
                print(f"[skip] seed={seed}/{target} (이미 runs.jsonl 에 있음)")
                continue

            print(f"[seed_sweep/{target}] seed={seed} 시작...")
            run = SeedRun(target, out_dir=os.path.join(out_root, "ckpt", f"seed{seed}"),
                          seed=seed, tab_epochs=args.tab_epochs, tab_batch=args.tab_batch,
                          img_epochs=args.img_epochs, img_batch=args.img_batch,
                          max_folds=args.max_folds, fix_brain_meta=args.fix_brain_meta,
                          text_encoder_fn=text_encoder_fn, cohort_df=cohort_df)
            os.makedirs(run.out_dir, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as log:
                ctx = () if args.verbose else (contextlib.redirect_stdout(log),
                                               contextlib.redirect_stderr(log))
                with contextlib.ExitStack() as stack:
                    for c in ctx:
                        stack.enter_context(c)
                    record, oof = run.run(report_encoder=args.report_encoder)

            with open(jsonl_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            if not args.smoke:
                with open(os.path.join(out_root, "oof", f"seed{seed}_{target}.json"),
                          "w", encoding="utf-8") as fh:
                    json.dump(oof, fh, ensure_ascii=False)

            print(f"[seed_sweep/{target}] seed={seed}: "
                  f"tabular={record['tabular_only']['mean']:.4f}  "
                  f"image={record['image_simplecnn_only']['mean']:.4f}  "
                  f"late={record['late_simplecnn']['mean']:.4f}  "
                  f"Δ={record['delta_late_minus_tab']:+.4f}  ({record['elapsed_s']:.0f}s)")
            if not args.smoke:
                anchor_check(seed, target, record)

    print(f"\n-> {jsonl_path}")
    print("다음: python experiments/실험1_기본융합_early_late/analyze_late_fusion.py seed-sweep")


# ===========================================================================
# CLI
# ===========================================================================
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True, metavar="COMMAND")

    b = sub.add_parser("method-b", help="2축 late fusion (채택 모델)")
    b.add_argument("--targets", type=comma_list, default=["os", "pfs"])
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--tab_epochs", type=int, default=60)
    b.add_argument("--img_epochs", type=int, default=30)
    b.add_argument("--resnet_epochs", type=int, default=30)
    b.add_argument("--max_folds", type=int, default=None)
    b.add_argument("--out_dir", default=None, help="기본 outputs/late_fusion_B")
    b.add_argument("--out", default=None, help="결과 JSON 경로 (기본 <out_dir>/results.json)")
    b.add_argument("--smoke", action="store_true", help="빠른 점검 (1 fold, 2 에포크)")
    b.add_argument("--no_fix_brain_meta", dest="fix_brain_meta", action="store_false",
                   help="brain_meta 누수 수정을 끄고 2026-08-02 이전 동작으로 (legacy 재현용)")
    b.set_defaults(func=cmd_method_b, fix_brain_meta=True)

    t = sub.add_parser("three-way", help="3축 가중합 (임상·판독지·영상)")
    t.add_argument("--target", default="os", choices=("os", "pfs"))
    t.add_argument("--batch_size", type=int, default=32)
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--out_dir", default=None, help="기본 outputs/late_fusion_3modal_rerun")
    t.set_defaults(func=cmd_three_way)

    s = sub.add_parser("seed-sweep", help="채택 레시피를 seed 만 바꿔 반복")
    s.add_argument("--seeds", type=lambda v: [int(x) for x in comma_list(v)],
                   default=list(DEFAULT_SEEDS))
    s.add_argument("--targets", type=comma_list, default=["os", "pfs"])
    s.add_argument("--tab_epochs", type=int, default=60)
    s.add_argument("--tab_batch", type=int, default=32)
    s.add_argument("--img_epochs", type=int, default=30)
    s.add_argument("--img_batch", type=int, default=16)
    s.add_argument("--max_folds", type=int, default=None)
    s.add_argument("--report_encoder", choices=("tfidf", "radbert"), default="tfidf",
                   help="tfidf(기본) 또는 radbert -- sclc.encoders 의 채택 레시피를 그대로 쓴다")
    s.add_argument("--out_dir", default=None,
                   help="기본: 인코더별로 분리 (tfidf -> outputs/late_fusion_seed_sweep, "
                        "radbert -> ..._radbert). 결과가 섞이지 않게 한다.")
    s.add_argument("--smoke", action="store_true", help="빠른 점검 (2 에포크) -- 통계에 안 씀")
    s.add_argument("--no_fix_brain_meta", dest="fix_brain_meta", action="store_false")
    s.add_argument("--allow_nonreference_train", action="store_true")
    s.add_argument("--verbose", action="store_true", help="학습 로그를 파일 대신 화면으로")
    s.set_defaults(func=cmd_seed_sweep, fix_brain_meta=True)
    return ap


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
