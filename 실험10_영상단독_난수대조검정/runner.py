# -*- coding: utf-8 -*-
"""복제(replicate) 실행기 — 팔(arm) 하나를 N번 돌려 jsonl 로 적재한다.

네 팔 전부 **완전히 같은 학습 절차**(`core.train.ImageOnlyEvaluator`, 30 epoch,
bs16, lr1e-4, wd1e-4, resize512, val 최고 체크포인트 선택)를 쓴다. 팔마다
달라지는 것은 딱 하나씩이다.

    A real_baseline  진짜 라벨, seed 42          → 기존 공개값 재현 확인용
    B real_seeds     진짜 라벨, seed 만 바뀜      → 관측값의 학습잡음 분포
    C perm_null      라벨 순열 (주력)            → 귀무분포
    D noise_image    이미지만 균일 난수          → 바닥 점검

복제 r 의 시드는 ``42 + 100*r`` 이며 순열 생성과 학습 초기화에 같이 쓴다.
r=0 이 곧 A(=기존 조건)다. C 가 순열과 초기화 양쪽으로 흔들려 B 보다 분산이
큰 것은 의도된 것이다 — 절차 전체의 무작위성이 귀무에 들어가야 한다.

체크포인트는 복제마다 2MB×5 개가 나오는데 쓸 데가 없으므로 즉시 지운다.
중단되어도 jsonl 에 남은 복제는 건너뛰고 이어서 돈다.
"""
import contextlib
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np

import nulls

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core import cohort  # noqa: E402
from core.train import ImageOnlyEvaluator  # noqa: E402

BASE_SEED = 42
ARMS = ("A", "B", "C", "D")
ARM_NAME = {"A": "real_baseline", "B": "real_seeds", "C": "perm_null", "D": "noise_image"}


def replicate_seed(replicate: int) -> int:
    return BASE_SEED + 100 * replicate


def _evaluator_kwargs(arm: str, target: str, seed: int, scope: str, ckpt_dir: str,
                      image_dir: str, epochs: int, num_workers: int) -> dict:
    kwargs = dict(target=target, epochs=epochs, seed=seed, save_dir=ckpt_dir,
                  num_workers=num_workers, image_dir=image_dir)
    if arm == "C":
        kwargs["cohort_transform"] = nulls.make_label_permuter(target, seed, scope)
    return kwargs


def run_replicate(arm: str, target: str, replicate: int, *, scope: str = "global",
                  epochs: int = 30, num_workers: int = 4, workspace: str,
                  keep_oof: bool = False) -> dict:
    """복제 1회 = 5-fold 학습·평가. 결과 dict 를 돌려준다."""
    seed = replicate_seed(replicate)
    started = time.time()
    tmp = tempfile.mkdtemp(prefix=f"{arm}{replicate}_", dir=workspace)
    image_dir = cohort.DEFAULT_IMAGE_DIR
    try:
        if arm == "D":
            ids = cohort.load_trimodal_cohort()["research_id"].drop_duplicates()
            size = nulls.median_image_size(cohort.DEFAULT_IMAGE_DIR, ids)
            image_dir = nulls.write_noise_images(os.path.join(tmp, "img"), ids, size, seed)

        ev = ImageOnlyEvaluator(**_evaluator_kwargs(
            arm, target, seed, scope, os.path.join(tmp, "ckpt"), image_dir, epochs, num_workers)).run()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    folds = [round(float(c), 6) for c in ev.c_indices]
    record = {
        "arm": arm, "arm_name": ARM_NAME[arm], "target": target, "replicate": replicate,
        "seed": seed, "perm_scope": scope if arm == "C" else None, "epochs": epochs,
        "num_workers": num_workers,
        "folds": folds, "mean": float(np.mean(folds)), "std": float(np.std(folds)),
        "elapsed_s": round(time.time() - started, 1),
    }
    if keep_oof:
        record["oof"] = ev.oof_predictions
    return record


def run_key(record: dict) -> tuple:
    """복제를 유일하게 식별하는 키.

    ``perm_scope``/``epochs`` 를 빼면 사고가 난다: global 로 1~10 을 돌린 뒤
    stratified 로 1~50 을 돌리면 1~10 이 "이미 했음"으로 건너뛰어져, **서로 다른
    귀무 구성 두 개가 한 p값에 섞인다.**
    """
    return (record["arm"], record["target"], record["replicate"],
            record.get("perm_scope"), record.get("epochs"))


def done_keys(jsonl_path: str) -> set:
    if not os.path.exists(jsonl_path):
        return set()
    with open(jsonl_path, encoding="utf-8") as fh:
        return {run_key(json.loads(line)) for line in fh if line.strip()}


def run_arm(arm: str, target: str, replicates, *, out_dir: str, scope: str = "global",
            epochs: int = 30, num_workers: int = 4, quiet: bool = True) -> None:
    """팔 하나를 복제 목록만큼 실행해 ``runs.jsonl`` 에 이어붙인다."""
    os.makedirs(out_dir, exist_ok=True)
    workspace = os.path.join(out_dir, f"_workspace_{os.getpid()}")
    os.makedirs(workspace, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "runs.jsonl")
    log_path = os.path.join(out_dir, "logs", f"{arm}_{target}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    planned = {"arm": arm, "target": target, "perm_scope": scope if arm == "C" else None,
               "epochs": epochs}

    for replicate in replicates:
        # 병렬 프로세스가 같은 jsonl 에 쓰므로 매 복제마다 다시 읽는다.
        if run_key({**planned, "replicate": replicate}) in done_keys(jsonl_path):
            print(f"[skip] {arm}/{target}/r{replicate} (already in runs.jsonl)")
            continue
        with open(log_path, "a", encoding="utf-8") as log:
            ctx = (contextlib.redirect_stdout(log), contextlib.redirect_stderr(log)) if quiet else ()
            with contextlib.ExitStack() as stack:
                for c in ctx:
                    stack.enter_context(c)
                record = run_replicate(arm, target, replicate, scope=scope, epochs=epochs,
                                       num_workers=num_workers, workspace=workspace,
                                       keep_oof=(arm == "A"))

        oof = record.pop("oof", None)
        with open(jsonl_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        if oof is not None:
            oof_name = f"oof_{target}_r{replicate}.json"
            with open(os.path.join(out_dir, oof_name), "w", encoding="utf-8") as fh:
                json.dump(oof, fh, ensure_ascii=False)
        print(f"[{arm}/{target}] r{replicate:<3d} seed={record['seed']:<5d} "
              f"mean={record['mean']:.4f}  folds={record['folds']}  ({record['elapsed_s']}s)")

    shutil.rmtree(workspace, ignore_errors=True)
