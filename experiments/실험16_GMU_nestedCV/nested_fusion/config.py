# -*- coding: utf-8 -*-
"""실험16 설정값. 데이터 분할 위치, seed, 인코더·융합 모델의 학습 규칙을 한곳에 모아 둔다."""

import os

from sclc import paths

# 경로 · seed
SPLIT_DIR = os.path.join(paths.SPLITS_DIR, "gmu_nested_v1")
SEED = 42
RUN_TAG = "run_v2"

# 인코더 학습 규칙 (tabular = 임상+판독지, image = PET MIP)
ENCODER_STOP = dict(monitor="loss", min_delta=1e-4, patience=10)
ENCODERS = {
    "tabular": dict(model="ConcatDeepSurv clin_report (clinical MLP 128x4 + report RadBERT 768->32->16, L2)",
                    max_epochs=60, batch_size=32, optimizer="Adam", lr=1e-4, weight_decay=1e-4,
                    stop_metric="S Efron NLL", min_epoch=20, **ENCODER_STOP),
    "image": dict(model="ImageOnlyDeepSurv (SimpleCNN 512 -> Dropout 0.2 -> Linear)",
                  max_epochs=30, batch_size=16, optimizer="Adam", lr=1e-4, weight_decay=1e-4,
                  stop_metric="S Efron NLL", min_epoch=1, **ENCODER_STOP),
}
# --smoke 경로 점검용 짧은 학습
SMOKE = dict(encoder_epochs=2, gmu=dict(max_epochs=3, patience=40, min_epoch=1, min_delta=1e-4))

# 모달리티
MODALITIES = ("clinical", "report", "image")
FEATURE_DIMS = {"clinical": 128, "report": 16, "image": 512}
ARMS = ("tabular", "image")

# 특징 수준 융합 모델(ConcatFusion · FixedGateFusion · JointGMU): 하이퍼파라미터 후보 G1~G6 와 선택 규칙
CANDIDATES = {
    "G1": dict(d=16, bottleneck=8, lr=1e-4, weight_decay=1e-3, dropout=0.30),
    "G2": dict(d=16, bottleneck=16, lr=1e-4, weight_decay=1e-3, dropout=0.30),
    "G3": dict(d=32, bottleneck=8, lr=1e-4, weight_decay=1e-3, dropout=0.30),
    "G4": dict(d=16, bottleneck=8, lr=1e-3, weight_decay=1e-3, dropout=0.30),
    "G5": dict(d=16, bottleneck=8, lr=1e-4, weight_decay=1e-4, dropout=0.30),
    "G6": dict(d=16, bottleneck=8, lr=1e-4, weight_decay=1e-3, dropout=0.40),
}
USED_KEYS = {"B_concat": ("d", "lr", "weight_decay", "dropout"),
             "C_fixed_gate": ("d", "lr", "weight_decay", "dropout"),
             "D_gmu": ("d", "bottleneck", "lr", "weight_decay", "dropout")}
SIMPLICITY_ORDER = ("G1", "G4", "G5", "G6", "G2", "G3")    # 성능이 비슷하면 앞쪽(단순한) 후보를 고른다
SELECTION_TOLERANCE = 0.005
# 학습률별 최대 epoch · patience
LR_RULES = {1e-4: dict(max_epochs=5000, patience=400), 1e-3: dict(max_epochs=500, patience=40)}
COMMON_RULE = dict(min_epoch=30, min_delta=1e-4)
TRAIN_RULE = {**LR_RULES[1e-3], **COMMON_RULE}

# GatedLateFusion: gate 계수 L2 페널티 후보
GATED_LATE_LAMBDAS = (0.01, 0.1, 1.0, 10.0, 100.0)


# 분할·누출 검사에 실패하면 결과를 저장하지 않고 바로 멈춘다
class GuardError(RuntimeError):
    """분할 불일치·데이터 누출·NaN 같은 문제."""


def guard(ok: bool, msg: str) -> None:
    if not ok:
        raise GuardError(f"[GUARD] {msg}")
