# -*- coding: utf-8 -*-
"""귀무가설(H0) 생성기 — 영상과 생존을 끊는 두 가지 방법.

이 실험이 묻는 것은 "영상 단독 C-index 0.657(OS)/0.615(PFS) 가 우연으로도
나올 수 있는 값인가" 이다. 그러려면 **영상과 생존의 연결만 끊고 나머지는 전부
그대로인** 데이터가 필요하다. 여기 두 방법이 있다.

라벨 순열 (permute_labels)
    (days, event) 를 **쌍으로 묶어** 환자 간에 재배치한다. 쌍을 깨고 따로
    섞으면 중도절단 구조가 무너져 인위적인 귀무가 되므로 절대 하면 안 된다.
    쌍으로 섞으면 KM 곡선·이벤트 총수·관찰기간 분포가 원본과 완전히 같다.

    scope="global"      238명 전체에 순열 하나. 교과서적 순열검정.
    scope="stratified"  event 상태 안에서만 섞는다. event=1 환자는 event=1 쌍만
                        받으므로 **fold 별 이벤트 수까지 정확히 보존**된다.
                        (보수적 변형. H0 아래에서는 둘 다 타당하다.)

잡음 이미지 (write_noise_images)
    라벨은 진짜로 두고 이미지를 균일 난수로 바꾼다. "정보가 0인 입력에서도
    파이프라인이 0.5 를 넘는가"를 보는 바닥 점검이다. 크기를 환자별 원본대로
    두면 원본 폭/높이가 그대로 남는데(폭 1개만으로 OS 0.571 이 나온다) 그러면
    바닥이 아니게 되므로, **전 환자를 하나의 고정 크기**로 만든다.
"""
import os

import numpy as np
import pandas as pd
from PIL import Image

LABEL_COLS = ("days", "event")


def permute_labels(cohort_df: pd.DataFrame, target: str, seed: int,
                   scope: str = "global") -> pd.DataFrame:
    """``{target}_days``/``{target}_event`` 를 쌍으로 묶어 환자 간에 재배치한 사본.

    ``cohort_df`` 는 환자당 5행(fold 마다 1행)이므로 research_id 단위로 라벨을
    한 번 뽑아 섞은 뒤 다시 전 행에 매핑한다. 한 환자의 라벨이 fold 마다 달라지는
    사고를 구조적으로 막는다.
    """
    if scope not in ("global", "stratified"):
        raise ValueError(f"scope must be 'global' or 'stratified', got {scope!r}")

    day_col, evt_col = f"{target}_days", f"{target}_event"
    per_patient = (cohort_df.drop_duplicates("research_id")
                            .set_index("research_id")[[day_col, evt_col]]
                            .sort_index())
    ids = per_patient.index.to_numpy()
    pairs = per_patient.to_numpy(dtype=float)          # (n, 2) = [days, event]

    rng = np.random.default_rng(seed)
    order = np.arange(len(ids))
    if scope == "global":
        order = rng.permutation(order)
    else:
        for value in np.unique(pairs[:, 1]):
            block = np.flatnonzero(pairs[:, 1] == value)
            order[block] = rng.permutation(block)

    shuffled = pd.DataFrame(pairs[order], index=ids, columns=[day_col, evt_col])
    out = cohort_df.copy()
    out[day_col] = out["research_id"].map(shuffled[day_col]).astype(cohort_df[day_col].dtype)
    out[evt_col] = out["research_id"].map(shuffled[evt_col]).astype(cohort_df[evt_col].dtype)
    return out


def make_label_permuter(target: str, seed: int, scope: str = "global"):
    """``ImageOnlyEvaluator(cohort_transform=...)`` 에 그대로 넣는 훅."""
    return lambda cohort_df: permute_labels(cohort_df, target, seed, scope)


def write_noise_images(dst_dir: str, research_ids, size: tuple[int, int], seed: int) -> str:
    """모든 환자를 같은 크기의 균일 난수 PNG 로 대체한 임시 이미지 폴더를 만든다."""
    os.makedirs(dst_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    w, h = size
    for rid in research_ids:
        arr = rng.integers(0, 256, size=(h, w), dtype=np.uint8)
        Image.fromarray(arr, mode="L").save(os.path.join(dst_dir, f"{int(rid)}.png"))
    return dst_dir


def median_image_size(image_dir: str, research_ids) -> tuple[int, int]:
    """코호트 원본 PNG 크기의 중앙값 (잡음 이미지의 고정 크기로 쓴다)."""
    sizes = np.array([Image.open(os.path.join(image_dir, f"{int(r)}.png")).size
                      for r in research_ids])
    return int(np.median(sizes[:, 0])), int(np.median(sizes[:, 1]))
