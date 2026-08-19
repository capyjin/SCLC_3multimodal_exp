# -*- coding: utf-8 -*-
"""영상 단독 난수대조 검정 — 학습 팔(A~D) 실행 CLI.

물음: **영상 단독** C-index(OS 0.6570 / PFS 0.6154)는 우연으로도 나오는 값인가?
기존 §10.4 난수대조는 이미 계산된 영상 위험점수를 tabular 와 결합한 뒤 섞은
것이라 "tabular 위에 얹는 증분"만 쟀다. 여기서는 영상 파이프라인 **자체**를
귀무 라벨 위에서 처음부터 다시 학습시켜 영상 단독의 귀무분포를 만든다.

Run:
    # ① 재현 확인 + 바닥 점검 (약 20분)
    python 실험10_영상단독_난수대조검정/exp_permutation.py --arms A,D
    # ② 파일럿 — 귀무 평균이 0.5 근처인지 먼저 본다 (약 80분)
    python 실험10_영상단독_난수대조검정/exp_permutation.py --arms C --replicates 1-10
    # ③ 본 실행. 3개 프로세스로 쪼개 돌려도 같은 runs.jsonl 에 안전하게 쌓인다
    python 실험10_영상단독_난수대조검정/exp_permutation.py --arms C --replicates 11-100
    python 실험10_영상단독_난수대조검정/exp_permutation.py --arms B

결과는 ``outputs/image_permutation/runs.jsonl`` 에 복제당 한 줄씩 쌓이고,
``analyze.py`` 가 그걸 읽어 p값·표·그림을 만든다.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import runner  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "outputs", "image_permutation")
DEFAULT_REPLICATES = {"A": (0, 0), "B": (1, 20), "C": (1, 100), "D": (1, 10)}


def parse_range(text: str) -> list[int]:
    out: list[int] = []
    for chunk in text.split(","):
        if "-" in chunk:
            lo, hi = chunk.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(chunk))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default="A", help=f"실행할 팔 (쉼표 구분). 선택지: {','.join(runner.ARMS)}")
    ap.add_argument("--targets", default="os,pfs")
    ap.add_argument("--replicates", default=None,
                    help="복제 번호 범위(예: 1-50 또는 1,3,7). 생략 시 팔별 기본값")
    ap.add_argument("--scope", default="global", choices=("global", "stratified"),
                    help="라벨 순열 범위 (C 팔에만 적용). global = 주력(귀무 중심 0.50), "
                         "stratified = 중도절단 조건부 검정(귀무 중심 0.53)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--verbose", action="store_true", help="학습 로그를 파일 대신 화면으로")
    args = ap.parse_args()

    for arm in args.arms.split(","):
        arm = arm.strip().upper()
        if arm not in runner.ARMS:
            raise SystemExit(f"unknown arm {arm!r}; choose from {runner.ARMS}")
        lo, hi = DEFAULT_REPLICATES[arm]
        replicates = parse_range(args.replicates) if args.replicates else list(range(lo, hi + 1))
        for target in args.targets.split(","):
            print(f"\n=== arm {arm} ({runner.ARM_NAME[arm]}) · {target} · "
                  f"복제 {replicates[0]}~{replicates[-1]} ({len(replicates)}회) ===")
            runner.run_arm(arm, target.strip(), replicates, out_dir=args.out_dir,
                           scope=args.scope, epochs=args.epochs,
                           num_workers=args.num_workers, quiet=not args.verbose)


if __name__ == "__main__":
    main()
