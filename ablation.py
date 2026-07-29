# -*- coding: utf-8 -*-
"""Ablation: same 238-patient split, same training loop (TrimodalEvaluator),
only the set of active branches changes. Isolates whether adding the image
branch to clinical+report actually helps or hurts.

Configs (OS target, 5-fold):
  all         : image + clinical + report   (== the reported early_fusion)
  clin_report : clinical + report           (reproduce the 0.7083 2-modal)
  clin_image  : clinical + image
  clin_only   : clinical
  report_only : report
  image_only  : image

Run:  python ablation.py --target os
"""
# ────────────────────────────────────────────────────────────────────────────
# [이 파일이 하는 일 — 큰 그림]
#
# 우리는 폐암(SCLC) 환자 238명의 "생존 예측" 모델을 만든다.
# 환자 한 명마다 3종류의 정보(모달리티)가 있다.
#   1) image    : CT 같은 의료 영상 사진
#   2) clinical : 나이·수치 같은 임상(숫자) 데이터
#   3) report   : 판독 리포트에서 뽑아낸 데이터
#
# 궁금한 것: "영상까지 넣으면 예측이 더 좋아질까, 오히려 방해가 될까?"
# 그래서 '재료를 하나씩 빼보는 실험'(=ablation, 절제 실험)을 한다.
#   예) 임상+리포트만 vs 임상+리포트+영상  → 점수 비교
#
# 이 파일은 그 실험을 자동으로 돌려서 점수표를 뽑아주는 역할을 한다.
# ────────────────────────────────────────────────────────────────────────────

# --- 필요한 도구(라이브러리) 불러오기 ---
import argparse       # 터미널에서 옵션(--target 등)을 받아오는 도구
import json           # 실험 결과를 파일로 저장할 때 쓰는 형식
import os             # 폴더 만들기·경로 다루기

import numpy as np    # 숫자 계산(평균·표준편차 등)을 쉽게 해주는 도구
import torch          # 딥러닝(인공신경망) 핵심 라이브러리
import torch.nn as nn # 신경망의 부품(레이어)들을 담고 있는 모듈
import torch.nn.functional as F  # 신경망에서 쓰는 함수들(normalize 등)

# 우리가 미리 만들어 둔 부품·훈련기를 가져온다.
from model import SimpleCNNBackbone, _MLPBranch  # 영상용/숫자용 신경망 부품
from train import TrimodalEvaluator              # 실제 학습·평가를 담당하는 '훈련기'


# ════════════════════════════════════════════════════════════════════════════
# [모델 설계도] 켜고 끌 수 있는(ablatable) 생존예측 신경망
#
# 비유하자면 '3개의 입'(image·clinical·report)이 각자 정보를 씹어서
# '요약 조각'을 만들고, 그 조각들을 이어붙여(concat) 마지막에 위험 점수 하나를
# 내뱉는 구조다. use_* 스위치로 어떤 입을 쓸지 껐다 켰다 할 수 있다.
# ════════════════════════════════════════════════════════════════════════════
class AblatableConcatDeepSurv(nn.Module):
    """Identical branch shapes to TrimodalConcatDeepSurv; concat only the
    branches enabled by the use_* flags. forward signature is unchanged
    (image, tabular) so it drops into the existing evaluator/data pipeline."""

    # __init__ : 모델을 처음 만들 때 딱 한 번 실행되는 '부품 조립' 단계.
    # (아직 계산은 안 하고, 필요한 신경망 부품들을 준비만 해 둔다.)
    def __init__(self, clinical_dim, report_dim, use_image=True, use_clinical=True,
                 use_report=True, gray_scale=True, image_proj_dim=128, image_dropout=0.2,
                 clinical_hidden=(128, 128, 128, 128), clinical_dropout=0.5,
                 report_hidden=(32, 16), report_dropout=0.3, fusion_dropout=0.3):
        super().__init__()  # 부모(nn.Module)의 준비 과정을 먼저 실행 (필수 규칙)
        # 세 개를 전부 끄면 쓸 재료가 하나도 없으니, 최소 하나는 켜져 있어야 함을 보장.
        assert use_image or use_clinical or use_report
        self.clinical_dim = int(clinical_dim)  # 임상 데이터의 칸(열) 개수 기억
        self.report_dim = int(report_dim)      # 리포트 데이터의 칸(열) 개수 기억
        # 어떤 입(모달리티)을 쓸지 스위치 상태를 저장해 둔다 (나중에 forward에서 참고).
        self.use_image, self.use_clinical, self.use_report = use_image, use_clinical, use_report

        # fused_dim : 마지막에 이어붙일 '요약 조각'들의 총 길이. 켠 만큼 늘어난다.
        fused_dim = 0

        # ── (1) 영상용 부품 ── 켜져 있을 때만 조립
        if use_image:
            # 사진에서 특징을 뽑는 CNN. 흑백이면 입력 채널 1, 컬러면 3.
            self.backbone = SimpleCNNBackbone(in_channels=1 if gray_scale else 3, output_dim=512)
            # CNN이 뽑은 512개 특징을 128개로 줄이는 작은 신경망(선형→ReLU→Dropout).
            # (Dropout = 학습 중 일부를 임의로 꺼서 '외우기'를 막는 장치)
            self.img_proj = nn.Sequential(
                nn.Linear(self.backbone.output_dim, image_proj_dim), nn.ReLU(), nn.Dropout(image_dropout))
            fused_dim += image_proj_dim  # 영상 조각 길이(128)만큼 총 길이 증가

        # ── (2) 임상(숫자)용 부품 ── 켜져 있을 때만 조립
        if use_clinical:
            # 숫자 데이터를 여러 층으로 처리하는 MLP(다층 신경망) 부품.
            self.clinical_branch = _MLPBranch(self.clinical_dim, clinical_hidden, clinical_dropout)
            fused_dim += self.clinical_branch.out_dim  # 임상 조각 길이만큼 증가

        # ── (3) 리포트용 부품 ── 켜져 있을 때만 조립
        if use_report:
            self.report_branch = _MLPBranch(self.report_dim, report_hidden, report_dropout)
            fused_dim += self.report_branch.out_dim   # 리포트 조각 길이만큼 증가

        # 이어붙인 전체 조각에 한 번 더 Dropout을 걸어 과적합(달달 외우기)을 방지.
        self.fusion_dropout = nn.Dropout(fusion_dropout)
        # 마지막 '머리': 이어붙인 특징들을 딱 1개의 위험 점수로 압축하는 선형층.
        self.head = nn.Linear(fused_dim, 1, bias=False)

    # forward : 실제로 데이터가 들어올 때마다 계산이 흘러가는 '작동' 단계.
    # image = 사진 묶음, tabular = 숫자 데이터 묶음(임상+리포트가 한 줄로 붙어 있음).
    def forward(self, image, tabular):
        feats = []  # 각 입에서 나온 '요약 조각'들을 담아둘 바구니

        # (1) 영상을 쓰면: CNN → 128차원 축소 → 길이 1로 정규화(normalize) 후 담기.
        #     normalize = 조각의 '크기'를 일정하게 맞춰 서로 공평하게 섞이도록 함.
        if self.use_image:
            feats.append(F.normalize(self.img_proj(self.backbone(image)), dim=1))

        # tabular(숫자 묶음)를 앞부분=임상, 뒷부분=리포트로 잘라 나눈다.
        clinical_x = tabular[:, : self.clinical_dim]  # 앞에서 clinical_dim개 = 임상
        report_x = tabular[:, self.clinical_dim:]     # 그 뒤 나머지 전부 = 리포트

        # (2) 임상을 쓰면: 임상 MLP 통과 → 정규화 → 바구니에 담기.
        if self.use_clinical:
            feats.append(F.normalize(self.clinical_branch(clinical_x), dim=1))

        # (3) 리포트를 쓰면: 리포트 MLP 통과 → 정규화 → 바구니에 담기.
        if self.use_report:
            feats.append(F.normalize(self.report_branch(report_x), dim=1))

        # 바구니 속 조각들을 옆으로 이어붙이고(concat), Dropout을 한 번 건다.
        fused = self.fusion_dropout(torch.cat(feats, dim=1))
        # 이어붙인 특징을 최종 위험 점수(숫자 1개)로 변환해 돌려준다.
        return self.head(fused)


# ── 실험 조합표 ──
# 이름(name) : 어떤 입(모달리티)을 켤지(True/False) 정의한 사전(dict).
# 예) "clin_report"는 영상은 끄고, 임상과 리포트만 켠 조합 → clin_only, report_only와
#     비교해서 "임상+리포트를 합치면 각각보다 나은가?"를 확인하는 용도.
CONFIGS = {
    "all":         dict(use_image=True,  use_clinical=True,  use_report=True),   # 셋 다 사용
    "clin_report": dict(use_image=False, use_clinical=True,  use_report=True),   # 임상+리포트만
    "clin_image":  dict(use_image=True,  use_clinical=True,  use_report=False),  # 임상+영상만
    "clin_only":   dict(use_image=False, use_clinical=True,  use_report=False),  # 임상만
    "report_only": dict(use_image=False, use_clinical=False, use_report=True),   # 리포트만
    "image_only":  dict(use_image=True,  use_clinical=False, use_report=False),  # 영상만
}


# make_factory : "이런 스위치 조합(flags)으로 모델을 만들어줘"라는 주문을 받아서,
# 실제로 모델을 찍어내는 '공장 함수(factory)'를 만들어 돌려준다.
# (TrimodalEvaluator가 나중에 clinical_dim, report_dim을 알게 됐을 때
#  이 factory(clinical_dim, report_dim)를 호출해서 모델을 생성함)
def make_factory(flags):
    def factory(clinical_dim, report_dim):
        return AblatableConcatDeepSurv(clinical_dim, report_dim, **flags)
    return factory


# ════════════════════════════════════════════════════════════════════════════
# [실행 시작점] 이 파일을 터미널에서 실행하면 여기부터 시작된다.
# ════════════════════════════════════════════════════════════════════════════
def main():
    # ── 터미널 옵션 정의 ──
    # 예) python ablation.py --target os --epochs 30
    #     처럼 실행할 때 뒤에 붙이는 옵션들을 여기서 받는다.
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="os", choices=("os", "pfs"))  # 예측 대상: 전체생존(os) or 무진행생존(pfs)
    ap.add_argument("--configs", default="all,clin_report,clin_image,clin_only,report_only,image_only")  # 돌릴 조합 목록(콤마로 구분)
    ap.add_argument("--epochs", type=int, default=30)     # 학습을 몇 바퀴(epoch) 돌릴지
    ap.add_argument("--batch_size", type=int, default=16) # 한 번에 몇 명씩 묶어서 학습할지
    ap.add_argument("--tag", default="")                  # 결과 저장 폴더 이름에 붙일 꼬리표
    args = ap.parse_args()  # 실제로 터미널 입력을 읽어서 args에 저장

    results = {}  # 각 조합(config)의 결과 점수를 모아둘 저장소

    # ── 조합 하나씩 순서대로 실험 ──
    # "--configs"로 받은 문자열을 콤마 기준으로 쪼개서, 조합 이름을 하나씩 꺼낸다.
    for name in args.configs.split(","):
        name = name.strip()        # 앞뒤 공백 제거 (예: " clin_only" → "clin_only")
        flags = CONFIGS[name]      # 위에서 정의한 조합표에서 해당 스위치 설정을 꺼냄

        print(f"\n########## CONFIG: {name}  ({flags})  target={args.target} bs={args.batch_size} ep={args.epochs} ##########")

        # TrimodalEvaluator = 실제로 데이터를 5등분(5-fold)해서
        # 학습→평가를 반복해주는 '훈련 관리자'. model_factory에 방금 만든
        # 공장 함수를 넘겨줘서, 이 조합에 맞는 모델을 안에서 찍어내게 한다.
        ev = TrimodalEvaluator(
            target=args.target, epochs=args.epochs, batch_size=args.batch_size,
            save_dir=f"outputs/ablation{args.tag}/{name}_{args.target}",  # 결과 저장 폴더
            model_factory=make_factory(flags),
        ).run()  # .run() 이 실제로 학습+평가를 전부 실행하는 부분

        cis = ev.c_indices  # 5-fold 각각의 성능 점수(C-index, 높을수록 예측을 잘한 것)

        # ── 과적합 정도 측정 ──
        # 학습 데이터 점수(train_cindex)와 검증 점수(val_cindex)의 차이가 크면
        # "외운 것"(과적합), 작으면 "제대로 배운 것". 각 fold에서 검증 점수가
        # 가장 좋았던 순간을 기준으로 둘의 격차를 재서 평균낸다.
        gaps = []
        for fold in {row["fold"] for row in ev.training_history}:
            rows = [r for r in ev.training_history if r["fold"] == fold]
            best = max(rows, key=lambda r: r["val_cindex"])  # 검증이 가장 좋았던 epoch
            gaps.append(best["train_cindex"] - best["val_cindex"])

        # 평균, 표준편차(점수가 얼마나 들쭉날쭉한지), 그리고 fold별 점수를 저장.
        results[name] = {
            "mean": float(np.mean(cis)),
            "std": float(np.std(cis)),
            "folds": [round(c, 4) for c in cis],
            "train_val_gap_mean": float(np.mean(gaps)) if gaps else None,
            "flags": flags,
            "fold_records": ev.fold_records,      # fold별 표본 수·사건(event) 수
            "training_history": ev.training_history,  # epoch별 학습곡선
        }
        print(f"[ABLATION] {name} {args.target}: {np.mean(cis):.4f} +/- {np.std(cis):.4f}  "
              f"folds={[round(c,4) for c in cis]}  train-val gap={np.mean(gaps):.4f}")

    # ── 모든 조합이 끝난 뒤, 표 형태로 요약 출력 ──
    print("\n================ ABLATION SUMMARY (target=%s) ================" % args.target)
    print(f"{'config':<14}{'mean':>8}{'std':>8}{'gap':>8}   folds")
    for name, r in results.items():
        print(f"{name:<14}{r['mean']:>8.4f}{r['std']:>8.4f}{r['train_val_gap_mean']:>8.4f}   {r['folds']}")

    # ── 결과를 JSON으로 저장 ──
    # 예전에는 터미널 출력만 남아서 나중에 숫자를 다시 찾을 수 없었음.
    # 이제는 파일로 남겨서 그래프 그리기·재현·비교에 바로 쓸 수 있게 한다.
    out_dir = f"outputs/ablation{args.tag}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"results_{args.target}.json")
    with open(out_path, "w") as f:
        json.dump({
            "target": args.target, "batch_size": args.batch_size, "epochs": args.epochs,
            "configs": results,
        }, f, indent=2)
    print(f"\nwrote {out_path}")


# 이 파일을 "직접" 실행했을 때만(예: python ablation.py) main()을 호출한다.
# 다른 파일에서 이 파일을 import만 할 때는 자동으로 실행되지 않도록 막아주는 관용구.
if __name__ == "__main__":
    main()
