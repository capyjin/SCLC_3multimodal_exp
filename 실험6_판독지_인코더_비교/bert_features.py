# -*- coding: utf-8 -*-
"""판독지 텍스트를 frozen 영어 biomedical BERT(RadBERT) 임베딩으로 바꾼다.

[동기]
  지금 판독지는 char n-gram TF-IDF(400차원)로만 쓰인다. TF-IDF 는 "글자 조각의
  빈도"라서 의미를 모른다 — "no evidence of metastasis" 와 "metastasis" 가
  거의 같은 벡터로 간다. 의학 코퍼스로 사전학습된 BERT 는 그 차이를 안다.
  그래서 텍스트 인코더만 TF-IDF -> RadBERT 로 바꿔서 이득이 있는지 본다.

[한국어 문제 — 이 실험이 3개 arm 으로 나뉘는 이유]
  판독지는 영어 의학용어(~64%)와 한국어(~9%, 주로 조사 + 정형화된 서술어)가
  섞여 있다. RadBERT 의 tokenizer 는 영어 전용이라 **한국어를 전부 [UNK] 로**
  바꿔 버린다 (이 코퍼스에서 실측 평균 16.3% 토큰이 [UNK]).
  [UNK] 가 성능을 깎는 원인인지 아닌지를 분리하려고 세 갈래로 잰다:
    bert_raw   : 원문 그대로 (한국어 -> [UNK])
    bert_nokr  : 한국어 글자를 지우고 영어만 남김 (strip_korean)
    bert_ko2en : 한국어 덩어리를 영어 구로 치환 (translate_korean)
  ``tokenization_stats()`` 로 각 arm 의 [UNK] 비율이 실제로 줄었는지 확인한다.

[누수 방지 — 이 저장소의 기존 규율과 동일]
  BERT forward pass 는 **문서 하나만 보고 계산되는 frozen 연산**이다. 다른
  환자의 정보가 개입할 여지가 구조적으로 없고 학습(fit)도 하지 않으므로,
  fold 와 무관하게 전역 1회 계산해도 누수가 아니다 (suv_features.py 의
  정규식 파싱과 같은 성격).
  반대로 **환자들을 가로질러 계산되는 통계는 전부 fold 별로** 잡는다:
  768차원을 줄이는 SVD 기저와 StandardScaler(mean/std) 두 가지이며, 둘 다
  train fold 환자 행으로만 fit 하고 val/test 는 transform 만 한다.
  실제 쓰인 기저의 설명분산·scaler 노름·표본수는 audit 리스트에 남겨서
  사후에 "정말 train 만 봤나"를 감사할 수 있게 한다.

[환자 정보 보호]
  raw 텍스트는 stdout/파일 어디에도 찍지 않는다. 캐시(.npz)에는 research_id 와
  float 임베딩만 들어가고, 캐시 파일명은 텍스트 내용의 sha256 앞 16자리라
  파일명만으로 내용이 드러나지 않는다. 통계는 전부 집계값(평균/비율)만 낸다.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import hashlib

import numpy as np
import torch

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from core import features

DEFAULT_MODEL = "StanfordAIMI/RadBERT"
DEFAULT_CACHE_DIR = "outputs/bert_cache"
# SVD 는 부호/초기화에 난수가 쓰이므로 재현성을 위해 고정한다.
SVD_RANDOM_STATE = 42


def _corpus_fingerprint(model_name: str, texts: dict[int, str], max_length: int) -> str:
    """(모델, max_length, 텍스트 내용) 에 대한 결정론적 지문.

    캐시 키로 쓴다. 텍스트가 한 글자라도 바뀌면(=arm 이 바뀌면) 다른 키가 되어
    엉뚱한 캐시를 재사용하는 사고가 나지 않는다. 앞 16자리만 파일명에 쓰므로
    파일명에서 원문을 되돌릴 수 없다.
    """
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))
    h.update(b"\x00")
    h.update(str(int(max_length)).encode("utf-8"))
    for rid in sorted(texts):
        h.update(b"\x00")
        h.update(str(int(rid)).encode("utf-8"))
        h.update(b"\x00")
        h.update(texts[rid].encode("utf-8"))
    return h.hexdigest()[:16]


def embed_corpus(model_name: str, texts: dict[int, str], max_length: int = 512,
                 batch_size: int = 16, device: str = "cuda",
                 cache_dir: str = DEFAULT_CACHE_DIR) -> dict[int, np.ndarray]:
    """{research_id: 768차원 float32 임베딩} 을 만든다 (frozen, 학습 없음).

    **누수 없음이 구조적으로 보장된다**: 모델 가중치는 사전학습된 채로 고정
    (``eval()`` + ``torch.no_grad()``)이고, 임베딩은 문서 **하나**만 입력으로
    받아 계산된다. 어떤 fit() 도 호출하지 않으므로 다른 환자(val/test)의 정보가
    train 쪽으로 흘러들 경로 자체가 없다. 따라서 fold 와 무관하게 전역 1회
    계산해도 된다. (fold 별로 잡아야 하는 통계는 make_text_encoder_fn 쪽에 있다.)

    풀링은 **[CLS] 가 아니라 attention mask 가중 mean-pooling** 이다.
    [CLS] 는 NSP/분류 objective 로 학습된 토큰이라 pretrain 그대로 쓰면
    문장 표현으로서 품질이 나쁘다는 게 정설이고, mean-pooling 이 frozen
    임베딩의 표준 선택이다. mask 로 pad 토큰을 빼고 평균내므로 배치 안
    padding 길이에 결과가 좌우되지 않는다.

    결과는 ``cache_dir`` 의 .npz 로 캐시된다 (키 = 모델명 + max_length +
    텍스트 내용 해시). 재실행하면 즉시 로드된다.
    """
    os.makedirs(cache_dir, exist_ok=True)
    fp = _corpus_fingerprint(model_name, texts, max_length)
    safe_name = model_name.replace("/", "__")
    cache_path = os.path.join(cache_dir, f"{safe_name}_len{max_length}_{fp}.npz")

    if os.path.exists(cache_path):
        blob = np.load(cache_path)
        ids, mat = blob["research_id"], blob["embedding"]
        print(f"[bert] cache hit {os.path.basename(cache_path)} "
              f"(n={len(ids)}, dim={mat.shape[1]})")
        return {int(r): mat[i].astype("float32") for i, r in enumerate(ids)}

    from transformers import AutoModel, AutoTokenizer

    if device == "cuda" and not torch.cuda.is_available():
        print("[bert] CUDA not available -> falling back to CPU")
        device = "cpu"
    torch_device = torch.device(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(torch_device)
    model.eval()          # dropout/batchnorm 등 학습 전용 동작 끄기
    for p in model.parameters():
        p.requires_grad_(False)   # frozen 임을 명시 (no_grad 와 별개로 이중 보증)

    # research_id 오름차순으로 고정 -> 배치 구성이 실행마다 동일 = 결정론적.
    rids = sorted(int(r) for r in texts)
    vectors = []
    with torch.no_grad():
        for start in range(0, len(rids), batch_size):
            chunk = rids[start:start + batch_size]
            batch_texts = [texts[r] for r in chunk]
            enc = tokenizer(batch_texts, padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt")
            enc = {k: v.to(torch_device) for k, v in enc.items()}
            hidden = model(**enc).last_hidden_state          # (B, T, H)
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)  # (B, T, 1)
            # mean-pooling: pad 토큰(mask=0)을 뺀 실제 토큰들의 평균
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            vectors.append(pooled.cpu().numpy().astype("float32"))

    mat = np.concatenate(vectors, axis=0).astype("float32")
    np.savez_compressed(cache_path, research_id=np.array(rids, dtype="int64"), embedding=mat)
    print(f"[bert] embedded n={len(rids)} docs -> dim={mat.shape[1]}, "
          f"cached to {os.path.basename(cache_path)}")
    return {int(r): mat[i] for i, r in enumerate(rids)}


def tokenization_stats(model_name: str, texts: dict[int, str], max_length: int = 512) -> dict:
    """tokenizer 가 이 텍스트를 어떻게 씹는지에 대한 **집계** 통계.

    한국어 처리 arm(bert_raw/nokr/ko2en)이 정말로 [UNK] 를 줄였는지 확인하는
    용도다. 원문은 일절 반환/출력하지 않고 비율·평균만 낸다 (환자정보 보호).
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    unk_id = tokenizer.unk_token_id
    n_tokens, unk_fracs, n_trunc, n_any_unk = [], [], 0, 0
    for rid in sorted(texts):
        ids = tokenizer(texts[rid], truncation=True, max_length=max_length)["input_ids"]
        full = tokenizer(texts[rid])["input_ids"]
        n_unk = sum(1 for i in ids if i == unk_id)
        n_tokens.append(len(ids))
        unk_fracs.append(n_unk / max(len(ids), 1))
        n_trunc += int(len(full) > max_length)
        n_any_unk += int(n_unk > 0)
    return {
        "n_docs": len(n_tokens),
        "mean_tokens": float(np.mean(n_tokens)),
        "median_tokens": float(np.median(n_tokens)),
        "max_tokens": int(np.max(n_tokens)),
        "unk_frac_mean": float(np.mean(unk_fracs)),
        "unk_frac_max": float(np.max(unk_fracs)),
        "pct_docs_with_unk": float(100.0 * n_any_unk / max(len(n_tokens), 1)),
        "pct_docs_truncated": float(100.0 * n_trunc / max(len(n_tokens), 1)),
    }


def _stack_embeddings(embeddings: dict[int, np.ndarray], ids, emb_dim: int) -> tuple[np.ndarray, int]:
    """id 순서대로 임베딩 행렬을 쌓는다. 임베딩이 없는 환자는 0 벡터로 채운다
    (features.build_fold_multimodal_tabular 가 판독지 없는 환자에 빈 문자열을
    주는 것과 같은 방어). 몇 명이 그랬는지도 같이 돌려준다."""
    ids = [int(i) for i in ids]
    rows, n_missing = [], 0
    for rid in ids:
        vec = embeddings.get(rid)
        if vec is None:
            rows.append(np.zeros(emb_dim, dtype="float32"))
            n_missing += 1
        else:
            rows.append(np.asarray(vec, dtype="float32"))
    if not rows:
        return np.empty((0, emb_dim), dtype="float32"), 0
    return np.vstack(rows).astype("float32"), n_missing


def _reduce_block(mats: dict[str, np.ndarray], out_dim: int, audit_extra: dict,
                  audit: list | None, do_svd: bool = True, do_scale: bool = True):
    """train fold 행으로만 SVD + StandardScaler 를 fit 하고 out_dim 으로 맞춘다.

    [폭을 out_dim 으로 고정하는 이유와 방법]
      SVD 가 뽑을 수 있는 성분 수는 최대 min(n_train, emb_dim) 이다. 이 코호트는
      fold 당 n_train=171 이라 out_dim=400 을 그대로 요구하면 sklearn 이
      **말없이 171개만** 돌려준다(에러도 경고도 없음) -> report 브랜치 폭이
      400 이 아니라 171 로 조용히 바뀌어 TF-IDF 와의 비교가 오염된다.
      그래서 (1) 성분 수를 min(out_dim, n_train, emb_dim) 으로 **명시적으로**
      깎고 경고를 찍은 뒤, (2) 남는 열을 0으로 패딩해 폭을 out_dim 으로 맞춘다.
      0 패딩은 이 저장소의 기존 관행 그대로다 — TfidfEncoder.transform() 도
      vocabulary 가 max_features 에 못 미치면 같은 방식으로 패딩해서
      "report_dim 은 항상 400 고정" 불변식을 지킨다 (exp_text_source.py 참고).
      덕분에 모델 구조·파라미터 수가 TF-IDF arm 과 완전히 동일해진다.
    """
    n_train, emb_dim = mats["train"].shape

    # ── (1) 차원 축소: SVD 기저는 train fold 행으로만 fit ──
    # do_svd=False 면 축소를 건너뛰고 원래 폭을 그대로 쓴다. 축소 파이프라인이
    # 성능을 깎는지(랭크 통제군에서 -0.037 관측) 원인을 SVD/스케일러로 분리하기 위한 스위치.
    if do_svd:
        n_comp = min(int(out_dim), int(n_train), int(emb_dim))
        if n_comp < out_dim:
            print(f"[bert] WARNING: out_dim={out_dim} > min(n_train={n_train}, emb_dim={emb_dim}) "
                  f"-> SVD 성분을 {n_comp}개로 줄이고 나머지 {out_dim - n_comp}열은 0 패딩한다 "
                  f"(폭은 {out_dim} 유지).")
        svd = TruncatedSVD(n_components=n_comp, random_state=SVD_RANDOM_STATE)
        svd.fit(mats["train"])
        reduced = {name: (svd.transform(arr) if len(arr) else np.empty((0, n_comp), dtype="float64"))
                   for name, arr in mats.items()}
        evr = round(float(svd.explained_variance_ratio_.sum()), 6)
    else:
        n_comp = int(emb_dim)
        out_dim = int(emb_dim)       # 축소를 안 하므로 폭은 원래 폭 그대로
        reduced = {name: np.asarray(arr, dtype="float64") for name, arr in mats.items()}
        evr = None

    # ── (2) 표준화: StandardScaler 도 train fold 로만 fit ──
    # do_scale=False 면 건너뛴다. SVD 는 성분을 분산 큰 순서로 정렬해 주는데
    # StandardScaler 가 각 성분을 자기 표준편차로 나누면 그 순서가 지워져서
    # 잡음 성분이 신호 성분과 같은 크기로 증폭된다 — 이게 범인인지 보는 스위치.
    if do_scale:
        scaler = StandardScaler().fit(reduced["train"])
        scaled = {name: (scaler.transform(arr) if len(arr) else np.empty((0, n_comp), dtype="float64"))
                  for name, arr in reduced.items()}
        scaler_mean_norm = round(float(np.linalg.norm(scaler.mean_)), 6)
        scaler_scale_norm = round(float(np.linalg.norm(scaler.scale_)), 6)
    else:
        scaled = reduced
        scaler_mean_norm = scaler_scale_norm = None

    # ── (3) 0 패딩으로 폭을 out_dim 에 고정 ──
    out = {}
    for name, arr in scaled.items():
        block = np.asarray(arr, dtype="float32")
        if n_comp < out_dim:
            pad = np.zeros((block.shape[0], out_dim - n_comp), dtype="float32")
            block = np.concatenate([block, pad], axis=1)
        out[name] = block

    if audit is not None:
        audit.append({
            **audit_extra,
            "n_train": int(n_train), "n_val": int(len(mats["val"])), "n_test": int(len(mats["test"])),
            "emb_dim": int(emb_dim), "out_dim": int(out_dim),
            "n_svd_components": int(n_comp), "n_zero_pad": int(out_dim - n_comp),
            "do_svd": bool(do_svd), "do_scale": bool(do_scale),
            # train fold 로만 fit 했다는 증거로 남기는 값들 (fold마다 달라야 정상)
            "svd_explained_variance_ratio_sum": evr,
            "scaler_mean_norm": scaler_mean_norm,
            "scaler_scale_norm": scaler_scale_norm,
        })
    return out


def make_text_encoder_fn(embeddings: dict[int, np.ndarray], out_dim: int = 400,
                         audit: list | None = None,
                         do_svd: bool = True, do_scale: bool = True):
    """features.build_fold_multimodal_tabular 에 넘길 fold-safe 텍스트 블록 생성기.

    반환 함수는 (train_ids, val_ids, test_ids) 를 받아
    {"train": X, "val": X, "test": X} (float32, 열 개수 = out_dim) 를 돌려준다.
    이 블록이 TF-IDF 블록을 **대체**한다.

    out_dim 기본값 400 은 TF-IDF 와 폭을 정확히 맞추기 위한 값이다. 폭이 달라지면
    모델 파라미터 수와 추정 부담이 같이 달라져서, "인코더가 좋아서" 이긴 건지
    "폭이 달라서" 이긴 건지 구분이 안 된다 (이 프로젝트는 폭을 늘리면 오히려
    나빠지는 걸 반복해서 확인했다). 그래서 폭은 고정한다.
    """
    def fn(train_ids, val_ids, test_ids):
        emb_dim = len(next(iter(embeddings.values())))
        mats, missing = {}, {}
        for name, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
            mats[name], missing[name] = _stack_embeddings(embeddings, ids, emb_dim)
        return _reduce_block(mats, out_dim, {"block": "bert", "n_missing_embedding": missing},
                             audit, do_svd=do_svd, do_scale=do_scale)

    return fn


def make_tfidf_svd_encoder_fn(corpus: dict[int, str], out_dim: int = 400,
                              tfidf_max_features: int = 400, tfidf_ngram_range=(2, 4),
                              audit: list | None = None,
                              do_svd: bool = True, do_scale: bool = True):
    """TF-IDF 를 **BERT arm 과 똑같은 축소 파이프라인**(train-only SVD + scaler)에 태운다.

    [왜 이 통제실험이 필요한가]
      BERT 블록은 fold당 n_train=171 이라 SVD 성분이 171개로 막히고 나머지는
      0 패딩이다 -> **실효 랭크 171**. 반면 TF-IDF 블록은 진짜 400차원이다.
      이 상태로 둘을 비교하면 "BERT 가 졌다"가 "랭크 171 이 랭크 400 에 졌다"와
      뒤섞여 구분되지 않는다.

      그래서 TF-IDF 를 같은 파이프라인에 통과시켜 똑같이 171 로 깎아 본다:
        - TF-IDF(171) 도 0.708 근처면  -> 랭크는 병목이 아니고, BERT 가 내용으로 진 것
        - TF-IDF(171) 이 0.67 근처로 떨어지면 -> BERT arm 들이 랭크 때문에 손해를 본
          것이므로, 지금까지의 BERT 수치는 **과소평가**이고 다시 재야 한다

    누수 방지는 make_text_encoder_fn 과 동일하다 — TF-IDF vocabulary, SVD 기저,
    StandardScaler 셋 다 train fold 로만 fit 한다.
    """
    def fn(train_ids, val_ids, test_ids):
        ids_by_split = {"train": train_ids, "val": val_ids, "test": test_ids}
        tfidf_enc = features.TfidfEncoder(max_features=tfidf_max_features,
                                          ngram_range=tfidf_ngram_range)
        texts = {name: [corpus.get(int(rid), "") for rid in ids]
                 for name, ids in ids_by_split.items()}
        tfidf_enc.fit(texts["train"])
        mats = {
            name: (tfidf_enc.transform(t) if len(t)
                   else np.empty((0, tfidf_enc.max_features), dtype="float32"))
            for name, t in texts.items()
        }
        return _reduce_block(mats, out_dim, {"block": "tfidf_svd",
                                             "tfidf_dim": int(tfidf_enc.max_features)},
                             audit, do_svd=do_svd, do_scale=do_scale)

    return fn


def make_tfidf_plus_bert_encoder_fn(corpus: dict[int, str], embeddings: dict[int, np.ndarray],
                                    tfidf_max_features: int = 400, tfidf_ngram_range=(2, 4),
                                    bert_out_dim: int = 200, audit: list | None = None):
    """[TF-IDF | BERT축소] 를 가로로 이어붙인 텍스트 블록 생성기.

    "BERT 가 TF-IDF 를 **대체**하는가"가 아니라 "**보완**하는가"를 보는 arm 이다.
    TF-IDF 쪽은 features.TfidfEncoder 를 그대로 재사용하므로 기준선 arm 과
    완전히 같은 계산이고(역시 train fold 로만 fit), BERT 쪽만 덧붙는다.
    bert_out_dim 은 기본 200 — 400+200=600 이면 기준선 400 대비 1.5배로,
    폭 증가를 최소로 누르면서 보완 효과를 볼 수 있는 절충이다.
    (폭을 400+400=800 으로 하면 폭 2배가 되어 '폭 때문에 나빠졌다'와
     'BERT 가 도움이 안 됐다'가 섞여 버린다.)
    """
    def fn(train_ids, val_ids, test_ids):
        ids_by_split = {"train": train_ids, "val": val_ids, "test": test_ids}

        # ── TF-IDF 절반: 기준선과 동일하게 train fold 텍스트로만 vocabulary fit ──
        tfidf_enc = features.TfidfEncoder(max_features=tfidf_max_features,
                                          ngram_range=tfidf_ngram_range)
        texts = {name: [corpus.get(int(rid), "") for rid in ids]
                 for name, ids in ids_by_split.items()}
        tfidf_enc.fit(texts["train"])
        tfidf_blocks = {
            name: (tfidf_enc.transform(t) if len(t)
                   else np.empty((0, tfidf_enc.max_features), dtype="float32"))
            for name, t in texts.items()
        }

        # ── BERT 절반: 위와 동일한 fold-safe 축소 ──
        emb_dim = len(next(iter(embeddings.values())))
        mats, missing = {}, {}
        for name, ids in ids_by_split.items():
            mats[name], missing[name] = _stack_embeddings(embeddings, ids, emb_dim)
        bert_blocks = _reduce_block(
            mats, bert_out_dim,
            {"block": "tfidf+bert", "tfidf_dim": int(tfidf_enc.max_features),
             "n_missing_embedding": missing}, audit)

        return {name: np.concatenate([tfidf_blocks[name], bert_blocks[name]], axis=1).astype("float32")
                for name in ids_by_split}

    return fn
