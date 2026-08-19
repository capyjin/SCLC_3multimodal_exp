# -*- coding: utf-8 -*-
"""판독지 안의 한글을 영어로 바꾸는 **고정 치환 사전**.

왜 필요한가
-----------
판독지는 글자 기준 영문 63.7% / 한글 8.8% 로, 의학 내용은 대부분 영어이고
한글은 조사(에·의·이·을·가·로)와 정형 서술어(관찰됨·생각됨·감별이 필요함)다.
그런데 영문 판독지로 학습된 BERT(RadBERT 등)의 어휘에는 한글이 없어서
**한 글자당 [UNK] 하나**가 되어버린다 — 실측 [UNK] 비율 평균 16.5%(최대 31%).

하필 파괴되는 한글이 판독지 의미를 뒤집는 말들이다:
    관찰되지 않음(248/253회) = 부정,  생각됨(329회)·감별이 필요함 = 확신도
"transition 관찰됨" 과 "transition 관찰되지 않음" 은 정반대인데 BERT 입장에선
둘 다 "transition [UNK][UNK][UNK]" 로 똑같이 보인다. 그래서 영어로 바꿔준다.

왜 LLM 번역이 아니라 사전인가
------------------------------
코퍼스의 고유 한글 덩어리는 **400개뿐**이고 상위 50개가 전체 등장의 89%,
100개면 94.8%를 덮는다. 즉 이 정도 규모면 사전 치환으로 충분하다. 사전 쪽이
오히려 낫다 — 결정론적이고(같은 입력 → 항상 같은 출력), 사람이 눈으로 감사할 수
있고, 환자 텍스트를 외부 API로 보내지 않으므로 IRB·국외이전 이슈가 없다.

누수(leakage) 관점
-------------------
이 사전은 **언어 지식**이지 데이터에서 적합한 통계가 아니다. 생존 결과(y)를
전혀 보지 않고 만들었고, train/val/test에 **똑같이** 적용된다. 사전제작에 쓴
빈도표도 텍스트(X)만 집계한 것이다. 사전 토크나이저를 쓰는 것과 같은 지위이므로
fold별로 다시 만들 필요가 없다.

사용법
------
    from ko2en import translate_korean, strip_korean
    translate_korean(text)   # 한글 -> 영어 (사전에 없는 한글은 삭제)
    strip_korean(text)       # 한글 전부 삭제 (대조군)

두 함수는 **같은 공백 정규화**를 거치므로, 둘의 출력 차이는 오직
"한글 의미를 영어로 살렸는가" 뿐이다 — 실험에서 원인을 분리하기 위한 설계.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import re

# ---------------------------------------------------------------------------
# 치환 사전.  긴 표현이 먼저 매칭되도록 아래에서 길이순 정렬하므로,
# 여기서는 의미 그룹별로 읽기 좋게 적어둔다.
# 값이 ""인 항목은 삭제(조사처럼 영어로 옮길 내용이 없는 것들).
# ---------------------------------------------------------------------------
KO2EN: dict[str, str] = {}

# --- 관찰되다 (be observed) : 부정형이 특히 중요 -----------------------------
KO2EN.update({
    "관찰되지 않았던": "was not observed",
    "관찰되지 않다가": "was not observed, then",
    "관찰되지는 않음": "is not clearly observed",
    "관찰되지 않으며": "is not observed and",
    "관찰되지 않으나": "is not observed, but",
    "관찰되지 않음": "is not observed",
    "관찰되지 않고": "is not observed and",
    "관찰되지 않던": "was not observed",
    "관찰되지 않아": "is not observed, so",
    "관찰되었으며": "was observed and",
    "관찰되는데": "is observed, and",
    "관찰되어": "is observed, so",
    "관찰되며": "is observed and",
    "관찰되나": "is observed, but",
    "관찰되던": "previously observed",
    "관찰되는": "observed",
    "관찰됨": "is observed",
})

# --- 보이다 (be seen) --------------------------------------------------------
KO2EN.update({
    "보이지 않으나": "is not seen, but",
    "보이지 않음": "is not seen",
    "보이지 않고": "is not seen and",
    "보이지 않아": "is not seen, so",
    "보이는데": "is seen, and",
    "보이며": "is seen and",
    "보이나": "is seen, but",
    "보이고": "is seen and",
    "보이는": "seen",
    "보이지": "seen",                    # 뒤의 않- 이 부정을 담당
    "보임": "is seen",
    "보여": "is seen, so",
    "보아며": "is seen and",
    "보는": "seen",
    "보인": "seen",
    "보시기": "for review",
    "보통의": "usual",
})

# --- 생각되다 / 판단되다 (be considered / judged) : 확신도 -------------------
KO2EN.update({
    "생각되어": "is considered, so",
    "생각되며": "is considered and",
    "생각되는": "considered",
    "생각됨": "is considered",
    "판단되며": "is judged and",
    "판단되나": "is judged, but",
    "판단되는": "judged",
    "판단됨": "is judged",
    "판독함": "interpreted",
    "판독된": "interpreted",
    "판독이": "interpretation",
    "판독": "interpretation",
})

# --- 의심되다 (be suspected) : 확신도 ---------------------------------------
KO2EN.update({
    "의심할만한": "suspicious for",
    "의심되며": "is suspected and",
    "의심되나": "is suspected, but",
    "의심되던": "previously suspected",
    "의심됨": "is suspected",
    "의심할": "to suspect",
})

# --- 감별 / 구별 (differentiation) ------------------------------------------
KO2EN.update({
    "감별이 필요하겠음": "differentiation would be needed",
    "감별이 곤란하여": "differentiation is difficult, so",
    "감별이 필요함": "differentiation is needed",
    "감별이 곤란함": "differentiation is difficult",
    "구별되지는 않음": "is not clearly distinguished",
    "구별되지 않습니다": "is not distinguished",
    "구별되지": "not distinguished",
    "구별되는": "distinguished",
    "구별이": "distinction",
    "구별할": "to distinguish",
    "감별이": "differentiation",
    "감별은": "differentiation",
})

# --- 가능성 (probability) : 확신도의 핵심 -----------------------------------
KO2EN.update({
    "가능성이 높지는 않음": "not highly probable",
    "가능성이 높지 않음": "low probability",
    "가능성이 높겠지만": "probable, but",
    "가능성이 높겠으나": "probable, but",
    "가능성이 높으며": "high probability and",
    "가능성이 높으나": "high probability, but",
    "가능성이 높겠음": "probable",
    "가능성이 높음": "high probability",
    "가능성이 높아": "high probability, so",
    "가능성을": "possibility",
    "가능성이": "possibility",
})

# --- 배제 / 제외 (exclude, rule out) ----------------------------------------
KO2EN.update({
    "배제하기는": "to rule out",
    "배제하기": "to rule out",
    "배제할": "to rule out",
    "제외한": "excluding",
})

# --- 있다 / 없다 (present / absent) -----------------------------------------
KO2EN.update({
    "되어있으며": "is present and",
    "있었던": "previously present",
    "있으며": "is present and",
    "있으나": "is present, but",
    "있는지는": "whether present",
    "있는지": "whether present",
    "있는데": "is present, and",
    "있겠음": "may be present",
    "있던": "previously present",
    "있고": "is present and",
    "있어": "is present, so",
    "있을": "may be present",
    "있는": "present",
    "있음": "is present",
    "없어졌다가": "disappeared, then",
    "없겠음": "would be absent",
    "없으나": "is absent, but",
    "없어": "is absent, so",
    "없고": "is absent and",
    "없음": "is absent",
    "없이": "without",
})

# --- 뚜렷하다 (distinct / conspicuous) --------------------------------------
KO2EN.update({
    # "-지 않-" 은 두 조각으로 쪼개지면 이중부정이 되므로 결합형을 먼저 잡는다.
    "뚜렷하지 않으나": "is not distinct, but",
    "뚜렷하지 않음": "is not distinct",
    "뚜렷하지 않아": "is not distinct, so",
    "뚜렸하지 않음": "is not distinct",   # 오타 변형
    "뚜렷하게": "distinctly",
    "뚜렷하지": "distinct",              # 뒤의 않- 이 부정을 담당
    "뚜렸하지": "distinct",
    "뚜렷한": "distinct",
    "뚜렷이": "distinctly",
})

# --- 높다 / 낮다 (high / low) -----------------------------------------------
KO2EN.update({
    "높지는 않음": "is not high",
    "높지 않음": "is not high",
    "높아보임": "appears high",
    "높겠으나": "would be high, but",
    "높겠지만": "would be high, but",
    "높으며": "is high and",
    "높으나": "is high, but",
    "높겠음": "would be high",
    "높지는": "high",                    # 뒤의 않- 이 부정을 담당
    "높지": "high",
    "높음": "is high",
    "높아": "is high, so",
    "높고": "is high and",
    "낮게": "low",
    "낮고": "is low and",
    "낮아": "is low, so",
})

# --- 증가 / 감소 / 크기 변화 (interval change) : 예후와 직결 ----------------
KO2EN.update({
    "증가하였으며": "has increased and",
    "증가하였음": "has increased",
    "증가없어": "no increase, so",
    "증가하여": "has increased, so",
    "감소하였으나": "has decreased, but",
    "감소하였고": "has decreased and",
    "감소하여": "has decreased, so",
    "커져있으며": "has enlarged and",
    "커져있으나": "has enlarged, but",
    "커져있고": "has enlarged and",
    "두꺼워져": "has thickened",
    "새롭게": "newly",
    "커져": "has enlarged",
    "작아": "has decreased in size",
    "증가는": "increase",
    "증가": "increase",
    "변화없는": "unchanged",
    "변화를": "change",
    "변화는": "change",
    "변화": "change",
    "호전을": "improvement",
    "정상화": "normalization",
    "여전히": "still",
    "새로": "newly",
})

# --- 동반 (accompanied by) --------------------------------------------------
KO2EN.update({
    "동반되어": "accompanied by, so",
    "동반하고": "accompanied by and",
    "동반한": "accompanied by",
    "동반함": "accompanied by",
    "동반됨": "is accompanied",
})

# --- 필요 / 권고 (recommendation) -------------------------------------------
KO2EN.update({
    "필요하겠음": "would be needed",
    "필요하면": "if needed",
    "바랍니다": "please",
    "하시기": "please",
    "필요함": "is needed",
    "필요할": "needed",
    "좋겠음": "is recommended",
    "확인해": "to confirm",
    "확인이": "confirmation",
    "시행이": "performing",
    "추가": "additional",
})

# --- 유의성 (significance) ---------------------------------------------------
KO2EN.update({
    "의의있는": "significant",
    "의미있는": "significant",
    "의미": "significance",
})

# --- 괄호 안 숫자 = maxSUV 범례 ---------------------------------------------
KO2EN.update({
    "괄호안의 숫자는": "the number in parentheses is",
    "괄호 안 숫자는": "the number in parentheses is",
    "괄호안 숫자는": "the number in parentheses is",
    "괄호안의": "in parentheses",
    "숫자는": "the number is",
    "괄호안": "in parentheses",
    "괄호": "parenthesis",
})

# --- 해부학적 위치 / 방향 ---------------------------------------------------
KO2EN.update({
    "전방부위에": "in the anterior region",
    "경계부위인": "at the border region",
    "측하방으로": "toward the inferolateral",
    "측상방으로": "toward the superolateral",
    "직후방에": "immediately posterior",
    "주행방향을": "the course",
    "부위부터": "from the region",
    "상방으로": "superiorly",
    "부위에는": "in the region",
    "후방의": "posterior",
    "후방에": "posterior",
    "후방을": "posterior",
    "전방과": "anterior and",
    "전방의": "anterior",
    "측방에": "lateral",
    "하방에도": "also inferior",
    "하방에": "inferior",
    "하방의": "inferior",
    "상방에": "superior",
    "상방": "superior",
    "전방": "anterior",
    "내측의": "medial",
    "외측의": "lateral",
    "외측에": "lateral",
    "우측의": "right-sided",
    "인접한": "adjacent",
    "인접해": "adjacent",
    "주변의": "surrounding",
    "주변에": "surrounding",
    "주변을": "surrounding",
    "주변": "surrounding",
    "인근에": "adjacent",
    "인근": "adjacent",
    "부위에": "in the region",
    "부위의": "of the region",
    "부위로": "to the region",
    "부위": "region",
    "전체에": "throughout",
    "전신의": "whole-body",
    "기시부": "origin",
    "경계": "border",
    "방향에": "in the direction",
    "위치임": "is located",
})

# --- 해부 / 검사 용어 --------------------------------------------------------
KO2EN.update({
    "부분지연영상에서는": "on the partial delayed image",
    "부분지연영상에서도": "also on the partial delayed image",
    "부분지연영상에서": "on the partial delayed image",
    "방사성의약품": "radiopharmaceutical",
    "지연영상에서": "on the delayed image",
    "검사에서": "on the examination",
    "병소들은": "the lesions",
    "복강내에": "in the peritoneal cavity",
    "영상의": "of the image",
    "복부의": "of the abdomen",
    "병소로": "as a lesion",
    "환자의": "the patient's",
    "환자가": "the patient",
    "통증이": "pain",
    "복강": "peritoneal cavity",
    "혈당": "blood glucose",
    "섭취가": "uptake",
    "섭취": "uptake",
    "병변": "lesion",
    "검사": "examination",
    "촬영": "scan",
    "주사": "injection",
    "임상": "clinical",
    "정보": "information",
    "목적": "purpose",
    "병원": "hospital",
    "평가할": "to evaluate",
    "평가": "evaluation",
    "제한": "limited",
    "유무": "presence or absence",
    "여부는": "whether",
    "크기와": "size and",
    "크기가": "size",
    "크기는": "size",
    "크기에": "in size",
    "크기의": "of size",
    "크기": "size",
    "양이": "amount",
    "질이": "quality",
    "정도가": "degree",
    "정도": "degree",
})

# --- 비교 / 시간 -------------------------------------------------------------
KO2EN.update({
    "비교하여": "compared with",
    "비교하면": "compared with",
    "참고하였을": "when referenced",
    "하였을": "when done",
    "언급된": "mentioned",
    "이전": "previous",
    "지난": "previous",
    "최근": "recent",
    "직전": "immediately before",
    "비해": "compared with",
    "보다는": "rather than",
    "그보다": "than that",
    "다시": "again",
    "아직": "still",
    "지나": "past",
    "이르는": "extending to",
    "발생된": "developed",
    "발생한": "developed",
    "남아": "remaining",
})

# --- 정도 / 범위 부사 --------------------------------------------------------
KO2EN.update({
    "전체적으로": "overall",
    "상대적으로": "relatively",
    "대부분의": "most of the",
    "대부분에": "in most",
    "대부분": "most",
    "일부분에서만": "only in part",
    "일부분에서": "in part",
    "일부에서": "in part",
    "일부는": "part of it",
    "일부를": "part of it",
    "일부에": "in part",
    "일부": "part",
    "비교적": "relatively",
    "그다지": "not particularly",
    "정확하게": "accurately",
    "정확한": "accurate",
    "심하고": "severe and",
    "심해": "severe, so",
    "심함": "is severe",
    "매우": "very",
    "다소": "somewhat",
    "특히": "especially",
    "거의": "almost",
    "주로": "mainly",
    "많음": "is abundant",
    "여러": "multiple",
    "다수": "multiple",
    "모든": "all",
    "이상": "or more",
    "정상": "normal",
    "같은": "same",
    "만한": "comparable to",
    "양상의": "in pattern",
    "국한한": "confined to",
})

# --- 접속 / 문장 연결 --------------------------------------------------------
KO2EN.update({
    "뿐만 아니라": "as well as",
    "하지만": "however",
    "그리고": "and",
    "또한": "also",
    "또는": "or",
    "때문에": "because",
    "그러나": "however",
    "따라": "according to",
    "의해": "by",
    "의한": "due to",
    "인해": "due to",
    "이로": "thereby",
    "그외": "other than that",
    "외에": "other than",
    "포함하여": "including",
    "포함되어": "included",
    "포함한": "including",
    "형성하고": "forming",
    "채우고": "filling",
    "걸쳐있는": "spanning",
    "걸쳐": "spanning",
    "연해있어": "abutting",
    "연해있는": "abutting",
    "둘러 싸고": "surrounding",
    "싸고": "surrounding",
    "연해": "abutting",
    "둘러": "surrounding",
    "시키고": "causing",
    "움직여": "moved",
    "반응의": "of the reaction",
})

# --- 잔여 저빈도 표현 (감사 결과 미커버로 잡힌 것들) -------------------------
KO2EN.update({
    "측정되었을": "when measured",
    "구분되는": "distinguished",
    "되었음": "has become",
    "되었을": "when it became",
    "곤란함": "is difficult",
    "곤란하여": "is difficult, so",
    "어려움": "is difficult",
    "군데에": "at sites",
    "군데": "sites",
    "아니라": "but rather",
    "뿐만": "not only",
    "외부": "external",
    "부분": "portion",
    "되어": "becoming",
})

# --- 조사·형식형태소 : 영어로 옮길 내용이 없어 삭제 --------------------------
for _p in ("에서는", "에서도", "로부터", "부터의", "에게", "에서", "에도", "에는",
           "내에도", "내에", "내의", "내로", "내", "으로", "로만", "까지", "부터",
           "이며", "이고", "들이", "들은", "들을", "들에서는", "들에", "개의", "개가",
           "개에", "것으로", "것이", "것일", "수", "그", "및", "등의", "등에", "등",
           "외", "와의", "와", "과", "은", "는", "이", "가", "을", "를", "의", "에",
           "로", "도", "된", "됨", "임", "함", "중", "두", "한", "큰", "더", "안",
           "때", "다", "좀", "약", "년", "전", "일", "지", "인", "결", "해", "하여",
           "하고", "하는", "여서", "인지", "인지는", "않음", "않아", "않고", "않으나",
           "않으며", "않던", "않았던", "않다가", "않는", "처럼", "좋지", "시"):
    KO2EN.setdefault(_p, "")

# 부정 보조어간은 삭제하면 의미가 뒤집히므로 위 루프의 기본값을 덮어쓴다.
# ("관찰되지 않음" 같은 결합형은 이미 위에서 통째로 잡히지만, 다른 동사와
#  결합해 남는 경우를 대비한 안전망.)
KO2EN.update({
    "않음": "not",
    "않아": "not, so",
    "않고": "not, and",
    "않으나": "not, but",
    "않으며": "not, and",
    "않던": "was not",
    "않았던": "was not",
    "않다가": "was not, then",
    "않는": "not",
    "않습니다": "not",
})

# ---------------------------------------------------------------------------
# 치환 실행부
# ---------------------------------------------------------------------------
# 긴 표현부터 매칭해야 "관찰되지 않음"이 "관찰되지"+"않음"으로 쪼개지지 않는다.
_KEYS = sorted(KO2EN, key=len, reverse=True)
_PATTERN = re.compile("|".join(re.escape(k) for k in _KEYS))
_HANGUL = re.compile(r"[가-힣]+")
_WS = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """공백 정리. translate/strip 양쪽이 똑같이 거쳐야 둘의 차이가
    '한글 의미를 살렸는가' 하나로만 남는다."""
    return _WS.sub(" ", text).strip()


def translate_korean(text: str) -> str:
    """한글을 영어로 치환한다. 사전에 없는 잔여 한글은 삭제한다.

    잔여 한글을 남기지 않는 이유: 남기면 BERT에서 다시 [UNK]가 되어
    '번역의 효과'와 '[UNK] 잔존의 효과'가 섞여버린다. 삭제해야
    strip_korean(대조군)과의 차이가 순수하게 '영어로 살린 의미'가 된다.
    """
    out = _PATTERN.sub(lambda m: " " + KO2EN[m.group(0)] + " ", text)
    out = _HANGUL.sub(" ", out)          # 사전에 없던 잔여 한글
    return _normalize(out)


def strip_korean(text: str) -> str:
    """한글을 전부 삭제한다 (대조군: '한글이 정말 신호인가?')."""
    return _normalize(_HANGUL.sub(" ", text))


def coverage(texts) -> dict:
    """사전이 한글 등장 횟수의 몇 %를 덮는지 계산한다 (감사용).

    환자 텍스트를 출력하지 않고 집계 수치만 돌려준다.
    """
    from collections import Counter
    seen, hit = Counter(), Counter()
    for t in texts:
        for chunk in _HANGUL.findall(t):
            seen[chunk] += 1
            # 이 덩어리가 사전 패턴으로 얼마나 소비되는지 글자 단위로 계산
            consumed = sum(len(m.group(0)) for m in _PATTERN.finditer(chunk))
            hit[chunk] += consumed / max(len(chunk), 1)
    total = sum(seen.values())
    covered = sum(hit.values())
    uncovered = sorted(((seen[k] * (1 - hit[k] / max(seen[k], 1)), k) for k in seen),
                       reverse=True)
    return {
        "unique_chunks": len(seen),
        "total_occurrences": total,
        "char_coverage": covered / max(total, 1),
        "dict_entries": len(KO2EN),
        "worst_uncovered": [(k, round(w, 1)) for w, k in uncovered[:15] if w > 0.5],
    }


if __name__ == "__main__":
    from core import cohort
    from core.features import load_text_corpus

    corpus, _ = load_text_corpus(cohort.DEFAULT_MERGED_CSV)
    docs = list(corpus.values())
    cov = coverage(docs)
    print(f"사전 항목 수      : {cov['dict_entries']}")
    print(f"고유 한글 덩어리  : {cov['unique_chunks']}")
    print(f"한글 등장 총횟수  : {cov['total_occurrences']}")
    print(f"글자 기준 커버율  : {cov['char_coverage']:.1%}")
    print(f"미커버 상위       : {cov['worst_uncovered']}")

    before = sum(len(_HANGUL.findall(d)) for d in docs)
    after = sum(len(_HANGUL.findall(translate_korean(d))) for d in docs)
    print(f"치환 후 잔여 한글 : {after} (원본 {before})")
