# -*- coding: utf-8 -*-
"""결과 JSON 저장 — arm 하나가 끝날 때마다 저장(crash-safe)하고,
**설정이 다른 실행 결과는 절대 한 파일에 섞지 않는다.**

[왜 이 클래스가 필요한가]
  정리 전에는 exp_bert_text / exp_text_source / exp_suv_features 세 파일이
  각자 같은 20여 줄을 복사해 갖고 있었다:

      prev = json.load(open(path)) if os.path.exists(path) else {}
      same = (prev.get("epochs") == args.epochs and ...)
      if prev and not same: print("[warn] ... 새로 시작한다")
      results = prev.get("arms", {}) if same else {}

  이 로직의 목적은 하나다 — **smoke test(--epochs 2) 결과가 본 실험(60 epoch)
  파일에 남아 있으면 한 표 안에 비교 불가능한 숫자가 섞여 잘못된 결론이 난다.**
  세 벌로 흩어져 있으면 한쪽만 고쳐도 아무도 모르므로 여기 한 곳에 둔다.
"""
import json
import os


class ResultStore:
    """``<out_dir>/results_<target>.json`` 하나를 관리한다.

    ``settings`` 는 "이 값들이 같아야 이어붙일 수 있다"는 판정 기준이다
    (epochs·batch_size·model_config 등). 하나라도 다르면 기존 파일을 이어붙이지
    않고 새로 시작한다 — 기존 파일을 지우지는 않고, 다음 save() 에서 덮어쓴다.

    ``item_key`` 는 항목들이 들어갈 최상위 키다 (실험마다 다르다:
    ``arms`` / ``variants`` / ``steps`` / ``axes``).
    """

    def __init__(self, path: str, settings: dict, item_key: str = "items",
                 header: dict | None = None, logger=None):
        self.path = path
        self.settings = dict(settings)
        self.item_key = item_key
        self.header = dict(header or {})
        self.log = logger
        self.items: dict = {}
        self.reused = False
        self._load()

    # ── 내부 ──────────────────────────────────────────────────────────────
    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            prev = json.load(open(self.path, encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            self._warn(f"[warn] {self.path} 를 읽지 못했다 ({exc}) -> 새로 시작한다.")
            return
        mismatched = {k: (prev.get(k), v) for k, v in self.settings.items() if prev.get(k) != v}
        if mismatched:
            detail = ", ".join(f"{k}: 파일={got!r} 이번={want!r}" for k, (got, want) in mismatched.items())
            self._warn(f"[warn] {self.path} 는 다른 설정으로 만들어졌다 ({detail}) "
                       "-> 이어붙이지 않고 새로 시작한다.")
            return
        self.items = prev.get(self.item_key, {})
        self.reused = bool(self.items)
        if self.reused:
            self._info(f"[store] 같은 설정의 기존 결과 {len(self.items)}개를 이어받는다: "
                       f"{sorted(self.items)}")

    def _warn(self, msg: str) -> None:
        (self.log.warning if self.log else print)(msg)

    def _info(self, msg: str) -> None:
        (self.log.info if self.log else print)(msg)

    # ── 공개 API ──────────────────────────────────────────────────────────
    def has(self, name: str) -> bool:
        return name in self.items

    def get(self, name: str, default=None):
        return self.items.get(name, default)

    def put(self, name: str, record: dict) -> None:
        """항목 하나를 넣고 **즉시 저장한다** (다음 arm 에서 죽어도 앞 결과가 남도록)."""
        self.items[name] = record
        self.save()

    def update_header(self, **kwargs) -> None:
        self.header.update(kwargs)

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        payload = {**self.settings, **self.header, self.item_key: self.items}
        tmp = f"{self.path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, self.path)   # 저장 도중에 죽어도 반쪽짜리 JSON 이 남지 않는다
