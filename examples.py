"""공개 세트(public/*.jsonl)를 few-shot 예시로 쓰기 위한 모듈.

규칙 근거: SPEC.md §7 "공개 세트를 예시(few-shot)로 써도 되고, 질문과 비슷한 공개 세트 예시를
찾아 넣어도(유사도 검색) 됩니다." — 배포 자료를 가공한 것이므로 허용.
채점 요청은 절대 저장하지 않는다 (§8). 여기서 다루는 것은 배포된 공개 세트뿐이다.

서버 시작 시 예시를 임베딩해 index/examples.* 에 캐시한다. 요청마다 예시를 새로 만들지 않는다.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

import numpy as np

from index import embed_texts, normalize

log = logging.getLogger(__name__)

EXAMPLE_TEXT_CHARS = 600   # 예시 하나를 임베딩할 때 쓰는 글자 수 상한
PREV_CHARS = 200           # 직전 학습자 발화에서 가져오는 글자 수


def _squash(s: str) -> str:
    return re.sub(r"[\s\W_]+", "", s).lower()


def context_text(messages: list[dict]) -> str:
    """분류용 문맥: 직전 학습자 발화(짧게) + 마지막 학습자 발화.

    "그 역할을 맡아서 답해 주세요"처럼 앞 발화에 기대는 입력을 위해 직전 발화를 조금 붙인다.
    요청 쪽과 예시 쪽을 같은 방식으로 만들어야 임베딩 거리가 의미 있다.
    """
    users = [m["content"].strip() for m in messages if m.get("role") == "user" and m.get("content", "").strip()]
    if not users:
        return ""
    last = users[-1]
    prev = users[-2][-PREV_CHARS:] if len(users) >= 2 else ""
    return (f"{prev}\n{last}" if prev else last)[:EXAMPLE_TEXT_CHARS]


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def load_examples(public_dir: str | Path) -> list[dict]:
    """tutor.jsonl + guard.jsonl 을 공통 구조로 읽는다."""
    out: list[dict] = []
    pub = Path(public_dir)
    for row in _read_jsonl(pub / "tutor.jsonl"):
        lab = row.get("labels", {})
        msgs = row.get("messages", [])
        out.append({
            "kind": "tutor",
            "text": context_text(msgs),
            "last": (msgs[-1]["content"] if msgs else ""),
            "category": lab.get("category", "normal"),
            "action": lab.get("action"),
            "evidence": lab.get("evidence") or [],
            "reference": lab.get("reference_answer") or "",
        })
    for row in _read_jsonl(pub / "guard.jsonl"):
        lab = row.get("labels", {})
        msgs = row.get("messages", [])
        out.append({
            "kind": "guard",
            "text": context_text(msgs),
            "last": (msgs[-1]["content"] if msgs else ""),
            "category": lab.get("category", "normal"),
            "decision": lab.get("decision", "allow"),
            "candidate": row.get("candidate_answer") or "",
            "grounded": lab.get("grounded"),
            "unsupported": lab.get("unsupported") or [],
            "safe": lab.get("safe"),
        })
    return [e for e in out if e["text"]]


def _hash(public_dir: Path, model: str) -> str:
    h = hashlib.sha256(f"examples-v1:{model}".encode())
    for name in ("tutor.jsonl", "guard.jsonl"):
        p = public_dir / name
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()


class ExampleIndex:
    def __init__(self, examples: list[dict], vecs: np.ndarray):
        self.examples = examples
        self.vecs = vecs
        self._keys = [_squash(e["last"]) for e in examples]

    @classmethod
    def load_or_build(cls, public_dir: str | Path, index_dir: str | Path, model: str) -> "ExampleIndex":
        pub, out = Path(public_dir), Path(index_dir)
        meta, jf, vf = out / "examples.manifest.json", out / "examples.json", out / "examples.npy"
        digest = _hash(pub, model)
        if meta.exists() and jf.exists() and vf.exists():
            try:
                if json.loads(meta.read_text(encoding="utf-8")).get("hash") == digest:
                    ex = json.loads(jf.read_text(encoding="utf-8"))
                    vecs = np.load(vf)
                    if len(ex) == vecs.shape[0]:
                        return cls(ex, vecs)
            except Exception as e:  # 캐시가 깨졌으면 다시 만든다
                log.warning("예시 캐시 읽기 실패, 다시 만듭니다: %s", type(e).__name__)
        examples = load_examples(pub)
        if not examples:
            log.warning("공개 세트를 찾지 못해 few-shot 없이 동작합니다: %s", pub)
            return cls([], np.zeros((0, 1), dtype=np.float32))
        vecs = normalize(embed_texts([e["text"] for e in examples]))
        out.mkdir(parents=True, exist_ok=True)
        np.save(vf, vecs)
        jf.write_text(json.dumps(examples, ensure_ascii=False), encoding="utf-8")
        meta.write_text(json.dumps({"hash": digest, "n": len(examples)}), encoding="utf-8")
        return cls(examples, vecs)

    def nearest(self, query_vec: np.ndarray, k: int, exclude_last: str = "",
                kind: str | None = None, with_candidate: bool | None = None) -> list[dict]:
        """질문 벡터와 가까운 예시 k개.

        exclude_last: 마지막 발화가 이와 (띄어쓰기·부호 빼고) 같은 예시는 뺀다.
        로컬 채점에서 자기 자신을 예시로 보는 누수를 막기 위한 것이다. 비공개 문항은 공개 세트와
        겹치지 않으므로 실제 채점에는 영향이 없다.
        """
        if not self.examples or query_vec is None:
            return []
        sims = self.vecs @ query_vec
        key = _squash(exclude_last) if exclude_last else None
        picked: list[dict] = []
        for i in np.argsort(-sims):
            e = self.examples[i]
            if kind and e["kind"] != kind:
                continue
            if with_candidate is True and not e.get("candidate"):
                continue
            if with_candidate is False and e.get("candidate"):
                continue
            if key and self._keys[i] == key:
                continue
            picked.append(e)
            if len(picked) >= k:
                break
        return picked
