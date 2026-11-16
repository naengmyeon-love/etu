"""공개 세트의 라벨 분포만 센다. 문항 본문은 출력하지 않는다.

    python3 inspect_public.py public/guard.jsonl public/tutor.jsonl

- guard: 범주(category)별 allow/block 개수 -> server.py 의 BLOCK_CATEGORIES 를 정하는 근거
- tutor: action 별 개수와, 인용이 있는 문항 수 -> NO_CITATION_ACTIONS 를 정하는 근거

파일 구조를 모르는 채로 쓰도록, 이름이 decision / category / action / evidence(또는 citations) 인 값을 어디에 있든 찾아 센다.
"""
from __future__ import annotations

import json
import sys
from collections import Counter


def find(obj, key, path=""):
    """중첩된 구조에서 이름이 key 인 값을 (경로, 값)으로 모두 찾는다."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else k
            if k == key:
                yield p, v
            yield from find(v, key, p)
    elif isinstance(obj, list):
        for v in obj:
            yield from find(v, key, path)


def main(paths: list[str]) -> None:
    for path in paths:
        rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        print(f"\n== {path}: {len(rows)}문항")
        print("최상위 키:", sorted({k for r in rows if isinstance(r, dict) for k in r}))

        pairs, actions = Counter(), Counter()
        for r in rows:
            cats = [v for p, v in find(r, "category") if isinstance(v, str) and "output" not in p]
            decs = [v for _, v in find(r, "decision") if isinstance(v, str)]
            if cats and decs:
                pairs[(cats[0], decs[0])] += 1
            acts = [v for _, v in find(r, "action") if isinstance(v, str)]
            if acts:
                cites = [v for k in ("evidence", "citations") for _, v in find(r, k) if isinstance(v, list)]
                actions[(acts[0], "인용 있음" if cites and cites[0] else "인용 없음")] += 1

        if pairs:
            print("범주별 판정:")
            for cat in sorted({c for c, _ in pairs}):
                print(f"  {cat:<22} allow {pairs[(cat, 'allow')]:>4}   block {pairs[(cat, 'block')]:>4}")
        if actions:
            print("action 별 인용:")
            for act in sorted({a for a, _ in actions}):
                print(f"  {act:<10} 인용 있음 {actions[(act, '인용 있음')]:>4}   인용 없음 {actions[(act, '인용 없음')]:>4}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["public/guard.jsonl", "public/tutor.jsonl"])
