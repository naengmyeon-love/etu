"""실험 B: 튜터 예시를 '비슷한 순'으로 고르되 같은 action 은 SHOT_CAP 개까지만 (라벨 균형).
처음 보는 질문에서 이웃 예시의 action 을 그대로 베끼는 것을 줄인다. SHOT_BALANCE=0 으로 끈다.
공개 세트 예시를 고르는 방식만 바꾼다 (SPEC §7.1 유사도 few-shot 허용 범위)."""
import sys
from pathlib import Path

p = Path(sys.argv[1])
src = p.read_text(encoding="utf-8")
if "SHOT_BALANCE" in src:
    sys.exit("already patched")
OLD = '    shots = app.state.examples.nearest(qvec, TUTOR_SHOTS, exclude_last=last, kind="tutor") if qvec is not None else []\n'
NEW = '    shots = balanced_tutor_shots(qvec, last) if qvec is not None else []\n'
FUNC = '''SHOT_BALANCE = os.environ.get("SHOT_BALANCE", "1") == "1"
SHOT_POOL = int(os.environ.get("SHOT_POOL", "24"))  # 비슷한 순으로 먼저 뽑는 후보 수
SHOT_CAP = int(os.environ.get("SHOT_CAP", "2"))     # 같은 action 예시 최대 수


def balanced_tutor_shots(qvec, last: str) -> list[dict]:
    """비슷한 순으로 후보를 뽑고, 같은 action 은 SHOT_CAP 개까지만 넣는다. 모자라면 남은 후보로 채운다."""
    if not SHOT_BALANCE:
        return app.state.examples.nearest(qvec, TUTOR_SHOTS, exclude_last=last, kind="tutor")
    pool = app.state.examples.nearest(qvec, SHOT_POOL, exclude_last=last, kind="tutor")
    picked, rest, count = [], [], {}
    for e in pool:
        a = e.get("action")
        if count.get(a, 0) < SHOT_CAP and len(picked) < TUTOR_SHOTS:
            picked.append(e)
            count[a] = count.get(a, 0) + 1
        else:
            rest.append(e)
    picked += rest[:max(0, TUTOR_SHOTS - len(picked))]
    return picked


async def run_tutor('''
assert src.count(OLD) == 1 and src.count("async def run_tutor(") == 1, "anchor not found"
src = src.replace(OLD, NEW).replace("async def run_tutor(", FUNC, 1)
p.write_text(src, encoding="utf-8")
print("patched B")
