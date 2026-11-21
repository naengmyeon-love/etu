"""G3e안: G3(가드 후보 답변 판정 3번 다수결)의 지연을 줄인다.
실제 채점(동시 64)에서 G3 가 가드 지연 p95 16.6 -> 19.4초, 20초 넘은 가드 요청 12 -> 68, 시간 초과 대체 응답 1 -> 13 으로 늘렸다.
- 3개를 동시에 보내되, 먼저 끝난 2개의 판정(안전·근거)이 같으면 바로 그 판정으로 결정하고 나머지는 취소한다.
- 첫 결과가 온 뒤 G3E_WAIT(기본 4초)가 지나도 결정이 안 나면, 그때까지 온 것들로 다수결(동률이면 먼저 온 것).
- 결정 규칙은 G3 와 같다: 과반이 unsafe → unsafe, 과반이 근거 없음 → 근거 없음(그 판정의 구간), 아니면 근거 있음.
끄기: G3E=0 (G3 그대로)
python3 patch_g3e.py <고칠 서버 파일>   (patch_d2g3.py 가 적용된 파일 위에)"""
import sys

p = sys.argv[1]
s = open(p, encoding="utf-8").read()
assert "G3E" not in s, "이미 적용됨"
old = '''    outs = await asyncio.gather(*[_guard_output_once(messages, candidate, deadline, qvec, input_result)
                                  for _ in range(GUARD_OUT_VOTES)], return_exceptions=True)
    outs = [o for o in outs if isinstance(o, dict)]'''
new = '''    if G3E:
        outs = await _vote_early(messages, candidate, deadline, qvec, input_result)
    else:
        outs = await asyncio.gather(*[_guard_output_once(messages, candidate, deadline, qvec, input_result)
                                      for _ in range(GUARD_OUT_VOTES)], return_exceptions=True)
    outs = [o for o in outs if isinstance(o, dict)]'''
assert s.count(old) == 1
s = s.replace(old, new)
anchor = "async def guard_output(messages: list[dict], candidate: str, deadline: float, qvec: np.ndarray | None,"
assert s.count(anchor) == 1
s = s.replace(anchor, '''G3E = os.environ.get("G3E", "1") == "1"                 # (G3e) 먼저 온 두 판정이 같으면 바로 결정
G3E_WAIT = float(os.environ.get("G3E_WAIT", "4.0"))       # 첫 결과 뒤 더 기다리는 최대 초


def _verdict_key(o: dict) -> tuple:
    return (o.get("safe") is not False, o.get("grounded") is not False)


async def _vote_early(messages, candidate, deadline, qvec, input_result) -> list:
    tasks = [asyncio.create_task(_guard_output_once(messages, candidate, deadline, qvec, input_result))
             for _ in range(GUARD_OUT_VOTES)]
    got: list = []
    first_at = None
    try:
        pending = set(tasks)
        while pending:
            timeout = None
            if first_at is not None:
                timeout = max(0.05, G3E_WAIT - (time.monotonic() - first_at))
            done, pending = await asyncio.wait(pending, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
            if not done:
                break  # 첫 결과 뒤 G3E_WAIT 초가 지남: 온 것들로 결정
            for t in done:
                try:
                    r = t.result()
                except Exception:
                    r = None
                if isinstance(r, dict):
                    got.append(r)
                    if first_at is None:
                        first_at = time.monotonic()
            keys = [_verdict_key(o) for o in got]
            if any(keys.count(k) >= 2 for k in keys):
                break  # 두 판정이 같다: 과반 확정
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
    return got


''' + anchor)
open(p, "w", encoding="utf-8").write(s)
print("G3e 적용:", p)
