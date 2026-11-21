"""V5안(팀장 v3.8 의 구간 다듬기): /guard 근거 없는 구간에서 앞의 주어(…은/는)와 코드 앞뒤 말을 떼고 틀린 어구만 남긴다. (SPAN_TRIM=0 이면 끔)
python3 patch_v5.py <고칠 서버 파일>"""
import sys

p = sys.argv[1]
s = open(p, encoding="utf-8").read()
assert "SPAN_TRIM" not in s, "이미 적용됨"
a = "async def run_tutor(messages: list[dict], deadline: float) -> dict:"
assert s.count(a) == 1
s = s.replace(a, r'''SPAN_TRIM = os.environ.get("SPAN_TRIM", "1") == "1"  # (V5) 근거 없는 구간에서 주어·코드 앞뒤 말을 뗀다
_SPAN_TOPIC_RE = re.compile(r"^((?:\S+\s+){0,2}?\S+(?:은|는|에는|에서는))\s+(.+)$")
_SPAN_CODE_RE = re.compile(r"`[^`]+`")


def trim_spans(spans: list[str]) -> list[str]:
    out: list[str] = []
    for s in spans:
        codes = _SPAN_CODE_RE.findall(s)
        for p in (codes if codes else [s]):
            m = _SPAN_TOPIC_RE.match(p)
            if m and len(squash(m.group(2))) >= 5:
                p = m.group(2)
            p = p.strip()
            if p and p not in out:
                out.append(p)
    return out


''' + a)
b = "    spans = spans[:2]\n"
assert s.count(b) == 1
s = s.replace(b, b + '''    if SPAN_TRIM and spans:  # (V5) 주어·코드 앞뒤 말을 떼고 틀린 어구만 남긴다
        try:
            trimmed = [x for x in trim_spans(spans) if squash(x) and squash(x) in flat][:3]
            spans = trimmed or spans
        except Exception as e:
            log.warning("구간 다듬기 실패: %s", type(e).__name__)
''')
open(p, "w", encoding="utf-8").write(s)
print("V5안 적용:", p)
