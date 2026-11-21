"""V4b안 (V4 위에): (가) 인용 섹션·질문에 없는 숫자(두 자리 이상·소수)가 든 문장도 뺀다,
(나) 프롬프트에서 answer 를 1문장으로. 원문 덧붙이기 2문장은 그대로.
NUM_TIDY=0 이면 (가) 끔 (프롬프트 문구는 그대로).
python3 patch_v4b.py <고칠 서버 파일>   (e2c8d8e 의 server.py 에 적용)"""
import sys

p = sys.argv[1]
s = open(p, encoding="utf-8").read()
assert "ANSWER_TIDY" in s, "V4 가 먼저 있어야 함"
assert "NUM_TIDY" not in s, "이미 적용됨"


def rep(old: str, new: str) -> None:
    global s
    assert s.count(old) == 1, old[:60]
    s = s.replace(old, new)


# (나) 1문장
rep("answer 는 질문이 묻는 핵심 사실만 1~2문장으로,", "answer 는 질문이 묻는 핵심 사실만 1문장으로,")

# (가) 숫자 거르기
rep('_PCT_RE = re.compile(r"\\d+(?:\\.\\d+)?\\s*%")\n',
    '_PCT_RE = re.compile(r"\\d+(?:\\.\\d+)?\\s*%")\n'
    'NUM_TIDY = os.environ.get("NUM_TIDY", "1") == "1"  # (V4b) 자료·질문에 없는 숫자가 든 문장도 뺀다\n'
    '_NUM_RE = re.compile(r"\\d+(?:[.,]\\d+)*")\n'
    '\n'
    '\n'
    'def _num_tokens(text: str) -> set[str]:\n'
    '    """글 속 숫자 모음. 천 단위 쉼표는 지우고 소수점은 남긴다."""\n'
    '    return {m.replace(",", "") for m in _NUM_RE.findall(SECTION_ID_RE.sub(" ", text))}\n'
    '\n'
    '\n'
    'def _needs_check(n: str) -> bool:\n'
    '    """한 자리 정수(1개·2가지 같은 흔한 말)는 보지 않는다. 두 자리 이상·소수만 본다."""\n'
    '    return "." in n or len(n) >= 2\n')

rep('def tidy_answer(answer: str, section_text: str) -> str:\n',
    'def tidy_answer(answer: str, section_text: str, extra_text: str = "") -> str:\n')

rep('    flat_sec = re.sub(r"\\s+", "", section_text)\n',
    '    flat_sec = re.sub(r"\\s+", "", section_text)\n'
    '    allowed = _num_tokens(section_text) | _num_tokens(extra_text) if NUM_TIDY else set()\n')

rep('        if i > 0 and _TIDY_DROP_RE.search(x):\n            continue\n',
    '        if i > 0 and _TIDY_DROP_RE.search(x):\n            continue\n'
    '        if NUM_TIDY and any(_needs_check(n) and n not in allowed for n in _num_tokens(x)):\n'
    '            log.warning("답변 다듬기: 자료에 없는 숫자 문장 뺌")\n'
    '            continue\n')

# 인용 섹션 전부 + 사용자 발화 전부를 허용 숫자의 출처로
rep('            raw = getattr(app.state.evidence, "raw", {}).get(result["citations"][0], "")\n'
    '            if raw:\n'
    '                result["answer"] = tidy_answer(result["answer"], raw)\n',
    '            raws = getattr(app.state.evidence, "raw", {})\n'
    '            raw = raws.get(result["citations"][0], "")\n'
    '            if raw:\n'
    '                every = "\\n".join(raws.get(c, "") for c in result["citations"])\n'
    '                said = "\\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "user")\n'
    '                result["answer"] = tidy_answer(result["answer"], every, said)\n')

open(p, "w", encoding="utf-8").write(s)
print("V4b안 적용:", p)
