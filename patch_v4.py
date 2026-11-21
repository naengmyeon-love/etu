"""V4안(팀장 v3.8 의 답 다듬기): answer 를 1~2문장·자료 표현 그대로 쓰게 하고,
인용 섹션에 없는 % 값이 든 문장과 '참고로/일반적으로'로 시작하는 덧붙임 문장을 뺀다. (ANSWER_TIDY=0 이면 다듬기 끔, 프롬프트 문구는 그대로)
python3 patch_v4.py <고칠 서버 파일>"""
import sys

p = sys.argv[1]
s = open(p, encoding="utf-8").read()
assert "ANSWER_TIDY" not in s, "이미 적용됨"

a = '4. 답변은 짧게 쓴다. answer 는 질문이 묻는 핵심 사실을 1~3문장으로 정확히 담고, 끝에 근거 섹션 id 를 "(PY-01-01)" 처럼 붙인다. scaffold 도 끝에 (섹션 id) 를 붙인다.'
assert s.count(a) == 1
s = s.replace(a, '4. 답변은 짧게 쓴다. answer 는 질문이 묻는 핵심 사실만 1~2문장으로, <자료>에 적힌 표현과 값을 그대로 옮겨 담고, 끝에 근거 섹션 id 를 "(PY-01-01)" 처럼 붙인다. scaffold 도 끝에 (섹션 id) 를 붙인다.\n'
              '   자료에 없는 덧붙임(일반적인 팁, "보통·일반적으로·참고로"로 시작하는 설명, 질문하지 않은 다른 실습·예제 이야기)은 쓰지 않는다.')

b = "            sents = split_kb_sentences(text)\n"
assert s.count(b) == 1
s = s.replace(b, "            if not hasattr(self, \"raw\"):\n                self.raw = {}\n            self.raw[path.stem] = text  # (V4) 섹션 원문 (본문 + 표)\n" + b)

c = "async def run_tutor(messages: list[dict], deadline: float) -> dict:"
assert s.count(c) == 1
s = s.replace(c, r'''ANSWER_TIDY = os.environ.get("ANSWER_TIDY", "1") == "1"  # (V4) answer 앞부분(LLM 이 쓴 문장) 다듬기
_TIDY_DROP_RE = re.compile(r"^\s*(?:참고로|일반적으로|덧붙이면|덧붙여)[\s,]")
_PCT_RE = re.compile(r"\d+(?:\.\d+)?\s*%")


def tidy_answer(answer: str, section_text: str) -> str:
    """인용 섹션에 없는 % 값이 든 문장과 '참고로/일반적으로'로 시작하는 덧붙임 문장을 뺀다.
    전부 빠지면 빈 문자열을 돌려준다 (뒤에 붙이는 원문 문장이 답이 된다)."""
    sents = [x for x in _SENT_SPLIT.split(answer.strip()) if x.strip()]
    if not sents:
        return answer
    flat_sec = re.sub(r"\s+", "", section_text)
    keep: list[str] = []
    for i, x in enumerate(sents):
        if any(re.sub(r"\s+", "", q) not in flat_sec for q in _PCT_RE.findall(x)):
            continue
        if i > 0 and _TIDY_DROP_RE.search(x):
            continue
        keep.append(x)
    out = " ".join(keep).strip()
    if not re.sub(r"[\s()]+", "", SECTION_ID_RE.sub("", out)):
        return ""  # (섹션 id) 만 남았으면 빈 답
    if not SECTION_ID_RE.search(out):  # 끝의 (섹션 id) 가 빠진 문장에 있었으면 다시 붙인다
        ids = SECTION_ID_RE.findall(answer)
        if ids:
            out = f"{out} ({ids[-1]})"
    return out


''' + c)

d = "    # (E1) 인용한 섹션 원문에서 질문과 가장 가까운 문장 하나를 그대로 덧붙인다\n"
assert s.count(d) == 1
s = s.replace(d, '''    # (V4) 인용 섹션에 없는 % 값·덧붙임 문장을 뺀다. 다 빠지면 아래에서 붙이는 원문 문장만 남긴다
    before_tidy = result["answer"]
    if ANSWER_TIDY and result["action"] == "answer" and result["citations"]:
        try:
            raw = getattr(app.state.evidence, "raw", {}).get(result["citations"][0], "")
            if raw:
                result["answer"] = tidy_answer(result["answer"], raw)
        except Exception as e:
            log.warning("답변 다듬기 실패: %s", type(e).__name__)
            result["answer"] = before_tidy
''' + d)

e = '                longer = result["answer"].rstrip() + lead + sent\n'
assert s.count(e) == 1
s = s.replace(e, '                longer = result["answer"].rstrip() + lead + sent if result["answer"].strip() else lead.strip() + " " + sent\n')

f = '            log.warning("근거 문장 덧붙이기 실패: %s", type(e).__name__)\n'
assert s.count(f) == 1
s = s.replace(f, f + '    if not result["answer"].strip():\n        result["answer"] = before_tidy  # 덧붙일 원문이 없으면 다듬기 전 답으로 되돌린다\n')
open(p, "w", encoding="utf-8").write(s)
print("V4안 적용:", p)
