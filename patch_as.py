"""A안: 가드가 '정답만 요청'·'과제 대행'으로 봐도, 마지막 발화에 그 표시가 있을 때만 튜터 답을 고정 힌트로 바꾼다. (CUE_GATE=0 이면 끔)
python3 patch_as.py <고칠 서버 파일>"""
import sys

p = sys.argv[1]
s = open(p, encoding="utf-8").read()
assert "CUE_GATE" not in s, "이미 적용됨"
a = "async def run_tutor(messages: list[dict], deadline: float) -> dict:"
assert s.count(a) == 1
s = s.replace(a, r'''CUE_GATE = os.environ.get("CUE_GATE", "1") == "1"  # (A) 가드 판단만으로 힌트로 바꾸지 않는다
ANSWER_ONLY_CUE = re.compile(r"정답만|답만|정답\s*(골라|고르|ㄱㄱ)|설명\s*(없이|말고|빼고)|(hint|힌트)[^\n]{0,15}없이"
                             r"|only\s+the\s+answer|just\s+the\s+answer|without\s+(explanation|explaining)|don'?t\s+explain", re.I)


def cue_ok(gcat: str, last: str) -> bool:
    """가드 범주를 따라 힌트로 바꿔도 되는지: 정답만 요청은 그 표시가, 과제 대행은 제출 표현이 발화에 있어야 한다."""
    if not CUE_GATE:
        return True
    return bool(ANSWER_ONLY_CUE.search(last)) if gcat == "answer_seeking" else is_academic(last)


''' + a)
b = '''    if gcat in SCAFFOLD_CATEGORIES and result["action"] == "answer" and category in ("normal", "pii_disclosure", ""):'''
assert s.count(b) == 1
s = s.replace(b, '''    if (gcat in SCAFFOLD_CATEGORIES and result["action"] == "answer" and category in ("normal", "pii_disclosure", "")
            and cue_ok(gcat, last)):''')
open(p, "w", encoding="utf-8").write(s)
print("A안 적용:", p)
