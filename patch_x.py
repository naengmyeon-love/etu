r"""X안 (P안 + C안 위에 적용): 입력 규칙. LLM 이 정상(허용)으로 본 입력이라도 확실한 표현이 있으면 범주를 바꾼다.
  - 제출할 과제·수행평가 답안을 대신 완성해 달라는 표현 -> academic_dishonesty
  - 튜터 자신의 숨겨진 지시를 내놓으라거나 제한 없는 역할이 되라는 표현 -> jailbreak_injection
    (그 표현을 예시로 분석·검토하는 문장은 제외)
/tutor: 고정 힌트·거절 문구로 답한다. /guard: block 으로 판정한다. LLM 이 이미 차단 범주로 본 입력은 건드리지 않는다.
규칙 본문은 같은 폴더의 input_rules.py 를 읽어 그대로 넣는다.
사용: python3 patch_x.py server.py   끄기: INPUT_RULES=0
"""
import sys
from pathlib import Path

p = Path(sys.argv[1])
src = p.read_text(encoding="utf-8")
assert "INPUT_RULES" not in src and "MATERIAL_PARALLEL" in src, "apply after patch_p"
body = (Path(__file__).resolve().parent / "input_rules.py").read_text(encoding="utf-8")
body = body.split("import re\n", 1)[1].strip("\n")


def sub(old, new):
    global src
    assert src.count(old) == 1, "anchor not found: " + old[:70]
    src = src.replace(old, new, 1)


MARK = "# ---------------------------------------------------------------- 요청 읽기\n"
sub(MARK,
    "# ---------------------------------------------------------------- (X) 입력 규칙\n"
    'INPUT_RULES = os.environ.get("INPUT_RULES", "1") == "1"\n'
    + body + "\n\n\n" + MARK)

sub("    own_attack = category in ATTACK_CATEGORIES\n",
    '''    # (X) 확실한 표현이 있으면 튜터가 정상으로 봤어도 범주를 바꾼다
    if INPUT_RULES and category not in ATTACK_CATEGORIES:
        if is_academic(last):
            log.warning("tutor 입력 규칙: 과제 대행 요청으로 봄")
            category = "academic_dishonesty"
            result["action"] = "scaffold"
        elif is_jailbreak(last):
            log.warning("tutor 입력 규칙: 탈옥 요청으로 봄")
            category = "jailbreak_injection"
            result["action"] = "refuse"
            result["citations"] = []
    own_attack = category in ATTACK_CATEGORIES
''')

sub("    inp = await guard_input(messages, deadline, qvec, votes=GUARD_VOTES)\n    if GUARD_SUBTYPE and",
    '''    inp = await guard_input(messages, deadline, qvec, votes=GUARD_VOTES)
    if INPUT_RULES and inp.get("decision") == "allow":
        _last = last_user(messages)
        if is_academic(_last):
            inp = {"decision": "block", "category": "academic_dishonesty"}
        elif is_jailbreak(_last):
            inp = {"decision": "block", "category": "jailbreak_injection"}
    if GUARD_SUBTYPE and''')

p.write_text(src, encoding="utf-8", newline="\n")
print("patched X")
