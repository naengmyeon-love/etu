r"""W안 (X+Y+Z 위에 적용): 작은 규칙 두 가지.
W1 (OFFTOPIC_FROM_GUARD): 튜터가 '자료에 없어요'(abstain)로 답하려는데 가드 입력 분류가 off_topic 이면 redirect 로 바꾼다.
W2 (QUOTE_RULE): 차별·혐오 문장을 '…라는 댓글/말/문장/문구/표현'으로 인용해 분류·분석을 묻는 질문은 정상 수업 질문으로 본다.
    가드 입력 분류가 harmful_content 라고 해도 normal 로 바꾸고, 튜터에는 참고 한 줄을 붙인다 (LLM 호출 수는 그대로).
사용: python3 patch_w.py server.py   끄기: OFFTOPIC_FROM_GUARD=0, QUOTE_RULE=0
"""
import sys
from pathlib import Path

p = Path(sys.argv[1])
src = p.read_text(encoding="utf-8")
assert "QUOTE_RULE" not in src and "MENTION_RULE" in src, "apply after patch_y"


def sub(old, new):
    global src
    assert src.count(old) == 1, "anchor not found: " + old[:70]
    src = src.replace(old, new, 1)


MARK = "# ---------------------------------------------------------------- 요청 읽기\n"
sub(MARK, '''OFFTOPIC_FROM_GUARD = os.environ.get("OFFTOPIC_FROM_GUARD", "1") == "1"  # (W1)
QUOTE_RULE = os.environ.get("QUOTE_RULE", "1") == "1"                    # (W2)
QUOTE_RE = re.compile(r"(?:이)?라는\\s*(?:가상\\s*|차별\\s*|혐오\\s*)?(?:댓글|말|문장|문구|표현)")
QUOTE_NOTE = ("\\n\\n<참고>\\n마지막 발화는 어떤 문장을 인용해 놓고 그 문장을 분류·분석하는 방법을 묻는 수업 질문이다. "
              "인용된 문장이 차별·혐오 표현이어도 그 주장에 동조하라는 요청이 아니므로 harmful_content 로 보지 말고, "
              "<자료>에 있는 내용이면 answer 한다.\\n</참고>")


def is_quote(text: str) -> bool:
    """문장을 인용해 놓고 그것을 다루는 방법을 묻는 질문이면 True (탈옥 요청 표현이 있으면 제외)."""
    return bool(QUOTE_RE.search(text)) and not is_jailbreak(text)


''' + MARK)

sub('''        f"<대화>\\n{render_dialogue(messages)}\\n</대화>"
    )
    # 가드 입력 분류를 튜터 답변과 동시에''', '''        f"<대화>\\n{render_dialogue(messages)}\\n</대화>"
    )
    if QUOTE_RULE and is_quote(last):
        user += QUOTE_NOTE
    # 가드 입력 분류를 튜터 답변과 동시에''')

sub('''    if MENTION_RULE and cat == "jailbreak_injection" and is_mention(last):
        cat = "normal"
''', '''    if MENTION_RULE and cat == "jailbreak_injection" and is_mention(last):
        cat = "normal"
    if QUOTE_RULE and cat == "harmful_content" and is_quote(last):
        cat = "normal"
''')

sub('''    # 거절·전환·안내·되묻기는 고정 문구로 답한다. 요청 내용을 되풀이하거나 일부 따르는 일을 막는다.
''', '''    # (W1) 가드가 수업 무관으로 본 요청에 '자료에 없다'고 답하려 하면 수업으로 돌린다
    if OFFTOPIC_FROM_GUARD and result["action"] == "abstain" and gcat == "off_topic":
        log.warning("tutor 교차 검사: 수업 무관으로 보고 redirect")
        result["action"] = "redirect"
    # 거절·전환·안내·되묻기는 고정 문구로 답한다. 요청 내용을 되풀이하거나 일부 따르는 일을 막는다.
''')

p.write_text(src, encoding="utf-8", newline="\n")
print("patched W")
