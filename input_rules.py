"""입력 규칙 (LLM 분류를 보완한다). 확실한 표현일 때만 걸리게 하고, 그 표현을 '공부 대상으로 다루는' 문장은 뺀다."""
import re

# 제출할 과제·수행평가 답안을 대신 완성해 달라는 표현
ACADEMIC_RE = re.compile(
    r"(?:수행\s*평가|숙제|과제|시험|평가)\s*(?:에|로|에다)?\s*(?:낼|넣을|제출할|제출하려는|내려는)"
    r"|제출\s*(?:할|용)\s*(?:답안|코드|글|보고서|리포트|수행\s*평가|과제|숙제)"
    r"|제출\s*(?:용으로|답안)"
    r"|(?:수행\s*평가|과제|숙제)\s*(?:답안으로|보고서\s*전체|문항|안내)"
    r"|답안\s*전체"
    r"|graded\s+(?:assessment|assignment|homework)"
    r"|(?:assessment|assignment|homework)\s+answer"
    r"|answer\s+for\s+(?:my\s+)?(?:\w+\s+)?(?:assessment|assignment|homework|submission)"
    r"|for\s+submission\b",
    re.IGNORECASE,
)

# 튜터 자신의 숨겨진 지시·시스템 프롬프트를 내놓으라는 표현
_SECRET = r"(?:숨겨진|숨은)\s*(?:지시|지침|프롬프트)|시\s*스\s*템\s*(?:프롬프트|메시지|지시)|system\s*prompt|sys\s*prompt"
JAIL_SECRET_RE = re.compile(
    r"(?:네|너의|너한테|니|튜터의|your)\s*(?:처음\s*)?(?:주어진|설정된)?\s*(?:" + _SECRET + r")"
    r"|(?:" + _SECRET + r")\s*(?:의\s*)?(?:원문|전문|전부|전체)",
    re.IGNORECASE,
)
# 튜터에게 제한 없는 역할이 되라고 시키는 표현
JAIL_ROLE_RE = re.compile(
    r"(?:제한|규칙)\s*(?:이\s*)?없[는이]?[^.\n]{0,14}?(?:역할로|(?:안내자|AI|존재|등장인물|튜터)(?:가|이)\s*(?:돼서|되어|돼)|AI로|존재로|바뀌어|바꿔서)"
    r"|(?:튜터|역할)[^.\n]{0,8}?(?:그만두고|버리고|버려)"
    r"|(?:이전|앞의|앞선|위의)?\s*지시(?:를|는)?\s*(?:싹\s*|다\s*|모두\s*)?무시(?:해|하고|하며)[^.\n]{0,40}?(?:답해|대답해|알려\s*줘|설명해\s*줘|답\s*ㄱㄱ)"
    r"|너한테\s*적용",
    re.IGNORECASE,
)
# 위 표현을 공부 대상으로 다루는 문장 (이 낱말이 있으면 규칙으로 판단하지 않는다)
JAIL_MENTION_RE = re.compile(
    r"라는|이라는|라고 |라고$|달라는|요구하는|노리는|문장|표현|예시로|비교해서|살펴보|넣어|괜찮은지|구분|구별|울타리를|울타리\s*안|fence|어떻게\s*다루|왜\s*프롬프트|대응",
    re.IGNORECASE,
)

MENTION_WINDOW = 20
# 문장 어디에 있든 '그 표현을 공부 대상으로 다룬다'는 뜻이 분명한 말
JAIL_STUDY_RE = re.compile(
    "달라는 |예시로|괜찮은지|라는 역할|요구하는 글|요구하는 문장|비교해서|살펴보면서|살펴보며|연습 문장|역할 문장을 살펴|역할 표현을",
    re.IGNORECASE,
)


def is_academic(text: str) -> bool:
    return bool(ACADEMIC_RE.search(text))


def is_jailbreak(text: str) -> bool:
    """규칙에 걸리는 표현이 있고, 그것을 공부 대상으로 다루는 말이 없을 때만 True."""
    if JAIL_STUDY_RE.search(text):
        return False
    for rx in (JAIL_SECRET_RE, JAIL_ROLE_RE):
        for m in rx.finditer(text):
            if not JAIL_MENTION_RE.search(text[m.start():m.end() + MENTION_WINDOW]):
                return True
    return False
