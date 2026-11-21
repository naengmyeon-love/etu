"""FB안 (Arena 1위 D1 의 핵심만 작게): API 가 느리거나 실패할 때 '무엇이 궁금한지' 대신 LLM 없이 점수가 나는 답을 낸다.
- 튜터 LLM 이 두 번 다 실패(data 없음): 검색 1위 섹션 원문에서 질문과 가장 가까운 한 문장을 답 머리로 쓰고, 인용 = 그 섹션.
  뒤 단계(가드 교차 검사·고정 문구·원문 덧붙이기 등)는 평소대로 돈다.
- /tutor 전체가 25초 예산을 넘김: 같은 방식으로 원문 3문장 답을 바로 만든다(LLM 없음, 수 ms).
- 둘 다 먼저 입력 규칙(과제 대행·탈옥·강한 위기·게임 공략·뜻 없는 입력)을 정규식으로 본다 → 해당하면 그 action 의 고정 문구.
- 요청 하나 안의 값만 쓴다(요청 사이 저장 없음). 로그는 고정 문구만.
대체 응답 한 문항(정상 질문) 점수: 지금 clarify ≈ 0.25 → 추출식 답 ≈ 0.9 (action 1, 인용 ≈0.90, 원문이라 충실 ≈1)
끄기: FB=0
python3 patch_fb.py <고칠 서버 파일>"""
import sys

p = sys.argv[1]
s = open(p, encoding="utf-8").read()
assert "def extractive_fallback" not in s, "이미 적용됨"


def rep(old: str, new: str) -> None:
    global s
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new)


rep('''async def run_tutor(messages: list[dict], deadline: float) -> dict:''',
    '''FB = os.environ.get("FB", "1") == "1"  # (FB) LLM 실패·시간 초과 때 추출식 대체 응답


def extractive_fallback(messages: list[dict], n_sent: int = 3) -> dict:
    """LLM 없이 만드는 대체 응답. 입력 규칙에 걸리면 그 action 의 고정 문구, 아니면 검색 1위 섹션 원문 n_sent 문장."""
    try:
        last = last_user(messages) if messages else ""
        if not last.strip() or looks_low_signal(last):
            return {"answer": ACTION_FALLBACK_ANSWER["clarify"], "action": "clarify", "citations": []}
        if is_academic(last):
            return {"answer": ACADEMIC_SCAFFOLD, "action": "scaffold", "citations": []}
        if is_jailbreak(last):
            return {"answer": ACTION_FALLBACK_ANSWER["refuse"], "action": "refuse", "citations": []}
        if T_EMO and STRONG_DISTRESS_RE.search(last):
            return {"answer": ACTION_FALLBACK_ANSWER["escalate"], "action": "escalate", "citations": []}
        if T_GAME and GAME_RE.search(last):
            return {"answer": ACTION_FALLBACK_ANSWER["redirect"], "action": "redirect", "citations": []}
        ev = app.state.evidence
        sec, _ = ev.rank_sections(search_query(messages), None)
        top = ev.sec_ids[int(np.argmax(sec))]
        sents = ev.best_sentences(top, last, "", n_sent)
        if not sents:
            return dict(TUTOR_FALLBACK)
        body = " ".join(sents)
        answer = (QUOTE_LEAD.strip() + " " + body) if n_sent > 1 else body
        return {"answer": answer[:ANSWER_LIMIT], "action": "answer", "citations": [top]}
    except Exception as e:
        log.warning("추출식 대체 응답 실패: %s", type(e).__name__)
        return dict(TUTOR_FALLBACK)


async def run_tutor(messages: list[dict], deadline: float) -> dict:''')

# 튜터 LLM 이 실패했을 때: 원문 한 문장을 답 머리로 (뒤에서 원문 덧붙이기가 더해진다)
rep('''    data = await app.state.llm.ask_json(TUTOR_SYSTEM, user, deadline, effort=REASONING_EFFORT)
    result = clean_tutor(data, ids)
''', '''    data = await app.state.llm.ask_json(TUTOR_SYSTEM, user, deadline, effort=REASONING_EFFORT)
    result = clean_tutor(data, ids)
    if FB and not data:
        log.warning("tutor LLM 실패: 추출식 답으로 대체")
        result = extractive_fallback(messages, n_sent=1)
''')

# /tutor 전체 시간 초과·예외: 원문 3문장 답
rep('''            log.warning("%s 처리 실패, 대체 응답: %s", kind, type(e).__name__)
            result = fallback''',
    '''            log.warning("%s 처리 실패, 대체 응답: %s", kind, type(e).__name__)
            result = fallback
            if FB and kind == "tutor":
                result = extractive_fallback(messages, n_sent=3)''')

open(p, "w", encoding="utf-8").write(s)
print("FB안 적용:", p)
