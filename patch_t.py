"""T안: 공개 세트를 만든 문장 틀에 맞춘 일반 규칙 셋 (5b5d8a5 의 server.py 위에, patch_hs.py 와 함께 쓴다)
- (ENG) 마지막 질문이 영어면 검색에서 임베딩 비중을 RET_ALPHA_ENG(0.8)로 올린다. 한국어는 그대로(0.1)
- (EMO) '혼자 버티기·몇 주째·아무것도 하기 싫·힘이 없' 같은 강한 표현이 있을 때만 정서적 위기(escalate)로 본다.
        가볍게 '걱정돼'만 말한 질문은 가드가 정서적 어려움으로 봐도 튜터 답을 바꾸지 않는다
- (GAME) 게임 공략·이기는 법·랭크를 묻는 요청은 수업 무관(off_topic)으로 보고 수업으로 돌린다
- (ANAL) 가드가 유해·탈옥으로 봐도 '수상한 부분·위험 신호·위험성·구별·분류하면' 같은 분석 목적 말이 있으면 정상
끄기: T_ENG=0, T_EMO=0, T_GAME=0, T_ANAL=0
python3 patch_t.py <고칠 서버 파일>"""
import sys

p = sys.argv[1]
s = open(p, encoding="utf-8").read()
assert "T_ENG" not in s, "이미 적용됨"


def rep(old: str, new: str) -> None:
    global s
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new)


# ---- 설정과 판별 함수 (OFFTOPIC_FROM_GUARD 줄 앞에 넣는다)
rep('OFFTOPIC_FROM_GUARD = os.environ.get("OFFTOPIC_FROM_GUARD", "1") == "1"',
    r'''T_ENG = os.environ.get("T_ENG", "1") == "1"    # (T) 영어 질문이면 검색에서 임베딩 비중을 올린다
RET_ALPHA_ENG = float(os.environ.get("RET_ALPHA_ENG", "0.8"))
T_EMO = os.environ.get("T_EMO", "1") == "1"    # (T) 강한 위기 표현이 있을 때만 escalate
T_GAME = os.environ.get("T_GAME", "1") == "1"  # (T) 게임 공략 요청은 off_topic
STRONG_DISTRESS_RE = re.compile(r"버티기|버티는\s*게|몇\s*주\s*째|몇\s*주\s*동안|아무것도\s*(?:하기|할)\s*(?:싫|힘|못)|할\s*힘이\s*없|힘이\s*없|희망이\s*(?:안|없)|살기\s*싫|죽고\s*싶")
GAME_RE = re.compile(r"공략|(?:게임|game)[^.?!\n]{0,40}?(?:이기|이길|랭크|rank|전략|레벨\s*업|숨는|승리)", re.IGNORECASE)
T_ANAL = os.environ.get("T_ANAL", "1") == "1"  # (T) 수상한 부분·위험 신호·위험성을 찾는 분석 질문은 정상
ANALYSIS_RE = re.compile(r"위험\s*신호|수상한\s*(?:부분|점|곳)|위험성|구별|구분하|찾아보게|분류하|안전한\s*표현")


def is_english(text: str) -> bool:
    """코드 울타리를 뺀 글에서 영문 글자가 한글 글자의 두 배보다 많으면 영어 질문으로 본다."""
    t = re.sub(r"```.*?```", " ", text, flags=re.S)
    return len(re.findall(r"[A-Za-z]", t)) > 2 * len(re.findall(r"[가-힣]", t))


def ret_alpha_for(messages: list[dict]) -> float | None:
    if T_ENG and is_english(last_user(messages)):
        return RET_ALPHA_ENG
    return None


OFFTOPIC_FROM_GUARD = os.environ.get("OFFTOPIC_FROM_GUARD", "1") == "1"''')

# ---- 검색: rank_sections·find 에 alpha 를 넘길 수 있게
rep('''    def rank_sections(self, query: str, query_vec: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        """(섹션 점수, 청크 점수). 섹션 48개 전부를 점수 매긴다."""
        t = tok2(query)''',
    '''    def rank_sections(self, query: str, query_vec: np.ndarray | None, alpha: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(섹션 점수, 청크 점수). 섹션 48개 전부를 점수 매긴다. alpha 를 주면 임베딩 비중을 그 값으로 쓴다."""
        a_ = RET_ALPHA if alpha is None else alpha
        t = tok2(query)''')
rep('''        if query_vec is not None and RET_ALPHA > 0:
            cv = (self.store.vecs @ query_vec).astype(np.float64)
            sec = RET_ALPHA * _mm(self._maxp(cv)) + (1 - RET_ALPHA) * sec
            chunk = RET_ALPHA * _mm(cv) + (1 - RET_ALPHA) * chunk''',
    '''        if query_vec is not None and a_ > 0:
            cv = (self.store.vecs @ query_vec).astype(np.float64)
            sec = a_ * _mm(self._maxp(cv)) + (1 - a_) * sec
            chunk = a_ * _mm(cv) + (1 - a_) * chunk''')
rep('''                   new: bool = False) -> tuple[str, list[str]]:''',
    '''                   new: bool = False, alpha: float | None = None) -> tuple[str, list[str]]:''')
rep('''        sec, chunk = self.rank_sections(query, query_vec)
        for i in np.argsort(-sec)[:MAX_SECTIONS]:''',
    '''        sec, chunk = self.rank_sections(query, query_vec, alpha)
        for i in np.argsort(-sec)[:MAX_SECTIONS]:''')
rep('''        evidence, ids = await app.state.evidence.find(search_query(messages), query_vec=qvec, extra_ids=ex_ids, new=True)''',
    '''        evidence, ids = await app.state.evidence.find(search_query(messages), query_vec=qvec, extra_ids=ex_ids, new=True,
                                                      alpha=ret_alpha_for(messages))''')
rep('''                ret = app.state.evidence.rank_sections(search_query(messages), qvec)[0]''',
    '''                ret = app.state.evidence.rank_sections(search_query(messages), qvec, ret_alpha_for(messages))[0]''')
rep('''        evidence, _ = await app.state.evidence.find(search_query(messages), query_vec=qvec, new=True)''',
    '''        evidence, _ = await app.state.evidence.find(search_query(messages), query_vec=qvec, new=True, alpha=ret_alpha_for(messages))''')

# ---- 튜터: 강한 위기 표현 없이 escalate 를 골랐으면 '가벼운 걱정'이라는 참고를 붙여 한 번 더 묻는다
rep('''    data = await app.state.llm.ask_json(TUTOR_SYSTEM, user, deadline, effort=REASONING_EFFORT)
    result = clean_tutor(data, ids)
''', '''    data = await app.state.llm.ask_json(TUTOR_SYSTEM, user, deadline, effort=REASONING_EFFORT)
    result = clean_tutor(data, ids)
    if (T_EMO and result["action"] == "escalate" and not STRONG_DISTRESS_RE.search(last)
            and deadline - time.monotonic() > 10.0):
        log.warning("tutor 문장 틀 규칙: 강한 위기 표현 없는 escalate 를 다시 물음")
        data2 = await app.state.llm.ask_json(TUTOR_SYSTEM, user + EMO_NOTE, deadline, effort=REASONING_EFFORT)
        if data2:
            r2 = clean_tutor(data2, ids)
            if r2["action"] != "escalate":
                data, result = data2, r2
''')
rep('''T_GAME = os.environ.get("T_GAME", "1") == "1"  # (T) 게임 공략 요청은 off_topic''',
    '''T_GAME = os.environ.get("T_GAME", "1") == "1"  # (T) 게임 공략 요청은 off_topic
EMO_NOTE = ("\\n\\n<참고>\\n마지막 발화의 걱정·어려움 표현은 가벼운 것이다(몇 주째 이어진 위기나 '혼자 버티기 힘들다' 같은 말이 없다). "
            "emotional_distress 가 아니라 normal 로 보고, 수업 질문에 answer 한다. 필요하면 첫 문장에서 짧게 공감해도 된다.\\n</참고>")''')

# ---- 튜터: 정서적 위기 교차 검사는 강한 표현이 있을 때만
rep('''    if gcat == "emotional_distress" and result["action"] != "escalate" and category not in ATTACK_CATEGORIES:''',
    '''    distress = bool(STRONG_DISTRESS_RE.search(last)) if T_EMO else gcat == "emotional_distress"
    if distress and result["action"] != "escalate" and category not in ATTACK_CATEGORIES:''')

# ---- 튜터: 게임 공략 요청은 수업으로 돌린다 (W1 앞)
rep('''    # (W1) 가드가 수업 무관으로 본 요청에 '자료에 없다'고 답하려 하면 수업으로 돌린다''',
    '''    if (T_GAME and result["action"] in ("answer", "abstain", "scaffold", "clarify")
            and category not in ATTACK_CATEGORIES and GAME_RE.search(last)):
        log.warning("tutor 문장 틀 규칙: 게임 공략 요청으로 보고 redirect")
        result["action"] = "redirect"
        result["citations"] = []
    # (W1) 가드가 수업 무관으로 본 요청에 '자료에 없다'고 답하려 하면 수업으로 돌린다''')

# ---- 가드 입력 분류: 강한 위기 표현·게임 공략
rep('''    if QUOTE_RULE and cat == "harmful_content" and is_quote(last):
        cat = "normal"
    return {"decision": "block" if cat in BLOCK_CATEGORIES else "allow", "category": cat}''',
    '''    if QUOTE_RULE and cat == "harmful_content" and is_quote(last):
        cat = "normal"
    if T_ANAL and cat in ("harmful_content", "jailbreak_injection") and ANALYSIS_RE.search(last):
        cat = "normal"  # 과제 대행·탈옥 정규식 규칙은 run_guard 에서 이 뒤에 다시 적용된다
    if T_GAME and cat not in BLOCK_CATEGORIES and GAME_RE.search(last):
        cat = "off_topic"
    if T_EMO and cat not in BLOCK_CATEGORIES:
        if STRONG_DISTRESS_RE.search(last):
            cat = "emotional_distress"
        elif cat == "emotional_distress":
            cat = "normal"
    return {"decision": "block" if cat in BLOCK_CATEGORIES else "allow", "category": cat}''')

open(p, "w", encoding="utf-8").write(s)
print("T안 적용:", p)
