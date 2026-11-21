"""D2 + G3 (D+R 서버 위에 적용)
(D2) LLM 이 쓴 문장을 하나씩 본다. 그 문장과 가장 많이 겹치는 섹션이 인용 섹션이 아니고 겹침이 인용 섹션의 D2_RATIO(2.0)배 이상이면 뺀다.
     (D안은 답 전체가 기울었을 때만 뺐다.) 인용은 그대로. 오프라인(800행): 바뀐 9행 근거 충실 0.556 -> 1.000, 정답 사실 1.000 그대로
(G3) 가드 후보 답변 판정을 동시에 GUARD_OUT_VOTES(3)번 돌려 다수결. 둘 이상이 unsafe 면 unsafe, 둘 이상이 근거 없음이면 근거 없음(그 판정의 구간)
     기존 덤프 3종으로 계산: 출력 점수 0.9301/0.9311/0.9281 -> 0.9343/0.9427/0.9321
끄기: D2=0, GUARD_OUT_VOTES=1
python3 patch_d2g3.py <고칠 서버 파일>"""
import sys

p = sys.argv[1]
s = open(p, encoding="utf-8").read()
assert "D2_RATIO" not in s, "이미 적용됨"


def rep(old: str, new: str) -> None:
    global s
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new)


# ---- D2: 문장 단위 빼기 함수 (drop_other_section 바로 뒤)
rep('''def tidy_answer(answer: str, section_text: str) -> str:''',
    '''D2 = os.environ.get("D2", "1") == "1"                    # (D2) 문장 단위로 다른 섹션 쪽 문장 빼기
D2_RATIO = float(os.environ.get("D2_RATIO", "2.0"))


def drop_other_sentences(answer: str, cite: str) -> str:
    """LLM 문장마다 가장 많이 겹치는 섹션이 인용 섹션이 아니고 그 겹침이 D2_RATIO 배 이상이면 그 문장을 뺀다."""
    ev = app.state.evidence
    if not getattr(ev, "dsec", None):
        drop_other_section(answer, cite)  # dsec 색인을 만든다
    if not getattr(ev, "dsec", None):
        return answer
    ids, bm = ev.dsec
    if cite not in ids:
        return answer
    ci = ids.index(cite)
    sents = [x for x in _SENT_SPLIT.split(answer.strip()) if x.strip()]
    keep = []
    for x in sents:
        xs = np.asarray(bm.get_scores(tok2(SECTION_ID_RE.sub(" ", x))), dtype=np.float64)
        bi = int(np.argmax(xs))
        if bi != ci and xs[bi] > 0 and xs[bi] >= D2_RATIO * max(float(xs[ci]), 1e-9):
            continue
        keep.append(x)
    if len(keep) == len(sents):
        return answer
    out = " ".join(keep).strip()
    if not re.sub(r"[\\s()]+", "", SECTION_ID_RE.sub("", out)):
        return ""
    return out


def tidy_answer(answer: str, section_text: str) -> str:''')

rep('''            if d_out != result["answer"]:
                log.warning("tutor 다른 섹션 문장 빼기")
                result["answer"] = d_out''',
    '''            if d_out != result["answer"]:
                log.warning("tutor 다른 섹션 문장 빼기")
                result["answer"] = d_out
            if D2 and result["answer"].strip():
                d2_out = drop_other_sentences(result["answer"], result["citations"][0])
                if d2_out != result["answer"]:
                    log.warning("tutor 다른 섹션 문장 빼기(문장 단위)")
                    result["answer"] = d2_out''')

# ---- G3: 가드 후보 답변 판정 다수결
rep('''async def guard_output(messages: list[dict], candidate: str, deadline: float, qvec: np.ndarray | None,
                       input_result: dict | None = None) -> dict:''',
    '''GUARD_OUT_VOTES = int(os.environ.get("GUARD_OUT_VOTES", "3"))  # (G3) 후보 답변 판정을 동시에 여러 번 돌려 다수결


async def guard_output(messages: list[dict], candidate: str, deadline: float, qvec: np.ndarray | None,
                       input_result: dict | None = None) -> dict:
    if GUARD_OUT_VOTES <= 1:
        return await _guard_output_once(messages, candidate, deadline, qvec, input_result)
    outs = await asyncio.gather(*[_guard_output_once(messages, candidate, deadline, qvec, input_result)
                                  for _ in range(GUARD_OUT_VOTES)], return_exceptions=True)
    outs = [o for o in outs if isinstance(o, dict)]
    if not outs:
        return {"grounded": True, "unsupported": [], "safe": True, "category": "normal"}
    half = len(outs) / 2
    unsafe = [o for o in outs if o.get("safe") is False]
    if len(unsafe) > half:
        return unsafe[0]
    ung = [o for o in outs if o.get("grounded") is False]
    if len(ung) > half:
        return ung[0]
    return next((o for o in outs if o.get("grounded") is not False and o.get("safe") is not False), outs[0])


async def _guard_output_once(messages: list[dict], candidate: str, deadline: float, qvec: np.ndarray | None,
                             input_result: dict | None = None) -> dict:''')

open(p, "w", encoding="utf-8").write(s)
print("D2+G3 적용:", p)
