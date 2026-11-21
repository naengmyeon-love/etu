"""D안: LLM 이 쓴 답이 인용 섹션보다 다른 섹션과 훨씬 더 겹치면(D_RATIO 배 이상), 그 다른 섹션 쪽 문장을 뺀다.
근거 충실 판정은 인용한 섹션 원문만 보므로, 인용하지 않은 섹션에서 가져온 문장은 '근거 없음'으로 판정되기 쉽다.
인용은 바꾸지 않는다. 전부 빠지면 뒤에 붙는 원문 문장만 남는다 (원문도 없으면 원래 답으로 되돌아감, 기존 코드).
공개 세트 600행 다시 채점(cite3.py): 바뀌는 행에서 근거 충실 0.59 -> 0.90, 정답 사실 그대로
끄기: D_DROP=0
python3 patch_d.py <고칠 서버 파일>   (server_t3 = T안 위에 적용)"""
import sys

p = sys.argv[1]
s = open(p, encoding="utf-8").read()
assert "D_DROP" not in s, "이미 적용됨"


def rep(old: str, new: str) -> None:
    global s
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new)


rep('''def tidy_answer(answer: str, section_text: str) -> str:''',
    '''D_DROP = os.environ.get("D_DROP", "1") == "1"            # (D) 다른 섹션 쪽으로 기운 LLM 문장 빼기
D_RATIO = float(os.environ.get("D_RATIO", "1.4"))         # 다른 섹션 겹침이 인용 섹션의 이 배 이상일 때만


def drop_other_section(answer: str, cite: str) -> str:
    """답 전체가 인용 섹션보다 다른 섹션과 D_RATIO 배 이상 더 겹치면, 그 섹션과 더 겹치는 문장을 뺀다."""
    ev = app.state.evidence
    if not hasattr(ev, "dsec"):
        raw = getattr(ev, "raw", {}) or {}
        ids = sorted(raw)
        ev.dsec = (ids, BM25Okapi([tok2(raw[k]) for k in ids])) if ids else None
    if not ev.dsec:
        return answer
    ids, bm = ev.dsec
    if cite not in ids:
        return answer
    ci = ids.index(cite)
    sc = np.asarray(bm.get_scores(tok2(SECTION_ID_RE.sub(" ", answer))), dtype=np.float64)
    bi = int(np.argmax(sc))
    if bi == ci or sc[bi] <= 0 or sc[bi] < D_RATIO * max(float(sc[ci]), 1e-9):
        return answer
    sents = [x for x in _SENT_SPLIT.split(answer.strip()) if x.strip()]
    keep = []
    for x in sents:
        xs = bm.get_scores(tok2(SECTION_ID_RE.sub(" ", x)))
        if xs[bi] > xs[ci] and xs[bi] > 0:
            continue
        keep.append(x)
    if len(keep) == len(sents):
        return answer
    out = " ".join(keep).strip()
    if not re.sub(r"[\\s()]+", "", SECTION_ID_RE.sub("", out)):
        return ""
    return out


def tidy_answer(answer: str, section_text: str) -> str:''')

rep('''    # (E1) 인용한 섹션 원문에서 질문과 가장 가까운 문장 하나를 그대로 덧붙인다
    quote_scaffold''',
    '''    # (D) 인용하지 않은 섹션 쪽으로 기운 LLM 문장을 뺀다 (근거 충실 판정은 인용 섹션 원문만 본다)
    if D_DROP and result["action"] == "answer" and len(result["citations"]) == 1 and result["answer"].strip():
        try:
            d_out = drop_other_section(result["answer"], result["citations"][0])
            if d_out != result["answer"]:
                log.warning("tutor 다른 섹션 문장 빼기")
                result["answer"] = d_out
        except Exception as e:
            log.warning("다른 섹션 문장 빼기 실패: %s", type(e).__name__)
    # (E1) 인용한 섹션 원문에서 질문과 가장 가까운 문장 하나를 그대로 덧붙인다
    quote_scaffold''')

open(p, "w", encoding="utf-8").write(s)
print("D안 적용:", p)
