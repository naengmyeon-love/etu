"""S안: 덧붙일 원문 문장을 학습한 선택기(sentsel.py)로 고른다. 실패하면 지금 방식(BM25) 그대로.
python3 patch_s.py <고칠 서버 파일>   (SENT_SEL=0 이면 끔)"""
import sys

p = sys.argv[1]
s = open(p, encoding="utf-8").read()
assert "SENT_SEL" not in s, "이미 적용됨"

a = 'QUOTE_LEAD = " 자료에는 이렇게 나와 있어요: "'
assert s.count(a) == 1
s = s.replace(a, 'SENT_SEL = os.environ.get("SENT_SEL", "1") == "1"  # (S) 덧붙일 문장을 학습한 선택기로 고른다\n' + a)

b = '''            sent = " ".join(app.state.evidence.best_sentences(
                result["citations"][0], last, result["answer"], QUOTE_K_ANSWER if result["action"] == "answer" else QUOTE_K_SCAFFOLD))'''
assert s.count(b) == 1
s = s.replace(b, '''            sent = " ".join(pick_quotes(
                result["citations"][0], last, messages, qvec, result["answer"], QUOTE_K_ANSWER if result["action"] == "answer" else QUOTE_K_SCAFFOLD))''')

c = "async def run_tutor(messages: list[dict], deadline: float) -> dict:"
assert s.count(c) == 1
s = s.replace(c, '''def pick_quotes(sid: str, last: str, messages: list[dict], qvec, answer: str, k: int) -> list[str]:
    """(S) 덧붙일 원문 문장 k 개. 선택기가 없거나 실패하면 질문과 가장 가까운 문장(BM25)으로 고른다."""
    sel = getattr(app.state, "sel", None)
    if sel is not None and qvec is not None:
        try:
            hist = " ".join(m["content"] for m in messages if m["role"] == "user")
            out = sel.pick(sid, last, hist, qvec, answer, k)
            if out:
                return out
        except Exception as e:
            log.warning("문장 선택 실패: %s", type(e).__name__)
    return app.state.evidence.best_sentences(sid, last, answer, k)


''' + c)

d = '    app.state.llm = LLM()\n    log.info("준비 완료'
assert s.count(d) == 1
s = s.replace(d, '''    app.state.sel = None
    if SENT_SEL:
        try:  # 문장 선택기는 시작 때 한 번 만든다. 못 만들어도 서버는 뜬다
            from index import embed_texts
            from sentsel import SentSelector
            app.state.sel = SentSelector.build(app.state.evidence, PUBLIC_DIR, INDEX_DIR, tok2, squash, context_text, embed_texts)
            log.info("문장 선택기 준비: 학습 문항 %d개", app.state.sel.n_train)
        except Exception as e:
            log.warning("문장 선택기 준비 실패: %s", type(e).__name__)
''' + d)
open(p, "w", encoding="utf-8").write(s)
print("S안 적용:", p)
