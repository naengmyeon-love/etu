"""F안 (H안 + E안 위에 적용): 인용 섹션을 고를 때 '질문과 가장 가까운 원문 문장이 있는 섹션'도 본다.
  점수 = 답변·질문 겹침 + CTX_LAM × 앞선 대화 겹침 + CTX_RET × 검색 점수 + CTX_SENT × 섹션 안 문장 최고점
LLM 호출 수·프롬프트는 그대로다. 사용: python3 patch_f.py server.py   끄기: CTX_SENT=0
"""
import sys
from pathlib import Path

p = Path(sys.argv[1])
src = p.read_text(encoding="utf-8")
assert "CTX_SENT" not in src and "QUOTE_ADD" in src and "CTX_CITE" in src, "apply after patch_h and patch_e"


def sub(old, new):
    global src
    assert src.count(old) == 1, "anchor not found: " + old[:70]
    src = src.replace(old, new, 1)


sub('CTX_HIST_CHARS = 6000',
    'CTX_SENT = float(os.environ.get("CTX_SENT", "1.0"))     # (F) 질문과 가장 가까운 원문 문장이 있는 섹션 비중\n'
    'CTX_HIST_CHARS = 6000')

sub('    def pick_with_history(self, text: str, hist: str, ret: np.ndarray | None = None) -> str | None:\n',
    '    def sent_scores(self, query: str) -> np.ndarray:\n'
    '        """(F) 섹션마다 질문과 가장 가까운 원문 문장의 점수."""\n'
    '        if not hasattr(self, "sent"):\n'
    '            self.build_sentences()\n'
    '        t = tok2(query)\n'
    '        out = np.zeros(len(self.sec_ids), dtype=np.float64)\n'
    '        for i, sid in enumerate(self.sec_ids):\n'
    '            ent = self.sent.get(sid)\n'
    '            if ent:\n'
    '                out[i] = float(np.max(ent[1].get_scores(t)))\n'
    '        return out\n'
    '\n'
    '    def pick_with_history(self, text: str, hist: str, ret: np.ndarray | None = None,\n'
    '                          sent: np.ndarray | None = None) -> str | None:\n')

sub('        score = np.where(a >= CTX_TAU, score, -1e9)\n',
    '        if sent is not None and CTX_SENT > 0:\n'
    '            score = score + CTX_SENT * _mm(sent)\n'
    '        score = np.where(a >= CTX_TAU, score, -1e9)\n')

sub('SECTION_ID_RE.sub(" ", result["answer"]) + " " + last, hist, ret)',
    'SECTION_ID_RE.sub(" ", result["answer"]) + " " + last, hist, ret,\n'
    '                                                            app.state.evidence.sent_scores(last) if CTX_SENT > 0 else None)')

p.write_text(src, encoding="utf-8", newline="\n")
print("patched F")
