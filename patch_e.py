"""E안 (v3.5 + B+R + H 위에 적용). LLM 호출 수·프롬프트는 그대로다. 답변을 다 만든 뒤 코드로만 다듬는다.

E1 (QUOTE_ADD): action=answer 이면 인용한 섹션 원문에서 마지막 질문과 가장 가까운 문장 하나를 답 뒤에 그대로 덧붙인다.
    답이 질문의 다른 면만 말해 핵심 사실을 빠뜨리는 일을 줄인다. 원문 그대로라 근거에서 벗어나지 않는다.
    QUOTE_SCAFFOLD=1 이면 정답만 달라는 요청(answer_seeking)의 힌트 답변에도 참고할 원문 한 문장을 붙인다.
E2 (ABSTAIN_SHORT): action=abstain 이면 "자료에 없어요" 첫 문장만 남기고 덧붙인 관련 사실과 인용을 뺀다.
    덧붙인 사실이 근거에서 확인되지 않는다고 판정되는 일이 많았다.
E3 (HATE_REDIRECT 문구): 혐오·차별 발언에 답하는 고정 문구를 바꾼다.
사용: python3 patch_e.py server.py   끄기: QUOTE_ADD=0, QUOTE_SCAFFOLD=0, ABSTAIN_SHORT=0
"""
import sys
from pathlib import Path

p = Path(sys.argv[1])
src = p.read_text(encoding="utf-8")
assert "QUOTE_ADD" not in src, "already patched"


def sub(old, new):
    global src
    assert src.count(old) == 1, "anchor not found: " + old[:70]
    src = src.replace(old, new, 1)


sub('SHOT_BALANCE = os.environ.get("SHOT_BALANCE", "1") == "1"\n',
    'SHOT_BALANCE = os.environ.get("SHOT_BALANCE", "1") == "1"\n'
    'QUOTE_ADD = os.environ.get("QUOTE_ADD", "1") == "1"            # (E1) 답 뒤에 인용 섹션 원문 한 문장을 덧붙인다\n'
    'QUOTE_SCAFFOLD = os.environ.get("QUOTE_SCAFFOLD", "1") == "1"  # (E1) 정답만 달라는 요청의 힌트 답변에도 붙인다\n'
    'ABSTAIN_SHORT = os.environ.get("ABSTAIN_SHORT", "1") == "1"    # (E2) 자료에 없다는 답은 첫 문장만, 인용 없이\n'
    'QUOTE_LEAD = " 자료에는 이렇게 나와 있어요: "\n'
    'QUOTE_LEAD_SCAFFOLD = " 자료의 이 부분을 참고해 보세요: "\n'
    '_SENT_SPLIT = re.compile(r"(?<=[.!?])\\s+")\n'
    '_ID_IN_TEXT = re.compile(r"\\s*\\(?\\b(?:PY|PR|FIN)-\\d{2}-\\d{2}\\b\\)?")\n'
    '\n'
    '\n'
    'def split_kb_sentences(text: str) -> list[str]:\n'
    '    """섹션 원문을 문장으로 나눈다. 제목·표 안내·코드 울타리·질문 문장은 뺀다."""\n'
    '    out: list[str] = []\n'
    '    for line in text.splitlines():\n'
    '        line = line.strip()\n'
    '        if not line or line.startswith("#") or line.startswith("(표:") or line.startswith("```"):\n'
    '            continue\n'
    '        line = re.sub(r"^A\\.\\s*", "", line)\n'
    '        for s in _SENT_SPLIT.split(line):\n'
    '            s = s.strip()\n'
    '            if 12 <= len(s) <= 220 and not s.endswith("?"):\n'
    '                out.append(s)\n'
    '    return list(dict.fromkeys(out))\n')

sub('    def pick_citation(self, text: str, candidates: list[str]) -> str | None:\n',
    '    def build_sentences(self) -> None:\n'
    '        """(E1) 섹션별 원문 문장 색인. 배포 KB 파일만 읽는다 (본문 .md + 표 .csv)."""\n'
    '        from pathlib import Path\n'
    '        self.sent: dict[str, tuple[list[str], BM25Okapi]] = {}\n'
    '        for path in sorted(Path(KB_DIR).glob("*/*.md")):\n'
    '            tables = sorted(path.parent.glob(f"{path.stem}-table*.csv"))\n'
    '            text = "\\n\\n".join([path.read_text(encoding="utf-8"), *(t.read_text(encoding="utf-8") for t in tables)])\n'
    '            sents = split_kb_sentences(text)\n'
    '            if sents:\n'
    '                self.sent[path.stem] = (sents, BM25Okapi([tok2(s) for s in sents]))\n'
    '\n'
    '    def best_sentence(self, sid: str, query: str, answer: str) -> str | None:\n'
    '        """(E1) 섹션 원문에서 질문과 가장 가까운 문장. 답에 이미 들어 있는 문장은 건너뛴다."""\n'
    '        if not hasattr(self, "sent"):\n'
    '            self.build_sentences()\n'
    '        ent = self.sent.get(sid)\n'
    '        if not ent:\n'
    '            return None\n'
    '        sents, bm = ent\n'
    '        scores = bm.get_scores(tok2(query))\n'
    '        flat = squash(answer)\n'
    '        for i in sorted(range(len(sents)), key=lambda i: -scores[i]):\n'
    '            if scores[i] <= 0:\n'
    '                break\n'
    '            if squash(sents[i]) in flat:\n'
    '                continue\n'
    '            return sents[i]\n'
    '        return None\n'
    '\n'
    '    def pick_citation(self, text: str, candidates: list[str]) -> str | None:\n')

sub('''        result["citations"] = []
    return result


async def answer_complied(''', '''        result["citations"] = []
    # (E1) 인용한 섹션 원문에서 질문과 가장 가까운 문장 하나를 그대로 덧붙인다
    quote_scaffold = (QUOTE_SCAFFOLD and result["action"] == "scaffold"
                      and (category == "answer_seeking" or (gcat == "answer_seeking" and category not in ATTACK_CATEGORIES)))
    if QUOTE_ADD and result["citations"] and (result["action"] == "answer" or quote_scaffold):
        try:
            sent = app.state.evidence.best_sentence(result["citations"][0], last, result["answer"])
            if sent:
                lead = QUOTE_LEAD if result["action"] == "answer" else QUOTE_LEAD_SCAFFOLD
                longer = result["answer"].rstrip() + lead + sent
                if len(longer) <= ANSWER_LIMIT:
                    result["answer"] = longer
        except Exception as e:
            log.warning("근거 문장 덧붙이기 실패: %s", type(e).__name__)
    # (E2) 자료에 없다는 답은 첫 문장만 남기고 인용을 뺀다
    if ABSTAIN_SHORT and result["action"] == "abstain":
        body = _ID_IN_TEXT.sub("", result["answer"]).strip()
        head = _SENT_SPLIT.split(body)[0].strip() if body else ""
        ok = bool(head) and len(head) <= 200 and ("없" in head or "않" in head)
        result["answer"] = head if ok else ACTION_FALLBACK_ANSWER["abstain"]
        result["citations"] = []
    return result


async def answer_complied(''')

# 서버 시작 때 문장 색인을 미리 만든다 (첫 요청에서 만들지 않게)
sub('    app.state.evidence = Evidence(',
    '    app.state.evidence = _evidence_with_sentences(')
sub('def split_kb_sentences(text: str) -> list[str]:\n',
    'def _evidence_with_sentences(*args, **kwargs):\n'
    '    ev = Evidence(*args, **kwargs)\n'
    '    try:\n'
    '        ev.build_sentences()\n'
    '    except Exception as e:  # 색인을 못 만들어도 서버는 뜬다 (덧붙이기만 빠진다)\n'
    '        log.warning("문장 색인 실패: %s", type(e).__name__)\n'
    '        ev.sent = {}\n'
    '    return ev\n'
    '\n'
    '\n'
    'def split_kb_sentences(text: str) -> list[str]:\n')

if len(sys.argv) > 2:  # (E3) 혐오 발언 대응 문구 교체: python3 patch_e.py server.py "<새 문구>"
    import re as _re
    m = _re.search(r'^HATE_REDIRECT = "[^"]*"$', src, _re.M)
    assert m and '"' not in sys.argv[2], "HATE_REDIRECT not found"
    src = src.replace(m.group(0), 'HATE_REDIRECT = "' + sys.argv[2] + '"', 1)

p.write_text(src, encoding="utf-8", newline="\n")
print("patched E")
