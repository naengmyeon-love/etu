"""H안: 인용 섹션을 고를 때 대화 흐름(앞선 대화)을 함께 본다. LLM 호출 수·프롬프트는 그대로다.

같은 사실이 여러 섹션에 겹쳐 적혀 있을 때, 답변 내용과 맞으면서 학습자가 지금 보고 있던 흐름(앞선 대화)과
가장 가까운 섹션을 인용한다. 앞선 대화가 없으면 답변 겹침과 검색 점수만 본다.
  점수 = 답변·질문 겹침(A) + CTX_LAM × 앞선 대화 겹침(H) + CTX_RET × 검색 점수(R),  A 가 CTX_TAU 미만인 섹션은 제외
사용: python3 patch_h.py server.py   끄기: CTX_CITE=0
"""
import sys
from pathlib import Path

p = Path(sys.argv[1])
src = p.read_text(encoding="utf-8")
assert "CTX_CITE" not in src, "already patched"


def sub(old, new):
    global src
    assert src.count(old) == 1, "anchor not found: " + old[:70]
    src = src.replace(old, new, 1)


sub('SHOT_BALANCE = os.environ.get("SHOT_BALANCE", "1") == "1"\n',
    'SHOT_BALANCE = os.environ.get("SHOT_BALANCE", "1") == "1"\n'
    'CTX_CITE = os.environ.get("CTX_CITE", "1") == "1"       # (H) 인용 섹션을 고를 때 앞선 대화도 본다\n'
    'CTX_LAM = float(os.environ.get("CTX_LAM", "1.0"))       # 앞선 대화 겹침 비중\n'
    'CTX_TAU = float(os.environ.get("CTX_TAU", "0.3"))       # 답변 겹침이 이보다 낮은 섹션은 인용하지 않는다\n'
    'CTX_RET = float(os.environ.get("CTX_RET", "1.0"))       # 검색 점수 비중\n'
    'CTX_HIST_CHARS = 6000                                    # 앞선 대화에서 보는 글자 수 상한 (최근 것부터)\n')

sub('    def render(self, by_section: dict[str, list[dict]], max_sections: int = MAX_SECTIONS) -> tuple[str, list[str]]:\n',
    '    def pick_with_history(self, text: str, hist: str, ret: np.ndarray | None = None) -> str | None:\n'
    '        """(H) 답변·질문과 겹치면서 앞선 대화·검색 결과와 가장 가까운 섹션. 볼 것이 없으면 None."""\n'
    '        use_ret = ret is not None and CTX_RET > 0\n'
    '        if self.sec2 is None or not (hist.strip() or use_ret):\n'
    '            return None\n'
    '        a = _mm(np.asarray(self.sec2.get_scores(tok2(text)), dtype=np.float64))\n'
    '        score = a.copy()\n'
    '        if hist.strip():\n'
    '            score = score + CTX_LAM * _mm(np.asarray(self.sec2.get_scores(tok2(hist)), dtype=np.float64))\n'
    '        if use_ret:\n'
    '            score = score + CTX_RET * _mm(ret)\n'
    '        score = np.where(a >= CTX_TAU, score, -1e9)\n'
    '        return self.sec_ids[int(np.argmax(score))]\n'
    '\n'
    '    def render(self, by_section: dict[str, list[dict]], max_sections: int = MAX_SECTIONS) -> tuple[str, list[str]]:\n')

sub('''        except Exception as e:
            log.warning("인용 재선택 실패: %s", type(e).__name__)

    # 가드 입력 분류(동시에 돌린 것)를 받아 튜터 판단과 교차 확인한다
''', '''        except Exception as e:
            log.warning("인용 재선택 실패: %s", type(e).__name__)
    if CTX_CITE and result["action"] in ("answer", "scaffold") and result["citations"]:
        try:  # (H) 같은 사실이 여러 섹션에 있으면 앞선 대화 흐름과 가까운 섹션을 인용한다
            hist = chr(10).join(m["content"] for m in messages[:-1])[-CTX_HIST_CHARS:]
            ret = None
            if CTX_RET > 0:
                ret = app.state.evidence.rank_sections(search_query(messages), qvec)[0]
            old = result["citations"][0]
            best = app.state.evidence.pick_with_history(SECTION_ID_RE.sub(" ", result["answer"]) + " " + last, hist, ret)
            if best and best != old:
                result["answer"] = result["answer"].replace(old, best)
                result["citations"] = [best]
        except Exception as e:
            log.warning("대화 흐름 인용 실패: %s", type(e).__name__)

    # 가드 입력 분류(동시에 돌린 것)를 받아 튜터 판단과 교차 확인한다
''')

p.write_text(src, encoding="utf-8", newline="\n")
print("patched H")
