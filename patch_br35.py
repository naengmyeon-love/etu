"""팀장 v3.5(768d5fa) server.py 에 B안 + R안을 얹는다.

B안: 튜터 few-shot 예시를 비슷한 순 24개 후보에서 같은 action 최대 2개까지 골라 6개로 채운다.
     v3.5 의 "가까운 예시 4개의 근거 섹션 추가"는 원래대로 가장 가까운 4개에서 모은다.
R안: 튜터 검색을 새 토큰화 BM25(섹션 + 청크 최고점) 90% + 임베딩 10% 로 바꾼다. 인용 재선택도 새 토큰화로.
     v3.5 의 근거 섹션 추가(extra_ids)는 그대로 살린다. 가드 검색은 그대로.
사용: python3 patch_br35.py server.py
끄기: SHOT_BALANCE=0, NEW_RETRIEVAL=0
"""
import sys
from pathlib import Path

p = Path(sys.argv[1])
src = p.read_text(encoding="utf-8")
assert "NEW_RETRIEVAL" not in src, "already patched"


def sub(old, new):
    global src
    assert src.count(old) == 1, "anchor not found: " + old[:70]
    src = src.replace(old, new, 1)


# ---------------- R: 토큰화와 정규화 함수
sub("from store import VectorStore, tokenize\n", '''from store import VectorStore, tokenize
from index import chunk_text_for_search

NEW_RETRIEVAL = os.environ.get("NEW_RETRIEVAL", "1") == "1"
RET_ALPHA = float(os.environ.get("RET_ALPHA", "0.1"))  # 임베딩 비중 (BM25 가 더 정확해서 낮게)
SHOT_BALANCE = os.environ.get("SHOT_BALANCE", "1") == "1"
SHOT_POOL = int(os.environ.get("SHOT_POOL", "24"))  # 비슷한 순으로 먼저 뽑는 예시 후보 수
SHOT_CAP = int(os.environ.get("SHOT_CAP", "2"))     # 같은 action 예시 최대 수
_TOK2_RE = re.compile(r"[a-z_][a-z0-9_]*(?:\\.[a-z_][a-z0-9_]*)*|\\d+(?:\\.\\d+)?|[가-힣]+")
_JOSA = ("으로", "에서", "에게", "까지", "부터", "이랑", "하고", "보다", "처럼",
         "은", "는", "이", "가", "을", "를", "에", "의", "도", "로", "와", "과", "만")


def tok2(text: str) -> list[str]:
    """검색용 토큰 v2 (사전·모델 없이 규칙만): 코드 식별자 원형+조각, 숫자, 한글 단어+조사 뗀 형태+2-gram."""
    out: list[str] = []
    for t in _TOK2_RE.findall(text.lower()):
        if "가" <= t[0] <= "힣":
            out.append(t)
            for j in _JOSA:
                if len(t) > len(j) + 1 and t.endswith(j):
                    out.append(t[: -len(j)])
                    break
            if len(t) >= 2:
                out.extend(t[i:i + 2] for i in range(len(t) - 1))
        else:
            out.append(t)
            parts = [x for x in re.split(r"[._]", t) if x]
            if len(parts) > 1:
                out.extend(parts)
    return out


def _mm(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)
''')

# ---------------- R: 새 색인
SEC = '        self.sec_bm25 = BM25Okapi([tokenize("\\n".join(c["title"] + " " + c["text"] for c in self.sections[sid])) for sid in self.sec_ids]) if self.sec_ids else None\n'
sub(SEC, SEC + '''        # (R) 새 검색용 색인: 배포 KB 만으로 만든다
        self.cidx = np.array([self.sec_pos[c["id"]] for c in store.chunks], dtype=np.int64)
        self.sec_chunk_idx: dict[str, list[int]] = {}
        for j, c in enumerate(store.chunks):
            self.sec_chunk_idx.setdefault(c["id"], []).append(j)
        self.kw2 = BM25Okapi([tok2(chunk_text_for_search(c)) for c in store.chunks]) if store.chunks else None
        self.sec2 = BM25Okapi([tok2("\\n".join(c["title"] + " " + c["text"] for c in self.sections[sid])) for sid in self.sec_ids], k1=0.9, b=0.4) if self.sec_ids else None
''')

# ---------------- R: 인용 재선택
sub("        scores = self.sec_bm25.get_scores(tokenize(text))\n",
    "        scores = self.sec2.get_scores(tok2(text)) if NEW_RETRIEVAL else self.sec_bm25.get_scores(tokenize(text))\n")

# ---------------- R: 새 find (v3.5 의 extra_ids 유지)
sub('''    async def find(self, query: str, query_vec: np.ndarray | None = None,
                   first_ids: list[str] | None = None, extra_ids: list[str] | None = None) -> tuple[str, list[str]]:
''', '''    def _maxp(self, chunk_scores: np.ndarray) -> np.ndarray:
        out = np.full(len(self.sec_ids), -1e9, dtype=np.float64)
        np.maximum.at(out, self.cidx, chunk_scores.astype(np.float64))
        return out

    def rank_sections(self, query: str, query_vec: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        """(섹션 점수, 청크 점수). 섹션 48개 전부를 점수 매긴다."""
        t = tok2(query)
        kc = np.asarray(self.kw2.get_scores(t), dtype=np.float64)
        sec = 0.5 * (_mm(np.asarray(self.sec2.get_scores(t), dtype=np.float64)) + _mm(self._maxp(kc)))
        chunk = _mm(kc)
        if query_vec is not None and RET_ALPHA > 0:
            cv = (self.store.vecs @ query_vec).astype(np.float64)
            sec = RET_ALPHA * _mm(self._maxp(cv)) + (1 - RET_ALPHA) * sec
            chunk = RET_ALPHA * _mm(cv) + (1 - RET_ALPHA) * chunk
        return sec, chunk

    async def find(self, query: str, query_vec: np.ndarray | None = None,
                   first_ids: list[str] | None = None, extra_ids: list[str] | None = None,
                   new: bool = False) -> tuple[str, list[str]]:
        if not (new and NEW_RETRIEVAL and query.strip() and self.kw2 is not None):
            return await self._find_old(query, query_vec, first_ids, extra_ids)
        by_section: dict[str, list[dict]] = {}
        for sid in first_ids or []:
            if sid in self.sections:
                by_section.setdefault(sid, [])
        sec, chunk = self.rank_sections(query, query_vec)
        for i in np.argsort(-sec)[:MAX_SECTIONS]:
            sid = self.sec_ids[int(i)]
            top = sorted(self.sec_chunk_idx[sid], key=lambda j: -chunk[j])[:3]
            by_section.setdefault(sid, []).extend(self.store.chunks[j] for j in top)
        if extra_ids:  # v3.5: 비슷한 공개 문항의 근거 섹션을 후보에 더한다 (최대 2개, 청크 3개씩)
            keys = list(by_section)[:MAX_SECTIONS]
            for sid in extra_ids:
                if sid in self.sections and sid not in keys:
                    keys.append(sid)
                    if not by_section.get(sid):
                        top = sorted(self.sec_chunk_idx.get(sid, []), key=lambda j: -chunk[j])[:3]
                        by_section[sid] = [self.store.chunks[j] for j in top]
            return self.render({k: by_section.get(k, []) for k in keys}, max_sections=len(keys))
        return self.render(by_section)

    async def _find_old(self, query: str, query_vec: np.ndarray | None = None,
                        first_ids: list[str] | None = None, extra_ids: list[str] | None = None) -> tuple[str, list[str]]:
''')

sub("        evidence, ids = await app.state.evidence.find(search_query(messages), query_vec=qvec, extra_ids=ex_ids)\n",
    "        evidence, ids = await app.state.evidence.find(search_query(messages), query_vec=qvec, extra_ids=ex_ids, new=True)\n")

# ---------------- B: 예시 골고루 (근거 섹션 힌트는 가장 가까운 예시에서)
sub('''    shots = app.state.examples.nearest(qvec, TUTOR_SHOTS, exclude_last=last,
                                          kind=None if TUTOR_SHOT_KIND == "all" else "tutor") if qvec is not None else []
    ex_ids: list[str] = []
    if TUTOR_SHOT_EVID:
        cnt: dict[str, int] = {}
        for e in shots[:EVID_SHOTS]:
''', '''    near = app.state.examples.nearest(qvec, max(TUTOR_SHOTS, SHOT_POOL) if SHOT_BALANCE else TUTOR_SHOTS,
                                      exclude_last=last,
                                      kind=None if TUTOR_SHOT_KIND == "all" else "tutor") if qvec is not None else []
    shots = balance_shots(near) if SHOT_BALANCE else near[:TUTOR_SHOTS]
    ex_ids: list[str] = []
    if TUTOR_SHOT_EVID:
        cnt: dict[str, int] = {}
        for e in near[:EVID_SHOTS]:
''')

sub("async def run_tutor(messages: list[dict], deadline: float) -> dict:\n", '''def balance_shots(pool: list[dict]) -> list[dict]:
    """(B) 비슷한 순 후보에서 같은 action 은 SHOT_CAP 개까지만 넣는다. 모자라면 남은 후보로 채운다."""
    picked, rest, count = [], [], {}
    for e in pool:
        a = e.get("action")
        if count.get(a, 0) < SHOT_CAP and len(picked) < TUTOR_SHOTS:
            picked.append(e)
            count[a] = count.get(a, 0) + 1
        else:
            rest.append(e)
    return picked + rest[:max(0, TUTOR_SHOTS - len(picked))]


async def run_tutor(messages: list[dict], deadline: float) -> dict:
''')

p.write_text(src, encoding="utf-8", newline="\n")
print("patched BR35")
