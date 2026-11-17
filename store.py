"""색인을 메모리에 올리고 검색한다 (벡터 + 키워드 하이브리드).

서버에서 쓰는 법:
    store = VectorStore.load_or_build("kb", "index")   # 서버를 띄울 때 한 번
    hits = await store.search("꿈틀이 실습 힌트 주세요", top_n=10)
    # hits[i] = {"id": "PY-01-02", "title": ..., "text": ..., "score": ...}

v2: search() 에 query_vec 을 넘기면 질문 임베딩을 다시 부르지 않는다.
    서버가 질문 임베딩 하나를 KB 검색과 공개 세트 예시 검색에 같이 쓰기 위한 것이다.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI
from rank_bm25 import BM25Okapi

from index import build_index, chunk_text_for_search, normalize

TOKEN_RE = re.compile(r"[a-z0-9_]+|[가-힣]+")
RRF_K = 60           # 순위 결합 상수. 클수록 상위 순위의 영향이 줄어든다
KW_MIN_RATIO = 0.25  # BM25 점수가 최고점의 이 비율보다 낮으면 키워드 쪽 점수를 주지 않는다
EMBED_TIMEOUT = 3.0  # 질문 임베딩 시도 한 번의 제한 시간(초)
EMBED_BUDGET = 6.0   # 재시도까지 포함한 질문 임베딩 전체 제한 시간(초). 요청 제한 30초 안에 LLM 몫을 남긴다

log = logging.getLogger(__name__)


def tokenize(text: str) -> list[str]:
    """키워드 검색용 토큰.

    영문·숫자는 단어 그대로, 한글은 단어와 글자 2개씩 묶음을 함께 낸다.
    형태소 분석기 없이도 '실습에서'가 '실습'과 맞도록 하기 위한 것이다.
    """
    tokens: list[str] = []
    for tok in TOKEN_RE.findall(text.lower()):
        tokens.append(tok)
        if "가" <= tok[0] <= "힣" and len(tok) > 2:
            tokens.extend(tok[i:i + 2] for i in range(len(tok) - 1))
    return tokens


class VectorStore:
    def __init__(self, index_dir: str | Path = "index"):
        d = Path(index_dir)
        self.chunks: list[dict] = json.loads((d / "chunks.json").read_text(encoding="utf-8"))
        self.vecs: np.ndarray = np.load(d / "embeddings.npy")
        if len(self.chunks) != self.vecs.shape[0]:
            raise RuntimeError("chunks.json 과 embeddings.npy 의 개수가 다릅니다. 색인을 다시 만드세요.")

        self.bm25 = BM25Okapi([tokenize(chunk_text_for_search(c)) for c in self.chunks])
        self._client = AsyncOpenAI(
            base_url=os.environ["EMBEDDING_BASE_URL"],
            api_key=os.environ["EMBEDDING_API_KEY"],
            timeout=EMBED_TIMEOUT,  # 기본값은 10분이라 API가 멈추면 요청이 그대로 매달린다
            max_retries=1,          # 일시적인 오류는 SDK가 한 번 다시 시도한다
        )
        self._model = os.environ["EMBEDDING_MODEL"]

    @classmethod
    def load_or_build(cls, kb_dir: str | Path = "kb", index_dir: str | Path = "index") -> "VectorStore":
        """색인이 없거나 KB가 바뀌었으면 만들고, 아니면 그대로 읽는다."""
        build_index(kb_dir, index_dir)
        return cls(index_dir)

    # ------------------------------------------------------------ 검색

    async def embed_query(self, query: str) -> np.ndarray:
        # 시도별 제한과 재시도가 겹쳐도 EMBED_BUDGET을 넘지 않게 전체를 한 번 더 묶는다
        resp = await asyncio.wait_for(
            self._client.embeddings.create(model=self._model, input=[query]),
            timeout=EMBED_BUDGET,
        )
        return normalize(np.asarray(resp.data[0].embedding, dtype=np.float32))

    def vector_scores(self, query_vec: np.ndarray) -> np.ndarray:
        """모든 chunk와의 코사인 유사도."""
        return self.vecs @ query_vec

    def keyword_scores(self, query: str) -> np.ndarray:
        """모든 chunk의 BM25 점수."""
        return np.asarray(self.bm25.get_scores(tokenize(query)), dtype=np.float32)

    @staticmethod
    def keyword_matched(kw: np.ndarray) -> np.ndarray:
        """키워드가 의미 있게 겹친 chunk만 True.

        겹친 단어가 없거나, 조사·어미 묶음("에서", "니다")만 겹쳐 최고점에 비해 아주 낮은 chunk는 뺀다.
        """
        return (kw > 0) & (kw >= KW_MIN_RATIO * float(kw.max()))

    async def search(
        self,
        query: str,
        top_n: int = 10,
        mode: str = "hybrid",
        min_vector_score: float | None = None,
        query_vec: np.ndarray | None = None,
    ) -> list[dict]:
        """관련 chunk를 점수 순으로 top_n개 돌려준다.

        mode: "hybrid"(기본) | "vector" | "keyword"
        hybrid는 두 검색의 순위를 RRF로 합친다. 점수 단위가 달라도 합칠 수 있다.
        질문 임베딩이 실패하면 hybrid는 키워드 검색만으로 답한다.
        키워드 검색만 쓸 때는 단어가 의미 있게 겹친 chunk만 돌려준다 (없으면 빈 목록).
        빈 질문은 API를 부르지 않고 빈 목록을 돌려준다.
        query_vec 을 주면 임베딩 API를 부르지 않고 그 벡터를 쓴다.

        min_vector_score: 코사인 유사도가 이 값보다 낮은 chunk는 버린다.
        """
        if not query.strip():
            return []
        n = len(self.chunks)
        top_n = min(top_n, n)
        vec = kw = None
        if mode in ("hybrid", "vector"):
            try:
                qv = query_vec if query_vec is not None else await self.embed_query(query)
                vec = self.vector_scores(qv)
            except Exception as e:
                if mode == "vector":
                    raise
                # 임베딩 API가 죽어도 답은 나가야 한다. 키워드 검색만으로 내려간다
                # 오류 메시지에 요청 내용이 섞일 수 있어 오류 종류만 남긴다 (본문 로그 금지)
                log.warning("질문 임베딩 실패, 키워드 검색만 사용: %s", type(e).__name__)
                mode = "keyword"
        if mode in ("hybrid", "keyword"):
            kw = self.keyword_scores(query)

        if mode == "vector":
            final = vec
        elif mode == "keyword":
            # 점수 0인 chunk로 top_n을 채우지 않는다. 겹친 것이 없으면 결과도 없다
            final = np.where(self.keyword_matched(kw), kw, -np.inf).astype(np.float32)
        else:
            final = np.zeros(n, dtype=np.float32)
            for scores in (vec, kw):
                order = np.argsort(-scores)
                ranks = np.empty(n, dtype=np.int64)
                ranks[order] = np.arange(n)
                contrib = 1.0 / (RRF_K + ranks + 1)
                if scores is kw:
                    contrib[~self.keyword_matched(kw)] = 0.0
                final += contrib

        hits = []
        for i in np.argsort(-final)[:top_n]:
            if not np.isfinite(final[i]):
                continue
            if min_vector_score is not None and vec is not None and vec[i] < min_vector_score:
                continue
            hit = dict(self.chunks[i])
            hit["score"] = float(final[i])
            if vec is not None:
                hit["vector_score"] = float(vec[i])
            if kw is not None:
                hit["keyword_score"] = float(kw[i])
            hits.append(hit)
        return hits


def section_ids(hits: list[dict]) -> list[str]:
    """검색 결과에서 섹션 ID를 순서를 지키며 중복 없이 뽑는다 (citations 후보)."""
    seen: dict[str, None] = {}
    for h in hits:
        seen.setdefault(h["id"], None)
    return list(seen)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="검색 확인용")
    ap.add_argument("query")
    ap.add_argument("--kb", default="kb")
    ap.add_argument("--index", default="index")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "vector", "keyword"])
    ap.add_argument("--min-vec", type=float, default=None, help="이 코사인 유사도보다 낮은 chunk는 버린다")
    args = ap.parse_args()

    async def main():
        store = VectorStore.load_or_build(args.kb, args.index)
        hits = await store.search(args.query, top_n=args.top, mode=args.mode, min_vector_score=args.min_vec)
        if not hits:
            print("관련 chunk 없음 (KB 밖 질문으로 판단)")
        for h in hits:
            preview = h["text"][:70].replace("\n", " ")
            cos = f"cos {h['vector_score']:.3f}" if "vector_score" in h else "cos -  "
            print(f"{h['score']:.4f} {cos} {h['id']:<10} {h['title'][:30]} | {preview}")

    asyncio.run(main())
