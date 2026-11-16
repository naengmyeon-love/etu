"""지식 베이스(kb/) -> chunk -> 임베딩 -> 디스크 색인(index/).

사용:
    python3 index.py --kb kb --out index      # 색인 생성 (KB가 그대로면 건너뜀)
    python3 index.py --kb kb --out index --force

만드는 파일:
    index/chunks.json      chunk 목록 (섹션 ID, 과목, 제목 경로, 본문)
    index/embeddings.npy   정규화된 임베딩 행렬 (chunk 수 x 차원)
    index/manifest.json    KB 해시. KB가 바뀌면 다시 만든다
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np
from openai import OpenAI

MAX_CHARS = 500      # chunk 하나의 최대 길이
OVERLAP = 80         # 길어서 자를 때 겹치는 길이
EMBED_BATCH = 64     # 임베딩 요청 하나에 넣는 chunk 수
CHUNKER_VERSION = 3  # 청킹 방식을 바꾸면 올린다. 올리면 기존 색인을 다시 만든다

HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$")


# ---------------------------------------------------------------- 청킹

def _pack(units: list[str], sep: str) -> list[str]:
    """units를 MAX_CHARS를 넘지 않게 sep로 이어 붙인다.

    unit 하나가 MAX_CHARS보다 길면 그것만 따로 한 조각으로 낸다 (호출한 쪽에서 더 자른다).
    """
    pieces, buf = [], ""
    for unit in units:
        if buf and len(buf) + len(sep) + len(unit) <= MAX_CHARS:
            buf = f"{buf}{sep}{unit}"
        else:
            if buf:
                pieces.append(buf)
            buf = unit
    if buf:
        pieces.append(buf)
    return pieces


def _cut_chars(text: str) -> list[str]:
    """글자 수로 겹쳐서 자른다. 줄 하나가 MAX_CHARS를 넘을 때만 쓰는 마지막 수단."""
    pieces, step = [], MAX_CHARS - OVERLAP
    for i in range(0, len(text), step):
        pieces.append(text[i:i + MAX_CHARS])
        if i + MAX_CHARS >= len(text):
            break
    return pieces


def _split_long(text: str) -> list[str]:
    """MAX_CHARS를 넘는 본문을 문단 -> 줄 -> 글자 순으로 점점 잘게 나눈다.

    표의 행이나 코드 한 줄이 중간에서 끊기지 않도록, 글자 수로 자르기 전에 줄 단위로 먼저 묶는다.
    """
    if len(text) <= MAX_CHARS:
        return [text]

    paras = [p.strip("\n") for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: list[str] = []
    for para in _pack(paras, "\n\n"):
        if len(para) <= MAX_CHARS:
            pieces.append(para)
            continue
        lines = [ln.rstrip() for ln in para.splitlines() if ln.strip()]
        for chunk in _pack(lines, "\n"):
            pieces.extend(_cut_chars(chunk) if len(chunk) > MAX_CHARS else [chunk])
    return pieces


def split_markdown(text: str) -> list[tuple[str, str]]:
    """마크다운을 헤더 기준으로 나눠 (제목 경로, 본문) 목록으로 돌려준다.

    코드 블록(```) 안의 '#'은 헤더로 보지 않는다 (파이썬 주석 보호).
    """
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []   # (헤더 수준, 제목)
    body: list[str] = []
    in_code = False

    def flush():
        content = "\n".join(body).strip()
        if content:
            path = " > ".join(t for _, t in stack)
            for piece in _split_long(content):
                sections.append((path, piece))
        body.clear()

    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
        m = None if in_code else HEADER_RE.match(line)
        if m:
            flush()
            level, title = len(m.group(1)), m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            body.append(line)
    flush()
    return sections


def csv_to_text(path: Path) -> str:
    """표 CSV를 '열이름: 값' 줄로 풀어 쓴다. 행 단위로 검색되기 쉽다."""
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return ""
    header, lines = rows[0], []
    for row in rows[1:]:
        cells = [f"{h}: {v}" for h, v in zip(header, row) if v.strip()]
        if cells:
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def section_csvs(md: Path) -> list[Path]:
    """섹션에 딸린 표 파일. <섹션ID>.csv 와 <섹션ID>-table1.csv 같은 이름을 모두 찾는다."""
    found = [md.with_suffix(".csv")] + sorted(md.parent.glob(f"{md.stem}-*.csv"))
    return [p for p in found if p.exists()]


def load_chunks(kb_dir: str | Path) -> list[dict]:
    """kb/<과목>/<섹션ID>.md (+ 딸린 표 .csv)를 읽어 chunk 목록을 만든다."""
    chunks: list[dict] = []
    for md in sorted(Path(kb_dir).glob("*/*.md")):
        section_id, subject = md.stem, md.parent.name
        text = md.read_text(encoding="utf-8")

        first = HEADER_RE.match(text.lstrip().splitlines()[0]) if text.strip() else None
        doc_title = first.group(2).strip() if first else section_id

        parts = split_markdown(text)
        for csv_path in section_csvs(md):
            table = csv_to_text(csv_path)
            for piece in _split_long(table) if table else []:
                parts.append((f"{doc_title} > 표", piece))

        for path, piece in parts:
            title = path or doc_title
            chunks.append({
                "id": section_id,          # 인용에 쓰는 섹션 ID
                "subject": subject,
                "title": title,
                "text": piece,
            })
    return chunks


def chunk_text_for_search(chunk: dict) -> str:
    """임베딩과 키워드 색인에 넣는 문자열. 제목을 앞에 붙여 맥락을 살린다."""
    return f"{chunk['title']}\n{chunk['text']}"


# ---------------------------------------------------------------- 임베딩

def _client() -> OpenAI:
    return OpenAI(
        base_url=os.environ["EMBEDDING_BASE_URL"],
        api_key=os.environ["EMBEDDING_API_KEY"],
    )


def embed_texts(texts: list[str]) -> np.ndarray:
    """텍스트 목록을 정규화된 벡터 행렬로 바꾼다. 실패하면 세 번까지 다시 시도한다."""
    client, model = _client(), os.environ["EMBEDDING_MODEL"]
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i:i + EMBED_BATCH]
        for attempt in range(3):
            try:
                resp = client.embeddings.create(model=model, input=batch)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
        vectors.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
    return normalize(np.asarray(vectors, dtype=np.float32))


def normalize(mat: np.ndarray) -> np.ndarray:
    """행마다 길이를 1로 맞춘다. 이후 내적이 곧 코사인 유사도가 된다."""
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    return mat / np.clip(norms, 1e-12, None)


# ---------------------------------------------------------------- 색인

def kb_hash(kb_dir: str | Path) -> str:
    """KB 파일 내용과 청킹 설정의 해시. 하나라도 바뀌면 색인을 다시 만든다."""
    settings = f"{CHUNKER_VERSION}:{MAX_CHARS}:{OVERLAP}:{os.environ.get('EMBEDDING_MODEL', '')}"
    h = hashlib.sha256(settings.encode())
    for p in sorted(Path(kb_dir).glob("*/*")):
        if p.suffix in (".md", ".csv"):
            h.update(p.as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def index_is_fresh(kb_dir: str | Path, out_dir: str | Path) -> bool:
    out = Path(out_dir)
    files = [out / "chunks.json", out / "embeddings.npy", out / "manifest.json"]
    if not all(f.exists() for f in files):
        return False
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    return manifest.get("kb_hash") == kb_hash(kb_dir)


def build_index(kb_dir: str | Path = "kb", out_dir: str | Path = "index", force: bool = False) -> int:
    """색인을 만들고 chunk 수를 돌려준다. KB가 그대로면 아무것도 하지 않는다."""
    out = Path(out_dir)
    if not force and index_is_fresh(kb_dir, out):
        return len(json.loads((out / "chunks.json").read_text(encoding="utf-8")))

    chunks = load_chunks(kb_dir)
    if not chunks:
        raise RuntimeError(f"{kb_dir} 에서 chunk를 하나도 만들지 못했습니다. 경로를 확인하세요.")
    vecs = embed_texts([chunk_text_for_search(c) for c in chunks])

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", vecs)
    (out / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    (out / "manifest.json").write_text(
        json.dumps({"kb_hash": kb_hash(kb_dir), "chunks": len(chunks), "dim": int(vecs.shape[1])}),
        encoding="utf-8",
    )
    return len(chunks)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="KB 색인 생성")
    ap.add_argument("--kb", default="kb")
    ap.add_argument("--out", default="index")
    ap.add_argument("--force", action="store_true", help="KB가 그대로여도 다시 만든다")
    args = ap.parse_args()

    started = time.time()
    n = build_index(args.kb, args.out, force=args.force)
    print(f"chunk {n}개 색인 완료 ({time.time() - started:.1f}초) -> {args.out}/")
