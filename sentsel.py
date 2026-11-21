"""(S안) 답 뒤에 덧붙일 원문 문장을 고르는 학습한 선택기.

서버 시작 때 한 번 만든다. 요청 사이에 바뀌는 상태는 없다.
- 문장 벡터: 배포 KB 문장을 운영진 임베딩 API 로 만든 것 (INDEX_DIR/sentvec.npz 에 저장해 다시 씀)
- 학습 데이터: 공개 세트(PUBLIC_DIR/tutor.jsonl)의 gold_facts 가 든 문장을 정답으로 본 로지스틱 회귀
외부 데이터·미리 학습된 모델 파일은 쓰지 않는다. 질문→답 표가 아니라 문장 특징 17개의 가중치만 배운다.

점검(개발용, 집계 숫자만): python3 sentsel.py <서버 모듈 이름>   → 5겹 교차 검증 적중률
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
_TOK_RE = re.compile(r"[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)*|\d+(?:\.\d+)?|[가-힣]+")
_JOSA = ("으로", "에서", "에게", "까지", "부터", "이랑", "하고", "보다", "처럼",
         "은", "는", "이", "가", "을", "를", "에", "의", "도", "로", "와", "과", "만")
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_CODE_RE = re.compile(r"[a-z_][a-z0-9_]{2,}")
_DECL_RE = re.compile(r"(이다|입니다|이에요|예요|한다|합니다|해요|돼요|된다)[.]?$")
_CODEISH_RE = re.compile(r"[a-z_]{3,}\(|=")
HIST_CHARS = 1500


def _key_tokens(text: str) -> set:
    out = set()
    for t in _TOK_RE.findall(text.lower()):
        if "가" <= t[0] <= "힣":
            for j in _JOSA:
                if len(t) > len(j) + 1 and t.endswith(j):
                    t = t[: -len(j)]
                    break
            if len(t) >= 2:
                out.add(t)
        else:
            out.add(t)
    return out


def _mm(v: np.ndarray) -> np.ndarray:
    lo, hi = float(v.min()), float(v.max())
    return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)


class SentSelector:
    def __init__(self, sent, vecs, tok2, squash):
        self.sent, self.vecs, self.tok2, self.squash = sent, vecs, tok2, squash
        self.w = None  # (가중치, 절편, 평균, 표준편차)
        self.n_train = 0

    # ------------------------------------------------------------ 특징
    def feats(self, sid: str, last: str, hist: str, qvec) -> np.ndarray:
        sents, bm = self.sent[sid]
        n = len(sents)
        a = np.asarray(bm.get_scores(self.tok2(last)), dtype=np.float64)
        b = np.asarray(bm.get_scores(self.tok2(hist[-HIST_CHARS:])), dtype=np.float64)
        ma, mb = float(a.max()) or 1.0, float(b.max()) or 1.0
        ra = np.argsort(np.argsort(-a, kind="stable"), kind="stable")
        c = (self.vecs[sid] @ np.asarray(qvec, dtype=np.float32)).astype(np.float64)
        rc = np.argsort(np.argsort(-c, kind="stable"), kind="stable")
        cm = _mm(c)
        qnum = set(_NUM_RE.findall(last))
        qcode = set(_CODE_RE.findall(last.lower()))
        X = np.zeros((n, 17), dtype=np.float64)
        for i, s in enumerate(sents):
            sl = s.lower()
            nums = set(_NUM_RE.findall(s))
            X[i] = [
                a[i] / ma, b[i] / mb, 1.0 / (1 + ra[i]), float(ra[i] == 0),
                float(bool(nums)), float("`" in s or bool(_CODEISH_RE.search(sl))),
                float(bool(nums & qnum)), float(any(x in sl for x in qcode)),
                i / max(1, n - 1), float(i < 3), min(len(s), 200) / 200.0,
                float(s.count(",") >= 2 and len(s) < 80),
                float(bool(_DECL_RE.search(s))),
                max(a[max(0, i - 1)], a[min(n - 1, i + 1)]) / ma,
                cm[i], 1.0 / (1 + rc[i]), float(c.max() - c[i]),
            ]
        return X

    def labels(self, sid: str, last: str, facts: list) -> np.ndarray:
        """문장별 정답 사실 낱말 포함 비율 (질문에 이미 있는 낱말은 뺀다)."""
        sents = self.sent[sid][0]
        fk = set()
        for f in facts:
            fk |= _key_tokens(f)
        qflat = self.squash(last)
        fk = {k for k in fk if self.squash(k) not in qflat} or fk
        keys = [self.squash(k) for k in fk]
        return np.array([sum(k in self.squash(s) for k in keys) / max(1, len(keys)) for s in sents])

    # ------------------------------------------------------------ 학습
    @staticmethod
    def _train(XY, epochs=300, lr=0.5, l2=1e-3):
        X = np.vstack([x for x, _ in XY])
        y = np.concatenate([t for _, t in XY])
        mu, sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - mu) / sd
        w = np.zeros(Z.shape[1])
        b0 = 0.0
        pos = float(y.mean())
        if not 0 < pos < 1:
            raise ValueError("labels")
        cw = np.where(y > 0, 0.5 / pos, 0.5 / (1 - pos))
        for _ in range(epochs):
            p = 1 / (1 + np.exp(-(Z @ w + b0)))
            g = (p - y) * cw
            w -= lr * (Z.T @ g / len(y) + l2 * w)
            b0 -= lr * g.mean()
        return w, b0, mu, sd

    def score(self, X: np.ndarray, model=None) -> np.ndarray:
        w, b0, mu, sd = model or self.w
        return ((X - mu) / sd) @ w + b0

    def training_rows(self, public_dir: str, context_text, embed_texts) -> list:
        items = []
        for line in (Path(public_dir) / "tutor.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            lab = r.get("labels") or {}
            if lab.get("action") not in ("answer", "scaffold") or not lab.get("gold_facts") or not lab.get("evidence"):
                continue
            sid = lab["evidence"][0]
            if sid not in self.sent or sid not in self.vecs:
                continue
            msgs = [m for m in r["messages"] if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)]
            users = [m["content"] for m in msgs if m["role"] == "user"]
            if not users:
                continue
            items.append(dict(sid=sid, last=users[-1], hist=" ".join(users), ctx=context_text(msgs), facts=lab["gold_facts"]))
        if not items:
            return []
        Q = np.asarray(embed_texts([it["ctx"][:2000] for it in items]), dtype=np.float32)
        Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)
        rows = []
        for it, q in zip(items, Q):
            cov = self.labels(it["sid"], it["last"], it["facts"])
            top = float(cov.max())
            if top <= 0:
                continue
            y = (cov >= max(0.34, 0.8 * top)).astype(np.float64)
            rows.append((self.feats(it["sid"], it["last"], it["hist"], q), y, cov))
        return rows

    # ------------------------------------------------------------ 만들기
    @classmethod
    def build(cls, evidence, public_dir: str, index_dir: str, tok2, squash, context_text, embed_texts, keep_rows: bool = False):
        sent = getattr(evidence, "sent", None) or {}
        if not sent:
            raise ValueError("sentences")
        sids = sorted(sent)
        digest = hashlib.sha256(json.dumps({s: sent[s][0] for s in sids}, ensure_ascii=False).encode("utf-8")).hexdigest()
        out = Path(index_dir)
        vf, mf = out / "sentvec.npz", out / "sentvec.json"
        vecs = None
        if vf.exists() and mf.exists():
            try:
                if json.loads(mf.read_text(encoding="utf-8")).get("hash") == digest:
                    z = np.load(vf)
                    vecs = {s: z[s] for s in sids}
            except Exception:
                vecs = None
        if vecs is None:
            flat = [x for s in sids for x in sent[s][0]]
            V = np.asarray(embed_texts(flat), dtype=np.float32)
            V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
            vecs, p = {}, 0
            for s in sids:
                n = len(sent[s][0])
                vecs[s] = V[p:p + n]
                p += n
            out.mkdir(parents=True, exist_ok=True)
            np.savez(vf, **vecs)
            mf.write_text(json.dumps({"hash": digest, "n": len(flat)}), encoding="utf-8")
        self = cls(sent, vecs, tok2, squash)
        rows = self.training_rows(public_dir, context_text, embed_texts)
        if len(rows) < 20:
            raise ValueError("too few training items")
        self.w = self._train([(x, y) for x, y, _ in rows])
        self.n_train = len(rows)
        if keep_rows:  # 점검용. 서버에서는 남기지 않는다
            self._rows = rows
        return self

    # ------------------------------------------------------------ 고르기
    def pick(self, sid: str, last: str, hist: str, qvec, answer: str, k: int) -> list:
        if self.w is None or sid not in self.sent or sid not in self.vecs or k <= 0:
            return []
        sents = self.sent[sid][0]
        s = self.score(self.feats(sid, last, hist, qvec))
        flat = self.squash(answer)
        out = []
        for i in np.argsort(-s, kind="stable"):
            if len(out) >= k:
                break
            if self.squash(sents[int(i)]) in flat:
                continue
            out.append(sents[int(i)])
        return out


if __name__ == "__main__":  # 개발용 점검: 5겹 교차 검증 적중률 (집계 숫자만 찍는다)
    import sys
    mod = __import__(sys.argv[1] if len(sys.argv) > 1 else "server")
    from index import embed_texts
    ev = mod._evidence_with_sentences(mod.VectorStore.load_or_build(mod.KB_DIR, mod.INDEX_DIR))
    sel = SentSelector.build(ev, mod.PUBLIC_DIR, mod.INDEX_DIR, mod.tok2, mod.squash, mod.context_text, embed_texts, keep_rows=True)
    rows = sel._rows
    order = list(range(len(rows)))
    random.Random(3).shuffle(order)
    pred = {}
    for f in range(5):
        te = set(order[f::5])
        m = sel._train([(rows[i][0], rows[i][1]) for i in order if i not in te])
        for i in te:
            pred[i] = sel.score(rows[i][0], m)
    print("문항", len(rows), "문장", sum(len(v[0]) for v in sel.sent.values()))
    for k in (1, 2, 3):
        hb = hl = cb = cl = 0.0
        for i, (X, y, cov) in enumerate(rows):
            oracle = int(np.argmax(cov))
            ib = list(np.argsort(-X[:, 0], kind="stable")[:k])
            il = list(np.argsort(-pred[i], kind="stable")[:k])
            hb += oracle in ib
            hl += oracle in il
            cb += float(np.mean([cov[j] > 0 for j in ib]))
            cl += float(np.mean([cov[j] > 0 for j in il]))
        n = len(rows)
        print(f"k={k} 적중: 지금(BM25) {hb/n:.3f} -> 선택기 {hl/n:.3f} | 쓸모문장 비율 {cb/n:.2f} -> {cl/n:.2f}")
    print("end")
