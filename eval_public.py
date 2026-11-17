"""공개 세트로 항목별 진단. harness/score_local.py 와 같은 채점 코드를 쓰되, 어디서 점수가 빠지는지 보여 준다.

    python3 eval_public.py --url http://localhost:8000 --token my-token            # 판정 모델 포함 (.env 읽은 터미널)
    python3 eval_public.py --url http://localhost:8000 --token my-token --no-judge # action·인용·가드만 (빠름)
    python3 eval_public.py ... --only tutor | --only guard

출력: 화면 + eval_out.txt (집계와 공개 문항 id 만. 요청·답변 본문은 기록하지 않는다)
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib import request as urlreq

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "grader"))
sys.path.insert(0, str(ROOT / "harness"))
import scoring  # noqa: E402
from judge import make_judge  # noqa: E402
from score_local import as_scoring_input, load_kb_dir  # noqa: E402

REFUSALS = ("refuse", "redirect")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _no_judge(kind, payload):
    raise RuntimeError("judge skipped")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--token", default=os.environ.get("SERVER_TOKEN", ""))
    ap.add_argument("--public", default="public")
    ap.add_argument("--kb", default="kb")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=30)
    ap.add_argument("--only", choices=["tutor", "guard"], default=None)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--out", default="eval_out.txt")
    a = ap.parse_args()

    endpoints = [a.only] if a.only else ["tutor", "guard"]
    public = [p for e in endpoints for p in read_jsonl(Path(a.public) / f"{e}.jsonl")]
    kb = load_kb_dir(a.kb)
    env = {k: os.environ.get(f"OPENAI_{k.upper()}") for k in ("base_url", "model", "api_key")}
    judge = None if a.no_judge or not all(env.values()) else make_judge(**env)

    def call(endpoint: str, payload: dict):
        req = urlreq.Request(
            f"{a.url.rstrip('/')}/{endpoint}", data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST", headers={"Content-Type": "application/json", "Authorization": f"Bearer {a.token}"},
        )
        with urlreq.urlopen(req, timeout=a.timeout) as resp:
            return json.loads(resp.read(256 * 1024 + 1))

    def one(p: dict):
        item, v = as_scoring_input(p)
        payload = {"spec_version": "1.0", "session_id": "s_" + secrets.token_hex(8), "messages": v["messages"]}
        if v["candidate_answer"]:
            payload["candidate_answer"] = v["candidate_answer"]
        t0 = time.monotonic()
        try:
            resp = call(p["endpoint"], payload)
            err = None
        except Exception as e:
            resp, err = None, type(e).__name__
        ms = (time.monotonic() - t0) * 1000
        if p["endpoint"] == "tutor":
            r = scoring.score_tutor(item, v, resp, judge or _no_judge, kb)
            if judge is None:
                r["judge_error"] = False
            pts = scoring.tutor_points(r)
            pred = (resp or {}).get("action") if isinstance(resp, dict) else None
            cites = (resp or {}).get("citations") if isinstance(resp, dict) else None
            return p, r, pts, ms, err, {"pred_action": pred, "n_cites": len(cites) if isinstance(cites, list) else None}
        r = scoring.score_guard(item, v, resp)
        pts = scoring.guard_points(r)
        inp = (resp or {}).get("input") if isinstance(resp, dict) else None
        return p, r, pts, ms, err, {"pred_cat": (inp or {}).get("category"), "pred_dec": (inp or {}).get("decision")}

    started = time.monotonic()
    with ThreadPoolExecutor(a.workers) as ex:
        results = list(ex.map(one, public))
    wall = time.monotonic() - started

    lines: list[str] = []
    say = lines.append
    lat = sorted(x[3] for x in results)
    say(f"요청 {len(results)}개, 벽시계 {wall:.1f}s, 지연 p50 {lat[len(lat)//2]:.0f}ms p95 {lat[max(0, int(len(lat)*0.95)-1)]:.0f}ms max {lat[-1]:.0f}ms")
    errs = Counter(x[4] for x in results if x[4])
    if errs:
        say(f"호출 오류: {dict(errs)}")

    tutor = [(p, r, pts, extra) for p, r, pts, ms, err, extra in results if p["endpoint"] == "tutor"]
    guard = [(p, r, pts, extra) for p, r, pts, ms, err, extra in results if p["endpoint"] == "guard"]

    if tutor:
        agg = scoring.aggregate([r for _, r, _, _ in tutor], [], {"tutor": 1, "guard": 0})
        say(f"\n===== 튜터 {agg['tutor']} (형식 오류 {agg['invalid']}, 판정 오류 {agg['judge_errors']}, 판정 {'없음' if judge is None else '있음'})")
        conf = Counter((p["labels"]["action"], extra["pred_action"]) for p, _, _, extra in tutor)
        golds = sorted({g for g, _ in conf})
        preds = sorted({str(x) for _, x in conf})
        say("action 혼동 행렬 (행=정답, 열=예측):")
        say("  " + f"{'':<10}" + "".join(f"{x[:8]:>9}" for x in preds))
        for g in golds:
            say("  " + f"{g:<10}" + "".join(f"{conf[(g, x if x != 'None' else None)]:>9}" for x in preds))
        acc = sum(r["action_correct"] for _, r, _, _ in tutor) / len(tutor)
        over = sum(r["over_refusal"] for _, r, _, _ in tutor)
        say(f"action 정확도 {acc:.3f}, 과잉 거절 {over}")
        cp = [r["citation_p"] for _, r, _, _ in tutor if r["citation_p"] is not None]
        cr = [r["citation_r"] for _, r, _, _ in tutor if r["citation_r"] is not None]
        if cp:
            say(f"인용 정밀도 평균 {sum(cp)/len(cp):.3f}, 재현율 평균 {sum(cr)/len(cr):.3f} (n={len(cp)})")
        nc = Counter(extra["n_cites"] for _, _, _, extra in tutor)
        say(f"인용 개수 분포 {dict(sorted(nc.items(), key=lambda kv: str(kv[0])))}")
        facts = [r["facts"] for _, r, _, _ in tutor if r["facts"] is not None]
        faith = [r["faithful"] for _, r, _, _ in tutor if r["faithful"] is not None]
        comp = [r["complied"] for _, r, _, _ in tutor if r["complied"] is not None]
        if facts:
            say(f"정답 사실 평균 {sum(facts)/len(facts):.3f} (n={len(facts)})")
        if faith:
            say(f"근거 충실 비율 {sum(faith)/len(faith):.3f} (n={len(faith)})")
        if comp:
            say(f"부적절 요청에 응함(0점) {sum(comp)}/{len(comp)}")
        per = defaultdict(list)
        for p, r, pts, _ in tutor:
            if not r.get("judge_error"):
                per[(p["labels"]["category"], p["labels"]["action"])].append(pts)
        say("범주|정답action 별 평균 점수:")
        for k in sorted(per):
            v = per[k]
            say(f"  {k[0]:<20} {k[1]:<9} n={len(v):>3}  {100*sum(v)/len(v):6.1f}")
        low = sorted(((pts, p["id"], p["labels"]["category"], p["labels"]["action"], extra["pred_action"]) for p, r, pts, extra in tutor))[:25]
        say("낮은 문항 (점수, id, 범주, 정답action, 예측action):")
        for row in low:
            say(f"  {100*row[0]:5.1f} {row[1]} {row[2]} {row[3]} -> {row[4]}")

    if guard:
        g = [r for _, r, _, _ in guard]
        say(f"\n===== 가드 {scoring.guard_score(g)} (형식 오류 {sum(not r['valid'] for r in g)})")
        say(f"block F1 {scoring.block_f1(g):.3f}  (정상 차단 {sum(r['false_block'] for r in g)}, 차단 누락 {sum(r['missed_block'] for r in g)})")
        cats = [r["category_correct"] for r in g if r["category_scored"]]
        say(f"범주 정확도(normal 제외) {sum(cats)/len(cats):.3f} (n={len(cats)})")
        conf = Counter((p["labels"]["category"], extra["pred_cat"]) for p, _, _, extra in guard)
        golds = sorted({x for x, _ in conf})
        preds = sorted({str(x) for _, x in conf})
        say("범주 혼동 행렬 (행=정답, 열=예측):")
        say("  " + f"{'':<20}" + "".join(f"{x[:9]:>10}" for x in preds))
        for gg in golds:
            say("  " + f"{gg:<20}" + "".join(f"{conf[(gg, x if x != 'None' else None)]:>10}" for x in preds))
        outs = [r for r in g if r["has_output"] and r["valid"]]
        if outs:
            say(f"후보 답변: grounded 정확도 {sum(r['output_correct'] for r in outs)/len(outs):.3f}, "
                f"구간 F1 평균 {sum(r['span_f1'] for r in outs)/len(outs):.3f}, safe 정확도 {sum(r['safe_correct'] for r in outs)/len(outs):.3f} (n={len(outs)})")
            by = defaultdict(list)
            for p, r, _, _ in guard:
                if r["has_output"] and r["valid"]:
                    lab = p["labels"]
                    key = "unsafe" if not lab["safe"] else ("grounded" if lab["grounded"] else "ungrounded")
                    by[key].append((r["output_correct"], r["span_f1"], r["safe_correct"]))
            for k, v in sorted(by.items()):
                say(f"  정답 {k:<10} n={len(v):>3} grounded {sum(x[0] for x in v)/len(v):.2f} spanF1 {sum(x[1] for x in v)/len(v):.2f} safe {sum(x[2] for x in v)/len(v):.2f}")
        low = sorted(((pts, p["id"], p["labels"]["category"], p["labels"]["decision"], extra["pred_cat"]) for p, r, pts, extra in guard))[:20]
        say("낮은 문항 (점수, id, 정답범주, 정답판정, 예측범주):")
        for row in low:
            say(f"  {100*row[0]:5.1f} {row[1]} {row[2]} {row[3]} -> {row[4]}")

    if tutor and guard:
        agg = scoring.aggregate([r for _, r, _, _ in tutor], [r for _, r, _, _ in guard], {"tutor": 0.5, "guard": 0.5})
        say(f"\n총점 {agg['total']} (튜터 {agg['tutor']}, 가드 {agg['guard']})")

    text = "\n".join(lines)
    print(text)
    Path(a.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
