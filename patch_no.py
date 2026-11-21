"""NAME_OOM안: 질문이 묻는 '이름 붙은 규칙·기준'이 KB 에 글자 그대로 없으면 abstain. (NAME_OOM=0 이면 끔)
python3 patch_no.py <고칠 서버 파일>"""
import sys

p = sys.argv[1]
s = open(p, encoding="utf-8").read()
assert "NAME_OOM" not in s, "이미 적용됨"
a = "async def run_tutor(messages: list[dict], deadline: float) -> dict:"
assert s.count(a) == 1
s = s.replace(a, r'''NAME_OOM = os.environ.get("NAME_OOM", "1") == "1"  # (NO) 이름 붙은 규칙·기준이 KB 에 없으면 abstain
_NO_HEAD = r"(규칙|기준|분류법|등급|지수|규정|기능)"
_NO_ASK = r"(?:은|는|이|가|을|를)?\s*(?:이\s*)?(?:뭐|머|무엇|어떤|설명|알려|얼마|몇)"
NAME_OOM_RE = re.compile(r"((?:[가-힣A-Za-z0-9_.‘’'\"]+\s*){1,2}?)\s?" + _NO_HEAD + _NO_ASK)
_NO_STOP = {"이", "그", "그럼", "말고", "과정", "과정의", "이과정의", "거절하지", "걍", "그냥", "좀", "다시"}
_NO_ADN = re.compile(r"(는|은|할|한|던|운|된|될)$")
_NO_JOSA = re.compile(r"(의|에서|에|으로|로|과|와|도|를|을)$")


def build_kb_flat() -> str:
    """KB 전체(본문 .md + 표 .csv)를 공백·부호 없이 이어 붙인 글. 서버 시작 때 한 번 만든다."""
    from pathlib import Path
    parts = []
    for path in sorted(Path(KB_DIR).glob("*/*")):
        if path.suffix in (".md", ".csv"):
            parts.append(path.read_text(encoding="utf-8"))
    return squash("".join(parts)).lower()


def missing_name(text: str, kb_flat: str) -> str | None:
    """질문이 묻는 '이름 붙은 규칙·기준'이 KB 에 글자 그대로 없으면 그 이름을 돌려준다."""
    if not kb_flat:
        return None
    for m in NAME_OOM_RE.finditer(text):
        words = [w.strip("‘’'\"") for w in m.group(1).split() if w.strip("‘’'\"")]
        if not words or words[-1] in _NO_STOP or _NO_ADN.search(words[-1]):
            continue
        last = _NO_JOSA.sub("", words[-1])
        two = len(words) >= 2 and words[-2] not in _NO_STOP and not _NO_ADN.search(words[-2])
        mod = (_NO_JOSA.sub("", words[-2]) + last) if two else last
        if len(mod) >= 2 and squash(mod + m.group(2)).lower() not in kb_flat:
            return " ".join(words[-2:] if two else words[-1:]) + " " + m.group(2)
    return None


''' + a)
b = "    if mat_task is not None and not mat_task.done():"
assert s.count(b) == 1
s = s.replace(b, '''    if (NAME_OOM and result["action"] == "answer" and not suspect
            and category in ("normal", "pii_disclosure", "out_of_material", "")):
        try:
            name = missing_name(last, getattr(app.state, "kb_flat", ""))
        except Exception as e:
            log.warning("이름 규칙 실패: %s", type(e).__name__)
            name = None
        if name:
            log.warning("tutor 이름 규칙: 자료에 없는 이름으로 보고 abstain")
            result["action"] = "abstain"
            result["answer"] = f"자료에는 '{name}'에 관한 내용이 나와 있지 않아요."
''' + b)
c = "    app.state.examples = ExampleIndex.load_or_build"
assert s.count(c) == 1
s = s.replace(c, '''    try:
        app.state.kb_flat = build_kb_flat() if NAME_OOM else ""
    except Exception as e:
        log.warning("KB 글 준비 실패: %s", type(e).__name__)
        app.state.kb_flat = ""
''' + c)
open(p, "w", encoding="utf-8").write(s)
print("NAME_OOM안 적용:", p)
