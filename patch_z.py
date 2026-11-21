"""Z안 (X+Y 위에 적용): /guard 에서 후보 답변이 없는 요청만, 입력 분류와 동시에 '자료에 있는 질문인지' 확인한다.
입력이 normal 로 분류되고 자료에 없다고 나오면 범주를 out_of_material 로 답한다 (판정은 allow 그대로).
후보 답변이 없는 요청은 LLM 호출이 1회뿐이라 여유가 있다. 사용: python3 patch_z.py server.py   끄기: GUARD_SUB_NOCAND=0
"""
import sys
from pathlib import Path

p = Path(sys.argv[1])
src = p.read_text(encoding="utf-8")
assert "GUARD_SUB_NOCAND" not in src and "MENTION_RULE" in src, "apply after patch_y"
old = "                if GUARD_SUBTYPE and GUARD_SUB_MATERIAL else None)\n"
assert src.count(old) == 1
src = src.replace(old, "                if GUARD_SUBTYPE and (GUARD_SUB_MATERIAL or (GUARD_SUB_NOCAND and candidate is None)) else None)\n")
old = 'GUARD_SPECULATIVE = os.environ.get("GUARD_SPECULATIVE"'
assert src.count(old) == 1
src = src.replace(old, 'GUARD_SUB_NOCAND = os.environ.get("GUARD_SUB_NOCAND", "1") == "1"      # (Z) 후보 답변이 없는 요청만 자료 확인\n' + old)
p.write_text(src, encoding="utf-8", newline="\n")
print("patched Z")
