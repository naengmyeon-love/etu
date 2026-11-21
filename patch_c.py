"""C안 (P안 위에 적용): 서버 안 동시 LLM 호출 제한 기본값을 64 에서 256 으로 올린다.
부하가 걸리면 이 제한에 줄이 생겨 호출이 시간 초과됐다 (부하 점검에서 확인). 사용: python3 patch_c.py server.py"""
import sys
from pathlib import Path
p = Path(sys.argv[1])
src = p.read_text(encoding="utf-8")
old = 'LLM_CONCURRENCY = int(os.environ.get("LLM_CONCURRENCY", "64"))'
assert src.count(old) == 1, "anchor"
p.write_text(src.replace(old, 'LLM_CONCURRENCY = int(os.environ.get("LLM_CONCURRENCY", "256"))'), encoding="utf-8", newline="\n")
print("patched C")
