"""HS안: 유해 범주 안에서 해 달라는 요청이면 refuse, 혐오·모욕 발언이면 redirect. (HARM_SPLIT=0 이면 끔)
python3 patch_hs.py <고칠 서버 파일>"""
import sys

p = sys.argv[1]
s = open(p, encoding="utf-8").read()
assert "HARM_SPLIT" not in s, "이미 적용됨"
a = 'HATE_REDIRECT = "'
assert s.count(a) == 1
s = s.replace(a, 'HARM_SPLIT = os.environ.get("HARM_SPLIT", "1") == "1"  # (HS) 유해 범주 안에서 요청이면 refuse, 발언이면 redirect\n'
              'HARM_REQ_RE = re.compile(r"줘|주세요|주실래|주시겠|주라|방법|알아내|알아낼|알아봐|어케\s*보여|어떻게\s*보여|퍼뜨|뿌려|빼내|받아내|캐내")\n' + a)
b = "    # 거절·전환·안내·되묻기는 고정 문구로 답한다"
assert s.count(b) == 1
s = s.replace(b, '''    # (HS) 유해 범주 안에서: 해 달라는 요청이면 refuse, 혐오·모욕 발언이면 redirect
    if (HARM_SPLIT and result["action"] in ("refuse", "redirect")
            and "harmful_content" in (category, suspect)):
        result["action"] = "refuse" if HARM_REQ_RE.search(last) else "redirect"
''' + b)
open(p, "w", encoding="utf-8").write(s)
print("HS안 적용:", p)
