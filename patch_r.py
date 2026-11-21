"""R안(안정성): API 가 한 번 400(BadRequest)을 주면 서버가 그 뒤 모든 호출에서 JSON 모드·추론 강도를 끄던 것을 고친다.
그 호출만 선택 인자 없이 다시 부르고, 서버 전체 설정은 그대로 둔다.
11/20 05:52 에 400 한 번으로 이 상태가 되어 D안 제출 3회가 81~82점으로 떨어졌다(JSON 아님 529회, 대체 응답 114회).
python3 patch_r.py <고칠 서버 파일>"""
import sys

p = sys.argv[1]
s = open(p, encoding="utf-8").read()
old = '''            except (BadRequestError, TypeError):
                if not self.use_extra:
                    raise
                self.use_extra = False
                resp = await self.client.chat.completions.create(**base)'''
new = '''            except (BadRequestError, TypeError):
                # (R) 이 호출만 선택 인자 없이 다시 부른다. 한 번 거절됐다고 서버 전체의 JSON 모드·추론 강도를 끄지 않는다
                resp = await self.client.chat.completions.create(**base)'''
assert s.count(old) == 1, "대상 코드를 못 찾음"
s = s.replace(old, new)
open(p, "w", encoding="utf-8").write(s)
print("R안 적용:", p)
