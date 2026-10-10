import subprocess, pathlib, json, time, hashlib
root=pathlib.Path('/Users/jordanh/Src/nanolang')
out=pathlib.Path('/private/tmp/nanolang-json-parser-integrated-20261008');out.mkdir(exist_ok=True)
def state():
 return {'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'status':subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True),'user_sha256':hashlib.sha256((root/'tests/user_guide/refresh_language_pure_function.nano').read_bytes()).hexdigest()}
before=state();(out/'before.json').write_text(json.dumps(before,indent=2));started=time.monotonic()
with (out/'gate.log').open('w') as log:
 result=subprocess.run(['make','-j1','test-selfhost-json-artifacts','test-file-service-parser','CC=/opt/homebrew/opt/llvm/bin/clang'],cwd=root,stdout=log,stderr=subprocess.STDOUT)
(out/'after.json').write_text(json.dumps({'state':state(),'exit':result.returncode,'seconds':time.monotonic()-started},indent=2))
raise SystemExit(result.returncode)
