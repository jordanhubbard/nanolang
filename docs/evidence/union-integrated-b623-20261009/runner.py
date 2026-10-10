import hashlib,json,os,pathlib,subprocess,time
root=pathlib.Path('/Users/jordanh/Src/nanolang')
out=pathlib.Path('/private/tmp/nanolang-union-integrated-20261009');out.mkdir(exist_ok=False)
def state():
 files=subprocess.check_output(['git','ls-files','-z','--','src','src_nano','modules','stdlib','tests','scripts','Makefile.gnu','spec.json'],cwd=root).decode().split('\0')
 return {'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'status':subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True),'sources':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files if name and (root/name).is_file()},'user_file':hashlib.sha256((root/'tests/user_guide/refresh_language_pure_function.nano').read_bytes()).hexdigest()}
before=state();assert before['head'].startswith('b623525f9');assert before['status']=='?? tests/user_guide/refresh_language_pure_function.nano\n'
(out/'before.json').write_text(json.dumps(before,indent=2)+'\n')
command=['make','-j2','test-owned-union-runtime','test-affine-state','test-affine-bytecode','test-affine-scalar-union-runtime','test-owned-runtime','test-owned-transfers','test-consuming-calls','test-multiple-consuming-calls','test-owned-result-descriptors','test-nested-owned-results','test-owned-value-results','test-ownership-contracts','test-ownership-declaration-projection','test-nvm2c','CC=/opt/homebrew/opt/llvm/bin/clang']
start=time.monotonic();print('START',command,flush=True)
with (out/'gate.log').open('w') as log:result=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CC':'/opt/homebrew/opt/llvm/bin/clang','ASAN_OPTIONS':'detect_leaks=1'})
after=state();(out/'after.json').write_text(json.dumps(after,indent=2)+'\n')
(out/'result.json').write_text(json.dumps({'command':command,'exit':result.returncode,'seconds':time.monotonic()-start,'unchanged':before==after},indent=2)+'\n')
assert before==after
print('END',result.returncode,flush=True);raise SystemExit(result.returncode)
