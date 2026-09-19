import pathlib,subprocess,hashlib,json,os,signal,time,tempfile
out=pathlib.Path(tempfile.mkdtemp(prefix='nanolang-public-c-darwin-',dir='/private/tmp'))
root=out/'source'
env={**os.environ,'PATH':'/opt/homebrew/opt/llvm/bin:/private/tmp/nanolang-darwin-wasmtime-venv.YxRmVW/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin','CC':'/opt/homebrew/opt/llvm/bin/clang','ASAN_OPTIONS':'detect_leaks=1:halt_on_error=1'}
sha='a59858424169ca52ed10f099c5692dd88e6636d4'
print(str(out),flush=True)
subprocess.run(['git','fetch','origin','main'],cwd='/Users/jordanh/Src/nanolang',check=True,env=env)
subprocess.run(['git','worktree','add','--detach',str(root),sha],cwd='/Users/jordanh/Src/nanolang',check=True,env=env)
files=subprocess.check_output(['git','ls-files','src','src_nano','spec','tests','scripts','Makefile','Makefile.gnu'],cwd=root,text=True).splitlines()
def hashes(paths):return {p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths if (root/p).is_file()}
(out/'source-before.json').write_text(json.dumps(hashes(files),indent=2))
results=[]
def run(name,cmd):
 start=time.monotonic();timeout=False
 with (out/(name+'.log')).open('w') as log:
  p=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=p.wait(timeout=900)
  except subprocess.TimeoutExpired:
   timeout=True;os.killpg(p.pid,signal.SIGTERM)
   try:code=p.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait()
 r=dict(name=name,command=cmd,status=code,timed_out=timeout,seconds=round(time.monotonic()-start,3));results.append(r)
 (out/'status.json').write_text(json.dumps(dict(source=sha,results=results),indent=2));print(json.dumps(r),flush=True)
 return code
with (out/'inventory.log').open('w') as log:
 for cmd in [['sw_vers'],['uname','-a'],['/usr/bin/clang','--version'],['clang','--version'],['opt','--version'],['node','--version'],['wasmtime','--version'],['python3','--version']]:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
code=run('build',['make','-j8','bin/nanoc_c','bin/nano','nano_virt','nano_vm','nvm2c','CC=/usr/bin/clang'])
if code:raise SystemExit(code)
tools=['bin/nanoc_c','bin/nano','bin/nano_virt','bin/nano_vm','bin/nvm2c']
before=hashes(files+tools);(out/'before.json').write_text(json.dumps(before,indent=2))
modules=['tests.test_public_c_binary64','tests.test_public_c_nonfinite_format','tests.test_public_c_string_equality']
code=run('suite',['python3','-m','unittest','-f','-v',*modules])
after=hashes(files+tools);(out/'after.json').write_text(json.dumps(after,indent=2))
(out/'git-status.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=root,text=True))
print(json.dumps(dict(unchanged=before==after,out=str(out))),flush=True)
raise SystemExit(code if code else int(before!=after))
