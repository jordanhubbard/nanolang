import pathlib,subprocess,hashlib,json,os,signal,time,tempfile
out=pathlib.Path(tempfile.mkdtemp(prefix='nanolang-aggregate-policy-darwin-',dir='/private/tmp'))
root=out/'source'
env={**os.environ,'PATH':'/opt/homebrew/opt/llvm/bin:/private/tmp/nanolang-darwin-wasmtime-venv.YxRmVW/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin','CC':'/usr/bin/clang'}
sha='acb656611e9509e266443411163ca3a8259a4872'
print(str(out),flush=True)
subprocess.run(['git','fetch','origin','fix/aggregate-binary64-policy'],cwd='/Users/jordanh/Src/nanolang',check=True,env=env)
subprocess.run(['git','worktree','add','--detach',str(root),sha],cwd='/Users/jordanh/Src/nanolang',check=True,env=env)
files=subprocess.check_output(['git','ls-files','src','src_nano','spec','tests','scripts','Makefile','Makefile.gnu'],cwd=root,text=True).splitlines()
def hashes(paths):return {p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths if (root/p).is_file()}
(out/'source-before.json').write_text(json.dumps(hashes(files),indent=2))
results=[]
def run(name,cmd):
 start=time.monotonic();timeout=False
 with (out/(name+'.log')).open('w') as log:
  p=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=p.wait(timeout=1200)
  except subprocess.TimeoutExpired:
   timeout=True;os.killpg(p.pid,signal.SIGTERM)
   try:code=p.wait(timeout=5)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait()
 r=dict(name=name,command=cmd,status=code,timed_out=timeout,seconds=round(time.monotonic()-start,3));results.append(r)
 (out/'status.json').write_text(json.dumps(dict(source=sha,results=results),indent=2));print(json.dumps(r),flush=True)
 return code
with (out/'inventory.log').open('w') as log:
 for cmd in [['sw_vers'],['uname','-a'],['/usr/bin/clang','--version'],['clang','--version'],['opt','--version'],['node','--version'],['wasmtime','--version'],['python3','--version']]:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
code=run('bootstrap',['make','-j8','bootstrap','bin/nano','nano_virt','nano_vm','nvm2c','nanoisa_dump','CC=/usr/bin/clang'])
if code:raise SystemExit(code)
tools=['bin/nano','bin/nanoc_c','bin/nanoc_stage1','bin/nanoc_stage2','bin/nano_virt','bin/nano_vm','bin/nvm2c','bin/nanoisa']
before=hashes(files+tools);(out/'before.json').write_text(json.dumps(before,indent=2))
code=run('suite',['python3','-m','unittest','-v','tests.test_aggregate_binary64_policy'])
if not code: code=run('adjacent',['make','-j8','CC=/usr/bin/clang','test-aggregate-binary64-eval','test-nanovm','test-binary64-arithmetic-eval'])
after=hashes(files+tools);(out/'after.json').write_text(json.dumps(after,indent=2))
(out/'git-status.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=root,text=True))
print(json.dumps(dict(unchanged=before==after,out=str(out))),flush=True)
raise SystemExit(code if code else int(before!=after))
