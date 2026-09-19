import hashlib,json,os,pathlib,subprocess,time
root=pathlib.Path('/private/tmp/nanolang-mixed-source-70c511dad');out=pathlib.Path('/private/tmp/nanolang-mixed-source-darwin-runtime-70c511dad')
def git(*a):return subprocess.check_output(['git',*a],cwd=root,text=True).strip()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def snap(paths):return {str(p):sha(p) for p in paths if p.is_file()}
assert git('rev-parse','HEAD')=='70c511dad439f77eba15fe0bdc189e90ff9c9486'
out.mkdir(exist_ok=False);(out/pathlib.Path(__file__).name).write_bytes(pathlib.Path(__file__).read_bytes())
paths=[root/p for p in git('ls-files','src','src_nano','runtime','modules','tests','scripts','spec','Makefile','Makefile.gnu').splitlines()]
before=snap(paths);(out/'sources-before.json').write_text(json.dumps(before,indent=2)+'\n')
inputs=list((root/'bin').glob('*'))+list((root/'obj').rglob('*.o'));before_inputs=snap(inputs);(out/'inputs-before.json').write_text(json.dumps(before_inputs,indent=2)+'\n')
env=os.environ.copy();env.update(PYTHONPATH=str(root),CC='/opt/homebrew/opt/llvm/bin/clang',SDKROOT=subprocess.check_output(['/usr/bin/xcrun','--show-sdk-path'],text=True).strip(),NMS_RUNTIME_CLANG='/opt/homebrew/opt/llvm/bin/clang',NMS_RUNTIME_OPT='/opt/homebrew/opt/llvm/bin/opt',NANO_MIXED_RUNTIME_DIR=str(out/'native'));(out/'native').mkdir()
commands=[('mixed-runtime',['make','-j2','test-mixed-samples-runtime','test-mixed-samples-runtime-alloc','test-mixed-samples-admission']),('owned-regressions',['make','-j2','test-owned-value-graphs','test-owned-value-results']),('service',['make','-j2','test-service-module'])]
report={'pin':git('rev-parse','HEAD'),'steps':[]};status=1
try:
 for name,cmd in commands:
  print('START',name,flush=True);start=time.monotonic()
  with (out/(name+'.log')).open('wb') as log:
   try:status=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600).returncode
   except subprocess.TimeoutExpired:status=124
  report['steps'].append({'name':name,'command':cmd,'status':status,'seconds':round(time.monotonic()-start,3)});(out/'status.json').write_text(json.dumps(report,indent=2)+'\n');print('END',name,status,flush=True)
  if status:break
finally:
 after=snap(paths);(out/'sources-after.json').write_text(json.dumps(after,indent=2)+'\n')
 current=list((root/'bin').glob('*'))+list((root/'obj').rglob('*.o'));(out/'inputs-after.json').write_text(json.dumps(snap(current),indent=2)+'\n')
 report.update(sources_unchanged=before==after,clean=not git('status','--porcelain','--untracked-files=no'));(out/'status.json').write_text(json.dumps(report,indent=2)+'\n')
raise SystemExit(status)
