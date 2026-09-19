import hashlib,json,os,pathlib,shutil,subprocess,time
root=pathlib.Path('/private/tmp/nanolang-mixed-source-70c511dad');out=pathlib.Path('/private/tmp/nanolang-mixed-source-darwin-qualified-70c511dad');out.mkdir(exist_ok=False)
def git(*a):return subprocess.check_output(['git',*a],cwd=root,text=True).strip()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
pin=git('rev-parse','HEAD');assert pin=='70c511dad439f77eba15fe0bdc189e90ff9c9486' and not git('status','--porcelain','--untracked-files=no')
files=git('ls-files','src','src_nano','runtime','modules','tests','scripts','spec','Makefile','Makefile.gnu').splitlines()
def sources():return {p:sha(root/p) for p in files if (root/p).is_file()}
def inputs():
 paths=list((root/'bin').glob('*'))+list((root/'obj').rglob('*.o'))+list((root/'lib').rglob('*.a'))
 paths += [pathlib.Path('/usr/bin/clang').resolve(),pathlib.Path('/opt/homebrew/opt/llvm/bin/clang').resolve(),pathlib.Path(shutil.which('python3')).resolve()]
 return {str(p):sha(p) for p in paths if p.is_file()}
before=sources();initial=inputs();(out/'sources-before.json').write_text(json.dumps(before,indent=2)+'\n');(out/'inputs-before.json').write_text(json.dumps(initial,indent=2)+'\n')
for path in (pathlib.Path(__file__),pathlib.Path('/private/tmp/nanolang-mixed-source-darwin-tests.py')):(out/path.name).write_bytes(path.read_bytes())
env=os.environ.copy()
for key in ('NANOC','NANO_VM','NANO_NVM2C','NANO_AOT_RUNTIME','NANO_BUILD_CACHE','NANO_MODULE_PATH'):env.pop(key,None)
env.update(NANO_BUILD_CACHE=str(root/'obj/module_cache'),NANO_VM=str(root/'bin/nano_vm'),NANO_NVM2C=str(root/'bin/nvm2c'),NANO_AOT_RUNTIME=str(root/'bin/nano_aot_runtime.o'),NANO_MODULE_PATH=str(root/'modules'),PYTHONPATH=str(root),CC='/opt/homebrew/opt/llvm/bin/clang',SDKROOT=subprocess.check_output(['/usr/bin/xcrun','--show-sdk-path'],text=True).strip())
cmd=['/opt/homebrew/bin/python3','/private/tmp/nanolang-mixed-source-darwin-tests.py',str(out/'tests')];status=1;start=time.monotonic();report={'pin':pin,'command':cmd}
try:
 with (out/'run.log').open('wb') as log:
  try:r=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=2400);status=r.returncode
  except subprocess.TimeoutExpired:status=124
finally:
 after=sources();final=inputs();(out/'sources-after.json').write_text(json.dumps(after,indent=2)+'\n');(out/'inputs-after.json').write_text(json.dumps(final,indent=2)+'\n')
 report.update(status=status,seconds=round(time.monotonic()-start,3),sources_unchanged=before==after,inputs_unchanged=initial==final,head_unchanged=pin==git('rev-parse','HEAD'),clean=not git('status','--porcelain','--untracked-files=no'))
 (out/'status.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
raise SystemExit(status)
