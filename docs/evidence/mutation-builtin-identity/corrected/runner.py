import hashlib,json,os,pathlib,shutil,subprocess,time
root=pathlib.Path('/home/jkh/Src/nanolang-mutation-builtin-corrected'); out=pathlib.Path('/tmp/nanolang-mutation-identity-95a1-corrected');out.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def git(*a):return subprocess.check_output(['git',*a],cwd=root,text=True).strip()
def save(n,v):(out/n).write_text(json.dumps(v,indent=2)+'\n')
assert git('rev-parse','--short=9','HEAD')=='95a1f6224'
assert not git('status','--porcelain','--untracked-files=no')
files=git('ls-files','src','src_nano','runtime','modules','tests','scripts','spec','Makefile','Makefile.gnu','GNUmakefile').splitlines()
def sources():return {p:sha(root/p) for p in files if (root/p).is_file()}
paths=[shutil.which(p) for p in ['gcc','g++','make','python3','git','as','ld']];paths.append(subprocess.check_output(['gcc','-print-prog-name=cc1'],text=True).strip())
def hosts():return {p:{'resolved_path':str(pathlib.Path(p).resolve()),'sha256':sha(p)} for p in paths}
def inputs():return {str(p.relative_to(root)):sha(p) for folder in ('bin','obj','lib') for p in (root/folder).rglob('*') if p.is_file()}
env=os.environ.copy()
for k in ('NANOC','NANO_VM','NANO_NVM2C','NANO_AOT_RUNTIME','NANO_BUILD_CACHE','NANO_MODULE_PATH'):env.pop(k,None)
env.update(CC=shutil.which('gcc'),CXX=shutil.which('g++'),NANO_BUILD_CACHE=str(root/'obj/module_cache'),NANO_VM=str(root/'bin/nano_vm'),NANO_NVM2C=str(root/'bin/nvm2c'),NANO_AOT_RUNTIME=str(root/'bin/nano_aot_runtime.o'),NANO_MODULE_PATH=str(root/'modules'),ASAN_OPTIONS='detect_leaks=1:halt_on_error=1')
(out/'runner.py').write_bytes(pathlib.Path(__file__).read_bytes())
save('environment.json',{'pin':git('rev-parse','HEAD'),'environment':{k:env[k] for k in ('CC','CXX','NANO_BUILD_CACHE','NANO_VM','NANO_NVM2C','NANO_AOT_RUNTIME','NANO_MODULE_PATH','ASAN_OPTIONS')}})
before=sources(); tools=hosts();save('sources-before.json',before);save('tools-before.json',tools);save('inputs-before-setup.json',inputs())
commands=[('unbound',['python3','/tmp/nanolang-mutation-unbound-only.py'],600),('adjacent',['python3','-m','unittest','-v','tests.test_canonical_prefix_conversion','tests.test_scalar_reduce_source.ScalarReduce.test_bound_reduce_function_and_local','tests.test_scalar_reduce_source.ScalarReduce.test_checked_signature_refusals_preserve_output'],600)]
(out/'unbound-runner.py').write_bytes(pathlib.Path('/tmp/nanolang-mutation-unbound-only.py').read_bytes())
results=[]
try:
 for label,command,bound in commands:
  save(label+'-inputs-before.json',inputs());start=time.monotonic();status='incomplete'
  try:
   with (out/(label+'.log')).open('wb') as log:status=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=bound).returncode
  except subprocess.TimeoutExpired:status='timeout'
  result={'phase':label,'command':command,'timeout':bound,'status':status,'elapsed_seconds':time.monotonic()-start};results.append(result);save(label+'-result.json',result);save(label+'-inputs-after.json',inputs());print(json.dumps(result),flush=True)
  if status!=0:break
finally:
 save('sources-after.json',sources());save('tools-after.json',hosts());save('results.json',{'phases':results,'sources_unchanged':before==sources(),'host_tools_unchanged':tools==hosts()})
raise SystemExit(0 if len(results)==2 and all(x['status']==0 for x in results) else 1)
