import hashlib,json,os,pathlib,shutil,subprocess,time
root=pathlib.Path('/home/jkh/Src/nanolang-mutation-builtin-identity');out=pathlib.Path('/tmp/nanolang-mutation-identity-66a-bootstrap');out.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def git(*a):return subprocess.check_output(['git',*a],cwd=root,text=True).strip()
assert git('rev-parse','HEAD')=='66a4686e18ec7d9153e98f8e7b1fa3c12d96af28'
assert not git('status','--porcelain','--untracked-files=no')
assert not (root/'bin').exists() and not (root/'obj').exists()
files=git('ls-files','src','src_nano','runtime','modules','tests','scripts','spec','Makefile','Makefile.gnu').splitlines()
def sources():return {p:sha(root/p) for p in files if (root/p).is_file()}
paths=[shutil.which(p) for p in ['gcc','g++','make','python3','git','as','ld']]
paths.append(subprocess.check_output(['gcc','-print-prog-name=cc1'],text=True).strip())
def hosts():return {p:{'resolved_path':str(pathlib.Path(p).resolve()),'sha256':sha(p)} for p in paths}
def save(n,x):(out/n).write_text(json.dumps(x,indent=2)+'\n')
before=sources();tools=hosts();save('sources-before.json',before);save('tools-before.json',tools)
(out/'runner.py').write_bytes(pathlib.Path(__file__).read_bytes())
env=os.environ.copy()
for key in ('NANOC','NANO_VM','NANO_NVM2C','NANO_AOT_RUNTIME','NANO_BUILD_CACHE','NANO_MODULE_PATH'):env.pop(key,None)
env.update(CC=shutil.which('gcc'),CXX=shutil.which('g++'),NANO_BUILD_CACHE=str(root/'obj/module_cache'),NANO_VM=str(root/'bin/nano_vm'),NANO_NVM2C=str(root/'bin/nvm2c'),NANO_AOT_RUNTIME=str(root/'bin/nano_aot_runtime.o'),NANO_MODULE_PATH=str(root/'modules'))
command=['make','-j2','bootstrap'];start=time.monotonic();status='incomplete'
save('environment.json',{'pin':git('rev-parse','HEAD'),'command':command,'environment':{k:env[k] for k in ['CC','CXX','NANO_BUILD_CACHE','NANO_VM','NANO_NVM2C','NANO_AOT_RUNTIME','NANO_MODULE_PATH']},'compiler':subprocess.check_output([env['CC'],'--version'],text=True)})
try:
 with (out/'bootstrap.log').open('wb') as log:
  status=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800).returncode
except subprocess.TimeoutExpired:status='timeout'
finally:
 save('sources-after.json',sources());save('tools-after.json',hosts())
 save('built-tools.json',{str(p.relative_to(root)):sha(p) for p in (root/'bin').glob('*') if p.is_file()})
 save('result.json',{'status':status,'elapsed_seconds':time.monotonic()-start,'sources_unchanged':sources()==before,'host_tools_unchanged':hosts()==tools})
 print((out/'result.json').read_text(),flush=True)
raise SystemExit(0 if status==0 else 1)
