import hashlib,json,os,pathlib,shutil,subprocess,time
root=pathlib.Path('/home/jkh/Src/nanolang-mixed-source-corrected'); out=pathlib.Path('/tmp/nanolang-mixed-source-bootstrap-70c511dad');out.mkdir(exist_ok=False)
def git(*a):return subprocess.check_output(['git',*a],cwd=root,text=True).strip()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
pin=git('rev-parse','HEAD');assert pin=='70c511dad439f77eba15fe0bdc189e90ff9c9486' and not git('status','--porcelain','--untracked-files=no')
files=git('ls-files','src','src_nano','runtime','modules','tests','scripts','spec','Makefile','Makefile.gnu').splitlines()
def sources():return {p:sha(root/p) for p in files if (root/p).is_file()}
def inventory():return {str(p.relative_to(root)):sha(p) for name in ('bin','obj','lib') for p in (root/name).rglob('*') if p.is_file()}
before=sources();(out/'source-before.json').write_text(json.dumps(before,indent=2)+'\n')
compiler={}
for name in ('cc','gcc','clang','make','python3','ar','ld'):
 p=pathlib.Path(shutil.which(name)).resolve();compiler[name]={'path':str(p),'sha256':sha(p)}
(out/'host-tools-before.json').write_text(json.dumps(compiler,indent=2)+'\n')
(out/pathlib.Path(__file__).name).write_bytes(pathlib.Path(__file__).read_bytes())
(out/'setup.mk').write_bytes(pathlib.Path('/tmp/nanolang-transitive-wrapper-setup.mk').read_bytes())
env=os.environ.copy()
for key in ('NANOC','NANO_VM','NANO_NVM2C','NANO_AOT_RUNTIME','NANO_BUILD_CACHE','NANO_MODULE_PATH'):env.pop(key,None)
env.update(NANO_BUILD_CACHE=str(root/'obj/module_cache'),NANO_VM=str(root/'bin/nano_vm'),NANO_NVM2C=str(root/'bin/nvm2c'),NANO_AOT_RUNTIME=str(root/'bin/nano_aot_runtime.o'),NANO_MODULE_PATH=str(root/'modules'),PYTHONPATH=str(root))
steps=[('bootstrap',['make','-j2','bootstrap']),('setup',['make','-j2','-f','Makefile.gnu','-f',str(out/'setup.mk'),'transitive-wrapper-setup','bin/nano'])]
report={'source':pin,'steps':[]};status=1
try:
 for name,cmd in steps:
  start=time.monotonic();print('START',name,flush=True)
  with (out/(name+'.log')).open('wb') as log:
   try:result=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800);status=result.returncode
   except subprocess.TimeoutExpired:status=124
  report['steps'].append({'name':name,'command':cmd,'status':status,'seconds':round(time.monotonic()-start,3)})
  (out/'status.json').write_text(json.dumps(report,indent=2)+'\n');print('END',name,status,flush=True)
  if status:break
finally:
 after=sources();(out/'source-after.json').write_text(json.dumps(after,indent=2)+'\n');(out/'built-inputs.json').write_text(json.dumps(inventory(),indent=2)+'\n')
 report['sources_unchanged']=before==after;report['unchanged_head']=pin==git('rev-parse','HEAD');report['clean']=not git('status','--porcelain','--untracked-files=no')
 (out/'host-tools-after.json').write_text(json.dumps({k:{'path':v['path'],'sha256':sha(pathlib.Path(v['path']))} for k,v in compiler.items()},indent=2)+'\n');(out/'status.json').write_text(json.dumps(report,indent=2)+'\n')
raise SystemExit(status)
