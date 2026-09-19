import hashlib,json,os,pathlib,shutil,subprocess,time
root=pathlib.Path('/home/jkh/Src/nanolang-owned-array-source-integrated')
out=pathlib.Path('/tmp/nanolang-owner-array-source-c9fb-bootstrap');out.mkdir(exist_ok=False)
pin='c9fb07ed55329f59e6b67ef0bf3f9dc756d6aa88'
def git(*a):return subprocess.check_output(['git',*a],cwd=root,text=True).strip()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert git('rev-parse','HEAD')==pin and not git('status','--porcelain')
files=git('ls-files','src','src_nano','runtime','modules','tests','scripts','spec','Makefile','GNUmakefile','Makefile.gnu').splitlines()
def sources():return {p:sha(root/p) for p in files if (root/p).is_file()}
def inputs():return {str(p.relative_to(root)):sha(p) for d in ('bin','obj','lib') for p in (root/d).rglob('*') if p.is_file()}
def tools():
 result={}
 for n in ('cc','gcc','clang','make','python3','ar','ld'):
  p=pathlib.Path(shutil.which(n)).resolve();result[n]={'path':str(p),'sha256':sha(p)}
 return result
def write(n,v):(out/n).write_text(json.dumps(v,indent=2)+'\n')
before=sources();write('source-before.json',before);write('host-tools-before.json',tools());write('inputs-before.json',inputs())
(out/pathlib.Path(__file__).name).write_bytes(pathlib.Path(__file__).read_bytes())
setup=out/'setup.mk';setup.write_text('''.PHONY: owned-array-source-setup
owned-array-source-setup: nanoisa_emit nano_virt nano_vm nvm2c nanoisa_dump test-local-binding-metadata
\t$(CC) $(CFLAGS) -o obj/borrow_shadow_names tests/nanovirt/borrow_shadow_names.c $(NANOVIRT_OBJECTS) $(NANOVM_OBJECTS) $(NANOISA_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(LDFLAGS)
''')
env=os.environ.copy()
for k in ('NANOC','NANO_VM','NANO_NVM2C','NANO_AOT_RUNTIME','NANO_BUILD_CACHE','NANO_MODULE_PATH'):env.pop(k,None)
env.update(CC='/usr/bin/gcc',NANO_VM=str(root/'bin/nano_vm'),NANO_NVM2C=str(root/'bin/nvm2c'),NANO_AOT_RUNTIME=str(root/'bin/nano_aot_runtime.o'),NANO_MODULE_PATH=str(root/'modules'),NANO_BUILD_CACHE=str(root/'obj/module_cache'),PYTHONPATH=str(root))
write('environment.json',{k:env[k] for k in ('CC','NANO_VM','NANO_NVM2C','NANO_AOT_RUNTIME','NANO_MODULE_PATH','NANO_BUILD_CACHE','PYTHONPATH')})
report={'pin':pin,'steps':[]};status=1
try:
 for name,cmd in [('bootstrap',['make','-j2','-f','Makefile.gnu','CC=/usr/bin/gcc','bootstrap']),('setup',['make','-j2','-f','Makefile.gnu','-f',str(setup),'CC=/usr/bin/gcc','owned-array-source-setup'])]:
  start=time.monotonic();print('START',name,flush=True)
  with (out/(name+'.log')).open('wb') as log:
   try:status=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800).returncode
   except subprocess.TimeoutExpired:status=124
  report['steps'].append({'name':name,'command':cmd,'status':status,'seconds':time.monotonic()-start});write('status.json',report);write(name+'-built-inputs.json',inputs());print('END',name,status,flush=True)
  if status:break
finally:
 write('source-after.json',sources());write('host-tools-after.json',tools());write('inputs-after.json',inputs());report.update(sources_unchanged=before==sources(),head_unchanged=pin==git('rev-parse','HEAD'));write('status.json',report)
raise SystemExit(status)
