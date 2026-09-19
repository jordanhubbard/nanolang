import hashlib,json,os,pathlib,subprocess,time,shutil,sys
root=pathlib.Path('/home/jkh/Src/nanolang-inline-owned-integrated')
out=pathlib.Path('/tmp/nanolang-inline-owned-integrated-qualified');out.mkdir(exist_ok=False)
def git(*a):return subprocess.check_output(['git',*a],cwd=root,text=True).strip()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
pin=git('rev-parse','HEAD');assert not git('status','--porcelain','--untracked-files=no')
files=git('ls-files','src','src_nano','runtime','tests','scripts','spec','modules','stdlib','Makefile','Makefile.gnu').splitlines()
def sources():return {p:sha(root/p) for p in files if (root/p).is_file()}
def tools():
 paths=list((root/'bin').glob('*'))
 paths += [root/'obj/borrow_shadow_names',root/'obj/test_local_bindings',pathlib.Path('/tmp/nanolang-inline-clang')]
 for name in ('cc','gcc','clang','make','python3','as','ld'):
  p=shutil.which(name)
  if p:paths.append(pathlib.Path(p).resolve())
 return {str(p):sha(p) for p in paths if p.is_file()}
def write(name,value):(out/name).write_text(json.dumps(value,indent=2)+'\n')
env=os.environ.copy()
for key in ('NANOC','NANO_VM','NANO_NVM2C','NANO_AOT_RUNTIME','NANO_BUILD_CACHE','NANO_MODULE_PATH'):env.pop(key,None)
env.update(NANO_BUILD_CACHE=str(root/'obj/module_cache'),NANO_VM=str(root/'bin/nano_vm'),NANO_NVM2C=str(root/'bin/nvm2c'),NANO_AOT_RUNTIME=str(root/'bin/nano_aot_runtime.o'),NANO_MODULE_PATH=str(root/'modules'),ASAN_OPTIONS='detect_leaks=1:halt_on_error=1')
before=sources();write('source-before.json',before);(out/'runner.py').write_bytes(pathlib.Path(__file__).read_bytes());(out/'test-driver.py').write_bytes(pathlib.Path('/tmp/nanolang-inline-integrated-tests.py').read_bytes())
report={'pin':pin,'steps':[]};status=1
prep='inline-prepare: nanoisa_emit nano_virt nano_vm nvm2c nanoisa_dump test-local-binding-metadata\n\t$(CC) $(CFLAGS) -o obj/borrow_shadow_names tests/nanovirt/borrow_shadow_names.c $(NANOVIRT_OBJECTS) $(NANOVM_OBJECTS) $(NANOISA_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(LDFLAGS)'
steps=[('prepare',['make','-j2','--eval='+prep,'inline-prepare'],{}),('focused-gcc',['python3','/tmp/nanolang-inline-integrated-tests.py',str(out/'gcc-producers.json')],{'CC':'gcc'}),('focused-clang',['python3','/tmp/nanolang-inline-integrated-tests.py',str(out/'clang-producers.json')],{'CC':'/tmp/nanolang-inline-clang'}),('runtime',['make','-j2','test-affine-state','test-affine-bytecode','test-owned-transfers','test-owned-runtime','test-owned-value-graphs','test-owned-value-results','test-owned-assertions','test-owned-binary64','test-nested-owned-results'],{})]
try:
 for name,cmd,overrides in steps:
  start=time.monotonic();tb=tools();write(name+'-tools-before.json',tb);print('START',name,flush=True)
  with (out/(name+'.log')).open('wb') as log:r=subprocess.run(cmd,cwd=root,env={**env,**overrides},stdout=log,stderr=subprocess.STDOUT)
  status=r.returncode;ta=tools();write(name+'-tools-after.json',ta)
  report['steps'].append({'name':name,'command':cmd,'environment':overrides,'status':status,'seconds':round(time.monotonic()-start,3),'tools_unchanged':tb==ta,'log_sha256':sha(out/(name+'.log'))});write('manifest.json',report)
  print('END',name,status,report['steps'][-1]['seconds'],'tools_equal',tb==ta,flush=True)
  if status:break
  if name.startswith('focused') or name=='source-regression':
   if tb!=ta:status=98;break
finally:
 after=sources();write('source-after.json',after);report.update(source_unchanged=after==before,head_unchanged=git('rev-parse','HEAD')==pin,tracked_tree_clean=not git('status','--porcelain','--untracked-files=no'),terminal=status);write('manifest.json',report);(out/'status').write_text(str(status)+'\n')
raise SystemExit(status)
