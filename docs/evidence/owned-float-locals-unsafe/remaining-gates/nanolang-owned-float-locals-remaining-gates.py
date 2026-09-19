import hashlib,json,os,pathlib,subprocess,time,shutil
root=pathlib.Path('/home/jkh/Src/nanolang-owned-float-locals-corrected');out=pathlib.Path('/tmp/nanolang-owned-float-locals-remaining-gates');out.mkdir(exist_ok=False)
def git(*a):return subprocess.check_output(['git',*a],cwd=root,text=True).strip()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
pin=git('rev-parse','HEAD');assert not git('status','--porcelain','--untracked-files=no')
files=git('ls-files','src','src_nano','runtime','modules','tests','scripts','spec','Makefile','Makefile.gnu').splitlines()
def sources():return {p:sha(root/p) for p in files if (root/p).is_file()}
def tools():
 paths=[*(root/'bin').glob('*'),root/'obj/borrow_shadow_names',*(root/'obj').glob('**/*.o')]
 paths += [pathlib.Path(shutil.which(x)).resolve() for x in ('gcc','cc','clang','python3','make')]
 paths += [pathlib.Path('/tmp/nanolang-projection-clang')]
 return {str(p):sha(p) for p in paths if p.is_file() and "/obj/nano_modules/" not in str(p) and "/obj/module_cache/" not in str(p) and not p.name.startswith("test_")}

before=sources();(out/'source-before.json').write_text(json.dumps(before,indent=2)+'\n')
for path in [pathlib.Path(__file__),pathlib.Path('/tmp/nanolang-float-corrected-unittest.py'),pathlib.Path('/tmp/nanolang-owned-float-locals-setup.mk')]: (out/path.name).write_bytes(path.read_bytes())
env=os.environ.copy()
for key in ('NANOC','NANO_VM','NANO_NVM2C','NANO_AOT_RUNTIME','NANO_BUILD_CACHE','NANO_MODULE_PATH'):env.pop(key,None)
env.update(NANO_BUILD_CACHE=str(root/'obj/module_cache'),NANO_VM=str(root/'bin/nano_vm'),NANO_NVM2C=str(root/'bin/nvm2c'),NANO_AOT_RUNTIME=str(root/'bin/nano_aot_runtime.o'),NANO_MODULE_PATH=str(root/'modules'),PYTHONPATH=str(root))
methods=['test_float_local_unsafe_original_pattern','test_float_local_unsafe_control_flow','test_float_local_in_borrowed_control_flow','test_float_local_unsafe_refusals_preserve_publication']
focus=[f'tests.test_source_borrow_emission.SourceBorrowEmission.{x}' for x in methods]+['tests.test_owned_record_patterns.OwnedRecordPatterns.test_unsafe_pattern_keeps_outer_shadow']
setup=['make','-j2','-f','Makefile.gnu','-f','/tmp/nanolang-owned-float-locals-setup.mk','transitive-wrapper-setup']
steps=[('setup',setup,None),('full-gcc',['python3','/tmp/nanolang-float-corrected-unittest.py',str(out/'full-gcc-producers.json'),'tests.test_source_borrow_emission','tests.test_owned_record_patterns.OwnedRecordPatterns.test_unsafe_pattern_keeps_outer_shadow'],'/usr/bin/gcc'),('focused-clang',['python3','/tmp/nanolang-float-corrected-unittest.py',str(out/'focused-clang-producers.json'),*focus],'/tmp/nanolang-projection-clang'),('runtime',['make','test-owned-binary64','test-owned-value-graphs','test-owned-value-results','test-consuming-calls','test-multiple-consuming-calls','test-owned-assertions'],None)]
steps=steps[2:]
report={'source':pin,'steps':[]};status=1
try:
 for name,cmd,cc in steps:
  assert git('rev-parse','HEAD')==pin and not git('status','--porcelain','--untracked-files=no')
  call_env=env.copy()
  if cc:call_env['CC']=cc
  before_tools=tools();(out/(name+'-tools-before.json')).write_text(json.dumps(before_tools,indent=2)+'\n')
  print('START',name,flush=True);start=time.monotonic()
  with (out/(name+'.log')).open('wb') as log:result=subprocess.run(cmd,cwd=root,env=call_env,stdout=log,stderr=subprocess.STDOUT)
  status=result.returncode;after_tools=tools();(out/(name+'-tools-after.json')).write_text(json.dumps(after_tools,indent=2)+'\n')
  item={'name':name,'command':cmd,'status':status,'seconds':round(time.monotonic()-start,3),'log_sha256':sha(out/(name+'.log')),'tools_unchanged':before_tools==after_tools}
  if cc:item.update(CC=cc,compiler_sha256=sha(pathlib.Path(cc).resolve()))
  report['steps'].append(item);(out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print('END',name,status,item['seconds'],flush=True)
  if status:break
  if name!='setup':assert before_tools==after_tools
finally:
 after=sources();(out/'source-after.json').write_text(json.dumps(after,indent=2)+'\n');report['source_unchanged']=after==before;report['unchanged_head']=git('rev-parse','HEAD')==pin;report['tracked_tree_clean']=not git('status','--porcelain','--untracked-files=no');(out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');(out/'status').write_text(str(status)+'\n')
raise SystemExit(status)
