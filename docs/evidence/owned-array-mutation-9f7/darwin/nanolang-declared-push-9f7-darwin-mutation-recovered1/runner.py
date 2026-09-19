import hashlib,json,os,pathlib,shutil,subprocess,sys,time,unittest,traceback
platform,phase=sys.argv[1:]
linux=platform=='linux'
root=pathlib.Path('/home/jkh/Src/nanolang-declared-push-evaluator-initializer' if linux else '/private/tmp/nanolang-declared-push-9f7ead937')
old=pathlib.Path('/home/jkh/Src/nanolang-declared-push-canonical-identity' if linux else '/private/tmp/nanolang-declared-push-0c5421d9b')
base=pathlib.Path(('/tmp/' if linux else '/private/tmp/')+'nanolang-declared-push-9f7-'+platform)
out=pathlib.Path(str(base)+'-'+phase+('-recovered1' if not linux and phase in ('mutation','patterns') else ''));out.mkdir(exist_ok=False)
pin='9f7ead937f983e6441f113488534835a046b2781';oldpin='0c5421d9b51abf8a48a01839824b3bb98dff2f68'
cc='/usr/bin/gcc' if linux else '/usr/bin/clang'
nativecc=cc if linux else '/opt/homebrew/opt/llvm/bin/clang'
make='/usr/bin/make'
os.chdir(root);sys.path.insert(0,str(root))
def git(*args,tree=root):return subprocess.check_output(['git',*args],cwd=tree,text=True).strip()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,value):(out/name).write_text(json.dumps(value,indent=2)+'\n')
def sources():return {p:sha(root/p) for p in git('ls-files','src','src_nano','runtime','modules','tests','scripts','spec','Makefile','GNUmakefile','Makefile.gnu').splitlines() if (root/p).is_file()}
def inputs():return {str(p.relative_to(root)):sha(p) for d in ('bin','obj','lib') for p in (root/d).rglob('*') if p.is_file()}
def tools():
 result={}
 for name in ('cc','gcc','clang','make','python3','ar','ld',cc,nativecc):
  path=pathlib.Path(shutil.which(name)).resolve();result[name]={'path':str(path),'sha256':sha(path)}
 return result
assert git('rev-parse','HEAD')==pin
status_text=git('status','--porcelain')
if linux and status_text:
 assert status_text=='?? tests/test_eval'
 retained=root/'tests/test_eval'
 assert sha(retained)=='f2afa578e828691d6563445d27f717285526f2536fed3dd6d7de2eb4d79aa47f'
 write('retained-failed-binary.json',{'path':str(retained),'sha256':sha(retained),'not_executed':True})
else:assert not status_text
for key in ('NANOC','NANO_VM','NANO_NVM2C','NANO_AOT_RUNTIME','NANO_BUILD_CACHE','NANO_MODULE_PATH'):os.environ.pop(key,None)
os.environ.update(CC=cc,NANO_VM=str(root/'bin/nano_vm'),NANO_NVM2C=str(root/'bin/nvm2c'),NANO_AOT_RUNTIME=str(root/'bin/nano_aot_runtime.o'),NANO_MODULE_PATH=str(root/'modules'),NANO_BUILD_CACHE=str(root/'obj/module_cache'),PYTHONPATH=str(root))
if not linux:os.environ['SDKROOT']=subprocess.check_output(['/usr/bin/xcrun','--sdk','macosx','--show-sdk-path'],text=True).strip()
write('environment.json',{k:v for k,v in os.environ.items() if k in ('CC','NANO_VM','NANO_NVM2C','NANO_AOT_RUNTIME','NANO_MODULE_PATH','NANO_BUILD_CACHE','PYTHONPATH','SDKROOT')})
(out/'runner.py').write_bytes(pathlib.Path(__file__).read_bytes())
initial=sources()
if not linux and phase in ('mutation','patterns'):
 free=shutil.disk_usage(root).free
 write('capacity-before.json',{'free_bytes':free,'minimum_bytes':2*1024**3})
 assert free>=2*1024**3
 failed=pathlib.Path(str(base)+'-mutation')
 old_sources=json.loads((failed/'source-before.json').read_text());assert old_sources==initial
 previous=json.loads((failed/'inputs-before.json').read_text());current=inputs()
 write('since-truncated-setup.json',{'added':{k:v for k,v in current.items() if k not in previous},'changed':{k:{'before':previous[k],'now':v} for k,v in current.items() if k in previous and v!=previous[k]},'missing':[k for k in previous if k not in current]})
 reuse=json.loads(pathlib.Path(str(base)+'-reuse/report.json').read_text())
 for name,item in reuse['stages'].items():assert sha(root/'bin'/name)==item['sha256']
write('source-before.json',initial);write('inputs-before.json',inputs());write('host-tools-before.json',tools())
status={'pin':pin,'phase':phase,'success':False};start=time.monotonic()
def command(name,args,timeout=1800):
 t=time.monotonic()
 with (out/(name+'.log')).open('wb') as log:
  code=subprocess.run(args,cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=timeout).returncode
 write(name+'.json',{'command':args,'status':code,'seconds':time.monotonic()-t})
 if code:raise RuntimeError(name+' failed with '+str(code))
def dependency(name,count=None):
 suffix='-recovered1' if not linux and name=='mutation' else ''
 d=json.loads(pathlib.Path(str(base)+'-'+name+suffix+'/status.json').read_text());assert d['success']
 if count is not None:assert d['testsRun']==count
try:
 if phase=='build':
  assert git('rev-parse','HEAD',tree=old)==oldpin and not git('status','--porcelain',tree=old)
  nano=git('ls-files','*.nano').splitlines()
  proof={p:sha(root/p) for p in nano};assert all(sha(old/p)==v for p,v in proof.items())
  write('unchanged-nano-sources.json',proof)
  (root/'bin').mkdir(exist_ok=True)
  stages={}
  for name in ('nanoc_stage1','nanoc_stage2'):
   source=old/'bin'/name;target=root/'bin'/name
   assert not target.exists();shutil.copy2(source,target)
   assert sha(source)==sha(target);stages[name]={'source':str(source),'target':str(target),'sha256':sha(source)}
  write('reused-stages.json',stages)
  setup=out/'setup.mk';setup.write_text('''.PHONY: owned-array-source-setup
owned-array-source-setup: nanoisa_emit nano_virt nano_vm nvm2c nanoisa_dump test-local-binding-metadata
\t$(CC) $(CFLAGS) -o obj/borrow_shadow_names tests/nanovirt/borrow_shadow_names.c $(NANOVIRT_OBJECTS) $(NANOVM_OBJECTS) $(NANOISA_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(LDFLAGS)
''')
  command('build',[make,'-j2','-f','Makefile.gnu','-f',str(setup),'CC='+cc,'stage1','owned-array-source-setup'])
 elif phase in ('eval-corrected2','lookup'):
  dependency('build')
  fixture=pathlib.Path(('/tmp/' if linux else '/private/tmp/')+'nanolang-eval-70c12-fixture.c')
  assert sha(fixture)=='b09487c2b23486282edd6cd28e1cc2eaeb5b6f78db595e7ca45388a98ecbec2f'
  (out/'tests').mkdir();(out/'src').symlink_to(root/'src',target_is_directory=True)
  overlay=out/'tests/test_eval.c';overlay.write_bytes(fixture.read_bytes())
  write('fixture.json',{'commit':'70c12fd68','sha256':sha(fixture),'path':str(fixture)})
  if phase=='lookup':
   driver=out/'tests/lookup.c'
   driver.write_text('#define main retained_full_evaluator_main\n#include "test_eval.c"\n#undef main\nint main(void) { test_eval_declared_push_initializer_bindings(); puts("I passed the unchanged declared push initializer control."); return 0; }\n')
   write('driver.json',{'path':str(driver),'sha256':sha(driver)})
   overlay=driver
  mk=out/'eval-overlay.mk'
  mk.write_text('eval-overlay: $(OBJ_DIR)/test_interpreter_ffi_native.so $(OBJ_DIR)/eval_io_faults.o $(OBJ_DIR)/eval_clock_test.o\n\t$(CC) $(CFLAGS) -o '+str(out/'test_eval')+' '+str(overlay)+' $(filter-out $(OBJ_DIR)/eval.o $(OBJ_DIR)/eval/eval_io.o,$(COMMON_OBJECTS)) $(OBJ_DIR)/eval_clock_test.o $(OBJ_DIR)/eval_io_faults.o $(RUNTIME_OBJECTS) $(LDFLAGS)\n')
  command('eval-build',[make,'-j2','-f','Makefile.gnu','-f',str(mk),'CC='+cc,'eval-overlay'])
  command('eval',[str(out/'test_eval')])
 else:
  if phase=='identity':
   dependency('build')
   if not linux:dependency('lookup')
   reuse=json.loads(pathlib.Path(str(base)+'-reuse/report.json').read_text())
   assert reuse['new_pin']==pin
   for name,item in reuse['stages'].items():assert sha(root/'bin'/name)==item['sha256']
   write('held-evaluator-dependency.json',{'status':'FAILED_PENDING_PR855','retained_linux_terminal':'/tmp/nanolang-declared-push-9f7-linux-eval-corrected2/status.json','acceptance_waived':False,'plan_commit':'d85425e89'})
   from tests import test_declared_array_push_identity as fixture
   cls=fixture.DeclaredArrayPushIdentity;suite=unittest.defaultTestLoader.loadTestsFromTestCase(cls);expected=6
  elif phase=='mutation':
   dependency('identity',6)
   from tests import test_source_owned_float_array_mutation as fixture
   cls=fixture.SourceOwnedFloatArrayMutation;original_setup=cls.setUpClass
   @classmethod
   def setup(c):
    t=time.monotonic()
    try:original_setup()
    finally:write('inputs-after-setup.json',inputs());write('setup.json',{'seconds':time.monotonic()-t,'work':str(getattr(c,'work',''))})
    os.environ['CC']=nativecc
    paths=[root/'bin'/n for n in ('nanoc_c','nanoc_stage1','nanoc_stage2','nano_virt','nano_vm','nvm2c','nanoisa','nanoisa_emit')]+c.emitters+c.shadow_tools
    write('producers-after-setup.json',{str(p):sha(p) for p in paths})
   cls.setUpClass=setup;suite=fixture.load_tests(None,None,None);expected=12
  else:
   assert phase=='patterns';dependency('mutation',12)
   from tests import test_owned_record_patterns as fixture
   cls=fixture.OwnedRecordPatterns;suite=unittest.defaultTestLoader.loadTestsFromTestCase(cls);expected=12
   import tempfile
   original_temp=tempfile.TemporaryDirectory
   class Retained(original_temp):
    def __exit__(self,*args):
     dest=out/('retained-%04d'%len(list(out.glob('retained-*'))));shutil.copytree(self.name,dest)
     return super().__exit__(*args)
   tempfile.TemporaryDirectory=Retained
   original_run=subprocess.run
   def retained_run(*args,**kwargs):
    result=original_run(*args,**kwargs)
    index=len(list(out.glob('command-*.json')))
    write('command-%04d.json'%index,{'args':[str(a) for a in args[0]],'status':result.returncode,'stdout':result.stdout.decode(errors='replace') if isinstance(result.stdout,bytes) else result.stdout,'stderr':result.stderr.decode(errors='replace') if isinstance(result.stderr,bytes) else result.stderr})
    return result
   subprocess.run=retained_run
  assert suite.countTestCases()==expected
  if phase!='mutation':os.environ['CC']=nativecc
  write('native-compiler.json',{'path':str(pathlib.Path(nativecc).resolve()),'sha256':sha(pathlib.Path(nativecc).resolve())})
  class Result(unittest.TextTestResult):
   def stopTest(self,test):
    super().stopTest(test)
    if hasattr(cls,'work'):
     dest=out/('artifacts-'+test._testMethodName);shutil.copytree(cls.work,dest)
     write('artifacts-'+test._testMethodName+'.json',{str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file()})
  result=unittest.TextTestRunner(verbosity=2,failfast=True,resultclass=Result).run(suite)
  status.update(testsRun=result.testsRun,errors=len(result.errors),failures=len(result.failures))
  if not result.wasSuccessful() or result.testsRun!=expected:raise RuntimeError('fixture gate failed')
 status['success']=True
except BaseException as error:
 status['error']=repr(error);traceback.print_exc()
finally:
 final=sources();write('source-after.json',final);write('inputs-after.json',inputs());write('host-tools-after.json',tools())
 status.update(seconds=time.monotonic()-start,sources_unchanged=initial==final,head_unchanged=git('rev-parse','HEAD')==pin)
 if not status['sources_unchanged'] or not status['head_unchanged']:status['success']=False
 write('status.json',status);print(json.dumps(status),flush=True)
raise SystemExit(0 if status['success'] else 1)
