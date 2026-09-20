import hashlib,json,os,pathlib,subprocess,time,shutil
root=pathlib.Path('/home/jkh/Src/nanolang-record-first-tuple-qualified')
out=pathlib.Path('/tmp/nanolang-record-first-2132-acceptance')
out.mkdir(exist_ok=False)
bootstrap=json.loads(pathlib.Path('/tmp/nanolang-record-first-2132-bootstrap/manifest.json').read_text())
assert bootstrap['steps'][-1]['status']==0 and bootstrap['source_unchanged'] and bootstrap['tracked_tree_clean']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(*a):return subprocess.check_output(['git',*a],cwd=root,text=True).strip()
assert git('rev-parse','HEAD')==bootstrap['source'] and not git('status','--porcelain','--untracked-files=no')
files=git('ls-files','src','src_nano','tests','scripts','modules','stdlib','spec','Makefile','GNUmakefile','Makefile.gnu').splitlines()
def sources():return {p:sha(root/p) for p in files if (root/p).is_file()}
def tools():return {str(p.relative_to(root)):sha(p) for p in (root/'bin').glob('*') if p.is_file()}
before=sources();(out/'source-before.json').write_text(json.dumps(before,indent=2)+'\n');(out/'runner.py').write_bytes(pathlib.Path(__file__).read_bytes())
env=os.environ.copy()
for key in ('NANOC','NANO_VM','NANO_NVM2C','NANO_AOT_RUNTIME','NANO_BUILD_CACHE','NANO_MODULE_PATH'):env.pop(key,None)
env.update(NANO_BUILD_CACHE=str(root/'obj/module_cache'),NANO_VM=str(root/'bin/nano_vm'),NANO_NVM2C=str(root/'bin/nvm2c'),NANO_AOT_RUNTIME=str(root/'bin/nano_aot_runtime.o'),NANO_MODULE_PATH=str(root/'modules'),NMS_NATIVE_CLANG_FLAGS='--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13')
report={'source':bootstrap['source'],'steps':[]};status=1
steps=[('prepare',['make','-j2','nano_virt','nanoisa_emit','nano_vm','nvm2c','nanoisa_dump'])]
for compiler in ('nanoc_c','nanoc_stage1','nanoc_stage2'):
 exe=out/(compiler+'-parser')
 steps += [(compiler+'-parser-build',[root/'bin'/compiler,root/'tests/parser_parenthesized.nano','-o',exe]),(compiler+'-parser-run',[exe])]
# I retain temporary artifacts without changing any test or assertion.
wrapper=out/'retained-test.py'
wrapper.write_text('''import tempfile,sys,unittest\nfrom pathlib import Path\nsys.path.insert(0,'''+repr(str(root))+''')\nclass Retained(tempfile.TemporaryDirectory):\n def __init__(self,*a,**k):\n  super().__init__(*a,**k);self._finalizer.detach();print("RETAINED",self.name,flush=True)\n def cleanup(self):pass\ntempfile.TemporaryDirectory=Retained\nfrom tests.test_source_borrow_emission import SourceBorrowEmission\nsuite=unittest.TestSuite([SourceBorrowEmission("test_inline_owner_wrappers_refuse_without_ordinary_fallback")])\nresult=unittest.TextTestRunner(verbosity=2).run(suite)\nraise SystemExit(not result.wasSuccessful())\n''')
steps.append(('paired-owner-refusal',['python3',wrapper]))
try:
 for name,cmd in steps:
  start=time.monotonic();print('START',name,flush=True)
  try:
   with (out/(name+'.log')).open('wb') as log:r=subprocess.run([str(x) for x in cmd],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=2400)
   status=r.returncode
  except subprocess.TimeoutExpired:status=124
  report['steps'].append({'name':name,'command':[str(x) for x in cmd],'status':status,'seconds':round(time.monotonic()-start,3),'log_sha256':sha(out/(name+'.log'))})
  (out/(name+'-tools.json')).write_text(json.dumps(tools(),indent=2)+'\n');(out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
  print('END',name,status,report['steps'][-1]['seconds'],flush=True)
  if status:break
finally:
 after=sources();(out/'source-after.json').write_text(json.dumps(after,indent=2)+'\n');report.update(source_unchanged=before==after,head_unchanged=git('rev-parse','HEAD')==bootstrap['source']);(out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
raise SystemExit(status)
