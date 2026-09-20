import hashlib,json,os,pathlib,shutil,subprocess,sys,time,unittest,traceback
platform,phase=sys.argv[1:]
linux=platform=='linux'
if not linux:os.environ['PATH']='/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin'
root=pathlib.Path('/home/jkh/Src/nanolang-affine-full-3c728c437' if linux else '/private/tmp/nanolang-affine-full-3c728c437-corrected')
old=pathlib.Path('/home/jkh/Src/nanolang-declared-push-canonical-identity' if linux else '/private/tmp/nanolang-declared-push-0c5421d9b')
base=pathlib.Path(('/tmp/' if linux else '/private/tmp/')+'nanolang-affine-full-3c728-corrected-'+platform)
out=pathlib.Path(str(base)+'-'+phase);out.mkdir(exist_ok=False)
pin='3c728c4371c6897f5ff08807adab1a528f3ca48d';oldpin='0c5421d9b51abf8a48a01839824b3bb98dff2f68'
cc='/usr/bin/gcc' if linux else '/usr/bin/clang'
nativecc=cc if linux else '/opt/homebrew/opt/llvm/bin/clang'
make='/usr/bin/make'
os.chdir(root);sys.path.insert(0,str(root))
def git(*args,tree=root):return subprocess.check_output(['git',*args],cwd=tree,text=True).strip()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,value):(out/name).write_text(json.dumps(value,indent=2)+'\n')
def sources():return {p:sha(root/p) for p in git('ls-files').splitlines() if (root/p).is_file()}
def inputs():return {str(p.relative_to(root)):sha(p) for d in ('bin','obj','lib') for p in (root/d).rglob('*') if p.is_file()}
def tools():
 result={}
 for name in ('cc','gcc','clang','make','python3','ar','ld',cc,nativecc):
  path=pathlib.Path(shutil.which(name)).resolve();result[name]={'path':str(path),'sha256':sha(path)}
 return result
assert shutil.disk_usage(root).free >= 4*1024**3
assert git('rev-parse','HEAD')==pin
assert git('status','--porcelain') in ('','?? tests/test_eval')
if (root/'tests/test_eval').exists():write('retained-evaluator.json',{'path':str(root/'tests/test_eval'),'sha256':sha(root/'tests/test_eval')})
for key in ('NANOC','NANO_VM','NANO_NVM2C','NANO_AOT_RUNTIME','NANO_BUILD_CACHE','NANO_MODULE_PATH'):os.environ.pop(key,None)
os.environ.update(ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',CC=cc,NANO_VM=str(root/'bin/nano_vm'),NANO_NVM2C=str(root/'bin/nvm2c'),NANO_AOT_RUNTIME=str(root/'bin/nano_aot_runtime.o'),NANO_MODULE_PATH=str(root/'modules'),NANO_BUILD_CACHE=str(root/'obj/module_cache'),PYTHONPATH=str(root))
if not linux:os.environ['SDKROOT']=subprocess.check_output(['/usr/bin/xcrun','--sdk','macosx','--show-sdk-path'],text=True).strip()
write('environment.json',{k:v for k,v in os.environ.items() if k in ('CC','NANO_VM','NANO_NVM2C','NANO_AOT_RUNTIME','NANO_MODULE_PATH','NANO_BUILD_CACHE','PYTHONPATH','SDKROOT','ASAN_OPTIONS')})
(out/'runner.py').write_bytes(pathlib.Path(__file__).read_bytes())
initial=sources();write('source-before.json',initial);write('inputs-before.json',inputs());write('host-tools-before.json',tools())
status={'pin':pin,'phase':phase,'success':False};start=time.monotonic()
def command(name,args,timeout=1800):
 t=time.monotonic()
 with (out/(name+'.log')).open('wb') as log:
  code=subprocess.run(args,cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=timeout).returncode
 write(name+'.json',{'command':args,'status':code,'seconds':time.monotonic()-t})
 if code:raise RuntimeError(name+' failed with '+str(code))
def dependency(name,count=None):
 d=json.loads(pathlib.Path(str(base)+'-'+name+'/status.json').read_text());assert d['success']
 if count is not None:assert d['testsRun']==count
try:
 assert phase=='module-identity'
 dependency('suite',33)
 for key in ('NANOLANG_AFFINE_COMPILERS','NANOLANG_AFFINE_COMPILER_ROOT'):os.environ.pop(key,None)
 dependency('build')
 prior=json.loads((pathlib.Path(str(base)+'-suite')/'inputs-after.json').read_text())
 assert inputs()==prior, 'Fresh suite providers changed before module identity'
 retention=out/'retention';retention.mkdir()
 shutil.copyfile(pathlib.Path(__file__).with_name('nanolang-affine-full-retention.py'),retention/'sitecustomize.py')
 os.environ['AFFINE_RETAIN_DIR']=str(out/'artifacts')
 os.environ['AFFINE_PROFILE_EVIDENCE_DIR']=str(out/'boundary-routes')
 os.environ['PYTHONPATH']=str(retention)+os.pathsep+str(root)
 write('suite-environment.json',{k:os.environ[k] for k in ('CC','PYTHONPATH','AFFINE_RETAIN_DIR','AFFINE_PROFILE_EVIDENCE_DIR')})
 command('module9',[sys.executable,'-m','unittest','-v','tests.test_affine_module_identity'],3600)
 log=(out/'module9.log').read_text()
 import re
 matches=re.findall(r'Ran (\d+) tests? in ',log)
 assert matches==['9'],matches
 assert re.search(r'\nOK\s*$',log),log[-2000:]
 status['testsRun']=9
 write('fresh-stages.json',{str(root/'bin'/n):sha(root/'bin'/n) for n in ('nanoc_c','nanoc_stage1','nanoc_stage2')})
 status['success']=True
except BaseException as error:
 status['error']=repr(error);traceback.print_exc()
finally:
 final=sources();write('source-after.json',final);write('inputs-after.json',inputs());write('host-tools-after.json',tools())
 status.update(seconds=time.monotonic()-start,sources_unchanged=initial==final,head_unchanged=git('rev-parse','HEAD')==pin)
 if not status['sources_unchanged'] or not status['head_unchanged']:status['success']=False
 write('status.json',status);print(json.dumps(status),flush=True)
raise SystemExit(0 if status['success'] else 1)
