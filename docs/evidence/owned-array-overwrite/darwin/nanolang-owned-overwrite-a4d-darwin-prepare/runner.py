import hashlib,json,os,pathlib,shlex,shutil,subprocess,sys,time,traceback
host,phase=sys.argv[1:]
linux=host=='linux'
root=pathlib.Path('/home/jkh/Src/nanolang-owned-array-overwrite-acceptance' if linux else '/private/tmp/nanolang-owned-array-overwrite-a4d3c0302')
base=pathlib.Path('/tmp' if linux else '/private/tmp')
prefix='nanolang-owned-overwrite-a4d-'+host
out=base/(prefix+'-'+phase);out.mkdir(exist_ok=False)
pin='a4d3c0302f7a0c3504ff6f9dc2f4f415925c6638'
cc='/usr/bin/gcc' if linux else '/usr/bin/clang'
if phase=='clang':cc='/usr/bin/clang-18 --gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13'
if phase=='sanitizer' and not linux:cc='/opt/homebrew/opt/llvm/bin/clang'
flags='-fsanitize=address,undefined -fno-omit-frame-pointer' if phase in ('sanitizer','clang') else ''
os.chdir(root)
def git(*args):return subprocess.check_output(['git',*args],text=True).strip()
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def save(name,value):(out/name).write_text(json.dumps(value,indent=2)+'\n')
def sources():return {p:sha(root/p) for p in git('ls-files').splitlines() if (root/p).is_file()}
def inputs():return {str(p.relative_to(root)):sha(p) for d in ('obj','bin','lib') for p in (root/d).rglob('*') if p.is_file()}
def tools():
 result={}
 for name in ('make','python3','ar','ld',shlex.split(cc)[0]):
  p=pathlib.Path(shutil.which(name)).resolve();result[name]={'path':str(p),'sha256':sha(p)}
 return result
assert git('rev-parse','HEAD')==pin and not git('status','--porcelain')
assert shutil.disk_usage(root).free>2*1024**3
for k in ('NANOC','NANO_VM','NANO_NVM2C','NANO_AOT_RUNTIME','NANO_MODULE_PATH','NANO_BUILD_CACHE'):os.environ.pop(k,None)
os.environ.update(CC=cc,OWNED_ARRAY_OVERWRITE_CFLAGS=flags,ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1',PYTHONPATH=str(root))
if not linux:os.environ['SDKROOT']=subprocess.check_output(['/usr/bin/xcrun','--sdk','macosx','--show-sdk-path'],text=True).strip()
save('environment.json',{k:os.environ[k] for k in ('CC','OWNED_ARRAY_OVERWRITE_CFLAGS','ASAN_OPTIONS','UBSAN_OPTIONS','PYTHONPATH','SDKROOT') if k in os.environ})
(out/'runner.py').write_bytes(pathlib.Path(__file__).read_bytes())
initial=sources();before=inputs();save('source-before.json',initial);save('inputs-before.json',before);save('tools-before.json',tools())
status={'pin':pin,'phase':phase,'success':False};start=time.monotonic()
def command(name,args,limit=1800):
 t=time.monotonic()
 with (out/(name+'.log')).open('wb') as f:p=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,timeout=limit)
 save(name+'.json',{'argv':args,'status':p.returncode,'seconds':time.monotonic()-t})
 if p.returncode:raise RuntimeError(name+' failed')
try:
 if phase=='prepare':
  mk=out/'prepare.mk';mk.write_text('overwrite-providers: $(NANOVM_OBJECTS) $(NANOISA_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS)\n')
  command('prepare',['make','-j2','-f','Makefile.gnu','-f',str(mk),'CC='+cc,'overwrite-providers'])
 else:
  assert json.loads((base/(prefix+'-prepare/status.json')).read_text())['success']
  if phase in ('sanitizer','clang'):assert json.loads((base/(prefix+'-ordinary/status.json')).read_text())['success']
  if phase=='clang':assert json.loads((base/(prefix+'-sanitizer/status.json')).read_text())['success']
  os.environ['OWNED_ARRAY_OVERWRITE_ARTIFACTS']=str(out/'artifacts')
  command('gate',['make','-f','Makefile.gnu','CC='+cc,'test-owned-array-overwrite'],600)
  log=(out/'gate.log').read_text();assert 'Ran 1 test' in log and '\nOK\n' in log
 status['success']=True
except BaseException as e:status['error']=repr(e);traceback.print_exc()
finally:
 after=sources();save('source-after.json',after);save('inputs-after.json',inputs());save('tools-after.json',tools())
 status.update(seconds=time.monotonic()-start,sources_unchanged=initial==after,head_unchanged=git('rev-parse','HEAD')==pin)
 if not status['sources_unchanged'] or not status['head_unchanged']:status['success']=False
 save('status.json',status);print(json.dumps(status),flush=True)
sys.exit(0 if status['success'] else 1)
