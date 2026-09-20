import os,sys,json,pathlib,hashlib,subprocess,shutil,signal,time
root=pathlib.Path(sys.argv[1]).resolve();report=pathlib.Path(sys.argv[2]).resolve();report.mkdir(exist_ok=False);os.chdir(root)
def write(n,x):(report/n).write_text(json.dumps(x,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
tracked=(root/'.qualification-tracked').read_text().splitlines() if (root/'.qualification-tracked').exists() else subprocess.check_output(['git','ls-files'],text=True).splitlines()
def source():return {p:sha(root/p) for p in tracked if (root/p).is_file()}
darwin=sys.platform=='darwin';clang='/opt/homebrew/opt/llvm/bin/clang' if darwin else '/usr/local/bin/clang';ordinary='/usr/bin/clang' if darwin else '/usr/bin/gcc';extra={}
llvmflags=[] if darwin else ['--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13']
for name,p in [('ordinary',ordinary),('llvm',clang),('python',sys.executable),('ld',shutil.which('ld')),('ar',shutil.which('ar'))]:extra[name]=str(pathlib.Path(p).resolve())
if darwin:
 extra['asan']='/opt/homebrew/Cellar/llvm/23.1.1/lib/clang/23/lib/darwin/libclang_rt.asan_osx_dynamic.dylib';extra['xcrun']='/usr/bin/xcrun'
else:
 for name in ['libasan.so','libubsan.so','cc1','collect2']:
  query='-print-prog-name=' if name in ('cc1','collect2') else '-print-file-name='
  p=subprocess.check_output([ordinary,query+name],text=True).strip();assert pathlib.Path(p).is_file();extra[name]=str(pathlib.Path(p).resolve())
 for name in ['libclang_rt.asan.so','libclang_rt.asan.a','libclang_rt.asan-preinit.a','libclang_rt.asan_static.a','libclang_rt.ubsan_standalone.a']:
  extra[name]='/usr/local/lib/clang/23/lib/aarch64-unknown-linux-gnu/'+name
assert all(pathlib.Path(p).is_file() for p in extra.values())
def tools():return {n:{'path':p,'sha256':sha(p)} for n,p in extra.items()}
shutil.copyfile(__file__,report/'driver.py');write('tool-selection.json',extra)
if darwin:
 sdk=subprocess.check_output(['/usr/bin/xcrun','--show-sdk-path'],text=True).strip();write('sdk.json',{'path':sdk,'version':subprocess.check_output(['/usr/bin/xcrun','--show-sdk-version'],text=True).strip()})
else:sdk=None
san=['-fsanitize=address,undefined','-fno-omit-frame-pointer']
phases=[('ordinary',ordinary,[]),('clang-ordinary',clang,llvmflags),('clang-sanitizer',clang,llvmflags+san)]
if not darwin:phases.insert(1,('gcc-sanitizer',ordinary,san))
results=[]
for name,cc,flags in phases:
 before=source();tb=tools();write(name+'-source-before.json',before);write(name+'-tools-before.json',tb)
 env=dict(os.environ,PORTABLE_ADAPTER_CC=cc,PORTABLE_ADAPTER_CLANG=clang,PORTABLE_ADAPTER_CFLAGS=' '.join(flags),PORTABLE_ADAPTER_LLVM_FLAGS=' '.join(llvmflags),PORTABLE_ADAPTER_EXTRA_TOOLS=json.dumps(extra),PORTABLE_ADAPTER_ARTIFACTS=str(report/name),ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1',LSAN_OPTIONS='')
 if sdk:env['SDKROOT']=sdk
 argv=[sys.executable,'-m','unittest','-f','-v','tests.test_portable_read_adapters'];write(name+'-command.json',{'argv':argv,'cwd':str(root),'env':{k:v for k,v in env.items() if k.startswith('PORTABLE_') or k in ['SDKROOT','ASAN_OPTIONS','UBSAN_OPTIONS','LSAN_OPTIONS']}})
 start=time.monotonic();p=None;status={'timeout':False,'errors':[],'signals':[]}
 with (report/(name+'.log')).open('wb') as log:
  try:
   p=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:p.wait(timeout=600)
   except subprocess.TimeoutExpired:status['timeout']=True
  except Exception as err:status['errors'].append(repr(err))
  finally:
   def exists():
    if p is None:return False
    try:os.killpg(p.pid,0);return True
    except ProcessLookupError:return False
    except OSError as err:status['errors'].append(repr(err));return True
   for sig in [signal.SIGTERM,signal.SIGKILL]:
    if not exists():break
    try:os.killpg(p.pid,sig);status['signals'].append(sig.name)
    except ProcessLookupError:pass
    deadline=time.monotonic()+5
    while time.monotonic()<deadline:
     p.poll()
     if not exists():break
     time.sleep(.05)
   status.update(returncode=p.poll() if p else None,leader_reaped=p is not None and p.returncode is not None,group_disappeared=not exists(),seconds=round(time.monotonic()-start,3));write(name+'-terminal.json',status)
 after=source();ta=tools();write(name+'-source-after.json',after);write(name+'-tools-after.json',ta);assert before==after and tb==ta
 results.append({'phase':name,**status});write('results.json',results);print(json.dumps(results[-1]),flush=True)
 if status['returncode']!=0 or status['timeout'] or status['errors'] or status['signals'] or not status['group_disappeared']:sys.exit(1)
