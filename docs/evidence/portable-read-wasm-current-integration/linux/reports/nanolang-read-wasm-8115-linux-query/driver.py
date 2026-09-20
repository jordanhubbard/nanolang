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
darwin=sys.platform=='darwin'
clang='/opt/homebrew/opt/llvm/bin/clang' if darwin else '/usr/local/bin/clang'
linker='/opt/homebrew/opt/lld/bin/wasm-ld' if darwin else '/usr/local/bin/wasm-ld'
node='/opt/homebrew/bin/node' if darwin else '/home/linuxbrew/.linuxbrew/bin/node'
wheel=pathlib.Path(sys.argv[3]).resolve()
extra={n:str(pathlib.Path(p).resolve()) for n,p in [('clang',clang),('linker',linker),('node',node),('python',sys.executable),('wheel',wheel)]}
assert all(pathlib.Path(p).is_file() for p in extra.values())
def tools():return {n:{'path':p,'sha256':sha(p)} for n,p in extra.items()}
shutil.copyfile(__file__,report/'driver.py');write('tool-selection.json',extra)
if darwin:
 sdk=subprocess.check_output(['/usr/bin/xcrun','--show-sdk-path'],text=True).strip();write('sdk.json',{'path':sdk,'version':subprocess.check_output(['/usr/bin/xcrun','--show-sdk-version'],text=True).strip()})
else:sdk=None
phases=['neighbors']
results=[]
for name in phases:
 before=source();tb=tools();write(name+'-source-before.json',before);write(name+'-tools-before.json',tb)
 env=dict(os.environ,PORTABLE_WASM_CLANG=clang,PORTABLE_WASM_LD=linker,PORTABLE_WASM_NODE=node,PORTABLE_WASM_PYTHON=sys.executable,PORTABLE_WASM_WHEEL=str(wheel),PORTABLE_WASM_FLAGS='',PORTABLE_WASM_EXTRA_TOOLS=json.dumps(extra),PORTABLE_WASM_ARTIFACTS=str(report/name),ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1',LSAN_OPTIONS='',PYTHONDONTWRITEBYTECODE='1')
 env.pop('PYTHONOPTIMIZE',None)
 if sdk:env['SDKROOT']=sdk
 argv=[sys.executable,str(pathlib.Path(sys.argv[4]).resolve()),str(root),str(report/'retained')];write(name+'-command.json',{'argv':argv,'cwd':str(root),'env':{k:v for k,v in env.items() if k.startswith('PORTABLE_') or k in ['SDKROOT','ASAN_OPTIONS','UBSAN_OPTIONS','LSAN_OPTIONS']}})
 start=time.monotonic();p=None;status={'timeout':False,'errors':[],'signals':[]}
 with (report/(name+'.log')).open('wb') as log:
  try:
   p=subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:p.wait(timeout=1800)
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
