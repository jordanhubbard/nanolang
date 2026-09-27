import hashlib,json,os,pathlib,platform,shlex,shutil,signal,subprocess,sys,time
root=pathlib.Path(sys.argv[1]).resolve(); out=pathlib.Path(sys.argv[2]).resolve(); out.mkdir(exist_ok=False)
store=out/'objects';store.mkdir();darwin=platform.system()=='Darwin'
if darwin:os.environ['PATH']='/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin'
def git(*args):return subprocess.check_output(['git',*args],cwd=root,text=True).strip()
def save(name,data):(out/name).write_text(json.dumps(data,indent=2)+'\n')
def entry(path):
 p=pathlib.Path(path).resolve(); data=p.read_bytes();h=hashlib.sha256(data).hexdigest();d=store/h
 if not d.exists():
  pools=[out.parent/'record-lists-144-linux-prepare/objects',out.parent/'record-lists-144-puck-prepare/objects',out.parent/'record-lists-06c-linux-prepare/objects',out.parent/'record-lists-06c-puck-prepare/objects',out.parent/'record-lists-833-linux-prepare/objects',out.parent/'record-lists-833-puck-prepare/objects',out.parent/'nanolang-record-lists-0e72-linux-prepare/objects',out.parent/'nanolang-record-lists-0e72-puck-prepare/objects',out.parent/'nanolang-record-lists-cb489-linux-prepare/objects',out.parent/'nanolang-record-lists-cb489-puck-prepare/objects',out.parent/'nanolang-record-lists-c923-linux-prepare/objects',out.parent/'nanolang-record-lists-c923-puck-prepare/objects',out.parent/'nanolang-record-lists-f54-linux-prepare/objects',out.parent/'nanolang-record-lists-f54-puck-prepare/objects',out.parent/'nanolang-record-lists-81cd-linux-prepare/objects',out.parent/'nanolang-record-lists-81cd-path-puck-prepare/objects',out.parent/'nanolang-record-lists-a448-linux-prepare/objects',out.parent/'nanolang-record-lists-a448-puck-prepare/objects',out.parent/'nanolang-record-lists-6cf-linux-matrix/objects',out.parent/'nanolang-record-lists-6cf-puck-matrix/objects',out.parent/'nanolang-record-lists-50de-linux-prepare/objects',out.parent/'nanolang-record-lists-50de-puck-prepare/objects',out.parent/'nanolang-record-lists-132f-linux-prepare/objects',out.parent/'nanolang-record-lists-132f-puck-prepare/objects',out.parent/'nanolang-record-lists-4382-linux-prepare/objects',out.parent/'nanolang-record-lists-4382-puck-prepare/objects',out.parent/'nanolang-record-lists-40870-linux-prepare/objects',out.parent/'nanolang-record-lists-40870-puck-prepare/objects',out.parent/'nanolang-record-lists-9f4e-linux-prepare/objects',out.parent/'nanolang-record-lists-9f4e-puck-prepare/objects',out.parent/'nanolang-record-lists-cc606-linux-prepare/objects',out.parent/'nanolang-record-lists-b915-linux-prepare/objects',out.parent/'nanolang-record-lists-cc606-puck-prepare/objects',out.parent/'nanolang-record-lists-b915-puck-prepare/objects']
  prior=next((pool/h for pool in pools if (pool/h).is_file()),None)
  if prior is not None:
   assert hashlib.sha256(prior.read_bytes()).hexdigest()==h
   os.link(prior,d)
  else:d.write_bytes(data)
 return {'sha256':h,'bytes':len(data),'archive':str(d)}
def snapshot(paths):return {str(p):entry(p) for p in sorted({pathlib.Path(x).resolve() for x in paths},key=str) if p.is_file()}
save('capacity-preflight.json',{'free_bytes':shutil.disk_usage(root).free,'minimum_bytes':2147483648})
assert shutil.disk_usage(root).free >= 2147483648, 'I require 2 GiB before preparation.'
pin=git('rev-parse','HEAD');assert pin=='73b61d06f04cf291a1960dc0b3e2b583ffefcb44'
assert not git('status','--porcelain','--untracked-files=no')
assert not (root/'obj').exists() and not (root/'bin').exists()
cc='/usr/bin/clang' if darwin else '/bin/gcc';native='/opt/homebrew/opt/llvm/bin/clang' if darwin else '/bin/gcc'
llvm='/opt/homebrew/opt/llvm/bin' if darwin else '/usr/local/bin'
selected=[cc,native,sys.executable,shutil.which('python3'),shutil.which('make'),shutil.which('ar'),shutil.which('ld'),shutil.which('perl'),shutil.which('git')]
selected += [llvm+'/'+x for x in ('clang','opt','llc','lli')]
sdk=None
if darwin:
 sdk=subprocess.check_output(['/usr/bin/xcrun','--show-sdk-path'],text=True).strip()
 assert pathlib.Path(sdk).is_dir(),sdk
 selected += ['/usr/bin/xcrun',str(pathlib.Path(sdk)/'SDKSettings.json')]
 save('sdk-selection.json',{'argv':['/usr/bin/xcrun','--show-sdk-path'],'path':sdk,'resolved':str(pathlib.Path(sdk).resolve())})
if darwin:selected += ['/opt/homebrew/bin/brew','/opt/homebrew/bin/pkg-config','/opt/homebrew/opt/openssl@3/include/openssl/sha.h','/opt/homebrew/opt/openssl@3/lib/libcrypto.dylib']
for p in selected:assert p and pathlib.Path(p).is_file(),p
tracked=git('ls-files').splitlines();tracked_set=set(tracked);source=[root/p for p in tracked if not p.startswith('docs/evidence/') and (root/p).is_file()]
save('sparse-policy.json',{'excluded':['/docs/evidence/'],'active_sources_complete':all((root/p).is_file() or (root/p).is_symlink() for p in tracked if not p.startswith('docs/evidence/'))});assert all((root/p).is_file() or (root/p).is_symlink() for p in tracked if not p.startswith('docs/evidence/'));save('tracked-paths.json',tracked);save('runner.json',entry(__file__))
source_initial=snapshot(source);tools_initial=snapshot(selected);save('source-initial.json',source_initial);save('tools-initial.json',tools_initial)
env=dict(os.environ)
for key in ('NANOC','NANOLANG_COMPILER','NANO_VM','NANO_NVM2C','NANO_AOT_RUNTIME','NANO_MODULE_PATH','NANO_BUILD_CACHE','CFLAGS','LDFLAGS','SDKROOT','DYLD_INSERT_LIBRARIES','LD_PRELOAD'):env.pop(key,None)
env.update(CC=cc,NANO_NATIVE_TEST_CC=native,NANOLANG_GUARD_SAN_CC=native,NMS_RUNTIME_CLANG=llvm+'/clang',NMS_RUNTIME_OPT=llvm+'/opt',NMS_NATIVE_CLANG_FLAGS='' if darwin else '--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13',ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1',LSAN_OPTIONS='',NANO_BUILD_CACHE=str(root/'obj/module_cache'),NANO_MODULE_PATH=str(root/'modules'))
if darwin:
 env.update(SDKROOT=sdk,NANO_NATIVE_TEST_CC=shlex.join([native,'-isysroot',sdk]),NANOLANG_GUARD_SAN_CC=shlex.join([native,'-isysroot',sdk]),NMS_NATIVE_CLANG_FLAGS=shlex.join(['-isysroot',sdk]))
save('environment.json',{k:v for k,v in env.items() if k.startswith(('NANO','NMS','ASAN','UBSAN','LSAN')) or k in ('CC','PATH','SDKROOT')})
report={'head':pin,'host':platform.node(),'platform':platform.platform(),'scope':'fresh ordinary COMMON_OBJECTS/RUNTIME_OBJECTS for focused parser/checker/lifetime fixtures only; no build/bootstrap/native corpus claim','phases':[]};save('manifest.json',report)
def products():return snapshot(p for folder in ('bin','obj','build','tests') for p in (root/folder).rglob('*') if p.is_file() and str(p.relative_to(root)) not in tracked_set)
def command(label,argv,bound):
 sb=snapshot(source);tb=snapshot(selected);save(label+'-sources-before.json',sb);save(label+'-tools-before.json',tb);save(label+'-products-before.json',products())
 state={'name':label,'argv':argv,'bound_seconds':bound,'status':None,'timeout':False,'cleanup':[]};save(label+'-status.json',state);start=time.monotonic();print('START',label,flush=True)
 try:
  with (out/(label+'.log')).open('wb') as log:
   p=subprocess.Popen(argv,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:
    deadline=time.monotonic()+bound
    while True:
     free=shutil.disk_usage(root).free
     with (out/'capacity-monitor.jsonl').open('a') as capacity:
      capacity.write(json.dumps({'phase':label,'elapsed':time.monotonic()-start,'free_bytes':free})+'\n')
     if free < 1073741824:
      state['capacity_stop']={'free_bytes':free,'minimum_bytes':1073741824}
      raise subprocess.TimeoutExpired(argv,bound)
     remaining=deadline-time.monotonic()
     if remaining <= 0: raise subprocess.TimeoutExpired(argv,bound)
     try:state['status']=p.wait(timeout=min(10,remaining));break
     except subprocess.TimeoutExpired:
      if time.monotonic()>=deadline:raise
   except subprocess.TimeoutExpired:
    state['timeout']='capacity_stop' not in state
    for sig in (signal.SIGTERM,signal.SIGKILL):
     try:os.killpg(p.pid,sig);state['cleanup'].append({'signal':sig.name,'sent':True})
     except ProcessLookupError:state['cleanup'].append({'signal':sig.name,'absent':True})
     try:p.wait(timeout=5)
     except subprocess.TimeoutExpired:state['cleanup'].append({'wait_expired':True})
    state['status']=124
   state['leader_reaped']=p.poll() is not None
   try:os.killpg(p.pid,0)
   except ProcessLookupError:state['group_gone']=True
   else:
    state['group_gone']=False
    try:os.killpg(p.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    try:p.wait(timeout=5)
    except subprocess.TimeoutExpired:state['cleanup'].append({'final_wait_expired':True})
 except BaseException as error:state['exception']=repr(error)
 state['seconds']=time.monotonic()-start;state['log']=entry(out/(label+'.log'));save(label+'-status.json',state)
 sa=snapshot(source);ta=snapshot(selected);save(label+'-sources-after.json',sa);save(label+'-tools-after.json',ta);save(label+'-products-after.json',products())
 state.update(sources_equal=sb==sa==source_initial,tools_equal=tb==ta==tools_initial)
 report['phases'].append(state);save('manifest.json',report);save(label+'-status.json',state);print('END',label,state['status'],round(state['seconds'],3),flush=True)
 return state['status']==0 and state.get('leader_reaped') and state.get('group_gone') and state['sources_equal'] and state['tools_equal']
for label,args in [('compiler',[cc,'--version']),('native-compiler',[native,'--version']),('llvm',[llvm+'/clang','--version']),('make',[shutil.which('make'),'--version'])]:
 if not command(label,args,30):sys.exit(1)
flags=['CC='+cc,'NMS_RUNTIME_CLANG='+llvm+'/clang','NMS_RUNTIME_OPT='+llvm+'/opt','NMS_NATIVE_CLANG_FLAGS='+env['NMS_NATIVE_CLANG_FLAGS']]
fragment=out/'prepare.mk'
fragment.write_text('include Makefile.gnu\n.PHONY: list-prepare\nlist-prepare: $(COMMON_OBJECTS) $(RUNTIME_OBJECTS)\n' + "\t@printf '%s\\n' '$(COMMON_OBJECTS) $(RUNTIME_OBJECTS)' > " + str(out/'providers.txt') + '\n' + "\t@printf '%s\\n' '$(CFLAGS)' > " + str(out/'cflags.txt') + '\n' + "\t@printf '%s\\n' '$(LDFLAGS)' > " + str(out/'ldflags.txt') + '\n')
save('prepare-fragment.json',entry(fragment))
for name,args,bound in [('provider-prepare',['make','-f',str(fragment),'-j2','list-prepare',*flags],1800)]:
 if not command(name,args,bound):sys.exit(1)
save('terminal.json',{'status':'PASS','head':pin,'scope':report['scope']})
