import hashlib,json,os,pathlib,platform,shutil,signal,subprocess,sys,time
root=pathlib.Path(sys.argv[1]).resolve(); out=pathlib.Path(sys.argv[2]).resolve(); out.mkdir(exist_ok=False)
store=out/'objects';store.mkdir();darwin=platform.system()=='Darwin'
def git(*args):return subprocess.check_output(['git',*args],cwd=root,text=True).strip()
def save(name,data):(out/name).write_text(json.dumps(data,indent=2)+'\n')
def entry(path):
 p=pathlib.Path(path).resolve(); data=p.read_bytes();h=hashlib.sha256(data).hexdigest();d=store/h
 if not d.exists():d.write_bytes(data)
 return {'sha256':h,'bytes':len(data),'archive':str(d)}
def snapshot(paths):return {str(p):entry(p) for p in sorted({pathlib.Path(x).resolve() for x in paths},key=str) if p.is_file()}
pin=git('rev-parse','HEAD');assert pin.startswith('e0e85441f')
assert not git('status','--porcelain','--untracked-files=no')
# I retain and inventory existing ordinary providers from the first qualified build.
mode=os.environ.get('CAST_U8_MATRIX_MODE','ordinary');assert mode in ('ordinary','asan')
cc='/usr/bin/clang' if darwin else '/bin/gcc';native='/opt/homebrew/opt/llvm/bin/clang' if darwin else '/bin/gcc'
llvm='/opt/homebrew/opt/llvm/bin' if darwin else '/usr/local/bin'
native=llvm+'/clang' if darwin or mode=='asan' else cc
selected=[cc,native,sys.executable,shutil.which('python3'),shutil.which('make'),shutil.which('ar'),shutil.which('ld'),shutil.which('perl'),shutil.which('git')]
selected += [llvm+'/'+x for x in ('clang','opt','llc','lli','llvm-as')]
selected += [shutil.which(x) for x in ('wasm-ld','wasmtime','node')]
for p in selected:assert p and pathlib.Path(p).is_file(),p
tracked=git('ls-files').splitlines();tracked_set=set(tracked);source=[root/p for p in tracked if not p.startswith('docs/evidence/') and (root/p).is_file()]
save('tracked-paths.json',tracked);save('runner.json',entry(__file__))
source_initial=snapshot(source);tools_initial=snapshot(selected);save('source-initial.json',source_initial);save('tools-initial.json',tools_initial)
env=dict(os.environ)
for key in ('NANOC','NANOLANG_COMPILER','NANO_VM','NANO_NVM2C','NANO_AOT_RUNTIME','NANO_MODULE_PATH','NANO_BUILD_CACHE','CFLAGS','LDFLAGS','SDKROOT','DYLD_INSERT_LIBRARIES','LD_PRELOAD'):env.pop(key,None)
env.update(CC=cc,NANO_NATIVE_TEST_CC=native,NANOLANG_GUARD_SAN_CC=native,NMS_RUNTIME_CLANG=llvm+'/clang',NMS_RUNTIME_OPT=llvm+'/opt',NMS_NATIVE_CLANG_FLAGS='' if darwin else '--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13',ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1',LSAN_OPTIONS='',NANO_BUILD_CACHE=str(root/'obj/module_cache'),NANO_MODULE_PATH=str(root/'modules'))
if mode=='asan':
 env['NANO_NATIVE_TEST_CC']=native+('' if darwin else ' --gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13')
 env['NANO_CAST_U8_NATIVE_FLAGS']='-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer -g'
 env['NANO_CAST_U8_IR_ASAN']='1'
save('environment.json',{k:v for k,v in env.items() if k.startswith(('NANO','NMS','ASAN','UBSAN','LSAN')) or k in ('CC','PATH','SDKROOT')})
report={'head':pin,'host':platform.node(),'platform':platform.platform(),'matrix_mode':mode,'scope':'Selected generated native C ASan/UBSan and emitted native IR ASan when requested; ordinary LLVM/Wasm. Explicit target ABI corrected generated CAST_U8 native/LLVM/Wasm acceptance with inventoried first-build ordinary providers; source and reconstruction remain unimplemented','phases':[]};save('manifest.json',report)
def products():return snapshot(p for folder in ('bin','obj','build','tests') for p in (root/folder).rglob('*') if p.is_file() and str(p.relative_to(root)) not in tracked_set)
def command(label,argv,bound):
 sb=snapshot(source);tb=snapshot(selected);save(label+'-sources-before.json',sb);save(label+'-tools-before.json',tb);save(label+'-products-before.json',products())
 state={'name':label,'argv':argv,'bound_seconds':bound,'status':None,'timeout':False,'cleanup':[]};save(label+'-status.json',state);start=time.monotonic();print('START',label,flush=True)
 try:
  with (out/(label+'.log')).open('wb') as log:
   p=subprocess.Popen(argv,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:state['status']=p.wait(timeout=bound)
   except subprocess.TimeoutExpired:
    state['timeout']=True
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
 except BaseException as error:state['exception']=repr(error)
 state['seconds']=time.monotonic()-start;state['log']=entry(out/(label+'.log'));save(label+'-status.json',state)
 sa=snapshot(source);ta=snapshot(selected);save(label+'-sources-after.json',sa);save(label+'-tools-after.json',ta);save(label+'-products-after.json',products())
 state.update(sources_equal=sb==sa==source_initial,tools_equal=tb==ta==tools_initial)
 report['phases'].append(state);save('manifest.json',report);save(label+'-status.json',state);print('END',label,state['status'],round(state['seconds'],3),flush=True)
 return state['status']==0 and state.get('leader_reaped') and state.get('group_gone') and state['sources_equal'] and state['tools_equal']

for label,args in [('compiler',[cc,'--version']),('native-compiler',[native,'--version']),('llvm',[llvm+'/clang','--version']),('make',[shutil.which('make'),'--version'])]:
 if not command(label,args,30):sys.exit(1)
flags=['CC='+cc,'NMS_RUNTIME_CLANG='+llvm+'/clang','NMS_RUNTIME_OPT='+llvm+'/opt','NMS_NATIVE_CLANG_FLAGS='+env['NMS_NATIVE_CLANG_FLAGS']]
phases=[('scalar-neighbors',[sys.executable,'-m','unittest','-v','tests.test_nvm2llvm','tests.test_nvm2llvm_floats','tests.test_nvm2wasm','tests.test_scalar_u8'],900),('managed-neighbors',[sys.executable,'-m','unittest','-v','tests.test_llvm_managed_strings','tests.test_llvm_managed_mutable_arrays','tests.test_managed_binary64_bits','tests.test_managed_binary64_format'],1200)]
for name,args,bound in phases:
 if not command(name,args,bound):sys.exit(1)
save('terminal.json',{'status':'PASS','head':pin,'scope':report['scope']})
