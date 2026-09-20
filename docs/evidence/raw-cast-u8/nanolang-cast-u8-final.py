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
pin=git('rev-parse','HEAD');assert pin.startswith('565a48d3f')
assert not git('status','--porcelain','--untracked-files=no')
# I reuse inventoried ordinary providers; each fixture and VM TU rebuilds.
cc='/usr/bin/clang' if darwin else '/bin/gcc';native='/opt/homebrew/opt/llvm/bin/clang' if darwin else '/bin/gcc'
llvm='/opt/homebrew/opt/llvm/bin' if darwin else '/usr/local/bin'
os.environ['PATH']=llvm+':'+('/private/tmp/nanolang-mixed-counted-tools-v43/wasmtime-v43.0.0-aarch64-macos:' if darwin else '')+os.environ['PATH']
selected=[cc,native,sys.executable,shutil.which('python3'),shutil.which('make'),shutil.which('ar'),shutil.which('ld'),shutil.which('perl'),shutil.which('git')]
selected += [llvm+'/'+x for x in ('clang','opt','llc','lli')]
selected += [shutil.which(x) for x in ('llvm-as','wasm-ld','wasmtime','node')]
for p in selected:assert p and pathlib.Path(p).is_file(),p
tracked=git('ls-files').splitlines();tracked_set=set(tracked);source=[root/p for p in tracked if not p.startswith('docs/evidence/') and (root/p).is_file()]
save('tracked-paths.json',tracked);save('runner.json',entry(__file__))
source_initial=snapshot(source);tools_initial=snapshot(selected);save('source-initial.json',source_initial);save('tools-initial.json',tools_initial)
env=dict(os.environ)
for key in ('NANOC','NANOLANG_COMPILER','NANO_VM','NANO_NVM2C','NANO_AOT_RUNTIME','NANO_MODULE_PATH','NANO_BUILD_CACHE','CFLAGS','LDFLAGS','SDKROOT','DYLD_INSERT_LIBRARIES','LD_PRELOAD'):env.pop(key,None)
env.update(CC=cc,NANO_NATIVE_TEST_CC=native,NANOLANG_GUARD_SAN_CC=native,NMS_RUNTIME_CLANG=llvm+'/clang',NMS_RUNTIME_OPT=llvm+'/opt',NMS_NATIVE_CLANG_FLAGS='' if darwin else '--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13',ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1',LSAN_OPTIONS='',NANO_BUILD_CACHE=str(root/'obj/module_cache'),NANO_MODULE_PATH=str(root/'modules'))
save('environment.json',{k:v for k,v in env.items() if k.startswith(('NANO','NMS','ASAN','UBSAN','LSAN')) or k in ('CC','PATH','SDKROOT')})
report={'head':pin,'host':platform.node(),'platform':platform.platform(),'scope':'raw CAST_U8 VM, schema and existing byte backend neighbors; source conversion remains unimplemented','phases':[]};save('manifest.json',report)
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


baseflags='-Wall -Wextra -Werror -std=c99 -g -O1 -fPIC -Isrc -D_GNU_SOURCE'
configs=[('homebrew-ordinary',native,False),('homebrew-sanitizers',native,True)] if darwin else [('clang-ordinary',llvm+'/clang',False),('clang-sanitizers',llvm+'/clang',True)]
if not darwin: configs=[]
for label,compiler,sanitize in configs:
 flags=baseflags+(' --gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13' if not darwin else '')+(' -fsanitize=address,undefined -fno-omit-frame-pointer' if sanitize else '')
 ldflags='-lm'+(' -fsanitize=address,undefined' if sanitize else '')
 if not command(label,['make','test-cast-u8','CC='+compiler,'CFLAGS='+flags,'LDFLAGS='+ldflags],300):sys.exit(1)
if not command('byte-neighbors',[sys.executable,'-m','unittest','-v','tests.test_scalar_u8'],900):sys.exit(1)
if not command('reconstruction-tool',['make','nvm2hl','CC='+cc],300):sys.exit(1)
asm=out/'convert.nasm';module=out/'convert.nvm'
asm.write_text('.entry main\n.function main 0 0 0 int 1\nPUSH_I64 257\nCAST_U8\nPOP\nPUSH_I64 0\nRET\n.end\n')
if not command('assemble-refusal',[str(root/'bin/nanoisa'),'asm',str(asm),'-o',str(module)],30):sys.exit(1)
probe=out/'check-refusal.py'
probe.write_text("import pathlib,subprocess,sys\np=pathlib.Path(sys.argv[1]);p.write_bytes(b'preserve-output-sentinel\\n');r=subprocess.run(sys.argv[2:],stdout=subprocess.PIPE,stderr=subprocess.PIPE);sys.stdout.buffer.write(r.stdout);sys.stderr.buffer.write(r.stderr);assert r.returncode!=0,r.returncode;assert p.read_bytes()==b'preserve-output-sentinel\\n';print('I refused conversion and preserved existing output.')\n")
for name,args in [('native',[str(root/'bin/nvm2c')]),('llvm',[str(root/'bin/nvm2llvm')]),('wasm',[str(root/'bin/nvm2wasm')]),('reconstructed-c',[str(root/'bin/nvm2hl'),'--language','c']),('reconstructed-nano',[str(root/'bin/nvm2hl'),'--language','nano'])]:
 destination=out/(name+'.out')
 if not command(name+'-refusal',[sys.executable,str(probe),str(destination),*args,str(module),'-o',str(destination)],60):sys.exit(1)
save('terminal.json',{'status':'PASS','head':pin,'scope':'Selected rebuilt VM and fixture sanitizers plus unsupported translator output preservation; ordinary linked providers separately inventoried.'})
