import hashlib,json,os,pathlib,shlex,shutil,signal,subprocess,sys,time
root=pathlib.Path(sys.argv[1]).resolve(); evidence=pathlib.Path(sys.argv[2]).resolve(); platform=sys.argv[3]
evidence.mkdir(parents=True,exist_ok=False); store=evidence/'objects';store.mkdir()
def dump(name,value): (evidence/name).write_text(json.dumps(value,indent=2)+'\n')
def item(p):
 p=pathlib.Path(p).resolve();data=p.read_bytes();h=hashlib.sha256(data).hexdigest();dest=store/h
 if not dest.exists():dest.write_bytes(data)
 return {'sha256':h,'bytes':len(data),'archive':str(dest)}
def snapshot(paths):return {str(p):item(p) for p in sorted(set(pathlib.Path(p).resolve() for p in paths),key=str)}
manifest=[]
def command(label,argv,env=None,timeout=180):
 state={'label':label,'argv':list(map(str,argv)),'timeout':False,'cleanup_complete':True,'returncode':None};start=time.monotonic()
 dump(label+'-command.json',state)
 try:
  with (evidence/(label+'.log')).open('wb') as output:
   proc=subprocess.Popen(state['argv'],cwd=root,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
   try:state['returncode']=proc.wait(timeout=timeout)
   except subprocess.TimeoutExpired:
    state['timeout']=True
    for sig in (signal.SIGTERM,signal.SIGKILL):
     try:os.killpg(proc.pid,sig)
     except ProcessLookupError:pass
     try:proc.wait(timeout=5)
     except subprocess.TimeoutExpired:continue
    state['returncode']=124;state['child_returncode']=proc.poll();state['cleanup_complete']=proc.poll() is not None
   try:os.killpg(proc.pid,0)
   except ProcessLookupError:pass
   else:
    state['cleanup_complete']=False
    try:os.killpg(proc.pid,signal.SIGKILL)
    except ProcessLookupError:pass
 except OSError as error:state['error']=repr(error);state['cleanup_complete']=False
 state['seconds']=time.monotonic()-start;dump(label+'-status.json',state);manifest.append(state);dump('manifest.json',manifest)
 print(label,state['returncode'],round(state['seconds'],3),flush=True)
 return state['returncode']==0 and state['cleanup_complete']
qualified=pathlib.Path(sys.argv[4]).resolve();python=sys.executable
ordinary='/bin/gcc' if platform=='linux' else '/usr/bin/clang'
native=ordinary if platform=='linux' else '/opt/homebrew/opt/llvm/bin/clang'
clang='/usr/local/bin/clang' if platform=='linux' else '/opt/homebrew/opt/llvm/bin/clang'
opt='/usr/local/bin/opt' if platform=='linux' else '/opt/homebrew/opt/llvm/bin/opt'
make=shutil.which('make');tools=[ordinary,native,clang,opt,python,make,shutil.which('ar'),shutil.which('ld'),shutil.which('git')]
tracked=subprocess.check_output(['git','ls-files'],cwd=root,text=True).splitlines();source=[root/p for p in tracked if not p.startswith('docs/evidence/')]
providers=[root/pathlib.Path(p) for p in shlex.split((qualified/'providers.txt').read_text())];providers.extend([root/'obj/nanovm/vm.o',root/'obj/nanoisa/service_bindings_module.o'])
assert all(p.is_file() for p in providers), [str(p) for p in providers if not p.is_file()]
flags=shlex.split((qualified/'cflags.txt').read_text());libs=shlex.split((qualified/'ldflags.txt').read_text())
baseenv=dict(os.environ,CC=ordinary,ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1',LSAN_OPTIONS='',NANO_VM_TRACE='0',NANO_MODULE_PATH=str(root/'modules'),NANO_BUILD_CACHE=str(root/'obj/module_cache'))
baseenv['NMS_RUNTIME_CLANG']=clang;baseenv['NMS_RUNTIME_OPT']=opt
baseenv['NMS_NATIVE_CLANG_FLAGS']='--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13' if platform=='linux' else ''
dump('llvm-selection.json',{k:baseenv[k] for k in ('NMS_RUNTIME_CLANG','NMS_RUNTIME_OPT','NMS_NATIVE_CLANG_FLAGS')})
dump('driver.json',item(__file__));dump('configuration.json',{'ordinary':ordinary,'native':native,'provider_source':str(qualified),'sanitizer_scope':'native generated mixed harness; no full-provider sanitizer claim','legacy_subprocess_scope':'outer command/status and final products; unchanged child test runners retain their own diagnostics'})
# Include previously measured selected compiler/runtime identities in this scope.
previous=json.loads((qualified/'prepare-after.json').read_text())
for name in previous:
 p=pathlib.Path(name)
 if not p.is_relative_to(root) and p.is_file():tools.append(p)
baseline=snapshot(source+tools+providers);dump('initial-inputs.json',baseline)
def products():return snapshot([p for d in (root/'bin',root/'obj',evidence/'products') for p in d.rglob('*') if p.is_file()])
(evidence/'products').mkdir()
def phase(label,argv,env=baseenv,bound=600):
 inputs=source+tools+providers+([] if label=='prepare-clis' else [p for p in (root/'bin').glob('*') if p.is_file()])
 before=snapshot(inputs);dump(label+'-before.json',before);dump(label+'-products-before.json',products())
 ok=command(label,argv,env,bound)
 after=snapshot(inputs);dump(label+'-after.json',after);dump(label+'-products-after.json',products())
 if not ok or before!=after:sys.exit(1)
phase('prepare-clis',[make,'-j4','CC='+ordinary,'nano_virt','nano_vm','nvm2c','nvm2llvm','nvm2wasm','nvm2hl','mixed-samples-runtime-fixture'],bound=900)
shared=['-dynamiclib'] if platform!='linux' else ['-shared']
for name,src,extra in [('ffi_callback_fixture','tests/nanovm/ffi_callback_fixture.c',['-pthread']),('ffi_artifact_first','tests/nanovm/ffi_artifact_fixture.c',['-DARTIFACT_ANSWER=42']),('ffi_artifact_second','tests/nanovm/ffi_artifact_identity_fixture.c',['-DARTIFACT_ANSWER=43'])]:
 phase(name+'-build',[ordinary,*flags,'-fPIC',*shared,*extra,src,'-o',root/'obj'/(name+'.so'),*libs])
for name,src,dependencies in [('callback-runtime','tests/nanovm/test_callback_runtime.c',['src/runtime/callback_runtime.c']),('callback-failures','tests/nanovm/test_callback_failures.c',[]),('vm-ffi','tests/nanovm/test_vm_ffi.c',providers)]:
 binary=evidence/'products'/name
 phase(name+'-build',[ordinary,*flags,'-UNDEBUG','-pthread','-Isrc/nanovm','-Isrc/nanoisa',src,*dependencies,*libs,'-o',binary])
 phase(name+'-run',[binary])
phase('callback-allocation',[make,'CC='+ordinary,'test-vm-callback-allocation'])
phase('mixed-admission',[make,'CC='+ordinary,'test-mixed-samples-admission','test-mixed-samples-runtime-alloc'])
mixed=evidence/'products'/'mixed';mixed.mkdir()
phase('mixed-query',[make,'CC='+ordinary,'test-mixed-samples'])
phase('owner-authority',[make,'CC='+ordinary,'test-owned-array-authority'])
phase('mixed-runtime',[python,'-m','unittest','-fv','tests.test_mixed_samples_runtime'],dict(baseenv,CC=native,NANO_MIXED_RUNTIME_DIR=str(mixed)))
owner_providers=[str(p) for p in providers if p.name not in ('vm.o','heap.o','nvm2c.o')]
phase('owner-array-public',[python,'-m','unittest','-fv','tests.test_private_owned_array_runtime'],dict(baseenv,CC=ordinary,NANO_OWNER_ARRAY_PUBLIC_TEST='1',PRIVATE_OWNER_ARRAY_OBJECTS=shlex.join(owner_providers),PRIVATE_OWNER_ARRAY_LDFLAGS=shlex.join(libs),PRIVATE_OWNER_ARRAY_ARTIFACTS=str(evidence/'products'/'owner')))
phase('ownership-contracts',[make,'CC='+ordinary,'test-ownership-contracts'])
phase('affine-bytecode',[make,'CC='+ordinary,'test-affine-bytecode'])
phase('declaration-projection',[make,'CC='+ordinary,'test-ownership-declaration-projection'])
phase('emitter',[root/'bin/nano_virt','src_nano/nanoisa_emit.nano','--emit-nvm','--strip-debug','-o',evidence/'products'/'emitter.nvm'],bound=120)
dump('terminal.json',{'status':'PASS','scope':'actual callback/FFI, mixed lifecycle/admission, owner ARRAY public/refusal plus original full imported emitter ten-second policy'})
