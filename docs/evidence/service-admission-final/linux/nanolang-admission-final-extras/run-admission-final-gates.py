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
python=sys.executable
if platform=='linux':
 ordinary='/bin/gcc';other='/usr/local/bin/clang';otherflags=['--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13']
 configs=[('gcc-ordinary',ordinary,[]),('gcc-sanitizer',ordinary,['-fsanitize=address,undefined','-fno-omit-frame-pointer']),('clang-sanitizer',other,[*otherflags,'-fsanitize=address,undefined','-fno-omit-frame-pointer'])]
else:
 ordinary='/usr/bin/clang';other='/opt/homebrew/opt/llvm/bin/clang';otherflags=[]
 configs=[('apple-ordinary',ordinary,[]),('homebrew-sanitizer',other,['-fsanitize=address,undefined','-fno-omit-frame-pointer'])]
configs=configs[:1]
make=shutil.which('make');tools=[ordinary,other,python,make,shutil.which('git'),shutil.which('ar'),shutil.which('ld')]
tracked=subprocess.check_output(['git','ls-files'],cwd=root,text=True).splitlines()
source=[root/p for p in tracked if not p.startswith('docs/evidence/')]
fragment=evidence/'prepare.mk';fragment.write_text("include Makefile.gnu\n.PHONY: admission-prepare\nadmission-prepare: $(NANOVM_OBJECTS) $(NANOISA_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS)\n\t@printf '%s\\n' '$(filter-out $(OBJ_DIR)/nanovm/vm.o,$(NANOVM_OBJECTS)) $(filter-out $(OBJ_DIR)/nanoisa/service_bindings_module.o,$(NANOISA_OBJECTS)) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS)' > "+str(evidence/'providers.txt')+"\n\t@printf '%s\\n' '$(CFLAGS)' > "+str(evidence/'cflags.txt')+"\n\t@printf '%s\\n' '$(LDFLAGS)' > "+str(evidence/'ldflags.txt')+"\n")
with fragment.open('a') as stream:
 stream.write("\t@printf '%s\\n' '$(filter-out $(OBJ_DIR)/nanoisa/affine_state.o $(OBJ_DIR)/nanoisa/nvm_v2_layouts.o,$(NANOISA_OBJECTS)) $(NANOISA_UTF8)' > "+str(evidence/'variant-providers.txt')+"\n")
dump('driver.json',item(__file__));dump('prepare-fragment.json',item(fragment));dump('configuration.json',{'platform':platform,'configs':configs,'source_scope':'all tracked files excluding historical docs/evidence','tools_scope':'named compiler, runtimes/subtools, Python, Make, Git, ar, ld; not complete transitive system tools','sanitizers':'constructor fixture includes affine TU plus rebuilt layout decoder; admission fixture includes VM plus observed service TU; other providers ordinary'})
for label,args in [('host',['uname','-a']),('head',['git','rev-parse','HEAD']),('ordinary-version',[ordinary,'--version']),('other-version',[other,*otherflags,'--version'])]:
 if not command(label,args):sys.exit(1)
if platform!='linux':
 for label,args in [('sdk',['xcrun','--show-sdk-path']),('sdk-version',['xcrun','--show-sdk-version'])]:
  if not command(label,args):sys.exit(1)
queries=[('ordinary-asan',[ordinary,'-print-file-name=libasan.so']),('ordinary-ubsan',[ordinary,'-print-file-name=libubsan.so']),('ordinary-cc1',[ordinary,'-print-prog-name=cc1']),('ordinary-collect2',[ordinary,'-print-prog-name=collect2'])] if platform=='linux' else []
queries += [('other-asan',[other,*otherflags,'-print-file-name='+('libclang_rt.asan.so' if platform=='linux' else 'libclang_rt.asan_osx_dynamic.dylib')])]
if platform=='linux':
 queries += [('other-asan-static',[other,*otherflags,'-print-file-name=libclang_rt.asan.a']),('other-asan-preinit',[other,*otherflags,'-print-file-name=libclang_rt.asan-preinit.a']),('other-ubsan-static',[other,*otherflags,'-print-file-name=libclang_rt.ubsan_standalone.a'])]
for label,args in queries:
 if not command(label,args):sys.exit(1)
 path=(evidence/(label+'.log')).read_text().strip()
 if pathlib.Path(path).is_file():tools.append(path)
 else:raise RuntimeError('I require exact runtime/tool path: '+path)
before=snapshot(source+tools);dump('prepare-before.json',before)
ok=command('prepare',[make,'-f',str(fragment),'-j4','CC='+ordinary,'admission-prepare'],timeout=900)
after=snapshot(source+tools);dump('prepare-after.json',after)
if not ok or before!=after:sys.exit(1)
providers=[root/p for p in shlex.split((evidence/'providers.txt').read_text())]
ldflags=shlex.split((evidence/'ldflags.txt').read_text());cflags=shlex.split((evidence/'cflags.txt').read_text())
all_providers=providers+[root/'obj/nanovm/vm.o',root/'obj/nanoisa/service_bindings_module.o']
variant_providers=[root/p for p in shlex.split((evidence/'variant-providers.txt').read_text())]
assert all(p.is_file() for p in variant_providers)
all_providers=list(set(all_providers+variant_providers))
baseenv=dict(os.environ,ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1',LSAN_OPTIONS='',NANO_VM_TRACE='0')
for label,cc,flags in configs:
 before=snapshot(source+tools+all_providers);dump(label+'-before.json',before)
 env=dict(baseenv,AFFINE_VARIANTS_CC=cc,AFFINE_VARIANTS_CFLAGS=shlex.join(flags),AFFINE_VARIANTS_OBJECTS=shlex.join(map(str,variant_providers)),AFFINE_VARIANTS_LDFLAGS=shlex.join(ldflags),AFFINE_VARIANTS_ARTIFACTS=str(evidence/(label+'-variants')))
 ok=command(label+'-variants',[python,'-m','unittest','-v','tests.test_affine_private_variants'],env,600)
 after_variants=snapshot(source+tools+all_providers);dump(label+'-variants-after.json',after_variants)
 if not ok or before!=after_variants:sys.exit(1)
 env=dict(baseenv,ORDINARY_ADMISSION_CC=cc,ORDINARY_ADMISSION_CFLAGS=shlex.join(flags),ORDINARY_ADMISSION_OBJECTS=shlex.join(map(str,providers)),ORDINARY_ADMISSION_LDFLAGS=shlex.join(ldflags),ORDINARY_ADMISSION_ARTIFACTS=str(evidence/label))
 ok=command(label,[python,'-m','unittest','-v','tests.test_vm_ordinary_admission'],env,600)
 after=snapshot(source+tools+all_providers);dump(label+'-after.json',after)
 if not ok or before!=after:sys.exit(1)
dump('terminal.json',{'status':'PASS','configs':[c[0] for c in configs],'input_pairs_equal':True,'sanitizer_scope':'fixture plus included VM and observed service TU; other providers ordinary','host_hooks':'modeled, not actual FFI or callback ABI'})
