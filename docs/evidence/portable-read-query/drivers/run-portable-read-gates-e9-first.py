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
make=shutil.which('make');tools=[ordinary,other,python,make,shutil.which('git')]
source=[p for p in (root/'src').rglob('*') if p.is_file() and p.suffix in ('.c','.h','.inc')]
source += [root/p for p in ['Makefile.gnu','tests/nanoisa/test_portable_host_plan.c','tests/test_portable_host_plan.py','tests/nanoisa/test_verifier.c','tests/nanoisa/test_verifier_cleanup.c','tests/nanoisa/test_nanoisa.c','docs/NANOISA_PORTABLE_READ_TEXT_LINKAGE.md']]
fragment=evidence/'prepare.mk';fragment.write_text('include Makefile.gnu\n.PHONY: portable-prepare\nportable-prepare: $(NANOISA_OBJECTS) $(NANOISA_UTF8)\n\t@printf \'%s\\n\' \'$(NANOISA_OBJECTS) $(NANOISA_UTF8)\' > '+str(evidence/'providers.txt')+'\n\t@printf \'%s\\n\' \'$(CFLAGS)\' > '+str(evidence/'cflags.txt')+'\n\t@printf \'%s\\n\' \'$(LDFLAGS)\' > '+str(evidence/'ldflags.txt')+'\n')
dump('driver.json',item(__file__));dump('prepare-fragment.json',item(fragment));dump('configuration.json',{'platform':platform,'configs':configs,'source_scope':'C/header/inc source tree plus exact fixtures/Make/contract','tools_scope':'named compiler, Python, Make, Git; not complete transitive system tools','sanitizers':'four query/decoder/verifier/type TUs + fixture; ordinary providers'})
for label,args in [('host',['uname','-a']),('head',['git','rev-parse','HEAD']),('ordinary-version',[ordinary,'--version']),('other-version',[other,*otherflags,'--version'])]:
 if not command(label,args):sys.exit(1)
if platform!='linux':
 for label,args in [('sdk',['xcrun','--show-sdk-path']),('sdk-version',['xcrun','--show-sdk-version'])]:
  if not command(label,args):sys.exit(1)
# Actual supported sanitizer runtime/compiler-subtool paths; retain those that resolve.
queries=[('ordinary-asan',[ordinary,'-print-file-name=libasan.so']),('ordinary-ubsan',[ordinary,'-print-file-name=libubsan.so']),('ordinary-cc1',[ordinary,'-print-prog-name=cc1'])] if platform=='linux' else []
queries += [('other-asan',[other,*otherflags,'-print-file-name='+('libclang_rt.asan-aarch64.so' if platform=='linux' else 'libclang_rt.asan_osx_dynamic.dylib')])]
for label,args in queries:
 if not command(label,args):sys.exit(1)
 path=(evidence/(label+'.log')).read_text().strip()
 if pathlib.Path(path).is_file():tools.append(path)
 else:raise RuntimeError('I require the exact runtime/tool path: '+path)
before=snapshot(source+tools);dump('prepare-before.json',before)
ok=command('prepare',[make,'-f',str(fragment),'-j4','CC='+ordinary,'portable-prepare'],timeout=900)
after=snapshot(source+tools);dump('prepare-after.json',after)
if not ok or before!=after:sys.exit(1)
providers=[root/p for p in shlex.split((evidence/'providers.txt').read_text())]
ldflags=shlex.split((evidence/'ldflags.txt').read_text());cflags=shlex.split((evidence/'cflags.txt').read_text())
excluded={'verifier.o','verifier_types.o','vm_decode.o'}
filtered=[p for p in providers if p.name not in excluded]
baseenv=dict(os.environ,ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1',LSAN_OPTIONS='')
for label,cc,flags in configs:
 before=snapshot(source+tools+providers);dump(label+'-before.json',before)
 env=dict(baseenv,PORTABLE_READ_CC=cc,PORTABLE_READ_CFLAGS=shlex.join(flags),PORTABLE_READ_OBJECTS=shlex.join(map(str,filtered)),PORTABLE_READ_LDFLAGS=shlex.join(ldflags),PORTABLE_READ_ARTIFACTS=str(evidence/label))
 ok=command(label,[python,'-m','unittest','-v','tests.test_portable_host_plan'],env,600)
 after=snapshot(source+tools+providers);dump(label+'-after.json',after)
 if not ok or before!=after:sys.exit(1)
# Exact unchanged generic verifier, allocation-cleanup and ISA fixture recipes;
# keep their fresh binaries externally instead of the Make recipe's removal.
for label,fixture,selected in [('generic-verifier','tests/nanoisa/test_verifier.c',providers),('verifier-cleanup','tests/nanoisa/test_verifier_cleanup.c',[p for p in providers if p.name!='verifier.o']),('nanoisa','tests/nanoisa/test_nanoisa.c',providers)]:
 before=snapshot(source+tools+providers);dump(label+'-before.json',before)
 binary=evidence/(label+'-binary')
 ok=command(label+'-build',[ordinary,*cflags,'-Isrc/nanoisa','-Isrc/nanoisa/module',fixture,*selected,*ldflags,'-o',binary],baseenv)
 if ok:ok=command(label+'-run',[binary],baseenv)
 if binary.exists():dump(label+'-binary.json',item(binary))
 after=snapshot(source+tools+providers);dump(label+'-after.json',after)
 if not ok or before!=after:sys.exit(1)
dump('terminal.json',{'status':'PASS','phases':len(configs)+3,'query_configs':[c[0] for c in configs],'input_pairs_equal':True})
