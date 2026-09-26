import hashlib,json,os,pathlib,platform,shlex,shutil,signal,subprocess,sys,time
root=pathlib.Path(sys.argv[1]).resolve(); prior=pathlib.Path(sys.argv[2]).resolve(); out=pathlib.Path(sys.argv[3]).resolve();out.mkdir(exist_ok=False)
store=out/'objects';store.mkdir();darwin=platform.system()=='Darwin'
def save(n,v):(out/n).write_text(json.dumps(v,indent=2)+'\n')
def item(p):
 p=pathlib.Path(p).resolve();b=p.read_bytes();h=hashlib.sha256(b).hexdigest();d=store/h
 if not d.exists():
  previous=prior/'objects'/h
  if previous.is_file():assert hashlib.sha256(previous.read_bytes()).hexdigest()==h;os.link(previous,d)
  else:d.write_bytes(b)
 return dict(sha256=h,bytes=len(b),archive=str(d))
def snapshot(paths):return {str(p):item(p) for p in sorted({pathlib.Path(x).resolve() for x in paths},key=str)}
pin='73b61d06f04cf291a1960dc0b3e2b583ffefcb44'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==pin
assert json.loads((prior/'terminal.json').read_text())['status']=='PASS'
source=json.loads((prior/'source-initial.json').read_text());prepared=json.loads((prior/'provider-prepare-products-after.json').read_text());tools=json.loads((prior/'tools-initial.json').read_text())
for path,fact in {**source,**prepared,**tools}.items():assert item(path)['sha256']==fact['sha256'],path
providers=[pathlib.Path(p) if pathlib.Path(p).is_absolute() else root/p for p in shlex.split((prior/'providers.txt').read_text())]
for p in providers:assert p.is_file(),p
links=shlex.split((prior/'ldflags.txt').read_text())
ordinary='/usr/bin/clang' if darwin else '/bin/gcc';other='/opt/homebrew/opt/llvm/bin/clang' if darwin else '/usr/local/bin/clang'
otherflags=shlex.split(json.loads((prior/'environment.json').read_text())['NMS_NATIVE_CLANG_FLAGS']) if darwin else ['--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13']
configs=[('apple-ordinary' if darwin else 'gcc-ordinary',ordinary,[])]
if not darwin:configs.append(('gcc-sanitizer',ordinary,['-fsanitize=address,undefined','-fno-omit-frame-pointer']))
configs.append(('homebrew-sanitizer' if darwin else 'clang-sanitizer',other,[*otherflags,'-fsanitize=address,undefined','-fno-omit-frame-pointer']))
env=dict(os.environ);env.update(json.loads((prior/'environment.json').read_text()))
for k in ('NANOC','NANOLANG_COMPILER','NANO_VM','NANO_NVM2C','NANO_AOT_RUNTIME','CFLAGS','CPPFLAGS','LDFLAGS','LD_PRELOAD','LD_AUDIT','DYLD_INSERT_LIBRARIES'):env.pop(k,None)
env.update(ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',LSAN_OPTIONS='',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
selected=list(tools)+[other,sys.executable,'/bin/ps'];immutable=list(source)+[str(p) for p in providers]+[p for p in prepared if pathlib.Path(p).is_relative_to(root/'bin')]
tracked=set(subprocess.check_output(['git','ls-files'],cwd=root,text=True).splitlines())
def products():return snapshot(p for d in ('bin','obj','build','tests') for p in (root/d).rglob('*') if p.is_file() and str(p.relative_to(root)) not in tracked)
manifest=[]
def descendants(pid):
 rows=[tuple(map(int,line.split())) for line in subprocess.check_output(['/bin/ps','-axo','pid=,ppid=,pgid='],text=True).splitlines() if line.strip()]
 owned={pid};groups=set();changed=True
 while changed:
  changed=False
  for child,parent,group in rows:
   if parent in owned and child not in owned:owned.add(child);groups.add(group);changed=True
 return groups

def command(label,argv,bound,childenv):
 before=snapshot(immutable+selected);save(label+'-inputs-before.json',before);save(label+'-products-before.json',products())
 status=dict(argv=list(map(str,argv)),bound=bound,status=None,timeout=False,cleanup=[],cwd=str(root));start=time.monotonic();p=None;seen=set()
 save(label+'-status.json',status)
 try:
  with (out/(label+'.log')).open('wb') as log:
   p=subprocess.Popen(status['argv'],cwd=root,env=childenv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   deadline=time.monotonic()+bound
   while True:
    seen.update(descendants(p.pid))
    free=shutil.disk_usage(out).free
    if free<1073741824:status['capacity_stop']=free;raise subprocess.TimeoutExpired(argv,bound)
    remaining=deadline-time.monotonic()
    if remaining<=0:raise subprocess.TimeoutExpired(argv,bound)
    try:status['status']=p.wait(timeout=min(2,remaining));break
    except subprocess.TimeoutExpired:pass
 except BaseException as error:
  status['error']=repr(error);status['timeout']=isinstance(error,subprocess.TimeoutExpired) and 'capacity_stop' not in status
 finally:
  if p is not None:
   seen.update(descendants(p.pid));seen.add(p.pid)
   for sig in (signal.SIGTERM,signal.SIGKILL):
    for pgid in seen:
     if pgid==os.getpgrp():raise RuntimeError('I refuse to signal my own group.')
     try:os.killpg(pgid,sig);status['cleanup'].append(dict(group=pgid,signal=sig.name))
     except ProcessLookupError:pass
    try:p.wait(timeout=5)
    except subprocess.TimeoutExpired:status['cleanup'].append('bounded wait expired')
   status['reaped']=p.poll() is not None
   alive=[]
   for pgid in seen:
    try:os.killpg(pgid,0);alive.append(pgid)
    except ProcessLookupError:pass
   status['remaining_groups']=alive
  status['seconds']=time.monotonic()-start
  if (out/(label+'.log')).exists():status['log']=item(out/(label+'.log'))
  save(label+'-status.json',status)
  after=snapshot(immutable+selected);save(label+'-inputs-after.json',after);save(label+'-products-after.json',products());status['inputs_equal']=before==after
  save(label+'-status.json',status);manifest.append(status);save('manifest.json',manifest)
 print(label,status['status'],round(status['seconds'],3),flush=True)
 assert status['status']==0 and not status.get('error') and status.get('reaped') and not status['remaining_groups'] and status['inputs_equal'],status
 return (out/(label+'.log')).read_text()
save('runner.json',item(__file__));save('scope.json',dict(pin=pin,configs=configs,source_scope='all active tracked inputs excluding historical docs/evidence, selected prepared provider objects and bin tools; complete fresh endpoints; generated module outputs separately mapped and may change',sanitizer_scope='five original fixture helpers rebuild their documented owning TUs; remaining prepared providers ordinary; no full native programs',tools_scope='selected compilers/subtools/runtimes; not full transitive tool closure',cleanup='bounded own observed descendant process groups; fixture also supervises each child'))
queries=[]
if not darwin:
 queries=[('gcc-asan',[ordinary,'-print-file-name=libasan.so']),('gcc-ubsan',[ordinary,'-print-file-name=libubsan.so']),('gcc-cc1',[ordinary,'-print-prog-name=cc1']),('gcc-collect2',[ordinary,'-print-prog-name=collect2'])]
queries += [('clang-asan',[other,*otherflags,'-print-file-name='+('libclang_rt.asan_osx_dynamic.dylib' if darwin else 'libclang_rt.asan.so')])]
if not darwin:
 for n in ('asan.a','asan-preinit.a','ubsan_standalone.a'):queries.append(('clang-'+n,[other,*otherflags,'-print-file-name=libclang_rt.'+n]))
for label,args in queries:
 name=command(label,args,30,env).strip();assert pathlib.Path(name).is_file(),name;selected.append(name)
focus=['tests.test_generic_record_lists.GenericRecordLists.test_checked_storage_scheduler_and_allocation_prefixes','tests.test_generic_record_lists.GenericRecordLists.test_array_intrinsic_and_declaration_identity']
for label,cc,flags in configs:
 reports=out/label;reports.mkdir()
 current=dict(env,NANO_LIST_CC=shlex.join([cc]),NANO_LIST_CFLAGS=shlex.join(['-O0','-g',*flags]),NANO_LIST_LDFLAGS=shlex.join(links),NANO_LIST_OBJECTS=shlex.join(map(str,providers)),NANO_LIST_REPORT_DIR=str(reports),NANO_NATIVE_LIST_CC=ordinary,NANO_NATIVE_LIST_CFLAGS=shlex.join(otherflags if darwin else []),NANO_NATIVE_LIST_LDFLAGS=shlex.join(links))
 save(label+'-config.json',{k:v for k,v in current.items() if k.startswith(('NANO','NMS','ASAN','LSAN','UBSAN'))})
 command(label,[sys.executable,'-m','unittest','-v','-f',*focus],900,current)
save('terminal.json',dict(status='PASS',pin=pin,scope='two original focused parser/checker/owned-lifetime methods under Apple ordinary and Homebrew ASan/UBSan; ordinary remaining prepared providers; no native or full source corpus claim'))
