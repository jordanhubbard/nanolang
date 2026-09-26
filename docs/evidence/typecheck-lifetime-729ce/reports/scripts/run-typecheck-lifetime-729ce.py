import hashlib,json,os,pathlib,shlex,signal,subprocess,time
root=pathlib.Path('/home/jkh/Src/nanolang-record-lists-qualified-40870')
source=pathlib.Path('/home/jkh/Src/nanolang-list-typecheck-shadow-timing')
prior=pathlib.Path('/tmp/nanolang-record-lists-40870-linux-prepare')
out=pathlib.Path('/tmp/nanolang-typecheck-lifetime-729ce-linux');out.mkdir(exist_ok=False)
toolroot=out/'compiler-root';toolroot.mkdir();(toolroot/'bin').mkdir()
for name in ('src','src_nano','modules','stdlib','include','lib','scripts','share','locales'):
 if (root/name).exists():(toolroot/name).symlink_to(root/name, target_is_directory=True)
store=out/'objects';store.mkdir()
assert __import__('shutil').disk_usage(out).free >= 2147483648, 'I require2GiB before diagnostic preparation.'
def save(n,x):(out/n).write_text(json.dumps(x,indent=2)+'\n')
def record(p):
 p=pathlib.Path(p).resolve(); b=p.read_bytes();h=hashlib.sha256(b).hexdigest();d=store/h
 if not d.exists():
  old=prior/'objects'/h
  if old.is_file():assert hashlib.sha256(old.read_bytes()).hexdigest()==h;os.link(old,d)
  else:d.write_bytes(b)
 return dict(sha256=h,bytes=len(b),archive=str(d))
def mapping(paths):return {str(p):record(p) for p in sorted(set(map(pathlib.Path,paths)))}
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()=='729ce12a874b9fa2ba0ad8950b00dd90c34d3a3d'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()=='40870d07805c85c2a6600c7d02b8440cd547f57a'
assert not subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True).strip()
base=json.loads((prior/'source-initial.json').read_text());tools=json.loads((prior/'tools-initial.json').read_text());products=json.loads((prior/'build-products-after.json').read_text())
for path,fact in {**base,**tools,**products}.items():assert record(path)['sha256']==fact['sha256'],path
paths=list(base)+list(tools)+list(products)
# I verify every diagnostic compile source against40870, allowing only reviewed observer/header dependency edits.
extra=[]
for path,fact in base.items():
 rel=pathlib.Path(path).relative_to(root);p=source/rel
 if rel.parts[0]=='docs':continue
 extra.append(p)
 if str(rel) not in ('src/env.c','src/eval.c','src/typechecker.c','src/env_record_lists.inc','src/env_signature_snapshot.inc','src/typechecker_nominal_arrays.inc','Makefile.gnu'):assert record(p)['sha256']==fact['sha256'],str(rel)
paths+=extra+[source/'src/evaluator_timing_private.h']
save('runner.json',record(__file__));save('inputs-before.json',mapping(paths))
env=dict(os.environ,CC='/bin/gcc',NANO_BUILD_CACHE=str(out/'module-cache'),NANO_MODULE_PATH=str(root/'modules'),NANO_AS_CAPTURE_HELPER=str(root/'bin/nano_as_capture.so'),ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',LSAN_OPTIONS='',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
for k in ('LD_PRELOAD','LD_AUDIT','DYLD_INSERT_LIBRARIES','NANOC','NANOLANG_COMPILER','NANO_CC','NANO_CFLAGS','CFLAGS','CPPFLAGS','LDFLAGS','NANO_CAPTURE_TIMEOUT_MS'):env.pop(k,None)
assert record(env['NANO_AS_CAPTURE_HELPER'])['sha256']==products[env['NANO_AS_CAPTURE_HELPER']]['sha256']
save('external-layout.json',{str(p):str(p.resolve()) for p in toolroot.iterdir() if p.is_symlink()})
save('environment.json',{k:v for k,v in env.items() if k.startswith(('NANO','ASAN','LSAN','UBSAN')) or k=='CC'})
def current_products():
 return mapping(p for p in out.rglob('*') if p.is_file() and not p.is_symlink() and store not in p.parents and p.suffix not in ('.json','.log','.jsonl'))
def command(name,args,cwd,bound):
 state=dict(argv=list(map(str,args)),cwd=str(cwd),bound=bound,timeout=False,cleanup=[]);start=time.monotonic();p=None
 before=mapping(paths);save(name+'-before.json',before);save(name+'-products-before.json',current_products())
 try:
  with (out/(name+'.log')).open('wb') as log:
   p=subprocess.Popen(state['argv'],cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:
    deadline=time.monotonic()+bound
    while True:
     free=__import__('shutil').disk_usage(out).free
     with (out/'capacity.jsonl').open('a') as capacity:
      capacity.write(json.dumps({'phase':name,'seconds':time.monotonic()-start,'free_bytes':free})+'\n')
     if free < 1073741824:
      state['capacity_stop']=free
      raise subprocess.TimeoutExpired(args,bound)
     remaining=deadline-time.monotonic()
     if remaining<=0:raise subprocess.TimeoutExpired(args,bound)
     try:state['returncode']=p.wait(timeout=min(10,remaining));break
     except subprocess.TimeoutExpired:
      if time.monotonic()>=deadline:raise
   except subprocess.TimeoutExpired:
    state['timeout']='capacity_stop' not in state
    for sig in (signal.SIGTERM,signal.SIGKILL):
     try:os.killpg(p.pid,sig)
     except ProcessLookupError:pass
     try:p.wait(timeout=5)
     except subprocess.TimeoutExpired:state['cleanup'].append('wait expired')
    state['returncode']=p.poll()
   state['reaped']=p.poll() is not None
   try:os.killpg(p.pid,0);state['group_gone']=False;os.killpg(p.pid,signal.SIGKILL)
   except ProcessLookupError:state['group_gone']=True
 except BaseException as error:state['exception']=repr(error)
 finally:
  state['seconds']=time.monotonic()-start
  if (out/(name+'.log')).exists():state['log']=record(out/(name+'.log'))
  after=mapping(paths);state['inputs_equal']=before==after
  save(name+'-status.json',state);save(name+'-after.json',after);save(name+'-products-after.json',current_products())
 print(name,state,flush=True)
 assert state.get('reaped') and state.get('group_gone') and not state['timeout'] and not state.get('exception') and not state.get('capacity_stop') and state['inputs_equal'],state
 return state['returncode']
fragment=out/'variables.mk';fragment.write_text('include Makefile.gnu\n.PHONY: diagnostic-variables\ndiagnostic-variables:\n\t@printf "%s\\n" "$(COMPILER_OBJECTS)" "$(CFLAGS)" "$(LDFLAGS)"\n')
assert command('variables',['make','-s','-f',fragment,'diagnostic-variables','CC=/bin/gcc'],root,30)==0
lines=(out/'variables.log').read_text().splitlines();assert len(lines)==3,lines
objects=shlex.split(lines[0]);flags=shlex.split(lines[1]);links=shlex.split(lines[2]);save('make-facts.json',dict(objects=objects,flags=flags,links=links))
for p in objects:assert str(root/p) in products,p
for unit in ('env','eval','typechecker'):
 args=['/bin/gcc',*flags,'-DNANO_EVALUATOR_LIFETIME_TIMING']
 if unit=='eval':args+=['-ffp-contract=off','-fno-fast-math']
 args+=['-c',str(source/'src'/f'{unit}.c'),'-o',str(out/f'{unit}.o')]
 assert command(unit+'-build',args,root,120)==0
 save(unit+'-object.json',record(out/f'{unit}.o'))
 paths.append(out/f'{unit}.o')
linked=[out/pathlib.Path(p).name if p in ('obj/env.o','obj/eval.o','obj/typechecker.o') else root/p for p in objects]
compiler=toolroot/'bin/nanoc-diagnostic'
assert command('link',['/bin/gcc',*flags,'-o',compiler,*linked,*links],root,120)==0
save('binary.json',record(compiler));save('actual-provider-inputs.json',mapping(linked))
save('inputs-before-run.json',mapping(paths))
result=command('full-typecheck',[compiler,root/'src_nano/typecheck_driver.nano','-o',out/'typecheck'],toolroot,120)
save('diagnostic-terminal.json',dict(status=result,scope='one full original typecheck_driver graph, unchanged ten-second shadow supervision; timing is diagnostic only'))
after=mapping(paths);save('inputs-after.json',after)
assert after==json.loads((out/'inputs-before-run.json').read_text())
save('generated-products.json',mapping(p for p in toolroot.rglob('*') if p.is_file() and not p.is_symlink() and 'obj' in p.relative_to(toolroot).parts))
