import hashlib,json,os,pathlib,shlex,signal,subprocess,time
root=pathlib.Path('/home/jkh/Src/nanolang-record-lists-qualified-cc606')
source=pathlib.Path('/home/jkh/Src/nanolang-generic-list-mutations')
prior=pathlib.Path('/tmp/nanolang-record-lists-cc606-linux-prepare')
out=pathlib.Path('/tmp/nanolang-evaluator-lifetime-192e-linux');out.mkdir(exist_ok=False)
toolroot=out/'compiler-root';toolroot.mkdir();(toolroot/'bin').mkdir()
for name in ('src','src_nano','modules','stdlib','include','lib','scripts','share','locales'):
 if (root/name).exists():(toolroot/name).symlink_to(root/name, target_is_directory=True)
store=out/'objects';store.mkdir()
def save(n,x):(out/n).write_text(json.dumps(x,indent=2)+'\n')
def record(p):
 p=pathlib.Path(p).resolve(); b=p.read_bytes();h=hashlib.sha256(b).hexdigest();d=store/h
 if not d.exists():
  old=prior/'objects'/h
  if old.is_file():assert hashlib.sha256(old.read_bytes()).hexdigest()==h;os.link(old,d)
  else:d.write_bytes(b)
 return dict(sha256=h,bytes=len(b),archive=str(d))
def mapping(paths):return {str(p):record(p) for p in sorted(set(map(pathlib.Path,paths)))}
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()=='192ef1c642ffa5827105e5ae93d7b25d7849e435'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()=='cc60606c9ca92fea83364021cd7268918cdd5c69'
assert not subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True).strip()
base=json.loads((prior/'source-initial.json').read_text());tools=json.loads((prior/'tools-initial.json').read_text());products=json.loads((prior/'build-products-after.json').read_text())
for path,fact in {**base,**tools,**products}.items():assert record(path)['sha256']==fact['sha256'],path
paths=list(base)+list(tools)+list(products)
# I verify every diagnostic compile source against cc606, allowing only the reviewed two source edits.
extra=[]
for path,fact in base.items():
 rel=pathlib.Path(path).relative_to(root);p=source/rel
 if rel.parts[0]=='docs':continue
 extra.append(p)
 if str(rel) not in ('src/eval.c','src/env_record_lists.inc'):assert record(p)['sha256']==fact['sha256'],str(rel)
paths+=extra
save('runner.json',record(__file__));save('inputs-before.json',mapping(paths))
env=dict(os.environ,CC='/bin/gcc',NANO_BUILD_CACHE=str(out/'module-cache'),NANO_MODULE_PATH=str(root/'modules'),NANO_AS_CAPTURE_HELPER=str(root/'bin/nano_as_capture.so'),ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',LSAN_OPTIONS='',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
for k in ('LD_PRELOAD','LD_AUDIT','DYLD_INSERT_LIBRARIES','NANOC','NANOLANG_COMPILER','NANO_CC','NANO_CFLAGS','CFLAGS','CPPFLAGS','LDFLAGS','NANO_CAPTURE_TIMEOUT_MS'):env.pop(k,None)
assert record(env['NANO_AS_CAPTURE_HELPER'])['sha256']==products[env['NANO_AS_CAPTURE_HELPER']]['sha256']
save('external-layout.json',{str(p):str(p.resolve()) for p in toolroot.iterdir() if p.is_symlink()})
save('environment.json',{k:v for k,v in env.items() if k.startswith(('NANO','ASAN','LSAN','UBSAN')) or k=='CC'})
def command(name,args,cwd,bound):
 state=dict(argv=list(map(str,args)),cwd=str(cwd),bound=bound,timeout=False,cleanup=[]);start=time.monotonic();p=None
 save(name+'-before.json',mapping(paths))
 try:
  with (out/(name+'.log')).open('wb') as log:
   p=subprocess.Popen(state['argv'],cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:state['returncode']=p.wait(timeout=bound)
   except subprocess.TimeoutExpired:
    state['timeout']=True
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
  save(name+'-status.json',state);save(name+'-after.json',mapping(paths))
 print(name,state,flush=True)
 assert state.get('reaped') and state.get('group_gone') and not state['timeout'] and not state.get('exception'),state
 return state['returncode']
fragment=out/'variables.mk';fragment.write_text('include Makefile.gnu\n.PHONY: diagnostic-variables\ndiagnostic-variables:\n\t@printf "%s\\n" "$(COMPILER_OBJECTS)" "$(CFLAGS)" "$(LDFLAGS)"\n')
assert command('variables',['make','-s','-f',fragment,'diagnostic-variables','CC=/bin/gcc'],root,30)==0
lines=(out/'variables.log').read_text().splitlines();assert len(lines)==3,lines
objects=shlex.split(lines[0]);flags=shlex.split(lines[1]);links=shlex.split(lines[2]);save('make-facts.json',dict(objects=objects,flags=flags,links=links))
for p in objects:assert str(root/p) in products,p
original=pathlib.Path('/tmp/nanolang-evaluator-lifetime-c1df-linux')
for unit in ('env','eval'):
 p=original/f'{unit}.o';fact=json.loads((original/(unit+'-object.json')).read_text())
 assert record(p)['sha256']==fact['sha256']
 save(unit+'-reused-object.json',record(p));paths.append(p)
linked=[original/'env.o' if p=='obj/env.o' else original/'eval.o' if p=='obj/eval.o' else root/p for p in objects]
compiler=toolroot/'bin/nanoc-diagnostic'
assert command('link',['/bin/gcc',*flags,'-o',compiler,*linked,*links],root,120)==0
save('binary.json',record(compiler));save('actual-provider-inputs.json',mapping(linked))
save('inputs-before-run.json',mapping(paths))
result=command('full-parser',[compiler,root/'src_nano/parser_driver.nano','-o',out/'parser'],toolroot,120)
save('diagnostic-terminal.json',dict(status=result,scope='one full original parser graph, unchanged ten-second shadow supervision; timing is diagnostic only'))
after=mapping(paths);save('inputs-after.json',after)
assert after==json.loads((out/'inputs-before-run.json').read_text())
save('generated-products.json',mapping(p for p in toolroot.rglob('*') if p.is_file() and not p.is_symlink() and 'obj' in p.relative_to(toolroot).parts))
