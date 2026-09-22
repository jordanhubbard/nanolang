import hashlib,json,os,pathlib,platform,shlex,shutil,signal,subprocess,sys,tarfile,time
base=pathlib.Path(sys.argv[1]).resolve(); archive=pathlib.Path(sys.argv[2]).resolve();base.mkdir(exist_ok=False)
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def file_identity(p):return dict(sha256=digest(p),bytes=p.stat().st_size,mode=p.stat().st_mode & 0o777)
def mapping(root):return {str(p.relative_to(root)):file_identity(p) for p in root.rglob('*') if p.is_file() and not p.is_symlink()}
darwin=platform.system()=='Darwin';env=dict(os.environ,ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
env.pop('LSAN_OPTIONS',None)
for key in ('CFLAGS','CPPFLAGS','LDFLAGS','SDKROOT','LD_PRELOAD','DYLD_INSERT_LIBRARIES','NANOLANG_COMPILER','NANOC','NANO_VM','NANO_NVM2C','NANO_AOT_RUNTIME'):env.pop(key,None)
if darwin:env['PATH']='/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin'
make=shutil.which('make',path=env['PATH']);cc='/usr/bin/clang' if darwin else '/usr/bin/gcc';sancc='/opt/homebrew/opt/llvm/bin/clang' if darwin else '/usr/bin/gcc'
sdk=subprocess.check_output(['/usr/bin/xcrun','--show-sdk-path'],text=True).strip() if darwin else ''
save(base/'identity.json',dict(pin='0eb14852e6718881f88f26f2afc94aa496ddefee',archive_sha=digest(archive),driver_sha=digest(pathlib.Path(__file__)),host=platform.uname()._asdict(),sdk=sdk,tools={p:digest(pathlib.Path(p).resolve()) for p in [make,cc,sancc,sys.executable]}))
assert shutil.disk_usage(base).free>2*1024**3
scope=json.loads(pathlib.Path(__file__).with_name('scope.json').read_text());assert scope['pin']=='0eb14852e6718881f88f26f2afc94aa496ddefee'
save(base/'scope-identity.json',dict(sha256=digest(pathlib.Path(__file__).with_name('scope.json'))))
fragment=base/'qualification.mk'
fragment.write_text('.PHONY: qualification-array-refusals\nqualification-array-refusals: $(NANOVIRT_OBJECTS) $(NANOVM_OBJECTS) $(NANOISA_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS)\n\t$(CC) $(CFLAGS) -o tests/nanovirt/test_codegen tests/nanovirt/test_codegen.c $(NANOVIRT_OBJECTS) $(NANOVM_OBJECTS) $(NANOISA_OBJECTS) $(COMMON_OBJECTS) $(RUNTIME_OBJECTS) $(LDFLAGS)\n\t./tests/nanovirt/test_codegen --array-arithmetic-refusals\n\trm -f tests/nanovirt/test_codegen\n')
fragment_before=file_identity(fragment);save(base/'fragment-before.json',fragment_before)
records=[]
def run(label,args,root,out,bound):
 status=dict(label=label,argv=args,bound=bound,timeout=False);start=time.monotonic();seen=set();p=None
 try:
  with (out/(label+'.log')).open('wb') as log:
   p=subprocess.Popen(args,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   status['pid']=p.pid;save(out/(label+'.json'),status)
   while p.poll() is None:
    rows=subprocess.check_output(['ps','-axo','pid=,ppid='],text=True);pairs=[tuple(map(int,r.split())) for r in rows.splitlines() if len(r.split())==2];parents={p.pid};changed=True
    while changed:
     child={a for a,b in pairs if b in parents};new=child-parents;changed=bool(new);parents|=new
    seen|=parents
    if time.monotonic()-start>bound:status['timeout']=True;break
    if shutil.disk_usage(base).free<1536*1024**2:status['capacity_stop']=True;break
    time.sleep(.2)
 finally:
  if p:
   for sig in (signal.SIGTERM,signal.SIGKILL):
    for pid in seen|{p.pid}:
     try:os.killpg(pid,sig)
     except ProcessLookupError:pass
    try:p.wait(timeout=5)
    except subprocess.TimeoutExpired:pass
   status['returncode']=p.poll();status['remaining_groups']=[]
   for pid in seen|{p.pid}:
    try:os.killpg(pid,0);status['remaining_groups'].append(pid)
    except ProcessLookupError:pass
  status['seconds']=time.monotonic()-start;save(out/(label+'.json'),status);records.append(status);save(base/'phases.json',records)
 print(label,status,flush=True)
 assert status.get('returncode')==0 and not status['timeout'] and not status.get('capacity_stop') and not status['remaining_groups'],status
 text=(out/(label+'.log')).read_text(errors='replace');assert 'ERROR: AddressSanitizer' not in text and 'runtime error:' not in text and 'ERROR: LeakSanitizer' not in text
try:
 for name,compiler,san in [('ordinary',cc,False),('sanitized',sancc,True)]:
  out=base/name;out.mkdir();root=out/'source'
  root.mkdir()
  with tarfile.open(archive) as tf:tf.extractall(root)
  with tarfile.open(archive) as tf:source_names=[m.name for m in tf if m.isfile()]
  for n,v in scope['files'].items():assert digest(root/n)==v['sha256'],n
  sources={n:file_identity(root/n) for n in source_names};save(out/'source-before.json',sources)
  tmp=out/'tmp';tmp.mkdir();env['TMPDIR']=str(tmp);env['NANOLANG_SDK_ROOT']=str(root)
  flags=['-Wall','-Wextra','-Werror','-std=c99','-g','-O1' if san else '-O2','-fPIC','-Isrc','-D_GNU_SOURCE'];ld=['-lm'];native=[]
  if darwin:flags+=['-isysroot',sdk];ld+=['-isysroot',sdk];native+=['-isysroot',sdk]
  if san:
   sf=['-fsanitize=address,undefined','-fno-omit-frame-pointer','-fno-sanitize-recover=all'];flags+=sf;ld+=sf;native+=sf
  env['NANO_NATIVE_TEST_CC']=compiler;env['NANO_CAST_U8_NATIVE_FLAGS']=shlex.join(native)
  args=[make,'-f','Makefile.gnu','-f',str(fragment),'-j2','CC='+compiler,'CFLAGS='+shlex.join(flags),'LDFLAGS='+shlex.join(ld),'PYTHON_WITH_YAML='+sys.executable]
  run(name+'-build',[*args,'stage1'],root,out,1200)
  providers={str(p.relative_to(root)):file_identity(p) for folder in ('bin','obj') for p in (root/folder).rglob('*') if p.is_file()}
  save(out/'providers-before.json',providers)
  for target,binary in [('test-typechecker','test_typechecker'),('test-eval','test_eval'),('qualification-array-refusals','nanovirt/test_codegen')]:
   shims=out/(target+'-shims');shims.mkdir();shim=shims/'rm'
   shim.write_text('#!'+sys.executable+'\n'+"import os,sys,shutil,hashlib,json\nfrom pathlib import Path\np=Path(os.environ['FULL_EVAL_BINARY']).resolve()\nif p.is_file() and any(Path(a).resolve()==p for a in sys.argv[1:] if not a.startswith('-')):\n d=Path(os.environ['FULL_EVAL_RETAIN']);shutil.copy2(p,d)\n h=hashlib.sha256(p.read_bytes()).hexdigest();assert hashlib.sha256(d.read_bytes()).hexdigest()==h\n d.with_suffix('.json').write_text(json.dumps({'sha256':h,'bytes':p.stat().st_size,'mode':p.stat().st_mode&511}))\nos.execv('/bin/rm',['/bin/rm',*sys.argv[1:]])\n")
   shim.chmod(0o755);save(out/(target+'-removal-wrapper-before.json'),file_identity(shim))
   oldpath=env['PATH'];env['PATH']=str(shims)+os.pathsep+oldpath
   env['FULL_EVAL_BINARY']=str(root/('tests/'+binary));env['FULL_EVAL_RETAIN']=str(out/(pathlib.Path(binary).name+'-retained'))
   try:run(name+'-'+target,[*args,target],root,out,1200)
   finally:
    env['PATH']=oldpath
    providers_after={n:file_identity(root/n) for n in providers}
    save(out/(target+'-providers-after.json'),providers_after)
    assert providers_after==providers,'existing provider bytes changed during '+target
    save(out/(target+'-removal-wrapper-after.json'),file_identity(shim));assert file_identity(shim)==json.loads((out/(target+'-removal-wrapper-before.json')).read_text())
    failed=root/('tests/'+binary)
    if failed.is_file():
     shutil.copy2(failed,out/(pathlib.Path(binary).name+'-retained'));save(out/(pathlib.Path(binary).name+'-retained.json'),file_identity(failed))
  after={n:file_identity(root/n) for n in sources};save(out/'source-after.json',after);assert after==sources
  save(out/'products.json',{str(p.relative_to(root)):digest(p) for folder in ['bin','obj'] for p in (root/folder).rglob('*') if p.is_file()})
 save(base/'fragment-after.json',file_identity(fragment));assert file_identity(fragment)==fragment_before
 tools_after={p:digest(pathlib.Path(p).resolve()) for p in [make,cc,sancc,sys.executable]};save(base/'tools-after.json',tools_after);assert tools_after==json.loads((base/'identity.json').read_text())['tools']
 save(base/'terminal.json',dict(status=0))
except BaseException as e:
 save(base/'fragment-after-failure.json',file_identity(fragment))
 if 'sources' in globals():save(out/'source-after-failure.json',{n:file_identity(root/n) for n in sources})
 save(base/'tools-after-failure.json',{p:digest(pathlib.Path(p).resolve()) for p in [make,cc,sancc,sys.executable]})
 save(base/'terminal.json',dict(status=1,error=repr(e)));raise
