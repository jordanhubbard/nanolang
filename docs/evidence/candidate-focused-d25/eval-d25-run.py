import hashlib,json,os,pathlib,platform,shlex,shutil,signal,subprocess,sys,tarfile,time
base=pathlib.Path(sys.argv[1]).resolve(); archive=pathlib.Path(sys.argv[2]).resolve();base.mkdir(exist_ok=False)
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def file_identity(p):return dict(sha256=digest(p),bytes=p.stat().st_size,mode=p.stat().st_mode & 0o777)
def mapping(root):return {str(p.relative_to(root)):file_identity(p) for p in root.rglob('*') if p.is_file() and not p.is_symlink()}
darwin=platform.system()=='Darwin';env=dict(os.environ,ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
env.pop('LSAN_OPTIONS',None)
if darwin:env['PATH']='/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin'
make=shutil.which('make',path=env['PATH']);cc='/usr/bin/clang' if darwin else '/usr/bin/gcc';sancc='/opt/homebrew/opt/llvm/bin/clang' if darwin else '/usr/bin/gcc'
sdk=subprocess.check_output(['/usr/bin/xcrun','--show-sdk-path'],text=True).strip() if darwin else ''
save(base/'identity.json',dict(pin='d25c42592b37be6fcf2c87c337ee7faa52f0babe',archive_sha=digest(archive),driver_sha=digest(pathlib.Path(__file__)),host=platform.uname()._asdict(),sdk=sdk,tools={p:digest(pathlib.Path(p).resolve()) for p in [make,cc,sancc,sys.executable]}))
assert shutil.disk_usage(base).free>2*1024**3
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
  sources={n:file_identity(root/n) for n in source_names};save(out/'source-before.json',sources)
  tmp=out/'tmp';tmp.mkdir();env['TMPDIR']=str(tmp);env['NANOLANG_SDK_ROOT']=str(root)
  flags=['-Wall','-Wextra','-Werror','-std=c99','-g','-O1' if san else '-O2','-fPIC','-Isrc','-D_GNU_SOURCE'];ld=['-lm'];native=[]
  if darwin:flags+=['-isysroot',sdk];ld+=['-isysroot',sdk];native+=['-isysroot',sdk]
  if san:
   sf=['-fsanitize=address,undefined','-fno-omit-frame-pointer','-fno-sanitize-recover=all'];flags+=sf;ld+=sf;native+=sf
  env['NANO_NATIVE_TEST_CC']=compiler;env['NANO_CAST_U8_NATIVE_FLAGS']=shlex.join(native)
  args=[make,'-f','Makefile.gnu','-j2','CC='+compiler,'CFLAGS='+shlex.join(flags),'LDFLAGS='+shlex.join(ld),'PYTHON_WITH_YAML='+sys.executable]
  run(name+'-build',[*args,'stage1'],root,out,1200)
  run(name+'-eval',[*args,'test-eval'],root,out,1200)
  after={n:file_identity(root/n) for n in sources};save(out/'source-after.json',after);assert after==sources
  save(out/'products.json',{str(p.relative_to(root)):digest(p) for folder in ['bin','obj'] for p in (root/folder).rglob('*') if p.is_file()})
 tools_after={p:digest(pathlib.Path(p).resolve()) for p in [make,cc,sancc,sys.executable]};save(base/'tools-after.json',tools_after);assert tools_after==json.loads((base/'identity.json').read_text())['tools']
 save(base/'terminal.json',dict(status=0))
except BaseException as e:
 if 'sources' in globals():save(out/'source-after-failure.json',{n:file_identity(root/n) for n in sources})
 save(base/'tools-after-failure.json',{p:digest(pathlib.Path(p).resolve()) for p in [make,cc,sancc,sys.executable]})
 save(base/'terminal.json',dict(status=1,error=repr(e)));raise
